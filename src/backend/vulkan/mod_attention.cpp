// Vulkan backend: KV store, decode / prefill attention and the Q/gate unpack. Attention runs the tiled kernels of fa.slang (fa_attention below),
// or with vulkan_attn_ref the portable reference kernel ref_attn_decode: row m attends causally to positions 0..pos[m] of slot slot[m], so
// prefill is the same dispatch over a run of ascending positions
#include "mod_common.h"
#include "mod_attention.h"
#include "engine/modules/attention.h"
#include "engine/modules/norm.h"
#include "engine/modules/rope.h"
#include "engine/backend/tiers.h"
#include "engine/runtime/options.h"
#include <algorithm>
#include <cmath>

// The option table lives in the host library, which the module-only smoke binaries do not link: bound weakly, as in mod_flash.cpp
namespace engine::runtime {
__attribute__(( weak )) Options& options();
}

namespace engine::modules {

namespace {
namespace vki = ::engine::backend::impl;
}

void Attention::store_kv_cache(
	const pod::KVRef& kv,
	const pod::TensorView& k,
	const pod::TensorView& v,
	size_t slot_stride,
	int layer_idx,
	const pod::BatchRows* d_rows,
	int count,
	int num_kv_heads,
	int head_dim,
	int max_seq,
	backend::stream_t stream
) {
	ENGINE_ASSERT( kv.ring == 0 || kv.ring == max_seq, "Attention::store_kv_cache: a ring cache has max_seq rows per layer" );
	const int total_elems = num_kv_heads * head_dim;
	if ( kv.kq != pod::KVQ::F16 || kv.vq != pod::KVQ::F16 ) {
		// Quantized formats: one invocation per quantization unit (kv_codec.slang), K and V each in their own format
		auto store = [&]( void* data, void* scales, const pod::TensorView& src, pod::KVQ q ) {
			const int unit = ( q == pod::KVQ::I8 || q == pod::KVQ::Q4 ) ? 64 : ( q == pod::KVQ::FP4 ? 16 : 1 );
			ENGINE_ASSERT( total_elems % unit == 0, "KV store: row width must be a multiple of the quantization group" );
			vki::KvStoreQParams p{ vki::dp( data ), vki::dp( scales ), vki::dp( src.data ), vki::rows_slot( d_rows ), vki::rows_pos( d_rows ),
				static_cast<int64_t>( slot_stride ), layer_idx, count, total_elems, max_seq, static_cast<int>( q ), kv.ring };
			vki::launch( "kv_store_q", p, stream, ( count * ( total_elems / unit ) + 127 ) / 128 );
		};
		store( kv.k, kv.ks, k, kv.kq );
		store( kv.v, kv.vs, v, kv.vq );
		return;
	}
	const int groups = ( count * total_elems + 127 ) / 128;
	vki::KvStoreParams pk{ vki::dp( kv.k ), vki::dp( k.data ), vki::rows_slot( d_rows ), vki::rows_pos( d_rows ),
		static_cast<int64_t>( slot_stride ), layer_idx, count, total_elems, max_seq, kv.ring };
	vki::launch( "kv_store_f16", pk, stream, groups );
	vki::KvStoreParams pv = pk;
	pv.data = vki::dp( kv.v );
	pv.src = vki::dp( v.data );
	vki::launch( "kv_store_f16", pv, stream, groups );
}

// One kernel for FP16 caches (attn_prep.slang, bit for bit the three steps); other formats and heads over 512 take the three steps, the norms
// as one run of independent ops
void Attention::norm_rope_store( const pod::KVRef& kv, pod::TensorView& q, pod::TensorView& k, pod::TensorView& v, const pod::TensorView& q_norm,
	const pod::TensorView& k_norm, float eps, int num_q_heads, int num_kv_heads, int head_dim, int rot_dim, int rot_pairs, float freq_base,
	size_t slot_stride, int layer_idx, const pod::BatchRows* d_rows, int count, int max_seq, backend::stream_t stream ) {
	const bool f16 = num_kv_heads == 0 || ( kv.kq == pod::KVQ::F16 && kv.vq == pod::KVQ::F16 );
	if ( f16 && head_dim <= 512 && rot_dim <= head_dim && rot_dim / 2 <= 256 ) {
		ENGINE_ASSERT( kv.ring == 0 || kv.ring == max_seq, "Attention::norm_rope_store: a ring cache has max_seq rows per layer" );
		vki::NormRopeStoreParams p{ vki::dp( q.data ), vki::dp( k.data ), vki::dp( v.data ), vki::dp( q_norm.data ), vki::dp( k_norm.data ), vki::dp( kv.k ),
			vki::dp( kv.v ), vki::rows_slot( d_rows ), vki::rows_pos( d_rows ), static_cast<int64_t>( slot_stride ), num_q_heads, num_kv_heads, head_dim, rot_dim,
			rot_pairs, layer_idx, max_seq, kv.ring, eps, freq_base, 0.0f };
		vki::launch( "norm_rope_store", p, stream, count, num_q_heads + 2 * num_kv_heads );
		return;
	}
	backend::concurrent_begin( stream );
	RMSNorm::forward_heads( q, q_norm, num_q_heads, head_dim, eps, 0.0f, stream );
	if ( num_kv_heads > 0 ) {
		backend::concurrent_next( stream );
		RMSNorm::forward_heads( k, k_norm, num_kv_heads, head_dim, eps, 0.0f, stream );
		backend::concurrent_next( stream );
		RMSNorm::forward_heads( v, pod::TensorView{}, num_kv_heads, head_dim, eps, 0.0f, stream );
	}
	backend::concurrent_end( stream );
	RoPE::forward( q, k, num_q_heads, num_kv_heads, head_dim, rot_dim, d_rows, count, freq_base, stream, false, rot_pairs );
	if ( num_kv_heads > 0 ) store_kv_cache( kv, k, v, slot_stride, layer_idx, d_rows, count, num_kv_heads, head_dim, max_seq, stream );
}

namespace vk_attn {

void reference( const pod::KVRef& kv, const pod::TensorView& q, const pod::TensorView& gate, pod::TensorView& out, size_t slot_stride, int layer_idx,
	const pod::BatchRows* d_rows, int count, int num_q_heads, int num_kv_heads, int head_dim, int max_seq, backend::stream_t stream, float scale, int window ) {
	ENGINE_ASSERT( kv.ring == 0 || ( window > 0 && window <= kv.ring && kv.ring == max_seq ), "Attention (Vulkan): a ring cache holds a sliding window of max_seq rows" );
	ENGINE_ASSERT( head_dim <= 512 && num_kv_heads > 0 && num_q_heads % num_kv_heads == 0, "Attention (Vulkan): head_dim <= 512 and GQA only" );
	vki::RefAttnParams p{ vki::dp( q.data ), vki::dp( kv.k ), vki::dp( kv.v ), vki::dp( gate.data ), vki::dp( out.data ), vki::rows_slot( d_rows ), vki::rows_pos( d_rows ),
		static_cast<int64_t>( slot_stride ), layer_idx, num_q_heads, num_kv_heads, head_dim, max_seq, gate.data != nullptr ? 1 : 0, scale > 0.0f ? scale : 1.0f / std::sqrt( static_cast<float>( head_dim ) ),
		vki::dp( kv.ks ), vki::dp( kv.vs ), static_cast<int>( kv.kq ), static_cast<int>( kv.vq ), window, kv.ring,
		vki::rows_key_end( d_rows ), vki::rows_bidir( d_rows ) };
	ENGINE_ASSERT( kv.vq != pod::KVQ::Q4 || head_dim % 64 == 0, "Attention (Vulkan): Q4 values need head_dim % 64 == 0" );
	vki::launch( "ref_attn_decode", p, stream, num_q_heads, count );
}

// The tiled attention (fa.slang). one_slot: the rows share a slot (a prefill chunk), so a workgroup takes up to 16 / (q heads per KV head) of
// them; else a row each. A group's keys go to `splits` workgroups (the kernel shares out the range it finds, so a decode graph replays at any
// length), as many as keep about kTargetGroups busy within what the range can reach: the window (plus a chunk and an image), the prefill's
// key_hint, or max_seq. The splits' partials take a fixed scratch, sized once (a captured forward must never see it move), which also bounds
// their number. A decode or short verify on an FP16 cache with head_dim 256 or 512 takes fa_decode instead: a group is every row of the call
// (one_slot) or one row, its query vectors over `parts` workgroups of at most kDecodeVectors, when two of those hold them
constexpr int kFaPoolWords = 10496, kTargetGroups = 160, kMaxSplits = 64;
constexpr size_t kFaScratch = size_t( 16 ) << 20;

void tiled( const pod::KVRef& kv, const pod::TensorView& q, const pod::TensorView& gate, pod::TensorView& out, size_t slot_stride, int layer_idx,
	const pod::BatchRows* d_rows, int count, int num_q_heads, int num_kv_heads, int head_dim, int max_seq, backend::stream_t stream, float scale, int window,
	bool one_slot, int key_hint ) {
	ENGINE_ASSERT( kv.ring == 0 || ( window > 0 && window <= kv.ring && kv.ring == max_seq ), "Attention (Vulkan): a ring cache holds a sliding window of max_seq rows" );
	ENGINE_ASSERT( head_dim <= 512 && head_dim % 16 == 0 && num_kv_heads > 0 && num_q_heads % num_kv_heads == 0, "Attention (Vulkan): head_dim a multiple of 16 up to 512, GQA" );
	ENGINE_ASSERT( kv.vq != pod::KVQ::Q4 || head_dim % 64 == 0, "Attention (Vulkan): Q4 values need head_dim % 64 == 0" );
	const int G = num_q_heads / num_kv_heads;
	ENGINE_ASSERT( G <= 16, "Attention (Vulkan): at most 16 query heads per KV head" );
	const int kDecodeVectors = head_dim == 256 ? 8 : 4;
	const int decode_rows = one_slot ? count : 1;
	const bool decode = kv.kq == pod::KVQ::F16 && kv.vq == pod::KVQ::F16 && ( head_dim == 256 || head_dim == 512 ) && decode_rows * G <= 2 * kDecodeVectors;
	const int group_rows = decode ? decode_rows : one_slot ? std::max( 1, 16 / G ) : 1;
	const int groups = ( count + group_rows - 1 ) / group_rows;
	const int parts = decode ? ( group_rows * G + kDecodeVectors - 1 ) / kDecodeVectors : 1;
	const int R = std::min( group_rows, count ) * G, hw = head_dim / 2;
	// keys per tile (fa_attn), per share of a split (fa_decode: four warps' steps of 4 keys)
	const int bc = decode ? 16 : R * hw + 32 * ( hw + 1 ) <= kFaPoolWords ? 32 : 16;
	// the keys a group can span
	const int rows_span = one_slot ? group_rows : 1;
	int span = window > 0 ? window + rows_span + pod::MAX_ROWS : max_seq;
	if ( key_hint > 0 ) span = std::min( span, key_hint );   // positions below it (an image never crosses a chunk)
	if ( kv.ring == 0 ) span = std::min( span, max_seq );
	const size_t per_split = static_cast<size_t>( count ) * num_q_heads * ( head_dim + 2 ) * sizeof( float );   // the output and its max and sum
	const int blocks = groups * parts * num_kv_heads;
	int splits = std::min( { kMaxSplits, ( kTargetGroups + blocks - 1 ) / blocks, ( span + bc - 1 ) / bc, static_cast<int>( kFaScratch / per_split ) } );
	splits = std::max( splits, 1 );
	const bool q4 = kv.vq == pod::KVQ::Q4;
	const bool direct = splits == 1;
	float* part = splits > 1 ? static_cast<float*>( vki::scratch( 40, kFaScratch, stream ) ) : nullptr;
	const bool has_gate = gate.data != nullptr;
	vki::FaParams p{ vki::dp( q.data ), vki::dp( kv.k ), vki::dp( kv.v ), vki::dp( gate.data ), vki::dp( out.data ), vki::rows_slot( d_rows ), vki::rows_pos( d_rows ),
		static_cast<int64_t>( slot_stride ), layer_idx, num_q_heads, num_kv_heads, head_dim, max_seq, has_gate && !( direct && q4 ) ? 1 : 0,
		scale > 0.0f ? scale : 1.0f / std::sqrt( static_cast<float>( head_dim ) ), vki::dp( kv.ks ), vki::dp( kv.vs ), static_cast<int>( kv.kq ), static_cast<int>( kv.vq ),
		window, kv.ring, vki::rows_key_end( d_rows ), vki::rows_bidir( d_rows ), count, group_rows, splits, bc, direct ? 1 : 0, vki::dp( part ), parts };
	const char* kernel = !decode ? "fa_attn" : head_dim == 256 ? "fa_decode_256" : "fa_decode_512";
	vki::launch( kernel, p, stream, static_cast<uint32_t>( groups * parts ), static_cast<uint32_t>( num_kv_heads ), static_cast<uint32_t>( splits ) );
	vki::FaReduceParams r{ vki::dp( part ), vki::dp( gate.data ), vki::dp( out.data ), count, num_q_heads, head_dim, splits, has_gate ? 1 : 0, static_cast<int>( kv.vq ) };
	if ( !direct ) vki::launch( "fa_reduce", r, stream, static_cast<uint32_t>( num_q_heads ), static_cast<uint32_t>( count ) );
	else if ( q4 ) vki::launch( "fa_finish", r, stream, static_cast<uint32_t>( num_q_heads ), static_cast<uint32_t>( count ) );
}

// head_dim 64: fa_dense_64 (a thread per query, 32 a workgroup); else fa_attn without slots, up to 16 query vectors a workgroup, the splits as
// tiled's (their partials in a scratch of their own: the callers are the TTS and vision towers, which may run beside a language model's forward)
void dense( const backend::half_t* q, const backend::half_t* k, const backend::half_t* v, backend::half_t* out, int rows, int keys, int num_q_heads, int num_kv_heads,
	int head_dim, float scale, backend::stream_t stream ) {
	ENGINE_ASSERT( head_dim <= 512 && head_dim % 16 == 0 && num_kv_heads > 0 && num_q_heads % num_kv_heads == 0, "Attention (Vulkan): head_dim a multiple of 16 up to 512, GQA" );
	if ( head_dim == 64 ) {
		struct FaDenseParams { uint64_t q, k, v, out_; int rows, keys, heads, kv_heads; float scale; };
		const FaDenseParams p{ vki::dp( q ), vki::dp( k ), vki::dp( v ), vki::dp( out ), rows, keys, num_q_heads, num_kv_heads, scale };
		vki::launch( "fa_dense_64", p, stream, static_cast<uint32_t>( ( rows + 31 ) / 32 ), static_cast<uint32_t>( num_q_heads ) );
		return;
	}
	const int G = num_q_heads / num_kv_heads;
	ENGINE_ASSERT( G <= 16, "Attention (Vulkan): at most 16 query heads per KV head" );
	const int group_rows = std::max( 1, 16 / G );
	const int groups = ( rows + group_rows - 1 ) / group_rows;
	const int R = std::min( group_rows, rows ) * G, hw = head_dim / 2;
	const int bc = R * hw + 32 * ( hw + 1 ) <= kFaPoolWords ? 32 : 16;
	const size_t per_split = static_cast<size_t>( rows ) * num_q_heads * ( head_dim + 2 ) * sizeof( float );
	const int blocks = groups * num_kv_heads;
	int splits = std::min( { kMaxSplits, ( kTargetGroups + blocks - 1 ) / blocks, ( keys + bc - 1 ) / bc, static_cast<int>( kFaScratch / per_split ) } );
	splits = std::max( splits, 1 );
	const bool direct = splits == 1;
	float* part = direct ? nullptr : static_cast<float*>( vki::scratch( 41, kFaScratch, stream ) );
	vki::FaParams p{ vki::dp( q ), vki::dp( k ), vki::dp( v ), 0, vki::dp( out ), 0, 0, 0, 0, num_q_heads, num_kv_heads, head_dim, keys, 0, scale, 0, 0,
		static_cast<int>( pod::KVQ::F16 ), static_cast<int>( pod::KVQ::F16 ), 0, 0, 0, 0, rows, group_rows, splits, bc, direct ? 1 : 0, vki::dp( part ), 1 };
	vki::launch( "fa_attn", p, stream, static_cast<uint32_t>( groups ), static_cast<uint32_t>( num_kv_heads ), static_cast<uint32_t>( splits ) );
	vki::FaReduceParams r{ vki::dp( part ), 0, vki::dp( out ), rows, num_q_heads, head_dim, splits, 0, static_cast<int>( pod::KVQ::F16 ) };
	if ( !direct ) vki::launch( "fa_reduce", r, stream, static_cast<uint32_t>( num_q_heads ), static_cast<uint32_t>( rows ) );
}

} // namespace vk_attn

namespace {

bool attn_ref() {
	static const bool ref = engine::runtime::options != nullptr && engine::runtime::options().vulkan_attn_ref;
	return ref;
}

}

void Attention::forward_decode( const pod::KVRef& kv, const pod::TensorView& q, const pod::TensorView& gate, pod::TensorView& out, size_t slot_stride, int layer_idx,
	const pod::BatchRows* d_rows, int count, int num_q_heads, int num_kv_heads, int head_dim, int max_seq, backend::stream_t stream, float scale, int window ) {
	if ( attn_ref() ) vk_attn::reference( kv, q, gate, out, slot_stride, layer_idx, d_rows, count, num_q_heads, num_kv_heads, head_dim, max_seq, stream, scale, window );
	else vk_attn::tiled( kv, q, gate, out, slot_stride, layer_idx, d_rows, count, num_q_heads, num_kv_heads, head_dim, max_seq, stream, scale, window, false, 0 );
}

// per_row false: the rows share a slot (the CUDA kernels' tensor-core form), so they share the K / V tiles of a workgroup
void Attention::forward_prefill( const pod::KVRef& kv, const pod::TensorView& q, const pod::TensorView& gate, pod::TensorView& out, size_t slot_stride, int layer_idx,
	const pod::BatchRows* d_rows, int count, int num_q_heads, int num_kv_heads, int head_dim, int max_seq, backend::stream_t stream, int key_hint, bool per_row, float scale,
	int window ) {
	if ( attn_ref() ) vk_attn::reference( kv, q, gate, out, slot_stride, layer_idx, d_rows, count, num_q_heads, num_kv_heads, head_dim, max_seq, stream, scale, window );
	else vk_attn::tiled( kv, q, gate, out, slot_stride, layer_idx, d_rows, count, num_q_heads, num_kv_heads, head_dim, max_seq, stream, scale, window, !per_row, key_hint );
}

void Attention::unpack_q_gate( const pod::TensorView& q_gate, pod::TensorView& q, pod::TensorView& gate, int num_q_heads, int head_dim, backend::stream_t stream ) {
	const int rows = ( q_gate.ndim >= 2 && q_gate.shape[0] > 0 ) ? q_gate.shape[0] : 1;
	const int total = rows * num_q_heads * head_dim;
	vki::RefUnpackParams p{ vki::dp( q_gate.data ), vki::dp( q.data ), vki::dp( gate.data ), total, head_dim };
	vki::launch( "ref_unpack_q_gate", p, stream, ( total + 127 ) / 128 );
}

} // namespace engine::modules
