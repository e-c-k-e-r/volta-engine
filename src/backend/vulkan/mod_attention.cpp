// Vulkan backend: FP16 KV store, decode / prefill attention and the Q/gate unpack. Attention runs the portable reference kernel ref_attn_decode:
// row m attends causally to positions 0..pos[m] of slot slot[m], so prefill is the same dispatch over a run of ascending positions
#include "mod_common.h"
#include "engine/modules/attention.h"
#include "engine/backend/tiers.h"
#include <cmath>

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
	if ( kv.kq != pod::KVQ::F16 || kv.vq != pod::KVQ::F16 ) {
		ENGINE_UNSUPPORTED( "Attention::store_kv_cache (quantized KV cache)" );
	}
	const int total_elems = num_kv_heads * head_dim;
	const int groups = ( count * total_elems + 127 ) / 128;
	vki::KvStoreParams pk{ vki::dp( kv.k ), vki::dp( k.data ), vki::rows_slot( d_rows ), vki::rows_pos( d_rows ),
		static_cast<int64_t>( slot_stride ), layer_idx, count, total_elems, max_seq };
	vki::launch( "kv_store_f16", pk, stream, groups );
	vki::KvStoreParams pv = pk;
	pv.data = vki::dp( kv.v );
	pv.src = vki::dp( v.data );
	vki::launch( "kv_store_f16", pv, stream, groups );
}

namespace {

void ref_attention( const pod::KVRef& kv, const pod::TensorView& q, const pod::TensorView& gate, pod::TensorView& out, size_t slot_stride, int layer_idx,
	const pod::BatchRows* d_rows, int count, int num_q_heads, int num_kv_heads, int head_dim, int max_seq, backend::stream_t stream ) {
	if ( kv.kq != pod::KVQ::F16 || kv.vq != pod::KVQ::F16 ) {
		ENGINE_UNSUPPORTED( "Attention decode/prefill (quantized KV cache)" );
	}
	ENGINE_ASSERT( head_dim <= 256 && num_kv_heads > 0 && num_q_heads % num_kv_heads == 0, "Attention (Vulkan): head_dim <= 256 and GQA only" );
	vki::RefAttnParams p{ vki::dp( q.data ), vki::dp( kv.k ), vki::dp( kv.v ), vki::dp( gate.data ), vki::dp( out.data ), vki::rows_slot( d_rows ), vki::rows_pos( d_rows ),
		static_cast<int64_t>( slot_stride ), layer_idx, num_q_heads, num_kv_heads, head_dim, max_seq, gate.data != nullptr ? 1 : 0, 1.0f / std::sqrt( static_cast<float>( head_dim ) ) };
	vki::launch( "ref_attn_decode", p, stream, num_q_heads, count );
}

}

void Attention::forward_decode( const pod::KVRef& kv, const pod::TensorView& q, const pod::TensorView& gate, pod::TensorView& out, size_t slot_stride, int layer_idx,
	const pod::BatchRows* d_rows, int count, int num_q_heads, int num_kv_heads, int head_dim, int max_seq, backend::stream_t stream ) {
	ref_attention( kv, q, gate, out, slot_stride, layer_idx, d_rows, count, num_q_heads, num_kv_heads, head_dim, max_seq, stream );
}

void Attention::forward_prefill( const pod::KVRef& kv, const pod::TensorView& q, const pod::TensorView& gate, pod::TensorView& out, size_t slot_stride, int layer_idx,
	const pod::BatchRows* d_rows, int count, int num_q_heads, int num_kv_heads, int head_dim, int max_seq, backend::stream_t stream, int, bool ) {
	ref_attention( kv, q, gate, out, slot_stride, layer_idx, d_rows, count, num_q_heads, num_kv_heads, head_dim, max_seq, stream );
}

void Attention::unpack_q_gate( const pod::TensorView& q_gate, pod::TensorView& q, pod::TensorView& gate, int num_q_heads, int head_dim, backend::stream_t stream ) {
	const int rows = ( q_gate.ndim >= 2 && q_gate.shape[0] > 0 ) ? q_gate.shape[0] : 1;
	const int total = rows * num_q_heads * head_dim;
	vki::RefUnpackParams p{ vki::dp( q_gate.data ), vki::dp( q.data ), vki::dp( gate.data ), total, head_dim };
	vki::launch( "ref_unpack_q_gate", p, stream, ( total + 127 ) / 128 );
}

} // namespace engine::modules
