// Vulkan backend: the GLM modules on the Slang kernels of src/kernels/slang/glm.slang: MLA latent absorb / expand / store / attention (dense and
// DSA-selected rows in one kernel), the kv_b projection, the DSA projections (dense FP16 GEMM) and indexer scoring, and the KDA gate projections and
// recurrence. The expanded-cache MLA entry points map onto the Attention ones (every head is its own KV head), as on the CPU.
#include "mod_common.h"
#include "engine/backend/runtime.h"
#include "engine/backend/tiers.h"
#include "engine/modules/attention.h"
#include "engine/modules/dsa.h"
#include "engine/modules/gdn.h"
#include "engine/modules/linear.h"
#include "engine/modules/mla.h"
#include <cmath>

namespace engine::modules {

namespace {
namespace vki = ::engine::backend::impl;

inline uint32_t tiles( int n, int tile ) { return static_cast<uint32_t>( ( n + tile - 1 ) / tile ); }
inline uint64_t rows_slot_at( const pod::SlotRef& s, int row ) { return s.rows != nullptr ? vki::rows_slot( s.rows ) + sizeof( int ) * static_cast<uint64_t>( row ) : 0; }
}

// ---------------------------------------------------------------- MLA latent

void MLALatent::absorb_q( const pod::TensorView& q, const pod::TensorView& kv_b, pod::TensorView& q_lat, int heads, int rows, backend::stream_t stream ) {
	vki::AbsorbParams p{ vki::dp( q.data ), vki::dp( kv_b.data ), vki::dp( q_lat.data ), heads };
	vki::launch( "latent_absorb_q", p, stream, tiles( LORA, 64 ), heads, rows );
}

void MLALatent::expand_out( const pod::TensorView& o_lat, const pod::TensorView& kv_b, pod::TensorView& out, int heads, int rows, backend::stream_t stream ) {
	vki::ExpandParams p{ vki::dp( o_lat.data ), vki::dp( kv_b.data ), vki::dp( out.data ), heads };
	vki::launch( "latent_expand_out", p, stream, tiles( HEAD, 64 ), heads, rows );
}

void MLALatent::store( const pod::KVRef& kv, const pod::TensorView& c, int layer, int max_seq, int rows, const pod::SlotRef& slot, backend::stream_t stream ) {
	ENGINE_ASSERT( kv.kq == pod::KVQ::F16, "MLALatent::store (Vulkan): FP16 latent cache only" );
	ENGINE_ASSERT( slot.rows != nullptr, "MLALatent::store: device-resolved positions only" );
	vki::LatentStoreParams p{ vki::dp( c.data ), vki::dp( kv.k ), rows_slot_at( slot, slot.row ), vki::rows_pos( slot.rows ) + sizeof( int ) * static_cast<uint64_t>( slot.row ),
		static_cast<int64_t>( slot.stride ), layer, max_seq };
	vki::launch( "latent_store", p, stream, tiles( LORA, 256 ), rows );
}

void MLALatent::attend( const pod::KVRef& kv, const pod::TensorView& q_lat, pod::TensorView& o_lat, int layer, int max_seq, int rows, int heads, const pod::SlotRef& slot,
	float*, int, backend::stream_t stream, const int* sel_idx, const int* sel_n ) {
	ENGINE_ASSERT( kv.kq == pod::KVQ::F16, "MLALatent::attend (Vulkan): FP16 latent cache only" );
	ENGINE_ASSERT( slot.rows != nullptr && slot.row == 0, "MLALatent::attend (Vulkan): device-resolved positions starting at row 0" );
	vki::LatentAttnParams p{ vki::dp( q_lat.data ), vki::dp( kv.k ), vki::dp( o_lat.data ), vki::rows_slot( slot.rows ), vki::rows_pos( slot.rows ), vki::dp( sel_idx ), vki::dp( sel_n ),
		static_cast<int64_t>( slot.stride ), layer, heads, max_seq, DSA::KPAD, ( sel_idx != nullptr && sel_n != nullptr ) ? 1 : 0, 1.0f / std::sqrt( static_cast<float>( HEAD ) ) };
	vki::launch( "latent_attend", p, stream, heads, rows );
}

void MLALatent::diag_scores( const backend::half_t* q_lat_row, const backend::half_t* cache_slot_layer, int n, int heads, float* scores, backend::stream_t stream ) {
	vki::DiagParams p{ vki::dp( q_lat_row ), vki::dp( cache_slot_layer ), vki::dp( scores ), n, heads, 1.0f / std::sqrt( static_cast<float>( HEAD ) ) };
	vki::launch( "latent_diag_scores", p, stream, tiles( n, 64 ), heads );
}

// ---------------------------------------------------------------- expanded-cache MLA

void MLA::project_kv_b( const pod::TensorView& kv_norm, const pod::TensorView& kv_b_proj, pod::TensorView& k_out, pod::TensorView& v_out, int num_heads, int head_dim, backend::stream_t stream, int rows ) {
	vki::KvBParams p{ vki::dp( kv_norm.data ), vki::dp( kv_b_proj.data ), vki::dp( k_out.data ), vki::dp( v_out.data ), num_heads, head_dim };
	vki::launch( "mla_project_kv_b", p, stream, tiles( num_heads * head_dim * 2, 64 ), rows );
}

void MLA::store_kv_cache( const pod::KVRef& kv, const pod::TensorView& k, const pod::TensorView& v, int mla_layer_idx, int, int total_elems, int max_seq, backend::stream_t stream,
	const pod::SlotRef& slot, int rows ) {
	if ( !slot.rows ) ENGINE_UNSUPPORTED( "MLA::store_kv_cache (host-resolved position)" );
	ENGINE_ASSERT( slot.row == 0, "MLA::store_kv_cache: row offset unsupported" );
	Attention::store_kv_cache( kv, k, v, slot.stride, mla_layer_idx, slot.rows, rows, total_elems / 256, 256, max_seq, stream );
}

void MLA::forward_decode( const pod::TensorView& q, const pod::KVRef& kv, pod::TensorView& out, int mla_layer_idx, int, int num_heads, int head_dim, int max_seq, backend::stream_t stream,
	const pod::SlotRef& slot, int rows, int key_hint ) {
	if ( !slot.rows ) ENGINE_UNSUPPORTED( "MLA::forward_decode (host-resolved position)" );
	ENGINE_ASSERT( slot.row == 0, "MLA::forward_decode: row offset unsupported" );
	pod::TensorView no_gate{};
	Attention::forward_decode( kv, q, no_gate, out, slot.stride, mla_layer_idx, slot.rows, rows, num_heads, num_heads, head_dim, max_seq, stream );
	( void )key_hint;
}

// ---------------------------------------------------------------- DSA

void DSA::project_small( const backend::half_t* w, const backend::half_t* x, backend::half_t* proj, int rows, int n, int k, backend::stream_t stream ) {
	ENGINE_ASSERT( rows >= 1 && rows <= 16 && k % 256 == 0, "DSA::project_small: up to 16 rows, k a multiple of 256" );
	LinearPlainGEMV::forward( w, x, proj, rows, n, k, stream );
}

void DSA::project_large( const backend::half_t* w, const backend::half_t* x, backend::half_t* proj, int rows, int n, int k, backend::stream_t stream ) {
	LinearPlainGEMV::forward( w, x, proj, rows, n, k, stream );
}

void DSA::score( const backend::half_t* q_idx, const backend::half_t* w, const backend::half_t* pool_plane, backend::half_t* scores, int score_stride, int layer, int max_seq, int rows,
	const pod::BatchRows* d_rows, size_t pool_slot_stride, backend::stream_t stream ) {
	ENGINE_ASSERT( max_seq % 64 == 0, "DSA score: max_seq must be a multiple of 64" );
	vki::DsaScoreParams p{ vki::dp( q_idx ), vki::dp( w ), vki::dp( pool_plane ), vki::dp( scores ), vki::rows_slot( d_rows ), vki::rows_pos( d_rows ), score_stride, layer, max_seq,
		static_cast<int64_t>( pool_slot_stride ), 1.0f / std::sqrt( static_cast<float>( DIM * HEADS ) ) };
	vki::launch( "dsa_score", p, stream, tiles( max_seq / POOL, 64 ), rows );
}

// ---------------------------------------------------------------- KDA

void GDN::project_kda_gates( const pod::TensorView& norm_hidden, const pod::TensorView& b_proj, const pod::TensorView& f_a_proj, const pod::TensorView& f_b_proj,
	const pod::TensorView& g_a_proj, const pod::TensorView& g_b_proj, const pod::TensorView& A_log, const pod::TensorView& dt_bias, pod::TensorView& beta_out,
	pod::TensorView& g_decay_out, pod::TensorView& z_out, pod::TensorView& f_a_buf, pod::TensorView& g_a_buf, int num_heads, int head_dim, int hidden_size,
	backend::stream_t stream, int rows ) {
	ENGINE_ASSERT( num_heads == 16 && head_dim == 128, "GDN::project_kda_gates (Vulkan): 16 heads of 128" );
	vki::KdaSmallParams a{ vki::dp( norm_hidden.data ), vki::dp( b_proj.data ), vki::dp( f_a_proj.data ), vki::dp( g_a_proj.data ), vki::dp( beta_out.data ), vki::dp( f_a_buf.data ),
		vki::dp( g_a_buf.data ), hidden_size, b_proj.data != nullptr ? 1 : 0, f_a_proj.data != nullptr ? 1 : 0, g_a_proj.data != nullptr ? 1 : 0 };
	vki::launch( "kda_gates_small", a, stream, 16 + 128 + 128, rows );
	const int channels = num_heads * head_dim;
	vki::KdaBigParams b{ vki::dp( f_a_buf.data ), vki::dp( g_a_buf.data ), vki::dp( f_b_proj.data ), vki::dp( g_b_proj.data ), vki::dp( A_log.data ), vki::dp( dt_bias.data ),
		vki::dp( g_decay_out.data ), vki::dp( z_out.data ), channels, f_b_proj.data != nullptr ? 1 : 0, g_b_proj.data != nullptr ? 1 : 0, A_log.data != nullptr ? 1 : 0, dt_bias.data != nullptr ? 1 : 0 };
	vki::launch( "kda_gates_big", b, stream, tiles( channels, 64 ), rows );
}

namespace {

// One launch of kda_step: rows rows in order, each in the slot slots[m] (null: slot 0), the state of slot s at state + s * slot_stride
void kda_launch( float* state, const pod::TensorView& qkv, const pod::TensorView& g_decay, const pod::TensorView& beta, const pod::TensorView& z, const pod::TensorView& o_norm,
	pod::TensorView& out, int num_heads, float eps, int rows, uint64_t slots, size_t slot_stride, float* snap, size_t snap_stride, int snap_rows, const int* snap_on,
	bool has_norm, bool decay_first, backend::stream_t stream ) {
	vki::KdaStepParams p{ vki::dp( state ), vki::dp( qkv.data ), vki::dp( g_decay.data ), vki::dp( beta.data ), vki::dp( z.data ), vki::dp( o_norm.data ), vki::dp( out.data ), slots,
		vki::dp( snap ), vki::dp( snap_on ), static_cast<int64_t>( slot_stride ), static_cast<int64_t>( snap_stride ), num_heads, has_norm ? 1 : 0, z.data != nullptr ? 1 : 0,
		o_norm.data != nullptr ? 1 : 0, decay_first ? 1 : 0, snap_rows, rows, eps };
	vki::launch( "kda_step", p, stream, num_heads );
}

}

// Unfused single step: the state is decayed while kv_mem is read from the undecayed column (as gdn.cu does); no norm
void GDN::recurrent_step_kda( float* gdn_state, const pod::TensorView& qkv, const pod::TensorView& g_decay, const pod::TensorView& beta, pod::TensorView& attn_out,
	int num_heads, int head_dim, backend::stream_t stream ) {
	ENGINE_ASSERT( head_dim == 128, "GDN::recurrent_step_kda: head_dim must be 128" );
	pod::TensorView none{};
	kda_launch( gdn_state, qkv, g_decay, beta, none, none, attn_out, num_heads, 0.0f, 1, 0, 0, nullptr, 0, 0, nullptr, false, false, stream );
}

void GDN::recurrent_step_kda_fused( float* gdn_state, const pod::TensorView& qkv, const pod::TensorView& g_decay, const pod::TensorView& beta, const pod::TensorView& z,
	const pod::TensorView& o_norm, pod::TensorView& attn_out, int num_heads, int head_dim, float eps, backend::stream_t stream, const pod::SlotRef& slot ) {
	ENGINE_ASSERT( head_dim == 128, "GDN::recurrent_step_kda_fused: head_dim must be 128" );
	kda_launch( gdn_state, qkv, g_decay, beta, z, o_norm, attn_out, num_heads, eps, 1, rows_slot_at( slot, slot.row ), slot.stride, nullptr, 0, 0, nullptr, true, true, stream );
}

void GDN::recurrent_step_kda_rows( float* gdn_state, const pod::TensorView& qkv, const pod::TensorView& g_decay, const pod::TensorView& beta, const pod::TensorView& z,
	const pod::TensorView& o_norm, pod::TensorView& attn_out, int num_heads, int head_dim, float eps, int rows, const pod::BatchRows* d_rows, size_t slot_stride,
	float* snap, size_t snap_stride, int snap_rows, const int* snap_on, backend::stream_t stream ) {
	ENGINE_ASSERT( head_dim == 128, "GDN::recurrent_step_kda_rows: head_dim must be 128" );
	kda_launch( gdn_state, qkv, g_decay, beta, z, o_norm, attn_out, num_heads, eps, rows, vki::rows_slot( d_rows ), slot_stride, snap, snap_stride, snap_rows, snap_on, true, true, stream );
}

} // namespace engine::modules
