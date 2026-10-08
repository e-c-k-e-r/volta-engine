// Vulkan backend: weak stubs for the mla entry points that have no Vulkan implementation (CUDA/SM70 only or not yet ported).
// A strong definition in another translation unit overrides these. Declarations come from engine/modules/mla.h.
#include "engine/modules/mla.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

__attribute__(( weak )) void MLA::forward_decode( const pod::TensorView& q, const pod::KVRef& kv, pod::TensorView& out, int mla_layer_idx, int pos, int num_heads, int head_dim, int max_seq, backend::stream_t stream, const pod::SlotRef& slot, int rows, int key_hint ) {
	ENGINE_UNSUPPORTED( "MLA::forward_decode" );
	__builtin_unreachable();
}

__attribute__(( weak )) void MLA::project_kv_b( const pod::TensorView& kv_norm, const pod::TensorView& kv_b_proj, pod::TensorView& k_out, pod::TensorView& v_out, int num_heads, int head_dim, backend::stream_t stream, int rows ) {
	ENGINE_UNSUPPORTED( "MLA::project_kv_b" );
	__builtin_unreachable();
}

__attribute__(( weak )) void MLA::store_kv_cache( const pod::KVRef& kv, const pod::TensorView& k, const pod::TensorView& v, int mla_layer_idx, int pos, int total_elems, int max_seq, backend::stream_t stream, const pod::SlotRef& slot, int rows ) {
	ENGINE_UNSUPPORTED( "MLA::store_kv_cache" );
	__builtin_unreachable();
}

__attribute__(( weak )) void MLADSA::forward_decode( const pod::TensorView& q_lat, const pod::TensorView& q_pe, const backend::half_t* latent_cache, const backend::half_t* rope_cache, pod::TensorView& out, int layer_idx, int current_pos, int dsa_top_k, int mla_dc, int mla_dr, int max_seq, int out_dim, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "MLADSA::forward_decode" );
	__builtin_unreachable();
}

__attribute__(( weak )) void MLADSA::store_cache( backend::half_t* latent_cache, backend::half_t* rope_cache, const backend::half_t* k_lat, const backend::half_t* k_pe, int layer_idx, int pos, int mla_dc, int mla_dr, int max_seq, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "MLADSA::store_cache" );
	__builtin_unreachable();
}

__attribute__(( weak )) void MLALatent::absorb_q( const pod::TensorView& q, const pod::TensorView& kv_b, pod::TensorView& q_lat, int heads, int rows, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "MLALatent::absorb_q" );
	__builtin_unreachable();
}

__attribute__(( weak )) void MLALatent::attend( const backend::half_t* cache, const pod::TensorView& q_lat, pod::TensorView& o_lat, int layer, int max_seq, int rows, int heads, const pod::SlotRef& slot, float* workspace, int key_hint, backend::stream_t stream, const int* sel_idx, const int* sel_n ) {
	ENGINE_UNSUPPORTED( "MLALatent::attend" );
	__builtin_unreachable();
}

__attribute__(( weak )) void MLALatent::diag_scores( const backend::half_t* q_lat_row, const backend::half_t* cache_slot_layer, int n, int heads, float* scores, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "MLALatent::diag_scores" );
	__builtin_unreachable();
}

__attribute__(( weak )) void MLALatent::expand_out( const pod::TensorView& o_lat, const pod::TensorView& kv_b, pod::TensorView& out, int heads, int rows, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "MLALatent::expand_out" );
	__builtin_unreachable();
}

__attribute__(( weak )) void MLALatent::store( backend::half_t* cache, const pod::TensorView& c, int layer, int max_seq, int rows, const pod::SlotRef& slot, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "MLALatent::store" );
	__builtin_unreachable();
}

} // namespace engine::modules
