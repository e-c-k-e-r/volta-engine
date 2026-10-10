// CPU backend: weak stubs for the mla entry points that have no CPU implementation (CUDA/SM70 only or not yet ported).
// A strong definition in another translation unit overrides these. Declarations come from engine/modules/mla.h.
#include "engine/modules/mla.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

__attribute__(( weak )) void MLADSA::forward_decode( const pod::TensorView& q_lat, const pod::TensorView& q_pe, const backend::half_t* latent_cache, const backend::half_t* rope_cache, pod::TensorView& out, int layer_idx, int current_pos, int dsa_top_k, int mla_dc, int mla_dr, int max_seq, int out_dim, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "MLADSA::forward_decode" );
	__builtin_unreachable();
}

__attribute__(( weak )) void MLADSA::store_cache( backend::half_t* latent_cache, backend::half_t* rope_cache, const backend::half_t* k_lat, const backend::half_t* k_pe, int layer_idx, int pos, int mla_dc, int mla_dr, int max_seq, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "MLADSA::store_cache" );
	__builtin_unreachable();
}

} // namespace engine::modules
