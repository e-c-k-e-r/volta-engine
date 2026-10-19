// Weak stubs: the DeepSeek-V4.1 kernels exist on the CUDA backend only.
#include "engine/modules/dsv4.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

__attribute__(( weak )) void DSV4::window_store( const backend::half_t* kv, backend::half_t* ring, const float* rope, int layer, int rows, const pod::BatchRows* d_rows, size_t ring_slot_stride, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "DSV4::window_store" );
	__builtin_unreachable();
}

__attribute__(( weak )) void DSV4::compress( const backend::half_t* ckv, const backend::half_t* cscore, const backend::half_t* norm_w, float* cring_kv, float* cring_sc, backend::half_t* latent_out, backend::half_t* comp_kv, const float* rope, int ratio, int rows, const pod::BatchRows* d_rows, size_t cring_slot_stride, size_t comp_slot_stride, float eps, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "DSV4::compress" );
	__builtin_unreachable();
}

__attribute__(( weak )) void DSV4::index_key_store( const backend::half_t* kraw, const backend::half_t* knorm_w, backend::half_t* idx_k, const float* rope, int ratio, int rows, const pod::BatchRows* d_rows, size_t idx_slot_stride, float eps, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "DSV4::index_key_store" );
	__builtin_unreachable();
}

__attribute__(( weak )) void DSV4::indexer( const backend::half_t* q, const backend::half_t* w, const backend::half_t* idx_k, float* scores, uint8_t* cand, int* sel, int* nsel, const float* rope, int ratio, int cap, int cand_mode, int cand_blocks, int cand_block, int rows, const pod::BatchRows* d_rows, size_t idx_slot_stride, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "DSV4::indexer" );
	__builtin_unreachable();
}

__attribute__(( weak )) void DSV4::attend( const backend::half_t* q, const backend::half_t* ring, const backend::half_t* comp_kv, const int* sel, const int* nsel, const float* sink, const float* rope, backend::half_t* o, int nh, int layer, int rows, const pod::BatchRows* d_rows, size_t ring_slot_stride, size_t comp_slot_stride, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "DSV4::attend" );
	__builtin_unreachable();
}

__attribute__(( weak )) void DSV4::concat_groups( const backend::half_t* in, backend::half_t* out, int groups, int n, int rows, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "DSV4::concat_groups" );
	__builtin_unreachable();
}

__attribute__(( weak )) void DSV4::route( const backend::half_t* x, const backend::half_t* w_t, const float* bias, const int* perm, int* idx_out, backend::half_t* w_out, float* scratch, int hidden, int experts, int top_k, float scale, int rows, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "DSV4::route" );
	__builtin_unreachable();
}

__attribute__(( weak )) void DSV4::hc_collapse_norm( const float* streams, const float* pre, const backend::half_t* norm_w, backend::half_t* out, float* post, float post_scale, int hidden, float eps, int rows, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "DSV4::hc_collapse_norm" );
	__builtin_unreachable();
}

__attribute__(( weak )) void DSV4::hc_identity_pre( float* pre, int rows, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "DSV4::hc_identity_pre" );
	__builtin_unreachable();
}

__attribute__(( weak )) void DSV4::engram_gate( float* streams, const backend::half_t* kv, const float* w, int hidden, float eps, int rows, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "DSV4::engram_gate" );
	__builtin_unreachable();
}

} // namespace engine::modules
