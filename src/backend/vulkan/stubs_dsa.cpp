// Vulkan backend: weak stubs for the dsa entry points that have no Vulkan implementation (CUDA/SM70 only or not yet ported).
// A strong definition in another translation unit overrides these. Declarations come from engine/modules/dsa.h.
#include "engine/modules/dsa.h"
#include "engine/modules/qsa.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

__attribute__(( weak )) void DSA::project_large( const backend::half_t* w, const backend::half_t* x, backend::half_t* proj, int rows, int n, int k, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "DSA::project_large" );
	__builtin_unreachable();
}

__attribute__(( weak )) void DSA::project_small( const backend::half_t* w, const backend::half_t* x, backend::half_t* proj, int rows, int n, int k, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "DSA::project_small" );
	__builtin_unreachable();
}

__attribute__(( weak )) void DSA::score( const backend::half_t* q_idx, const backend::half_t* w, const backend::half_t* pool_plane, backend::half_t* scores, int score_stride, int layer, int max_seq, int rows, const pod::BatchRows* d_rows, size_t pool_slot_stride, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "DSA::score" );
	__builtin_unreachable();
}

} // namespace engine::modules
