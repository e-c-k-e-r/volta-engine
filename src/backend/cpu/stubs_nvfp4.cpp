// CPU backend: weak stubs for the nvfp4 entry points that have no CPU implementation (CUDA/SM70 only or not yet ported).
// A strong definition in another translation unit overrides these. Declarations come from engine/modules/nvfp4.h.
#include "engine/modules/nvfp4.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

__attribute__(( weak )) void LinearNVFP4::forward( const pod::TensorView& x, const pod::TensorView& codes, const pod::TensorView& scales, float gscale, pod::TensorView& out, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "LinearNVFP4::forward" );
	__builtin_unreachable();
}

} // namespace engine::modules
