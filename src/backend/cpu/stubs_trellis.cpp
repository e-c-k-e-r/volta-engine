// CPU backend: weak stubs for the trellis entry points that have no CPU implementation (CUDA/SM70 only or not yet ported).
// A strong definition in another translation unit overrides these. Declarations come from engine/modules/trellis.h.
#include "engine/modules/trellis.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

__attribute__(( weak )) void LinearTrellis::forward( const pod::TensorView& x, const pod::TensorView& trellis, const pod::TensorView& suh, const pod::TensorView& svh, pod::TensorView& out, int bits, bool mcg, bool mul1, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "LinearTrellis::forward" );
	__builtin_unreachable();
}

__attribute__(( weak )) void LinearTrellis::forward_recon( const pod::TensorView& x, const pod::TensorView& trellis, const pod::TensorView& suh, const pod::TensorView& svh, pod::TensorView& out, int bits, bool mcg, bool mul1, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "LinearTrellis::forward_recon" );
	__builtin_unreachable();
}

} // namespace engine::modules
