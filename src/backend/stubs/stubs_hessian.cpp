// Weak stub: Hessian capture (quantization calibration) exists on the CUDA backend only.
#include "engine/modules/hessian.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

__attribute__(( weak )) void Hessian::accumulate( float* h, const backend::half_t* x, int rows, int k, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "Hessian::accumulate" );
	__builtin_unreachable();
}

} // namespace engine::modules
