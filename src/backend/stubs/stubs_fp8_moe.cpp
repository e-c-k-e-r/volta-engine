// Weak stubs: the FP8 routed-expert kernels (original FP8 checkpoints) exist on the CUDA backend only.
#include "engine/modules/fp8_moe.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

__attribute__(( weak )) size_t FP8MoE::scratch_bytes( int, int, int ) { return 0; }

__attribute__(( weak )) void FP8MoE::forward( const FP8ExpertTable&, const backend::half_t*, int, const int*, const backend::half_t*, int, backend::half_t*, void*,
	uint32_t*, float, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "FP8MoE::forward" );
}

} // namespace engine::modules
