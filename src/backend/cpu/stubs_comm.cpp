// CPU backend: weak stubs for the multi-GPU collective (NVLink IPC ring, src/comm). A single-device CPU backend has no peers.
// A strong definition in another translation unit overrides these.
#include "engine/core/comm.h"
#include "engine/backend/tiers.h"

namespace engine::comm {

__attribute__(( weak )) void CollectiveRing::init( int local_rank, int ranks, void* const peer_staging[ENGINE_NUM_GPUS], void* const peer_flags[ENGINE_NUM_GPUS] ) {
	ENGINE_UNSUPPORTED( "CollectiveRing::init" );
	__builtin_unreachable();
}

__attribute__(( weak )) void CollectiveRing::all_reduce_rms_norm(
	pod::TensorView& x,
	pod::TensorView& residual,
	const pod::TensorView& weight,
	float eps,
	backend::stream_t stream,
	float constant_bias,
	pod::TensorView* out
) {
	ENGINE_UNSUPPORTED( "CollectiveRing::all_reduce_rms_norm" );
	__builtin_unreachable();
}

__attribute__(( weak )) void CollectiveRing::all_reduce_hc_apply( pod::TensorView&, float*, const float*, const float*, backend::stream_t ) {
	ENGINE_UNSUPPORTED( "CollectiveRing::all_reduce_hc_apply" );
	__builtin_unreachable();
}

__attribute__(( weak )) void CollectiveRing::all_reduce_gated_norm( pod::TensorView&, float*, const float*, const backend::half_t*, float*, int, float, backend::stream_t, const backend::half_t*, const float* ) {
	ENGINE_UNSUPPORTED( "CollectiveRing::all_reduce_gated_norm" );
	__builtin_unreachable();
}

__attribute__(( weak )) void CollectiveRing::all_reduce( pod::TensorView& x, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "CollectiveRing::all_reduce" );
	__builtin_unreachable();
}

__attribute__(( weak )) void init_4gpu_mesh( CollectiveRing rings[ENGINE_NUM_GPUS], void* staging_ptrs[ENGINE_NUM_GPUS], void* flag_ptrs[ENGINE_NUM_GPUS], size_t staging_bytes, int ranks ) {
	ENGINE_UNSUPPORTED( "init_4gpu_mesh" );
	__builtin_unreachable();
}

__attribute__(( weak )) void destroy_4gpu_mesh( void* staging_ptrs[ENGINE_NUM_GPUS], void* flag_ptrs[ENGINE_NUM_GPUS] ) {
	ENGINE_UNSUPPORTED( "destroy_4gpu_mesh" );
	__builtin_unreachable();
}

} // namespace engine::comm
