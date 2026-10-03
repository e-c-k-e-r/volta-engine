#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace engine::comm {

struct CollectiveRing {
	int   rank;
	void* peer_ptrs[ENGINE_NUM_GPUS];
	void* sync_flags[ENGINE_NUM_GPUS];
	uint32_t seq;

	void init( int local_rank, void* const peer_staging[ENGINE_NUM_GPUS], void* const peer_flags[ENGINE_NUM_GPUS] );

	void all_reduce_rms_norm(
		pod::TensorView& x,
		pod::TensorView& residual,
		const pod::TensorView& weight,
		float eps,
		cudaStream_t stream = nullptr
	);

	void all_reduce(
		pod::TensorView& x,
		cudaStream_t stream = nullptr
	);
};

void init_4gpu_mesh(
	CollectiveRing rings[ENGINE_NUM_GPUS],
	void* staging_ptrs[ENGINE_NUM_GPUS],
	void* flag_ptrs[ENGINE_NUM_GPUS],
	size_t staging_bytes = 1024 * 1024
);

void destroy_4gpu_mesh(
	void* staging_ptrs[ENGINE_NUM_GPUS],
	void* flag_ptrs[ENGINE_NUM_GPUS]
);

} // namespace engine::comm
