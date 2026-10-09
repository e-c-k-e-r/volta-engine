#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/backend/types.h"
#include <cstdint>
#include <cstddef>

namespace engine::comm {

// Staging bytes per device the mesh needs: 16 slots (4 stages x 4 ranks) of 3 MiB.
static constexpr size_t kMeshStagingBytes = 16 * 6 * 1024 * 1024;

struct CollectiveRing {
	int   rank;
	int   nranks = ENGINE_NUM_GPUS;   // 2 or 4
	void* peer_ptrs[ENGINE_NUM_GPUS];
	void* sync_flags[ENGINE_NUM_GPUS]; // flag pages; sequence counter is device-resident (graph-safe)

	void init( int local_rank, int ranks, void* const peer_staging[ENGINE_NUM_GPUS], void* const peer_flags[ENGINE_NUM_GPUS] );

	void all_reduce_rms_norm(
		pod::TensorView& x,
		pod::TensorView& residual,
		const pod::TensorView& weight,
		float eps,
		backend::stream_t stream = nullptr,
		float constant_bias = 1.0f,
		pod::TensorView* out = nullptr
	);

	void all_reduce(
		pod::TensorView& x,
		backend::stream_t stream = nullptr
	);
};

void init_4gpu_mesh(
	CollectiveRing rings[ENGINE_NUM_GPUS],
	void* staging_ptrs[ENGINE_NUM_GPUS],
	void* flag_ptrs[ENGINE_NUM_GPUS],
	size_t staging_bytes = kMeshStagingBytes,
	int ranks = ENGINE_NUM_GPUS
);

void destroy_4gpu_mesh(
	void* staging_ptrs[ENGINE_NUM_GPUS],
	void* flag_ptrs[ENGINE_NUM_GPUS]
);

} // namespace engine::comm
