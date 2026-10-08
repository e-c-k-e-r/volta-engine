#pragma once

#include "engine/config.h"
#include "engine/core/batch.h"
#include "engine/backend/types.h"
#include <cuda_fp16.h>

namespace engine::spec {

// Publishes row r of `hidden` ([m, H]) as tap `tap` of the position-indexed ring (taps [cap, 5 * H]) when the row's
// slot owns the ring. Slot, position and owner come from device memory so the launch is graph-replayable.
void dflash_publish_taps(
	half* taps,
	const half* hidden,
	const pod::BatchRows* d_rows,
	const int* d_owner,
	int m,
	int tap,
	int H,
	int cap,
	backend::stream_t stream
);

// Reduces mHC residual streams [m, 4, H] (FP32) to one FP16 hidden row per token: mode 0 = mean of the four streams,
// mode k in 1..4 = stream k - 1 (diagnostic selectors for matching the drafter's training-time tap).
void dflash_reduce_streams(
	const float* streams,
	half* out,
	int m,
	int H,
	int mode,
	backend::stream_t stream
);

} // namespace engine::spec
