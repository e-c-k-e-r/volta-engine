#pragma once

#include "engine/config.h"
#include "engine/core/batch.h"
#include "engine/backend/types.h"

namespace engine::spec {

// The columns of a position's tap row (taps_n taps of hidden, one after the other) that a rank's ring keeps: the whole row, or the rank's
// contiguous share of it when the drafter's fc is row-parallel over the row (each rank then projects its own columns and the partial sums are
// all-reduced)
struct TapSlice {
	int hidden = 0, taps_n = 0;   // the whole row: taps_n * hidden columns
	int col0 = 0, cols = 0;       // the kept columns [col0, col0 + cols)
	bool whole() const { return col0 == 0 && cols == taps_n * hidden; }
	// Tap `tap`'s part of the kept columns: hidden columns [*src0, *src0 + *n) land at column *dst0 of the kept row; false when the rank keeps none of it
	bool span( int tap, int* src0, int* dst0, int* n ) const {
		const int a = MAX( tap * hidden, col0 ), b = MIN( ( tap + 1 ) * hidden, col0 + cols );
		if ( a >= b ) return false;
		*src0 = a - tap * hidden;
		*dst0 = a - col0;
		*n = b - a;
		return true;
	}
};

// Publishes row r of `hidden` ([m, hidden]) as tap `tap` of the position-indexed ring (taps [cap, slice.cols]) when the row's
// slot owns the ring. Slot, position and owner come from device memory so the launch is graph-replayable.
void dflash_publish_taps(
	backend::half_t* taps,
	const backend::half_t* hidden,
	const pod::BatchRows* d_rows,
	const int* d_owner,
	int m,
	int tap,
	const TapSlice& slice,
	int cap,
	backend::stream_t stream
);

// The same with one ring per slot (taps [slots, cap, slice.cols], slot_stride elements apart): every row publishes to its own slot's ring, so several slots can
// feed a drafter at once. CUDA only (the backends without it keep the single owner ring).
void dflash_publish_taps_slots(
	backend::half_t* taps,
	const backend::half_t* hidden,
	const pod::BatchRows* d_rows,
	int m,
	int tap,
	const TapSlice& slice,
	int cap,
	size_t slot_stride,
	backend::stream_t stream
);

// The same from FP32 rows (a model whose residual stream is FP32), rounded to FP16
void dflash_publish_taps_slots_f32(
	backend::half_t* taps,
	const float* hidden,
	const pod::BatchRows* d_rows,
	int m,
	int tap,
	const TapSlice& slice,
	int cap,
	size_t slot_stride,
	backend::stream_t stream
);

// Reduces mHC residual streams [m, 4, H] (FP32) to one FP16 hidden row per token: mode 0 = mean of the four streams,
// mode k in 1..4 = stream k - 1 (diagnostic selectors for matching the drafter's training-time tap).
void dflash_reduce_streams(
	const float* streams,
	backend::half_t* out,
	int m,
	int H,
	int mode,
	backend::stream_t stream
);

} // namespace engine::spec
