#pragma once

#include "engine/config.h"
#include "engine/backend/types.h"
#include <cstddef>

namespace pod {

// Maximum rows of a decode / speculative forward: the sizes a CUDA graph is captured for.
static constexpr int MAX_BATCH = 16;
// Capacity of a BatchRows: a prefill chunk may run up to this many rows in one eager forward.
static constexpr int MAX_ROWS = 512;

// Per-row execution descriptor for a single forward pass, passed by value into kernels.
//
// Mode A (speculative verify / chunked prefill, single request):
//   rows form a causal candidate sequence inside one slot: slot[m] = s, pos[m] = base_pos + m.
// Mode B (concurrent batching, multi-client):
//   row m is one token of an independent request: slot[m] = s_m, pos[m] = p_m (slots distinct).
//
// Stateful modules (KV cache, GDN recurrence, conv history) walk rows in order, so both modes
// share one code path: a causal block simply re-uses the same slot state row after row.
struct BatchRows {
	// Which rows need logits. Only honoured for chunks above MAX_BATCH rows (the eager prefill path); a prefill
	// chunk whose logits nobody reads skips the final norm and LM head, and the last chunk computes only its
	// final row (written to logits row 0).
	enum LogitsMode : int { LOGITS_ALL = 0, LOGITS_NONE = 1, LOGITS_LAST = 2 };

	int count = 1;
	int logits_mode = LOGITS_ALL;
	int slot[MAX_ROWS] = {};
	int pos[MAX_ROWS] = {};
	// Multimodal rotary positions (M-RoPE). With mrope != 0, row m rotates by rope_pos[0..2][m] (time, height, width) instead of
	// pos[m]; pos[m] stays the KV index. Only the prefix up to rope_pos is uploaded when mrope is 0.
	int mrope = 0;
	int rope_pos[3][MAX_ROWS];

	size_t upload_bytes() const { return mrope ? sizeof( BatchRows ) : offsetof( BatchRows, rope_pos ); }

	static BatchRows causal( int count, int base_pos, int slot_id, int logits_mode = LOGITS_ALL ) {
		ENGINE_ASSERT( count >= 1 && count <= MAX_ROWS, "BatchRows::causal: count out of range" );
		BatchRows r;
		r.count = count;
		r.logits_mode = logits_mode;
		for ( int m = 0; m < count; ++m ) {
			r.slot[m] = slot_id;
			r.pos[m] = base_pos + m;
		}
		return r;
	}

	static BatchRows batched( int count, const int* slot_ids, const int* positions ) {
		ENGINE_ASSERT( count >= 1 && count <= MAX_ROWS, "BatchRows::batched: count out of range" );
		BatchRows r;
		r.count = count;
		for ( int m = 0; m < count; ++m ) {
			r.slot[m] = slot_ids[m];
			r.pos[m] = positions[m];
		}
		return r;
	}

	__host__ __device__ int max_pos() const {
		int p = 0;
		for ( int m = 0; m < count; ++m ) {
			p = MAX( p, pos[m] );
		}
		return p;
	}
};

// Device-side slot resolution for per-slot state buffers (graph-capturable: the slot is read
// from a device-resident BatchRows at kernel time, not baked into launch arguments).
// rows == nullptr => base pointer used as-is.
struct SlotRef {
	const BatchRows* rows = nullptr; // device pointer
	int row = 0;
	size_t stride = 0;               // elements per slot

	template <typename T>
	__host__ __device__ T* apply( T* base ) const {
		return rows ? base + static_cast<size_t>( rows->slot[row] ) * stride : base;
	}
};

} // namespace pod
