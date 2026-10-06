#pragma once

#include "engine/config.h"

namespace pod {

// Maximum number of rows (M) evaluated by a single model forward pass.
static constexpr int MAX_BATCH = 16;

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
	int count = 1;
	int slot[MAX_BATCH] = {};
	int pos[MAX_BATCH] = {};

	static BatchRows causal( int count, int base_pos, int slot_id ) {
		ENGINE_ASSERT( count >= 1 && count <= MAX_BATCH, "BatchRows::causal: count out of range" );
		BatchRows r;
		r.count = count;
		for ( int m = 0; m < count; ++m ) {
			r.slot[m] = slot_id;
			r.pos[m] = base_pos + m;
		}
		return r;
	}

	static BatchRows batched( int count, const int* slot_ids, const int* positions ) {
		ENGINE_ASSERT( count >= 1 && count <= MAX_BATCH, "BatchRows::batched: count out of range" );
		BatchRows r;
		r.count = count;
		for ( int m = 0; m < count; ++m ) {
			r.slot[m] = slot_ids[m];
			r.pos[m] = positions[m];
		}
		return r;
	}

	int max_pos() const {
		int p = 0;
		for ( int m = 0; m < count; ++m ) {
			p = MAX( p, pos[m] );
		}
		return p;
	}
};

} // namespace pod
