#pragma once

#include "engine/config.h"
#include "engine/backend/types.h"
#include <cstdint>

// Rows per pass of the NVFP4 expert kernels: the dispatch lists, the hidden-state scratch and the route partials are sized for this many. A pass
// reads each active expert's weights again, so the larger it is the fewer times a prefill chunk streams them (512 = one pass per chunk)
#ifndef MOE_NVFP4_ROWS
#define MOE_NVFP4_ROWS 512
#endif

namespace engine::modules {

// Per-rank NVFP4 expert table. Pointer arrays are device-resident, indexed by local expert id; a route whose id is >= num_local is skipped.
// Codes / scales use the LinearNVFP4 prepacked tile layout; g* are the per-matrix global scales (weight_scale_2).
struct NVFP4ExpertTable {
	const uint8_t* const* gate_c;
	const uint8_t* const* gate_s;
	const uint8_t* const* up_c;
	const uint8_t* const* up_s;
	const uint8_t* const* down_c;
	const uint8_t* const* down_s;
	const float* gate_g;
	const float* up_g;
	const float* down_g;
	int num_local;
	int min_expert;   // global id of local expert 0; routes outside [min_expert, min_expert + num_local) are skipped
	int hidden;       // H
	int inter;        // I
};

inline int nvfp4_disp_ints( int num_local ) { return 4 + 2 * num_local + num_local * MOE_NVFP4_ROWS; }

struct NVFP4MoE {
	// rows x topk routes: h[r][k] = silu(x[r] @ gate[e]^T) * (x[r] @ up[e]^T), then out[r] = sum_k w[r][k] * (h[r][k] @ down[e]^T) in fixed route order.
	// Routes are grouped by expert (a dispatch pass), so an expert's weights are read once for up to 8 rows (more rows: once per chunk of 8).
	// rows <= MOE_NVFP4_ROWS. disp: nvfp4_disp_ints(num_local) ints; counters: hidden/32 zero-initialised ints (left zero); partial: rows * topk * hidden floats;
	// h: rows * topk * inter halves; heat optional. act_limit > 0: the swiglu limit (up clamped both ways, the activated gate from above).
	static void forward_rows(
		const NVFP4ExpertTable& t, const backend::half_t* x, const int* route_idx, const backend::half_t* route_w, int rows, int topk,
		backend::half_t* h, float* partial, int* disp, int* counters, uint32_t* heat, float act_limit, backend::half_t* out, backend::stream_t stream
	);
};

} // namespace engine::modules
