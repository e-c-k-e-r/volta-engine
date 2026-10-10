#pragma once

#include "engine/config.h"
#include "engine/backend/types.h"
#include <cstdint>

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

struct NVFP4MoE {
	// rows x topk routes: h[r][k] = silu(x[r] @ gate[e]^T) * (x[r] @ up[e]^T), then out[r] = sum_k w[r][k] * (h[r][k] @ down[e]^T) in fixed route order.
	// One thread block per (route, 32-column tile): every route streams its expert's weights (rows that share an expert read them again).
	// counters: rows * hidden/32 zero-initialised ints (left zero); partial: rows * topk * hidden floats; h: rows * topk * inter halves; heat optional.
	static void forward_rows(
		const NVFP4ExpertTable& t, const half* x, const int* route_idx, const half* route_w, int rows, int topk,
		half* h, float* partial, int* counters, uint32_t* heat, half* out, backend::stream_t stream
	);
};

} // namespace engine::modules
