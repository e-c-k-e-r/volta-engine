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
	int hidden;   // H
	int inter;    // I
};

struct NVFP4MoE {
	// One row: h[k] = silu(x @ gate[e_k]^T) * (x @ up[e_k]^T) for each of the n_routes routes, then out = sum_k w[k] * (h[k] @ down[e_k]^T).
	// counters: at least hidden/32 zero-initialised ints (the kernel leaves them zero); partial: n_routes * hidden floats; h: n_routes * inter halves.
	static void forward_row(
		const NVFP4ExpertTable& t, const half* x, const int* route_idx, const half* route_w, int n_routes,
		half* h, float* partial, int* counters, half* out, backend::stream_t stream
	);
};

} // namespace engine::modules
