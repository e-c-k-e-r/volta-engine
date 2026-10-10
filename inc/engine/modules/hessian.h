#pragma once

#include "engine/config.h"
#include "engine/backend/types.h"

namespace engine::modules {

// Calibration support: running sum of x^T x over the rows of a projection input (the Hessian of the layer-wise quantization loss).
struct Hessian {
	// h [k, k] fp32 += x^T x for x [rows, k] fp16 (fp32 accumulation)
	static void accumulate( float* h, const backend::half_t* x, int rows, int k, backend::stream_t stream );
};

} // namespace engine::modules
