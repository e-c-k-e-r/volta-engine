#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include <cuda_runtime.h>

namespace engine::modules {

struct LinearNVFP4 {
	static void forward(
		const pod::TensorView& x,
		const pod::TensorView& codes,
		const pod::TensorView& scales,
		float gscale,
		pod::TensorView& out,
		cudaStream_t stream = nullptr
	);
};

} // namespace engine::modules
