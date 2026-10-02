#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include <cuda_runtime.h>

namespace engine::modules {

struct LinearTrellis {
	static void forward(
		const pod::TensorView& x,
		const pod::TensorView& trellis,
		const pod::TensorView& suh,
		const pod::TensorView& svh,
		pod::TensorView& out,
		int bits,
		bool mcg,
		cudaStream_t stream = nullptr
	);
};

} // namespace engine::modules
