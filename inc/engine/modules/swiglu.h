#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace engine::modules {

struct SwiGLU {
	static void forward(
		const pod::TensorView& gate,
		const pod::TensorView& up,
		pod::TensorView& out,
		cudaStream_t stream = nullptr
	);
};

} // namespace engine::modules
