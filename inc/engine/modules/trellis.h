#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/backend/types.h"

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
		bool mul1 = false,
		backend::stream_t stream = nullptr,
		bool volta_tiles = false
	);

	// Large-M path: Hadamard, dequantize the weight to FP16, Gemm::nn (FP32 accumulate), Hadamard. FP16 output.
	// forward() routes here above a row threshold; exposed so tests can compare it against the fused kernels.
	static void forward_recon(
		const pod::TensorView& x,
		const pod::TensorView& trellis,
		const pod::TensorView& suh,
		const pod::TensorView& svh,
		pod::TensorView& out,
		int bits,
		bool mcg,
		bool mul1 = false,
		backend::stream_t stream = nullptr,
		bool volta_tiles = false
	);
};

// Whether this build reads EXL3 tiles in the Volta order (LinearWeight::volta_tiles): CPU and Vulkan builds do, CUDA builds with SM_70 code on SM_70 devices
bool trellis_volta_tiles_supported();

} // namespace engine::modules
