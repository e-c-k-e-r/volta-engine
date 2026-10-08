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
		backend::stream_t stream = nullptr
	);

	// Large-M path: Hadamard, dequantize the weight to FP16, cuBLAS HGEMM (FP32 accumulate), Hadamard. FP16 output.
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
		backend::stream_t stream = nullptr
	);
};

} // namespace engine::modules
