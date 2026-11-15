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

// count (2 to 4) projections in one launch at one or two rows (the int8 GEMV over their columns together), projection t of input x[t]
// (the same row count and width for all): every weight of width bits, the same codebook and tile order, each output a multiple of 256
// columns. False when the call does not qualify (nothing is launched: the caller runs the projections one by one)
bool trellis_forward_multi( const pod::TensorView* x, int count, const pod::TensorView* trellis, const pod::TensorView* suh, const pod::TensorView* svh,
	pod::TensorView* const* out, int bits, bool mcg, bool mul1, backend::stream_t stream, bool volta_tiles );

// Whether this build reads EXL3 tiles in the Volta order (LinearWeight::volta_tiles): CPU and Vulkan builds do, CUDA builds with SM_70 code on SM_70 devices
bool trellis_volta_tiles_supported();

} // namespace engine::modules
