#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace engine::modules {

struct RMSNorm {
	static void forward(
		const pod::TensorView& x,
		const pod::TensorView& weight,
		pod::TensorView& out,
		float eps = 1e-6f,
		float constant_bias = 1.0f,
		cudaStream_t stream = nullptr
	);

	static void forward_heads(
		pod::TensorView& qk,
		const pod::TensorView& weight,
		int num_heads,
		int head_dim,
		float eps = 1e-6f,
		float constant_bias = 1.0f,
		cudaStream_t stream = nullptr
	);
};

struct Elementwise {
	static void add_residual(
		pod::TensorView& hidden,
		const pod::TensorView& residual,
		cudaStream_t stream = nullptr
	);

	static void argmax(
		const pod::TensorView& logits,
		int* d_out_idx,
		int numel,
		cudaStream_t stream = nullptr
	);
};

} // namespace engine::modules
