#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/backend/types.h"
#include <cuda_fp16.h>

namespace engine::modules {

struct RMSNorm {
	static void forward(
		const pod::TensorView& x,
		const pod::TensorView& weight,
		pod::TensorView& out,
		float eps = 1e-6f,
		float constant_bias = 1.0f,
		backend::stream_t stream = nullptr
	);

	static void forward_heads(
		pod::TensorView& qk,
		const pod::TensorView& weight,
		int num_heads,
		int head_dim,
		float eps = 1e-6f,
		float constant_bias = 1.0f,
		backend::stream_t stream = nullptr
	);
};

struct Elementwise {
	static void add_residual(
		pod::TensorView& hidden,
		const pod::TensorView& residual,
		backend::stream_t stream = nullptr
	);

	static void argmax(
		const pod::TensorView& logits,
		int* d_out_idx,
		int numel,
		backend::stream_t stream = nullptr
	);

	static void batched_argmax(
		const pod::TensorView& logits,
		int* d_out_idx,
		int numel,
		int batch_size,
		backend::stream_t stream = nullptr
	);

	// Scans only the first `columns` entries of each row and adds `index_offset` to the index, for a
	// vocabulary-parallel LM head whose shard is padded past its real width. Also returns the winning
	// logit so the caller can reduce across shards. d_out_val may be null.
	static void batched_argmax_range(
		const pod::TensorView& logits,
		int* d_out_idx,
		float* d_out_val,
		int columns,
		int batch_size,
		int index_offset,
		backend::stream_t stream = nullptr
	);
};

} // namespace engine::modules
