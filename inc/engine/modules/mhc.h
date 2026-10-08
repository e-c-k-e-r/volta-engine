#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace engine::modules {

struct HyperConnection {
	static void init_streams(
		const pod::TensorView& in,
		pod::TensorView& streams,
		int hidden_size = 4096,
		cudaStream_t stream = nullptr,
		int rows = 1                    // independent rows: streams [rows, 4, hidden]
	);

	static void init_streams(
		const half* in,
		float* streams,
		int hidden_size = 4096,
		cudaStream_t stream = nullptr,
		int rows = 1                    // independent rows: streams [rows, 4, hidden]
	);

	static void collapse(
		const pod::TensorView& streams,
		const pod::TensorView& fn,
		const pod::TensorView& base,
		const pod::TensorView& scale,
		pod::TensorView& collapsed_out,
		pod::TensorView& post_out,
		pod::TensorView& comb_out,
		pod::TensorView& inv_rms_buf,
		pod::TensorView& mix_logits_buf,
		pod::TensorView& pre_buf,
		int hidden_size = 4096,
		int sinkhorn_iters = 20,
		float eps = 1e-6f,
		cudaStream_t stream = nullptr,
		int rows = 1                    // independent rows: streams [rows, 4, hidden]
	);

	static void collapse(
		const float* streams,
		const float* fn,
		const float* base,
		const float* scale,
		half* collapsed_out,
		float* post_out,
		float* comb_out,
		float* inv_rms_buf,
		float* mix_logits_buf,
		float* pre_buf,
		int hidden_size = 4096,
		int sinkhorn_iters = 20,
		float eps = 1e-6f,
		cudaStream_t stream = nullptr,
		int rows = 1                    // independent rows: streams [rows, 4, hidden]
	);

	// collapse followed by RMSNorm(weight) of the collapsed row, in one launch and bit-identical to the pair. Returns
	// false (nothing launched) when the shape does not fit, so the caller runs collapse + RMSNorm::forward instead.
	static bool collapse_norm(
		const pod::TensorView& streams,
		const pod::TensorView& fn,
		const pod::TensorView& base,
		const pod::TensorView& scale,
		const pod::TensorView& norm_weight,
		pod::TensorView& norm_out,
		pod::TensorView& post_out,
		pod::TensorView& comb_out,
		pod::TensorView& inv_rms_buf,
		pod::TensorView& mix_logits_buf,
		pod::TensorView& pre_buf,
		int hidden_size,
		int sinkhorn_iters,
		float eps,
		float norm_eps,
		float norm_bias,
		cudaStream_t stream,
		int rows
	);

	static void update(
		pod::TensorView& streams,
		const pod::TensorView& sublayer_out,
		const pod::TensorView& post,
		const pod::TensorView& comb,
		int hidden_size = 4096,
		cudaStream_t stream = nullptr,
		int rows = 1                    // independent rows: streams [rows, 4, hidden]
	);

	static void update(
		pod::TensorView& streams_out,
		const pod::TensorView& streams_in,
		const pod::TensorView& sublayer_out,
		const pod::TensorView& post,
		const pod::TensorView& comb,
		int hidden_size = 4096,
		cudaStream_t stream = nullptr,
		int rows = 1                    // independent rows: streams [rows, 4, hidden]
	);

	static void update(
		float* streams,
		const half* sublayer_out,
		const float* post,
		const float* comb,
		int hidden_size = 4096,
		cudaStream_t stream = nullptr,
		int rows = 1                    // independent rows: streams [rows, 4, hidden]
	);

	static void update(
		float* streams_out,
		const float* streams_in,
		const half* sublayer_out,
		const float* post,
		const float* comb,
		int hidden_size = 4096,
		cudaStream_t stream = nullptr,
		int rows = 1                    // independent rows: streams [rows, 4, hidden]
	);
};

struct HyperHead {
	static void forward(
		const pod::TensorView& streams,
		pod::TensorView& final_hidden,
		int hidden_size = 4096,
		bool mean = true,
		cudaStream_t stream = nullptr,
		int rows = 1                    // independent rows: streams [rows, 4, hidden]
	);

	static void mean(
		const pod::TensorView& streams,
		pod::TensorView& final_hidden,
		int hidden_size = 4096,
		cudaStream_t stream = nullptr,
		int rows = 1                    // independent rows: streams [rows, 4, hidden]
	);

	static void mean(
		const float* streams,
		half* final_hidden,
		int hidden_size = 4096,
		cudaStream_t stream = nullptr,
		int rows = 1                    // independent rows: streams [rows, 4, hidden]
	);
};

} // namespace engine::modules
