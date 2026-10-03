#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace engine::modules {

struct GDN {
	static void project_gates(
		const pod::TensorView& norm_hidden,
		const pod::TensorView& in_proj_a,
		const pod::TensorView& in_proj_b,
		const pod::TensorView& A_log,
		const pod::TensorView& dt_bias,
		pod::TensorView& gate_out,
		pod::TensorView& beta_out,
		int num_v_heads,
		int hidden_size,
		cudaStream_t stream = nullptr
	);

	static void causal_conv1d_silu(
		pod::TensorView& qkv,
		pod::TensorView& conv_history,
		const pod::TensorView& conv_weights,
		int channels,
		cudaStream_t stream = nullptr
	);

	static void l2_norm_heads(
		pod::TensorView& qkv,
		int num_k_heads,
		int head_dim,
		cudaStream_t stream = nullptr
	);

	static void recurrent_step(
		float* gdn_state,
		const pod::TensorView& qkv,
		const pod::TensorView& gate,
		const pod::TensorView& beta,
		pod::TensorView& attn_out,
		int num_k_heads,
		int num_v_heads,
		int head_dim,
		cudaStream_t stream = nullptr
	);

	static void gated_rmsnorm(
		pod::TensorView& attn_out,
		const pod::TensorView& z,
		const pod::TensorView& norm_weight,
		int num_v_heads,
		int head_dim,
		cudaStream_t stream = nullptr
	);
};

} // namespace engine::modules
