#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace engine::modules {

struct MoERouter {
	static void route_topk(
		const pod::TensorView& x,
		const pod::TensorView& router_weights,
		const pod::TensorView& e_score_bias,
		int* d_topk_indices,
		half* d_topk_weights,
		int hidden_size,
		int num_experts = 288,
		int top_k = 8,
		float routed_scaling_factor = 2.5f,
		cudaStream_t stream = nullptr
	);

	static void combine(
		const half* expert_parts,
		const half* topk_weights,
		const half* shared_out,
		half* output,
		int hidden_size,
		int top_k = 8,
		cudaStream_t stream = nullptr
	);
};

} // namespace engine::modules
