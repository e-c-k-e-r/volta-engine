#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/modules/linear.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace pod {

struct MoELayerWeights {
	QuantType type = QuantType::TRELLIS;
	int num_local_experts = 72;
	LinearWeight gate[72];
	LinearWeight up[72];
	LinearWeight down[72];
	void* device_ptrs = nullptr;
};

} // namespace pod

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

	static void accumulate(
		half* accum_out,
		const half* expert_out,
		half weight,
		int hidden_size,
		cudaStream_t stream = nullptr
	);

	static void accumulate(
		pod::TensorView& accum_out,
		const pod::TensorView& expert_out,
		half weight,
		int hidden_size,
		cudaStream_t stream = nullptr
	);

	static void exl3_moe_dispatch(
		const int* selected_experts,
		const half* routing_weights,
		int64_t* expert_count,
		int64_t* token_sorted,
		half* weight_sorted,
		int* cursors,
		int num_experts,
		int routing_first,
		int routing_last,
		int4* active_experts,
		cudaStream_t stream = nullptr
	);
};

struct MoE {
	static void init_device_ptrs(
		pod::MoELayerWeights& weights,
		pod::MemoryArena& arena,
		cudaStream_t stream = nullptr
	);

	static void forward(
		const pod::TensorView& x,
		const pod::MoELayerWeights& weights,
		const pod::TensorView& topk_idx,
		const pod::TensorView& topk_weight,
		pod::TensorView& out,
		int start_expert,
		int end_expert,
		half* temp_state_g,
		half* temp_state_u,
		half* temp_intermediate_g,
		half* temp_intermediate_u,
		int64_t* expert_count,
		int64_t* token_sorted,
		half* weight_sorted,
		int* cursors,
		int4* active_experts,
		cudaStream_t stream = nullptr
	);
};

} // namespace engine::modules

namespace ext {

void exl3_moe(
	const void* hidden_state,
	void* output_state,
	const int64_t* expert_count,
	const int64_t* token_sorted,
	const half* weight_sorted,
	half* temp_state_g,
	half* temp_state_u,
	half* temp_intermediate_g,
	half* temp_intermediate_u,
	int act_function,
	int K_gate, int K_up, int K_down,
	const uint16_t** gate_trellis, const half** gate_suh, const half** gate_svh,
	const uint16_t** up_trellis, const half** up_suh, const half** up_svh,
	const uint16_t** down_trellis, const half** down_suh, const half** down_svh,
	bool gate_mcg, bool gate_mul1,
	bool up_mcg, bool up_mul1,
	bool down_mcg, bool down_mul1,
	float act_limit,
	int num_active,
	const int4* active_experts,
	cudaStream_t stream = nullptr
);

} // namespace ext
