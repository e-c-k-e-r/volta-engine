#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/modules/linear.h"
#include "engine/backend/types.h"

#define MOE_MAX_GROUPS 8
#define MOE_MAX_EXPERTS 288   // experts per rank: 72 expert-parallel, all 288 with each expert sharded over the ranks

namespace pod {

struct MoELayerWeights {
	QuantType type = QuantType::TRELLIS;
	int num_local_experts = 72;
	LinearWeight gate[MOE_MAX_EXPERTS];
	LinearWeight up[MOE_MAX_EXPERTS];
	LinearWeight down[MOE_MAX_EXPERTS];
	void* device_ptrs = nullptr;
	uint32_t* heat = nullptr;          // [num_local_experts] routes per expert since the last reset (device), optional
};

// Device scratch for one MoE dispatch (single token, top_k routes). Allocated once at init.
struct MoEScratch {
	int64_t* expert_count = nullptr;   // [num_local_experts + 1]
	int64_t* token_sorted = nullptr;   // [top_k]
	engine::backend::half_t*    weight_sorted = nullptr;  // [top_k]
	int*     cursors = nullptr;        // [num_local_experts + 1]
	int4*    active_experts = nullptr; // [num_local_experts + 1]
	engine::backend::half_t*    state_g = nullptr;        // [groups, top_k, hidden]
	engine::backend::half_t*    state_u = nullptr;
	engine::backend::half_t*    interm_g = nullptr;       // [groups, top_k, intermediate]
	engine::backend::half_t*    interm_u = nullptr;
	engine::backend::half_t*    planes = nullptr;         // [MOE_MAX_GROUPS, max_rows, hidden] per-group output planes
	// Fused decode path (exl3_moe_coop.cuh): scratch for up to 16 rows
	engine::backend::half_t*    coop_had_g = nullptr;     // [128 slots, hidden]
	engine::backend::half_t*    coop_had_u = nullptr;
	float*   coop_gu_g = nullptr;      // [2 split-k, 128 slots, intermediate]
	float*   coop_gu_u = nullptr;
	engine::backend::half_t*    coop_act = nullptr;       // [128 slots, intermediate]
	float*   coop_d_out = nullptr;     // [128 slots, hidden]
	int*     coop_ctr = nullptr;       // completion counters + run table, zero between calls
};

} // namespace pod

namespace engine::modules {

struct MoERouter {
	static void route_topk(
		const pod::TensorView& x,
		const pod::TensorView& router_weights,
		const pod::TensorView& e_score_bias,
		int* d_topk_indices,
		backend::half_t* d_topk_weights,
		int hidden_size,
		int num_experts = 288,
		int top_k = 8,
		float routed_scaling_factor = 2.5f,
		backend::stream_t stream = nullptr,
		int rows = 1                    // x [rows, hidden] -> indices / weights [rows, top_k]
	);

	static void combine(
		const backend::half_t* expert_parts,
		const backend::half_t* topk_weights,
		const backend::half_t* shared_out,
		backend::half_t* output,
		int hidden_size,
		int top_k = 8,
		backend::stream_t stream = nullptr
	);

	static void accumulate(
		backend::half_t* accum_out,
		const backend::half_t* expert_out,
		backend::half_t weight,
		int hidden_size,
		backend::stream_t stream = nullptr
	);

	static void accumulate(
		pod::TensorView& accum_out,
		const pod::TensorView& expert_out,
		backend::half_t weight,
		int hidden_size,
		backend::stream_t stream = nullptr
	);

	static void exl3_moe_dispatch(
		const int* selected_experts,
		const backend::half_t* routing_weights,
		int64_t* expert_count,
		int64_t* token_sorted,
		backend::half_t* weight_sorted,
		int* cursors,
		int num_experts,
		int routing_first,
		int routing_last,
		int4* active_experts,
		backend::stream_t stream = nullptr,
		int rows = 1,                   // routes = 8 * rows, token index = route / 8
		int chunk_rows = 4,             // rows per work item (the expert GEMM tile)
		uint32_t* heat = nullptr        // optional per-expert route counters, incremented by this dispatch
	);
};

struct MoE {
	static void init_device_ptrs(
		pod::MoELayerWeights& weights,
		pod::MemoryArena& arena,
		backend::stream_t stream = nullptr
	);

	static pod::MoEScratch alloc_scratch(
		pod::MemoryArena& arena,
		int num_local_experts,
		int top_k,
		int hidden_size,
		int intermediate_size,
		int max_rows = 1
	);

	// Routed-expert FFN for one token: out = sum_k w_k * expert_k( x ) over the experts in
	// [start_expert, end_expert). Fully device-resident (no host sync); dispatches on weights.type.
	static void forward(
		const pod::TensorView& x,
		const pod::MoELayerWeights& weights,
		const int* d_topk_indices,
		const backend::half_t* d_topk_weights,
		pod::TensorView& out,
		int start_expert,
		int end_expert,
		const pod::MoEScratch& scratch,
		float act_limit = 10.0f,
		backend::stream_t stream = nullptr
	);
};

} // namespace engine::modules

namespace ext {

void exl3_moe(
	const void* hidden_state,
	void* output_state,
	const int64_t* expert_count,
	const int64_t* token_sorted,
	const engine::backend::half_t* weight_sorted,
	engine::backend::half_t* temp_state_g,
	engine::backend::half_t* temp_state_u,
	engine::backend::half_t* temp_intermediate_g,
	engine::backend::half_t* temp_intermediate_u,
	int act_function,
	int K_gate, int K_up, int K_down,
	const uint16_t** gate_trellis, const engine::backend::half_t** gate_suh, const engine::backend::half_t** gate_svh,
	const uint16_t** up_trellis, const engine::backend::half_t** up_suh, const engine::backend::half_t** up_svh,
	const uint16_t** down_trellis, const engine::backend::half_t** down_suh, const engine::backend::half_t** down_svh,
	bool gate_mcg, bool gate_mul1,
	bool up_mcg, bool up_mul1,
	bool down_mcg, bool down_mul1,
	float act_limit,
	int num_active,
	const int4* active_experts,
	int num_groups,
	size_t out_plane_stride,        // elements between the per-group output planes
	int wide,                       // expert GEMM tile: 0: 4 rows, 2: 16 rows above 4
	int hidden_dim,
	int intermediate_dim,           // this rank's width of the expert intermediate
	int num_experts,
	engine::backend::stream_t stream = nullptr
);

} // namespace ext
