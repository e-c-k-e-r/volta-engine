// CPU backend: weak stubs for the moe entry points that have no CPU implementation (CUDA/SM70 only or not yet ported).
// A strong definition in another translation unit overrides these. Declarations come from engine/modules/moe.h.
#include "engine/modules/moe.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

__attribute__(( weak )) pod::MoEScratch MoE::alloc_scratch( pod::MemoryArena& arena, int num_local_experts, int top_k, int hidden_size, int intermediate_size, int max_rows ) {
	ENGINE_UNSUPPORTED( "MoE::alloc_scratch" );
	__builtin_unreachable();
}

__attribute__(( weak )) void MoE::forward( const pod::TensorView& x, const pod::MoELayerWeights& weights, const int* d_topk_indices, const backend::half_t* d_topk_weights, pod::TensorView& out, int start_expert, int end_expert, const pod::MoEScratch& scratch, float act_limit, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "MoE::forward" );
	__builtin_unreachable();
}

__attribute__(( weak )) void MoE::init_device_ptrs( pod::MoELayerWeights& weights, pod::MemoryArena& arena, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "MoE::init_device_ptrs" );
	__builtin_unreachable();
}

__attribute__(( weak )) void MoERouter::accumulate( backend::half_t* accum_out, const backend::half_t* expert_out, backend::half_t weight, int hidden_size, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "MoERouter::accumulate" );
	__builtin_unreachable();
}

__attribute__(( weak )) void MoERouter::accumulate( pod::TensorView& accum_out, const pod::TensorView& expert_out, backend::half_t weight, int hidden_size, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "MoERouter::accumulate" );
	__builtin_unreachable();
}

__attribute__(( weak )) void MoERouter::combine( const backend::half_t* expert_parts, const backend::half_t* topk_weights, const backend::half_t* shared_out, backend::half_t* output, int hidden_size, int top_k, backend::stream_t stream ) {
	ENGINE_UNSUPPORTED( "MoERouter::combine" );
	__builtin_unreachable();
}

__attribute__(( weak )) void MoERouter::exl3_moe_dispatch( const int* selected_experts, const backend::half_t* routing_weights, int64_t* expert_count, int64_t* token_sorted, backend::half_t* weight_sorted, int* cursors, int num_experts, int routing_first, int routing_last, int4* active_experts, backend::stream_t stream, int rows, int chunk_rows, uint32_t* heat ) {
	ENGINE_UNSUPPORTED( "MoERouter::exl3_moe_dispatch" );
	__builtin_unreachable();
}

__attribute__(( weak )) void MoERouter::route_topk( const pod::TensorView& x, const pod::TensorView& router_weights, const pod::TensorView& e_score_bias, int* d_topk_indices, backend::half_t* d_topk_weights, int hidden_size, int num_experts, int top_k, float routed_scaling_factor, backend::stream_t stream, int rows ) {
	ENGINE_UNSUPPORTED( "MoERouter::route_topk" );
	__builtin_unreachable();
}

} // namespace engine::modules
