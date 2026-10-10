// CPU backend: weak stubs for the moe entry points that have no CPU implementation (CUDA/SM70 only or not yet ported).
// A strong definition in another translation unit overrides these. Declarations come from engine/modules/moe.h.
#include "engine/modules/moe.h"
#include "engine/backend/tiers.h"

namespace engine::modules {

__attribute__(( weak )) void MoERouter::exl3_moe_dispatch( const int* selected_experts, const backend::half_t* routing_weights, int64_t* expert_count, int64_t* token_sorted, backend::half_t* weight_sorted, int* cursors, int num_experts, int routing_first, int routing_last, int4* active_experts, backend::stream_t stream, int rows, int chunk_rows, uint32_t* heat, int top_k ) {
	ENGINE_UNSUPPORTED( "MoERouter::exl3_moe_dispatch" );
	__builtin_unreachable();
}

} // namespace engine::modules
