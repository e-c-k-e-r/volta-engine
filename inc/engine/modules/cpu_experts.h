#pragma once

#include "engine/backend/types.h"
#include "engine/runtime/cpu_expert_mailbox.h"
#include <cstdint>

namespace engine::modules {

// Device half of the CPU expert path (see engine/runtime/cpu_expert_service.h). In a MoE layer, after the router and before the experts:
// post() takes the routes to some cold experts (owner[e] = 1: the expert is cold and has a CPU copy; the last floor( frac * d ) of the layer's d
// distinct cold experts) out of the route list (their index becomes -1, so the expert kernels skip them), counts them in
// `heat`, and posts them with the input rows to the rank's mailbox; the GPU experts then run beside the CPU. wait_add() after them waits for the
// service and adds its result into the layer output. Both are no-ops in a layer where no route went to the CPU (ctl[0] = 0). ctl: two device
// uint64 (the sequence of this layer's request, the last sequence used), zero at start
struct CpuExperts {
	static void post( const backend::half_t* x, int* sel, const backend::half_t* rw, int rows, int topk, int hidden, int min_expert, int max_expert,
		const uint8_t* owner, float frac, uint32_t* heat, cpu_moe::Mailbox* mb, uint64_t* ctl, int layer, backend::stream_t stream );
	static void wait_add( backend::half_t* out, int rows, int hidden, cpu_moe::Mailbox* mb, const uint64_t* ctl, backend::stream_t stream );
};

}   // namespace engine::modules
