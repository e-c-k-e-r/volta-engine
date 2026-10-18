#pragma once

#include "engine/backend/types.h"
#include "engine/core/arena.h"
#include "engine/modules/expert_tier.h"
#include "engine/runtime/cpu_expert_mailbox.h"
#include <cstdint>

namespace engine::cpu_moe {

// What a model keeps to use the CPU expert path (engine/modules/cpu_experts.h): the rank's mailbox and the device sequence counters
struct CpuExpertHandle {
	Mailbox* mb = nullptr;
	uint64_t* ctl = nullptr;
	float frac = 0.0f;               // share of a layer's distinct cold experts handed to the CPU (cpu_experts)
	bool on() const { return mb != nullptr; }
	// Whether tier layer `li` takes part in a step of `rows` rows (prefill chunks read the cold experts as usual)
	bool usable( const modules::ExpertTier& tier, int li, int rows ) const {
		return on() && rows <= kMailboxRows && li >= 0 && li < static_cast<int>( tier.layers.size() ) && tier.layers[li].cpu_owner_dev != nullptr;
	}
};

// cpu_experts > 0 with an active expert tier: starts the service (once), creates the rank's mailbox, gives every cold expert of every tier layer a
// CPU-layout copy (trellis or NVFP4, from its pinned host codes and device scale vectors) and keeps that true across swaps. The experts of tier
// layer i are numbered first_expert + e; act_limit is the SwiGLU clamp of the model (0 = plain SiLU)
CpuExpertHandle attach_cpu_experts( modules::ExpertTier& tier, int rank, int first_expert, int hidden, float act_limit, pod::MemoryArena& device, backend::stream_t stream );

}   // namespace engine::cpu_moe
