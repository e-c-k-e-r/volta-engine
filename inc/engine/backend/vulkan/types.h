#pragma once

// Vulkan backend vocabulary (selected by engine/backend/types.h). Host-only: no Vulkan headers are included here; the
// handles are pointers to opaque structs owned by src/backend/vulkan.
#include <cstdint>

// Qualifier and vector-type spellings that shared headers (core/batch.h, modules/*.h) use in declarations; no-ops / plain
// structs for host code on this backend
#ifndef __host__
	#define __host__
#endif
#ifndef __device__
	#define __device__
#endif
struct int4 { int x, y, z, w; };

namespace engine::backend {
	struct vk_stream;   // command stream: records into a command buffer, submitted on sync, ordered by a timeline semaphore
	struct vk_graph;    // pre-recorded command buffer replayed with vkQueueSubmit
	using stream_t = vk_stream*;
	using graph_t = vk_graph*;
	using graph_exec_t = vk_graph*;

	// 16-bit storage types (the SPIR-V kernels use Float16 storage); conversions live in numeric.h
	struct half_t { uint16_t bits; };
	struct bf16_t { uint16_t bits; };

	// Direction of a copy; Auto is resolved by the allocation table (device addresses are known, host pointers are not)
	enum class Copy { H2D, D2H, D2D, Auto };

	// Pinned host allocation flavors (Vulkan: Default only, mapped flavors report failure with nullptr)
	enum class Pin { Default, Mapped, MappedPortable };
}
