#pragma once

#include "engine/config.h"

// Compile-time backend selection. Exactly one backend is active; CUDA is the default so existing builds are unchanged.
// Types here are aliases of the backend's native types (no marshalling in kernel-facing code); calls live in
// engine/backend/runtime.h as inline forwards. There are no virtual calls anywhere on this path.
#if !defined(ENGINE_BACKEND_CUDA) && !defined(ENGINE_BACKEND_VULKAN) && !defined(ENGINE_BACKEND_CPU)
	#define ENGINE_BACKEND_CUDA 1
#endif

#if defined(ENGINE_BACKEND_CUDA) && ENGINE_BACKEND_CUDA == 1
	#include <cuda_runtime.h>
	#include <cuda_fp16.h>
	#include <cuda_bf16.h>

namespace engine::backend {
	using stream_t = cudaStream_t;
	using graph_t = cudaGraph_t;
	using graph_exec_t = cudaGraphExec_t;
	using half_t = __half;
	using bf16_t = __nv_bfloat16;

	// Direction of a copy; the call sites name it instead of the backend's enum
	// Auto lets the runtime infer the direction from the pointers (unified addressing), as for loads from mapped files
	enum class Copy { H2D, D2H, D2D, Auto };

	// Pinned host allocation flavors
	enum class Pin { Default, Mapped, MappedPortable };
}
#else
	#error "engine/backend/types.h: no backend implemented for this configuration (set ENGINE_BACKEND_CUDA=1)"
#endif
