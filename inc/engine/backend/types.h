#pragma once

#include "engine/config.h"

// Compile-time backend selection. Exactly one backend is active; CUDA is the default so existing builds are unchanged.
// Each backend supplies the same vocabulary in namespace engine::backend (stream_t, graph_t, graph_exec_t, half_t, bf16_t,
// enum Copy, enum Pin) as aliases of its native types (no marshalling in kernel-facing code); calls live in
// engine/backend/runtime.h as inline forwards. There are no virtual calls anywhere on this path.
#if !defined(ENGINE_BACKEND_CUDA) && !defined(ENGINE_BACKEND_VULKAN) && !defined(ENGINE_BACKEND_CPU)
	#define ENGINE_BACKEND_CUDA 1
#endif

#if defined(ENGINE_BACKEND_CUDA) && ENGINE_BACKEND_CUDA == 1
	#include "engine/backend/cuda/types.h"
#elif defined(ENGINE_BACKEND_CPU) && ENGINE_BACKEND_CPU == 1
	#include "engine/backend/cpu/types.h"
#elif defined(ENGINE_BACKEND_VULKAN) && ENGINE_BACKEND_VULKAN == 1
	#include "engine/backend/vulkan/types.h"
#else
	#error "engine/backend/types.h: unknown backend configuration"
#endif
