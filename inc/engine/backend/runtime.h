#pragma once

#include "engine/backend/types.h"

// Thin runtime layer the models, runtime and core use instead of naming the vendor API (device, stream, copies, memset,
// pinned host memory, graph capture/replay, ENGINE_CHECK). The function set is the same for every backend; see the
// per-backend headers.
#if defined(ENGINE_BACKEND_CUDA) && ENGINE_BACKEND_CUDA == 1
	#include "engine/backend/cuda/runtime.h"
#elif defined(ENGINE_BACKEND_CPU) && ENGINE_BACKEND_CPU == 1
	#include "engine/backend/cpu/runtime.h"
#elif defined(ENGINE_BACKEND_VULKAN) && ENGINE_BACKEND_VULKAN == 1
	#include "engine/backend/vulkan/runtime.h"
#endif
