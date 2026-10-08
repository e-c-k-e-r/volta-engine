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

// 1 when the backend can capture and replay a launch sequence (graph_* functions); 0 when GraphCache must run the body
// directly. A backend's runtime.h may define it first; the default is 1 (CUDA).
#ifndef ENGINE_BACKEND_HAS_GRAPHS
	#define ENGINE_BACKEND_HAS_GRAPHS 1
#endif
