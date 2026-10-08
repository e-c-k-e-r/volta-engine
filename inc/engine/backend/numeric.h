#pragma once

#include "engine/backend/types.h"

// Scalar conversions between the backend's 16-bit float types and float, for host-side code (weight conversion in the
// loaders): to_float( half_t / bf16_t ), from_float_half( float ), from_float_bf16( float ), all round to nearest even.
#if defined(ENGINE_BACKEND_CUDA) && ENGINE_BACKEND_CUDA == 1
	#include "engine/backend/cuda/numeric.h"
#elif defined(ENGINE_BACKEND_CPU) && ENGINE_BACKEND_CPU == 1
	#include "engine/backend/cpu/numeric.h"
#elif defined(ENGINE_BACKEND_VULKAN) && ENGINE_BACKEND_VULKAN == 1
	#include "engine/backend/vulkan/numeric.h"
#endif
