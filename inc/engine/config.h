#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define ENGINE_ARCH_SM70 1
#define ENGINE_NUM_GPUS  4
// SM count the persistent / cooperative launches size their grids to. 0 (default): ask the device at startup (80 on a V100, 82 on a
// 3090, 60 on a 4070 Ti; the architecture alone does not fix it). Nonzero: use this count, capped at the device's (CMake -DENGINE_SM_COUNT=N)
#ifndef ENGINE_SM_COUNT
	#define ENGINE_SM_COUNT 0
#endif
// Opt-in dynamic shared memory per block the big kernels request: V100 allows 96 KB, SM_86 / SM_89 99 KB, so the default fits all three
#ifndef ENGINE_MAX_SMEM
	#define ENGINE_MAX_SMEM 98304
#endif

#if defined(__GNUC__) || defined(__clang__)
	#define ENGINE_LIKELY(x)   __builtin_expect(!!(x), 1)
	#define ENGINE_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
	#define ENGINE_LIKELY(x)   (x)
	#define ENGINE_UNLIKELY(x) (x)
#endif

#ifndef MIN
	#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif
#ifndef MAX
	#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

#define ENGINE_ASSERT(cond, msg) \
	do { \
		if ( ENGINE_UNLIKELY(!(cond)) ) { \
			std::fprintf(stderr, "[FATAL] Assertion failed: %s (%s:%d): %s\n", #cond, __FILE__, __LINE__, msg); \
			std::_Exit(1); \
		} \
	} while (0)

#define CUDA_CHECK(call) \
	do { \
		cudaError_t err = (call); \
		if ( ENGINE_UNLIKELY(err != cudaSuccess) ) { \
			std::fprintf(stderr, "[CUDA ERROR] %s (%s:%d): %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
			std::_Exit(1); \
		} \
	} while (0)

namespace pod {}
namespace impl {}
namespace engine {
	namespace core {}
	namespace comm {}
	namespace modules {}
	namespace models {}
	namespace spec {}
}
