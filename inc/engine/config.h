#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define ENGINE_ARCH_SM70 1
#define ENGINE_NUM_GPUS  4
#define ENGINE_SM_COUNT  80
#define ENGINE_MAX_SMEM  98304

#if defined(__GNUC__) || defined(__clang__)
	#define ENGINE_LIKELY(x)   __builtin_expect(!!(x), 1)
	#define ENGINE_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
	#define ENGINE_LIKELY(x)   (x)
	#define ENGINE_UNLIKELY(x) (x)
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
