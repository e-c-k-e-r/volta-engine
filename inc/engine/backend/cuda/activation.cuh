#pragma once

// Gate activations shared by the CUDA / HIP kernels, spelled once with the fast exponential the kernels have always used.
#include <cmath>

namespace engine::math {

__device__ __forceinline__ float sigmoid( float x ) {
	return 1.0f / ( 1.0f + __expf( -x ) );
}

__device__ __forceinline__ float silu( float x ) {
	return x / ( 1.0f + __expf( -x ) );
}

} // namespace engine::math
