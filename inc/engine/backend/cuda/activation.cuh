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

// GELU, tanh approximation (gelu_pytorch_tanh)
__device__ __forceinline__ float gelu_tanh( float x ) {
	return 0.5f * x * ( 1.0f + tanhf( 0.7978845608028654f * ( x + 0.044715f * x * x * x ) ) );
}

} // namespace engine::math
