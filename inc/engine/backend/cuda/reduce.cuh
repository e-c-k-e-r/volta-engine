#pragma once

// Warp reductions shared by the CUDA / HIP kernels. The shuffle order (xor, offsets 16 down to 1) fixes the summation order, so every
// kernel that sums through it gets the same bits.
#include <cstdint>

namespace engine::math {

__device__ __forceinline__ float warp_sum( float v ) {
	#pragma unroll
	for ( int o = 16; o > 0; o >>= 1 ) v += __shfl_xor_sync( 0xffffffff, v, o );
	return v;
}

} // namespace engine::math
