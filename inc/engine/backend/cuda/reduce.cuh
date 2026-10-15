#pragma once

// Warp reductions shared by the CUDA / HIP kernels. The shuffle order (xor, offsets 16 down to 1) fixes the summation order, so every
// kernel that sums through it gets the same bits.
#include <cstdint>

namespace engine::math {

// Sum over the lanes of a warp (Top = 16), or over the first 2 * Top lanes (Top = 4 sums the 8 per-warp partials of a block)
template <int Top = 16>
__device__ __forceinline__ float warp_sum( float v ) {
	#pragma unroll
	for ( int o = Top; o > 0; o >>= 1 ) v += __shfl_xor_sync( 0xffffffff, v, o );
	return v;
}

template <int Top = 16>
__device__ __forceinline__ float warp_max( float v ) {
	#pragma unroll
	for ( int o = Top; o > 0; o >>= 1 ) v = fmaxf( v, __shfl_xor_sync( 0xffffffff, v, o ) );
	return v;
}

} // namespace engine::math
