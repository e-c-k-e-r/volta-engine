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

// Sum of v over the whole block (a multiple of 32 threads, at most MaxWarps warps, 1D to 3D): a butterfly inside each warp, the warp totals
// through shared memory, then the totals through warp 0's butterfly; every thread gets the sum. One call per __syncthreads interval (the
// shared array is reused).
template <int MaxWarps>
__device__ __forceinline__ float block_sum( float v ) {
	__shared__ float s_warp[MaxWarps];
	v = warp_sum( v );
	const int tid = threadIdx.x + blockDim.x * ( threadIdx.y + blockDim.y * threadIdx.z );   // blocks may be 2D (32 x 4)
	const int warp = tid >> 5, lane = tid & 31;
	if ( lane == 0 ) s_warp[warp] = v;
	__syncthreads();
	if ( warp == 0 ) {
		float b = ( lane < ( static_cast<int>( blockDim.x * blockDim.y * blockDim.z ) >> 5 ) ) ? s_warp[lane] : 0.0f;
		b = warp_sum<MaxWarps / 2>( b );
		if ( lane == 0 ) s_warp[0] = b;
	}
	__syncthreads();
	return s_warp[0];
}

// The same block sum with the warp totals added in warp order (0, 1, 2, ...) in every thread, which rounds differently from the butterfly
// of block_sum; kernels that always summed that way keep their bits. Same one-call-per-__syncthreads rule.
template <int MaxWarps>
__device__ __forceinline__ float block_sum_ordered( float v ) {
	__shared__ float s_warp[MaxWarps];
	v = warp_sum( v );
	if ( ( threadIdx.x & 31 ) == 0 ) s_warp[threadIdx.x >> 5] = v;
	__syncthreads();
	float total = 0.0f;
	for ( int i = 0; i < static_cast<int>( blockDim.x >> 5 ); ++i ) total += s_warp[i];
	return total;
}

} // namespace engine::math
