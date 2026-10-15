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

// Sum of v over the whole block (a multiple of 32 threads is expected, at most MaxWarps warps, 1D to 3D): a butterfly inside each warp,
// the warp totals through shared memory, then every warp runs the butterfly over the totals itself, so there is one barrier and no
// broadcast; every thread gets the sum. The shared array is reused by a second call: put a __syncthreads() between two calls.
template <int MaxWarps>
__device__ __forceinline__ float block_sum( float v ) {
	__shared__ float s_warp[MaxWarps];
	v = warp_sum( v );
	const int tid = threadIdx.x + blockDim.x * ( threadIdx.y + blockDim.y * threadIdx.z );   // blocks may be 2D (32 x 4)
	const int lane = tid & 31;
	if ( lane == 0 ) s_warp[tid >> 5] = v;
	__syncthreads();
	const int warps = ( static_cast<int>( blockDim.x * blockDim.y * blockDim.z ) + 31 ) >> 5;
	return warp_sum( lane < warps ? s_warp[lane] : 0.0f );   // all 32 lanes: adding the zero lanes first does not change a bit, and every lane ends with the sum
}

// Maximum of v over the block, the same way (the lanes without a warp total contribute -infinity)
template <int MaxWarps>
__device__ __forceinline__ float block_max( float v ) {
	__shared__ float s_warp[MaxWarps];
	v = warp_max( v );
	const int tid = threadIdx.x + blockDim.x * ( threadIdx.y + blockDim.y * threadIdx.z );
	const int lane = tid & 31;
	if ( lane == 0 ) s_warp[tid >> 5] = v;
	__syncthreads();
	const int warps = ( static_cast<int>( blockDim.x * blockDim.y * blockDim.z ) + 31 ) >> 5;
	return warp_max( lane < warps ? s_warp[lane] : -INFINITY );
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
