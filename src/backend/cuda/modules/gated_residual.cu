#include "engine/modules/gated_residual.h"
#include <cuda_fp16.h>
#include <cmath>

namespace engine::modules {

namespace {

__device__ __forceinline__ float warp_sum( float v ) {
	#pragma unroll
	for ( int o = 16; o > 0; o >>= 1 ) v += __shfl_xor_sync( 0xffffffff, v, o );
	return v;
}

__global__ void gr_init_kernel( const half* __restrict__ in, float* __restrict__ x, int hidden, int hc ) {
	const int r = blockIdx.y;
	for ( int d = blockIdx.x * blockDim.x + threadIdx.x; d < hidden; d += gridDim.x * blockDim.x ) {
		const float v = __half2float( in[static_cast<size_t>( r ) * hidden + d] );
		for ( int s = 0; s < hc; ++s ) x[( static_cast<size_t>( r ) * hc + s ) * hidden + d] = v;
	}
}

// One block per (row, stream): normed[r, s, :] = x * rsqrt(mean(x^2) + eps) * (1 + w[s, :])
__global__ void gr_norm_kernel( const float* __restrict__ x, const half* __restrict__ w, float* __restrict__ normed, int hidden, int hc, float eps ) {
	const int r = blockIdx.x, s = blockIdx.y;
	const size_t base = ( static_cast<size_t>( r ) * hc + s ) * hidden;
	float sq = 0.0f;
	for ( int d = threadIdx.x; d < hidden; d += blockDim.x ) {
		const float v = x[base + d];
		sq += v * v;
	}
	sq = warp_sum( sq );
	__shared__ float ws[32];
	if ( ( threadIdx.x & 31 ) == 0 ) ws[threadIdx.x >> 5] = sq;
	__syncthreads();
	float total = 0.0f;
	for ( int i = 0; i < ( blockDim.x >> 5 ); ++i ) total += ws[i];
	const float inv = rsqrtf( total / hidden + eps );
	for ( int d = threadIdx.x; d < hidden; d += blockDim.x ) {
		normed[base + d] = x[base + d] * inv * ( 1.0f + __half2float( w[static_cast<size_t>( s ) * hidden + d] ) );
	}
}

// t[r, j] = silu( dot(normed[r], down[j]) / hc ): a warp per (row, output), 8 outputs per block
__global__ void gr_down_kernel( const float* __restrict__ normed, const half* __restrict__ down, float* __restrict__ t, int flat, int rank, int hc ) {
	const int r = blockIdx.x;
	const int j = blockIdx.y * 8 + ( threadIdx.x >> 5 );
	const int lane = threadIdx.x & 31;
	if ( j >= rank ) return;
	const float* a = normed + static_cast<size_t>( r ) * flat;
	const half* b = down + static_cast<size_t>( j ) * flat;
	float acc = 0.0f;
	for ( int i = lane * 2; i < flat; i += 64 ) {
		const half2 bv = *reinterpret_cast<const half2*>( b + i );
		acc = fmaf( a[i], __low2float( bv ), acc );
		acc = fmaf( a[i + 1], __high2float( bv ), acc );
	}
	acc = warp_sum( acc );
	if ( lane == 0 ) {
		const float z = acc / hc;
		t[static_cast<size_t>( r ) * rank + j] = z / ( 1.0f + __expf( -z ) );
	}
}

// mixed[r, d] = mean_s( sigmoid( up[s * hidden + d] . t[r] ) * normed[r, s, d] )
__global__ void gr_up_mix_kernel( const float* __restrict__ normed, const half* __restrict__ up, const float* __restrict__ t,
	half* __restrict__ mixed, int hidden, int hc, int rank ) {
	extern __shared__ float s_t[];
	const int r = blockIdx.y;
	for ( int k = threadIdx.x; k < rank; k += blockDim.x ) s_t[k] = t[static_cast<size_t>( r ) * rank + k];
	__syncthreads();
	const int d = blockIdx.x * blockDim.x + threadIdx.x;
	if ( d >= hidden ) return;
	float out = 0.0f;
	for ( int s = 0; s < hc; ++s ) {
		const half* u = up + static_cast<size_t>( s * hidden + d ) * rank;
		float acc = 0.0f;
		for ( int k = 0; k < rank; k += 2 ) {
			const half2 uv = *reinterpret_cast<const half2*>( u + k );
			acc = fmaf( s_t[k], __low2float( uv ), acc );
			acc = fmaf( s_t[k + 1], __high2float( uv ), acc );
		}
		const float w = 1.0f / ( 1.0f + __expf( -acc ) );
		out += w * normed[( static_cast<size_t>( r ) * hc + s ) * hidden + d];
	}
	mixed[static_cast<size_t>( r ) * hidden + d] = __float2half_rn( out / hc );
}

// post[r, s] = 2 * sigmoid( dot(normed[r], inject[s]) / hc ): a block per (row, stream)
__global__ void gr_post_kernel( const float* __restrict__ normed, const half* __restrict__ inject, float* __restrict__ post, int flat, int hc ) {
	const int r = blockIdx.x, s = blockIdx.y;
	const float* a = normed + static_cast<size_t>( r ) * flat;
	const half* b = inject + static_cast<size_t>( s ) * flat;
	float acc = 0.0f;
	for ( int i = threadIdx.x; i < flat; i += blockDim.x ) acc = fmaf( a[i], __half2float( b[i] ), acc );
	acc = warp_sum( acc );
	__shared__ float ws[32];
	if ( ( threadIdx.x & 31 ) == 0 ) ws[threadIdx.x >> 5] = acc;
	__syncthreads();
	if ( threadIdx.x == 0 ) {
		float total = 0.0f;
		for ( int i = 0; i < ( blockDim.x >> 5 ); ++i ) total += ws[i];
		post[static_cast<size_t>( r ) * hc + s] = 2.0f / ( 1.0f + __expf( -total / hc ) );
	}
}

__global__ void gr_apply_kernel( float* __restrict__ x, const float* __restrict__ post, const half* __restrict__ y, int hidden, int hc ) {
	const int r = blockIdx.y, s = blockIdx.z;
	const int d = blockIdx.x * blockDim.x + threadIdx.x;
	if ( d >= hidden ) return;
	const float p = post[static_cast<size_t>( r ) * hc + s];
	x[( static_cast<size_t>( r ) * hc + s ) * hidden + d] += p * __half2float( y[static_cast<size_t>( r ) * hidden + d] );
}

} // namespace

void GatedResidual::init_streams( const backend::half_t* in, float* x, int rows, int hidden, int hc, backend::stream_t stream ) {
	gr_init_kernel<<<dim3( ( hidden + 255 ) / 256, rows ), 256, 0, stream>>>( reinterpret_cast<const half*>( in ), x, hidden, hc );
}

void GatedResidual::mix(
	const float* x, const backend::half_t* norm_w, const backend::half_t* down, const backend::half_t* up, const backend::half_t* inject,
	backend::half_t* mixed, float* post, float* scratch, int rows, int hidden, int hc, int rank, float eps, backend::stream_t stream
) {
	const int flat = hc * hidden;
	float* normed = scratch;
	float* t = scratch + static_cast<size_t>( rows ) * flat;
	gr_norm_kernel<<<dim3( rows, hc ), 256, 0, stream>>>( x, reinterpret_cast<const half*>( norm_w ), normed, hidden, hc, eps );
	gr_down_kernel<<<dim3( rows, ( rank + 7 ) / 8 ), 256, 0, stream>>>( normed, reinterpret_cast<const half*>( down ), t, flat, rank, hc );
	gr_up_mix_kernel<<<dim3( ( hidden + 127 ) / 128, rows ), 128, static_cast<size_t>( rank ) * sizeof( float ), stream>>>(
		normed, reinterpret_cast<const half*>( up ), t, reinterpret_cast<half*>( mixed ), hidden, hc, rank );
	if ( inject != nullptr ) {
		gr_post_kernel<<<dim3( rows, hc ), 256, 0, stream>>>( normed, reinterpret_cast<const half*>( inject ), post, flat, hc );
	}
}

void GatedResidual::apply( float* x, const float* post, const backend::half_t* y, int rows, int hidden, int hc, backend::stream_t stream ) {
	gr_apply_kernel<<<dim3( ( hidden + 255 ) / 256, rows, hc ), 256, 0, stream>>>( x, post, reinterpret_cast<const half*>( y ), hidden, hc );
}

} // namespace engine::modules
