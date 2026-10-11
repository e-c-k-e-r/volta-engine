// Device kernels shared by the H3 DiT and the H3 decoders.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace engine::video {
namespace kern {

__device__ __forceinline__ float block_sum( float v, float* red ) {
	for ( int d = 16; d; d >>= 1 ) {
		v += __shfl_xor_sync( 0xffffffffu, v, d );
	}
	const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
	__syncthreads();
	if ( l == 0 ) { red[w] = v; }
	__syncthreads();
	const int nw = ( blockDim.x + 31 ) >> 5;
	float r = l < nw ? red[l] : 0.0f;
	for ( int d = 16; d; d >>= 1 ) {
		r += __shfl_xor_sync( 0xffffffffu, r, d );
	}
	return r;
}

__device__ __forceinline__ float block_max( float v, float* red ) {
	for ( int d = 16; d; d >>= 1 ) {
		v = fmaxf( v, __shfl_xor_sync( 0xffffffffu, v, d ) );
	}
	const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
	__syncthreads();
	if ( l == 0 ) { red[w] = v; }
	__syncthreads();
	const int nw = ( blockDim.x + 31 ) >> 5;
	float r = l < nw ? red[l] : 0.0f;
	for ( int d = 16; d; d >>= 1 ) {
		r = fmaxf( r, __shfl_xor_sync( 0xffffffffu, r, d ) );
	}
	return r;
}

template <typename Tin, typename Tout>
__global__ void k_convert( const Tin* src, Tout* dst, size_t n ) {
	for ( size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		dst[i] = static_cast<Tout>( static_cast<float>( src[i] ) );
	}
}

__device__ __forceinline__ void store_out( half* p, float v ) { *p = __float2half_rn( v ); }
__device__ __forceinline__ void store_out( float* p, float v ) { *p = v; }

constexpr int kMaxPerThread = 24; // hidden <= 256 * 24

// x[row] (+= gate[idx] * branch[row]) -> normalized, optionally scale/shift modulated, written as Tout.
template <typename Tout>
__global__ void k_norm_mod( float* x, const float* branch, const float* gate, long gstride, const float* w, const float* shift, const float* scale, long mstride,
	const int* idx, int H, float eps, Tout* out ) {
	__shared__ float red[32];
	const int row = blockIdx.x;
	const long ii = idx ? idx[row] : 0;
	float v[kMaxPerThread];
	float ss = 0.0f;
	#pragma unroll
	for ( int e = 0; e < kMaxPerThread; ++e ) {
		const int c = threadIdx.x + e * 256;
		float a = 0.0f;
		if ( c < H ) {
			a = x[static_cast<size_t>( row ) * H + c];
			if ( branch ) {
				const float g = gate ? gate[ii * gstride + c] : 1.0f;
				a = a + g * branch[static_cast<size_t>( row ) * H + c];
				x[static_cast<size_t>( row ) * H + c] = a;
			}
		}
		v[e] = a;
		ss += a * a;
	}
	ss = block_sum( ss, red );
	const float inv = rsqrtf( ss / static_cast<float>( H ) + eps );
	#pragma unroll
	for ( int e = 0; e < kMaxPerThread; ++e ) {
		const int c = threadIdx.x + e * 256;
		if ( c < H ) {
			float o = v[e] * inv * w[c];
			if ( scale ) { o = o * ( 1.0f + scale[ii * mstride + c] ) + shift[ii * mstride + c]; }
			store_out( out + static_cast<size_t>( row ) * H + c, o );
		}
	}
}

// x += gate[idx] * (branch * rowscale)
__global__ void k_add_gated( float* x, const float* branch, const float* rowscale, const float* gate, long gstride, const int* idx, int H, const float* bias = nullptr ) {
	const int row = blockIdx.x;
	const long ii = idx ? idx[row] : 0;
	const float rs = rowscale ? rowscale[row] : 1.0f;
	for ( int c = threadIdx.x; c < H; c += blockDim.x ) {
		const float g = gate ? gate[ii * gstride + c] : 1.0f;
		x[static_cast<size_t>( row ) * H + c] += g * ( branch[static_cast<size_t>( row ) * H + c] * rs + ( bias ? bias[c] : 0.0f ) );
	}
}

__global__ void k_add_bias( float* y, const float* b, int rows, int cols ) {
	const size_t n = static_cast<size_t>( rows ) * cols;
	for ( size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		y[i] += b[i % cols];
	}
}

// silu(gate) * up with a per-row power-of-two scale that keeps the FP16 result in range (restored after the down projection).
__global__ void k_silu_mul( const half* f1, half* f2, float* rowscale, int F, const float* bias = nullptr ) {
	__shared__ float red[32];
	const int row = blockIdx.x;
	const half* g = f1 + static_cast<size_t>( row ) * 2 * F;
	const half* u = g + F;
	const float* bg = bias;
	const float* bu = bias ? bias + F : nullptr;
	float mx = 0.0f;
	for ( int c = threadIdx.x; c < F; c += blockDim.x ) {
		const float a = __half2float( g[c] ) + ( bg ? bg[c] : 0.0f ), b = __half2float( u[c] ) + ( bu ? bu[c] : 0.0f );
		mx = fmaxf( mx, fabsf( a / ( 1.0f + __expf( -a ) ) * b ) );
	}
	mx = block_max( mx, red );
	int e = 0;
	if ( isfinite( mx ) ) { frexpf( mx, &e ); }
	const float s = ldexpf( 1.0f, max( e - 11, 0 ) );
	if ( threadIdx.x == 0 ) { rowscale[row] = s; }
	for ( int c = threadIdx.x; c < F; c += blockDim.x ) {
		const float a = __half2float( g[c] ) + ( bg ? bg[c] : 0.0f ), b = __half2float( u[c] ) + ( bu ? bu[c] : 0.0f );
		f2[static_cast<size_t>( row ) * F + c] = __float2half_rn( a / ( 1.0f + __expf( -a ) ) * b / s );
	}
}


} // namespace kern
} // namespace engine::video
