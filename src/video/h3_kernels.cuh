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
static __global__ void k_convert( const Tin* src, Tout* dst, size_t n ) {
	for ( size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		dst[i] = static_cast<Tout>( static_cast<float>( src[i] ) );
	}
}

__device__ __forceinline__ void store_out( half* p, float v ) { *p = __float2half_rn( v ); }
__device__ __forceinline__ void store_out( float* p, float v ) { *p = v; }

constexpr int kMaxPerThread = 24; // hidden <= 256 * 24

// x[row] (+= gate[idx] * branch[row]) -> normalized, optionally scale/shift modulated, written as Tout.
template <typename Tout>
static __global__ void k_norm_mod( float* x, const float* branch, const float* gate, long gstride, const float* w, const float* shift, const float* scale, long mstride,
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
static __global__ void k_add_gated( float* x, const float* branch, const float* rowscale, const float* gate, long gstride, const int* idx, int H, const float* bias = nullptr ) {
	const int row = blockIdx.x;
	const long ii = idx ? idx[row] : 0;
	const float rs = rowscale ? rowscale[row] : 1.0f;
	for ( int c = threadIdx.x; c < H; c += blockDim.x ) {
		const float g = gate ? gate[ii * gstride + c] : 1.0f;
		x[static_cast<size_t>( row ) * H + c] += g * ( branch[static_cast<size_t>( row ) * H + c] * rs + ( bias ? bias[c] : 0.0f ) );
	}
}

static __global__ void k_add_bias( float* y, const float* b, int rows, int cols ) {
	const size_t n = static_cast<size_t>( rows ) * cols;
	for ( size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		y[i] += b[i % cols];
	}
}

// silu(gate) * up with a per-row power-of-two scale that keeps the FP16 result in range (restored after the down projection).
static __global__ void k_silu_mul( const half* f1, half* f2, float* rowscale, int F, const float* bias = nullptr ) {
	// F a multiple of 8 and at most 16384: each thread holds its 8-wide pieces of the activation between the max pass and the write pass (one read of f1)
	__shared__ float red[32];
	const int row = blockIdx.x;
	const half* g = f1 + static_cast<size_t>( row ) * 2 * F;
	const half* u = g + F;
	const float* bg = bias;
	const float* bu = bias ? bias + F : nullptr;
	float val[8][8];
	float mx = 0.0f;
	#pragma unroll
	for ( int it = 0; it < 8; ++it ) {
		const int c0 = ( it * blockDim.x + threadIdx.x ) * 8;
		if ( c0 >= F ) { break; }
		const uint4 gv = *reinterpret_cast<const uint4*>( g + c0 ), uv = *reinterpret_cast<const uint4*>( u + c0 );
		const half* gh = reinterpret_cast<const half*>( &gv );
		const half* uh = reinterpret_cast<const half*>( &uv );
		#pragma unroll
		for ( int e = 0; e < 8; ++e ) {
			const float a = __half2float( gh[e] ) + ( bg ? bg[c0 + e] : 0.0f ), b = __half2float( uh[e] ) + ( bu ? bu[c0 + e] : 0.0f );
			const float v = a / ( 1.0f + __expf( -a ) ) * b;
			val[it][e] = v;
			mx = fmaxf( mx, fabsf( v ) );
		}
	}
	mx = block_max( mx, red );
	int e2 = 0;
	if ( isfinite( mx ) ) { frexpf( mx, &e2 ); }
	const float s = ldexpf( 1.0f, max( e2 - 11, 0 ) );
	if ( threadIdx.x == 0 ) { rowscale[row] = s; }
	#pragma unroll
	for ( int it = 0; it < 8; ++it ) {
		const int c0 = ( it * blockDim.x + threadIdx.x ) * 8;
		if ( c0 >= F ) { break; }
		uint4 ov;
		half* oh = reinterpret_cast<half*>( &ov );
		#pragma unroll
		for ( int e = 0; e < 8; ++e ) { oh[e] = __float2half_rn( val[it][e] / s ); }
		*reinterpret_cast<uint4*>( f2 + static_cast<size_t>( row ) * F + c0 ) = ov;
	}
}


// Tensor-parallel SiLU-mul: the per-row power-of-two scale must come from the whole F, so the max over this rank's slice is written to amax[row], reduced (max) over the
// ranks, and k_silu_apply then derives the scale from the global value (same arithmetic as k_silu_mul).
static __global__ void k_silu_amax( const half* f1, float* amax, int F ) {
	__shared__ float red[32];
	const int row = blockIdx.x;
	const half* g = f1 + static_cast<size_t>( row ) * 2 * F;
	const half* u = g + F;
	float mx = 0.0f;
	for ( int c0 = threadIdx.x * 8; c0 < F; c0 += blockDim.x * 8 ) {
		const uint4 gv = *reinterpret_cast<const uint4*>( g + c0 ), uv = *reinterpret_cast<const uint4*>( u + c0 );
		const half* gh = reinterpret_cast<const half*>( &gv );
		const half* uh = reinterpret_cast<const half*>( &uv );
		#pragma unroll
		for ( int e = 0; e < 8; ++e ) {
			const float a = __half2float( gh[e] ), b = __half2float( uh[e] );
			mx = fmaxf( mx, fabsf( a / ( 1.0f + __expf( -a ) ) * b ) );
		}
	}
	mx = block_max( mx, red );
	if ( threadIdx.x == 0 ) { amax[row] = mx; }
}

static __global__ void k_silu_apply( const half* f1, half* f2, float* rowscale, int F ) {
	const int row = blockIdx.x;
	const half* g = f1 + static_cast<size_t>( row ) * 2 * F;
	const half* u = g + F;
	const float mx = rowscale[row];
	int e2 = 0;
	if ( isfinite( mx ) ) { frexpf( mx, &e2 ); }
	const float s = ldexpf( 1.0f, max( e2 - 11, 0 ) );
	__syncthreads(); // every thread has read the maximum before it is replaced by the scale
	if ( threadIdx.x == 0 ) { rowscale[row] = s; }
	for ( int c0 = threadIdx.x * 8; c0 < F; c0 += blockDim.x * 8 ) {
		const uint4 gv = *reinterpret_cast<const uint4*>( g + c0 ), uv = *reinterpret_cast<const uint4*>( u + c0 );
		const half* gh = reinterpret_cast<const half*>( &gv );
		const half* uh = reinterpret_cast<const half*>( &uv );
		uint4 ov;
		half* oh = reinterpret_cast<half*>( &ov );
		#pragma unroll
		for ( int e = 0; e < 8; ++e ) {
			const float a = __half2float( gh[e] ), b = __half2float( uh[e] );
			oh[e] = __float2half_rn( a / ( 1.0f + __expf( -a ) ) * b / s );
		}
		*reinterpret_cast<uint4*>( f2 + static_cast<size_t>( row ) * F + c0 ) = ov;
	}
}


} // namespace kern
} // namespace engine::video
