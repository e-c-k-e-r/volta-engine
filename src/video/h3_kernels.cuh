// Device kernels shared by the H3 DiT and the H3 decoders.
#pragma once

#include <cuda_fp16.h>
#include "engine/backend/cuda/reduce.cuh"
#include "engine/backend/cuda/activation.cuh"
#include "engine/kernels/dense.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace engine::video {
namespace kern {

// the generic pieces live in engine/kernels/dense.cuh (shared with the TTS model)
using engine::dense::block_sum;
using engine::dense::block_max;
using engine::dense::k_convert;
using engine::dense::store_out;
using engine::dense::kMaxPerThread;
using engine::dense::k_norm_mod;
using engine::dense::k_add_gated;
using engine::dense::k_add_bias;

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
			const float v = engine::math::silu( a ) * b;
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
			mx = fmaxf( mx, fabsf( engine::math::silu( a ) * b ) );
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
			oh[e] = __float2half_rn( engine::math::silu( a ) * b / s );
		}
		*reinterpret_cast<uint4*>( f2 + static_cast<size_t>( row ) * F + c0 ) = ov;
	}
}


} // namespace kern
} // namespace engine::video
