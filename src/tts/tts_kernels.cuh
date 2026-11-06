// Device kernels of the TTS model that aren't generic (those are in engine/kernels/dense.cuh).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "engine/backend/cuda/reduce.cuh"

namespace engine::tts::kern {

// FSQ digits (level indices, NeMo 44 kHz: groups of 4 scalars at [8, 5, 5, 5] levels) -> values in [-1, 1]
__global__ void k_digits_values( const uint8_t* digits, half* vals, int n ) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= n ) { return; }
	const float h = ( i % 4 == 0 ) ? 4.0f : 2.0f;
	vals[i] = __float2half( ( static_cast<float>( digits[i] ) - h ) / h );
}

// The input embedding of one row (vall_e/model.py V3.hidden): the text embedding at text positions (kind 1), else the
// audio projection a (bias included); the mask embedding at masked positions, the length query at kind 5; plus the
// segment embedding (kind - 1, clamped to [0, 2]) and the row's mode embedding.
__global__ void k_embed_mix( float* x, const float* a, const half* text_emb, const int* ids, const uint8_t* kind, const uint8_t* masked,
	const float* mask_emb, const float* len_q, const float* segment, const float* mode_vec, int d ) {
	const int p = blockIdx.x;
	const int k = kind[p];
	const int seg = min( max( k - 1, 0 ), 2 );
	for ( int c = threadIdx.x; c < d; c += blockDim.x ) {
		float v = k == 1 ? __half2float( text_emb[static_cast<size_t>( ids[p] ) * d + c] ) : a[static_cast<size_t>( p ) * d + c];
		if ( masked[p] ) { v = mask_emb[c]; }
		if ( k == 5 && len_q ) { v = len_q[c]; }
		x[static_cast<size_t>( p ) * d + c] = v + segment[seg * d + c] + mode_vec[c];
	}
}

// qkv [L, 3, H, hd] -> q, k, v [H, L, hd]: q and k RMS-normalised per head (weights qn / kn, eps) and rotated (RoPE on
// the first `rot` dims, rotate-half, theta), v copied. One block of hd threads per (position, head).
__global__ void k_qk_norm_rope( const half* qkv, const float* qn, const float* kn, half* q, half* k, half* v, int L, int H, int hd, int rot, float theta, float eps ) {
	const int p = blockIdx.x / H, h = blockIdx.x % H, c = threadIdx.x;
	const size_t src = ( static_cast<size_t>( p ) * 3 * H + h ) * hd;
	const size_t dst = ( static_cast<size_t>( h ) * L + p ) * hd;
	extern __shared__ float sh[];  // [2 * hd]
	for ( int which = 0; which < 2; ++which ) {
		const float x = __half2float( qkv[src + static_cast<size_t>( which ) * H * hd + c] );
		__syncthreads();  // block_sum reuses its shared array
		float ss = engine::math::block_sum<32>( x * x );
		const float* w = which == 0 ? qn : kn;
		const float y = w ? x * rsqrtf( ss / hd + eps ) * w[c] : x;
		sh[which * hd + c] = y;
	}
	__syncthreads();
	for ( int which = 0; which < 2; ++which ) {
		float o = sh[which * hd + c];
		if ( c < rot ) {
			const int half_rot = rot / 2, i = c % half_rot;
			const float freq = powf( theta, -2.0f * i / rot );
			float s, co;
			sincosf( p * freq, &s, &co );
			const float x1 = sh[which * hd + i], x2 = sh[which * hd + i + half_rot];
			o = c < half_rot ? x1 * co - x2 * s : x2 * co + x1 * s;
		}
		( which == 0 ? q : k )[dst + c] = __float2half( o );
	}
	v[dst + c] = qkv[src + static_cast<size_t>( 2 ) * H * hd + c];
}

// Row softmax of scale * S ([rows, n] FP32) into P (FP16)
__global__ void k_softmax( const float* S, half* P, int n, float scale ) {
	const size_t r = blockIdx.x;
	const float* s = S + r * n;
	float m = -INFINITY;
	for ( int j = threadIdx.x; j < n; j += blockDim.x ) { m = fmaxf( m, s[j] ); }
	m = engine::math::block_max<32>( m );
	float sum = 0.0f;
	for ( int j = threadIdx.x; j < n; j += blockDim.x ) { sum += __expf( ( s[j] - m ) * scale ); }
	__syncthreads();
	sum = engine::math::block_sum<32>( sum );
	const float inv = 1.0f / sum;
	for ( int j = threadIdx.x; j < n; j += blockDim.x ) { P[r * n + j] = __float2half( __expf( ( s[j] - m ) * scale ) * inv ); }
}

// o [H, L, hd] -> y [L, H * hd], times sigmoid(g [L, H * hd]) if g (the attention output gate)
__global__ void k_heads_gate( const half* o, const half* g, half* y, int L, int H, int hd ) {
	const int p = blockIdx.x;
	for ( int c = threadIdx.x; c < H * hd; c += blockDim.x ) {
		const int h = c / hd, e = c % hd;
		float v = __half2float( o[( static_cast<size_t>( h ) * L + p ) * hd + e] );
		if ( g ) { v *= 1.0f / ( 1.0f + __expf( -__half2float( g[static_cast<size_t>( p ) * H * hd + c] ) ) ); }
		y[static_cast<size_t>( p ) * H * hd + c] = __float2half( v );
	}
}

// up [L, 2F] (a | b) -> silu(a) * b [L, F]
__global__ void k_swiglu( const half* up, half* act, int F ) {
	const size_t p = blockIdx.x;
	for ( int c = threadIdx.x; c < F; c += blockDim.x ) {
		const float a = __half2float( up[p * 2 * F + c] ), b = __half2float( up[p * 2 * F + F + c] );
		act[p * F + c] = __float2half( a / ( 1.0f + __expf( -a ) ) * b );
	}
}

} // namespace engine::tts::kern
