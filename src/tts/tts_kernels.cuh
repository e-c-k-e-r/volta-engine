// Device kernels of the TTS model that aren't generic (those are in engine/kernels/dense.cuh).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "engine/backend/cuda/activation.cuh"
#include "engine/backend/cuda/reduce.cuh"
#include "engine/backend/cuda/rope_math.cuh"

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

// The rotary angles' cos / sin of every position and pair, [n, rot / 2] each
__global__ void k_rope_table( float* cs, float* sn, int n, int rot, float theta ) {
	const int half_rot = rot / 2;
	for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * half_rot; i += gridDim.x * blockDim.x ) {
		engine::math::rope_cos_sin<false>( i / half_rot, i % half_rot, rot, theta, cs[i], sn[i] );
	}
}

// qkv [L, 3, H, hd] -> q, k, v, each [H, L, hd] (head_major, for the batched GEMMs) or [L, H, hd] (the fused attention): q and k
// RMS-normalised per head (weights qn / kn, eps) and rotated (RoPE on the first `rot` dims, rotate-half, the angles from k_rope_table at
// each position's place in its row, pos), v copied. A warp per (position, head), hd a multiple of 32 up to 128; blocks of QK_WARPS warps
// with 2 * hd floats of shared memory each.
constexpr int QK_WARPS = 8;
__global__ void k_qk_norm_rope( const half* qkv, const float* qn, const float* kn, half* q, half* k, half* v, const int* pos, int L, int H, int hd, int rot,
	const float* rope_cs, const float* rope_sn, float eps, bool head_major ) {
	extern __shared__ float sh[];  // [QK_WARPS][2 * hd]
	const int lane = threadIdx.x % 32, w = threadIdx.x / 32, item = blockIdx.x * QK_WARPS + w;
	if ( item >= L * H ) { return; }
	const int p = item / H, h = item % H, n = hd / 32;
	const size_t src = ( static_cast<size_t>( p ) * 3 * H + h ) * hd;
	const size_t dst = head_major ? ( static_cast<size_t>( h ) * L + p ) * hd : ( static_cast<size_t>( p ) * H + h ) * hd;
	float* s = sh + w * 2 * hd;
	for ( int which = 0; which < 2; ++which ) {
		float x[4], ss = 0.0f;
		#pragma unroll
		for ( int e = 0; e < 4; ++e ) {
			if ( e < n ) {
				x[e] = __half2float( qkv[src + static_cast<size_t>( which ) * H * hd + lane + 32 * e] );
				ss += x[e] * x[e];
			}
		}
		ss = engine::math::warp_sum( ss );
		const float* wt = which == 0 ? qn : kn;
		const float r = rsqrtf( ss / hd + eps );
		#pragma unroll
		for ( int e = 0; e < 4; ++e ) {
			const int c = lane + 32 * e;
			if ( e < n ) { s[which * hd + c] = wt ? x[e] * r * wt[c] : x[e]; }
		}
	}
	__syncwarp();
	const int half_rot = rot / 2;
	#pragma unroll
	for ( int e = 0; e < 4; ++e ) {
		const int c = lane + 32 * e;
		if ( e >= n ) { continue; }
		float sn = 0.0f, co = 1.0f;
		if ( c < rot ) {
			const size_t t = static_cast<size_t>( pos[p] ) * half_rot + c % half_rot;
			co = rope_cs[t];
			sn = rope_sn[t];
		}
		for ( int which = 0; which < 2; ++which ) {
			float o = s[which * hd + c];
			if ( c < rot ) {
				const int i = c % half_rot;
				const float x1 = s[which * hd + i], x2 = s[which * hd + i + half_rot];
				o = c < half_rot ? x1 * co - x2 * sn : x2 * co + x1 * sn;
			}
			( which == 0 ? q : k )[dst + c] = __float2half( o );
		}
		v[dst + c] = qkv[src + static_cast<size_t>( 2 ) * H * hd + c];
	}
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

// k_softmax over FP16 scores already scaled (ROCm: the scores GEMM writes FP16 to stay on the matrix cores); FP32 inside
__global__ void k_softmax_h( const half* S, half* P, int n ) {
	const size_t r = blockIdx.x;
	const half* s = S + r * n;
	float m = -INFINITY;
	for ( int j = threadIdx.x; j < n; j += blockDim.x ) { m = fmaxf( m, __half2float( s[j] ) ); }
	m = engine::math::block_max<32>( m );
	float sum = 0.0f;
	for ( int j = threadIdx.x; j < n; j += blockDim.x ) { sum += __expf( __half2float( s[j] ) - m ); }
	__syncthreads();
	sum = engine::math::block_sum<32>( sum );
	const float inv = 1.0f / sum;
	for ( int j = threadIdx.x; j < n; j += blockDim.x ) { P[r * n + j] = __float2half( __expf( __half2float( s[j] ) - m ) * inv ); }
}

// o [H, L, hd] (head_major) or [L, H * hd] -> y [L, H * hd], times sigmoid(g [L, H * hd]) if g (the attention output gate)
__global__ void k_heads_gate( const half* o, const half* g, half* y, int L, int H, int hd, bool head_major ) {
	const int p = blockIdx.x;
	for ( int c = threadIdx.x; c < H * hd; c += blockDim.x ) {
		const int h = c / hd, e = c % hd;
		float v = __half2float( o[head_major ? ( static_cast<size_t>( h ) * L + p ) * hd + e : static_cast<size_t>( p ) * H * hd + c] );
		if ( g ) { v *= engine::math::sigmoid( __half2float( g[static_cast<size_t>( p ) * H * hd + c] ) ); }
		y[static_cast<size_t>( p ) * H * hd + c] = __float2half( v );
	}
}

// up [L, 2F] (a | b) -> silu(a) * b [L, F]
__global__ void k_swiglu( const half* up, half* act, int F ) {
	const size_t p = blockIdx.x;
	for ( int c = threadIdx.x; c < F; c += blockDim.x ) {
		const float a = __half2float( up[p * 2 * F + c] ), b = __half2float( up[p * 2 * F + F + c] );
		act[p * F + c] = __float2half( engine::math::silu( a ) * b );
	}
}

} // namespace engine::tts::kern
