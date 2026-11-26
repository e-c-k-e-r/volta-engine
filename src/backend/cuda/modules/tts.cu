// CUDA (and ROCm) backend: the vall-e TTS and NeMo codec operators (engine/modules/tts.h). GEMMs on cuBLAS / hipBLAS (one handle per device);
// the backbone's attention on the fused WMMA kernel (h3_attention: head_dim 64 or 128); on ROCm, whose WMMA accumulator layout that kernel
// doesn't follow, scores by batched GEMM, a row softmax and a second batched GEMM.
#include "engine/modules/tts.h"
#include "engine/backend/cuda/activation.cuh"
#include "engine/backend/cuda/reduce.cuh"
#include "engine/backend/cuda/rope_math.cuh"
#include "engine/core/blas_check.h"
#include "engine/core/log.h"
#include "engine/kernels/dense.cuh"
#include "engine/video/h3_attn.h"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <mutex>

namespace engine::modules {

namespace {

// The attention scores (FP32) and probabilities (FP16) of the heads taken at once: all 24 of v2.0's for rows up to ~960 positions, 3 at
// 2560 (one head needs 39 MiB there)
constexpr size_t ATTN_BYTES = 128ull << 20;
constexpr size_t ATTN_S_BYTES = ( ATTN_BYTES / 6 * 4 + 255 ) / 256 * 256;   // the scores, then the probabilities at a 256-byte boundary

bool fused_attention( int head_dim ) {
#if defined(USE_ROCM)
	( void ) head_dim;
	return false;
#else
	return head_dim == 64 || head_dim == 128;
#endif
}

constexpr int kMaxDev = 64;
std::mutex g_mu;
cublasHandle_t g_blas[kMaxDev] = {};
half* g_c16[kMaxDev] = {};   // ROCm: the FP16 result of an FP32-out GEMM before it's widened
size_t g_c16_elems[kMaxDev] = {};

cublasHandle_t blas() {
	int dev = 0;
	CUDA_CHECK( cudaGetDevice( &dev ) );
	std::lock_guard<std::mutex> lk( g_mu );
	if ( !g_blas[dev] ) { CUBLAS_CHECK( cublasCreate( &g_blas[dev] ) ); }
	return g_blas[dev];
}

int grid( size_t n ) { return static_cast<int>( std::min<size_t>( 4096, ( n + 255 ) / 256 ) ); }

// ---- the backbone

__global__ void k_digits_values( const uint8_t* digits, half* vals, int n ) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= n ) { return; }
	const float h = ( i % 4 == 0 ) ? 4.0f : 2.0f;
	vals[i] = __float2half( ( static_cast<float>( digits[i] ) - h ) / h );
}

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

__global__ void k_rope_table( float* cs, float* sn, int n, int rot, float theta ) {
	const int half_rot = rot / 2;
	for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * half_rot; i += gridDim.x * blockDim.x ) {
		engine::math::rope_cos_sin<false>( i / half_rot, i % half_rot, rot, theta, cs[i], sn[i] );
	}
}

// A warp per (position, head), hd a multiple of 32 up to 128; blocks of QK_WARPS warps with 2 * hd floats of shared memory each
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

__global__ void k_heads_gate( const half* o, const half* g, half* y, int L, int H, int hd, bool head_major ) {
	const int p = blockIdx.x;
	for ( int c = threadIdx.x; c < H * hd; c += blockDim.x ) {
		const int h = c / hd, e = c % hd;
		float v = __half2float( o[head_major ? ( static_cast<size_t>( h ) * L + p ) * hd + e : static_cast<size_t>( p ) * H * hd + c] );
		if ( g ) { v *= engine::math::sigmoid( __half2float( g[static_cast<size_t>( p ) * H * hd + c] ) ); }
		y[static_cast<size_t>( p ) * H * hd + c] = __float2half( v );
	}
}

__global__ void k_swiglu( const half* up, half* act, int F ) {
	const size_t p = blockIdx.x;
	for ( int c = threadIdx.x; c < F; c += blockDim.x ) {
		const float a = __half2float( up[p * 2 * F + c] ), b = __half2float( up[p * 2 * F + F + c] );
		act[p * F + c] = __float2half( engine::math::silu( a ) * b );
	}
}

// ---- the flow head

__global__ void k_gather_half( const float* src, const int* pos, int offset, half* dst, int d ) {
	const int i = blockIdx.x;
	const size_t s = static_cast<size_t>( offset + pos[i] ) * d;
	for ( int c = threadIdx.x; c < d; c += blockDim.x ) { dst[static_cast<size_t>( i ) * d + c] = __float2half( src[s + c] ); }
}

__global__ void k_time_embedding( float t, half* out ) {
	const int i = threadIdx.x;  // 128 threads
	const float f = expf( -logf( 10000.0f ) * i / 128.0f );
	const float a = t * 1000.0f * f;
	out[i] = __float2half( cosf( a ) );
	out[128 + i] = __float2half( sinf( a ) );
}

__global__ void k_bias_act_half( const float* x, const float* b, half* y, int rows, int cols, int act ) {
	const size_t n = static_cast<size_t>( rows ) * cols;
	for ( size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		float v = x[i] + ( b ? b[i % cols] : 0.0f );
		if ( act ) { v = engine::math::silu( v ); }
		y[i] = __float2half( v );
	}
}

__global__ void k_cond_silu( const float* cond, const float* temb, half* sc, int rows, int W ) {
	const size_t n = static_cast<size_t>( rows ) * W;
	for ( size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		const float v = cond[i] + temb[i % W];
		sc[i] = __float2half( engine::math::silu( v ) );
	}
}

// One block of 256 threads per row
__global__ void k_ln_mod( const float* x, const float* mod, int mstride, half* out, int W ) {
	const int r = blockIdx.x;
	const float* xr = x + static_cast<size_t>( r ) * W;
	float s = 0.0f;
	for ( int c = threadIdx.x; c < W; c += blockDim.x ) { s += xr[c]; }
	const float mean = engine::math::block_sum<32>( s ) / W;
	float q = 0.0f;
	for ( int c = threadIdx.x; c < W; c += blockDim.x ) { const float dlt = xr[c] - mean; q += dlt * dlt; }
	__syncthreads();
	const float inv = rsqrtf( engine::math::block_sum<32>( q ) / W + 1e-6f );
	const float* m = mod + static_cast<size_t>( r ) * mstride;
	for ( int c = threadIdx.x; c < W; c += blockDim.x ) {
		out[static_cast<size_t>( r ) * W + c] = __float2half( ( xr[c] - mean ) * inv * ( 1.0f + m[W + c] ) + m[c] );
	}
}

__global__ void k_gate_add( float* x, const float* branch, const float* bias, const float* mod, int mstride, int rows, int W ) {
	const size_t n = static_cast<size_t>( rows ) * W;
	for ( size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		const size_t r = i / W, c = i % W;
		x[i] += mod[r * mstride + 2 * W + c] * ( branch[i] + bias[c] );
	}
}

__global__ void k_replicate_half( const float* x, half* y, int n, int R ) {
	for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * R; i += gridDim.x * blockDim.x ) { y[i] = __float2half( x[i % n] ); }
}

__global__ void k_guide_step( const float* v, const float* bias, float* x, float* vt, int n, int V, int R, float cfg, float cfg_prompt, float dt ) {
	for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
		const float b = bias[i % V];
		const float vc = v[i] + b;
		float g = vc;
		if ( R >= 2 ) {
			const float vn = v[n + i] + b;
			g = vn + ( vc - vn ) * cfg;
			if ( R == 3 ) { g += ( vc - ( v[2 * n + i] + b ) ) * cfg_prompt; }
		}
		if ( vt ) { vt[i] = g; }
		x[i] += g * dt;
	}
}

__global__ void k_snap( const float* x, uint8_t* out, int n ) {
	for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x ) {
		const int L = ( i % 4 == 0 ) ? 8 : 5;
		const float h = static_cast<float>( L / 2 );
		const float r = fmaxf( rintf( x[i] * h + h ), 0.0f );
		out[i] = static_cast<uint8_t>( fminf( r, static_cast<float>( L - 1 ) ) );
	}
}

// ---- the codec

__global__ void k_dequant( const uint8_t* digits, float* x, int T, int4 hl ) {
	for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < 32 * T; i += gridDim.x * blockDim.x ) {
		const int c = i / T, t = i % T, j = c % 4;
		const int h = j == 0 ? hl.x : j == 1 ? hl.y : j == 2 ? hl.z : hl.w;
		x[i] = ( static_cast<float>( digits[static_cast<size_t>( t ) * 32 + c] ) - h ) / h;
	}
}

// round(s tanh(x + tan(offset / s)) - offset) + levels / 2, with s = (levels - 1) / 2 (1 - eps) and offset 0.5 for an even number of levels
__global__ void k_quant( const float* x, uint8_t* digits, int T, int4 levels, float eps ) {
	for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < 32 * T; i += gridDim.x * blockDim.x ) {
		const int c = i / T, t = i % T, j = c % 4;
		const int L = j == 0 ? levels.x : j == 1 ? levels.y : j == 2 ? levels.z : levels.w;
		const float scale = ( L - 1 ) * 0.5f * ( 1.0f - eps ), offset = L % 2 == 0 ? 0.5f : 0.0f, shift = tanf( offset / scale );
		const float code = rintf( scale * tanhf( x[i] + shift ) - offset );
		digits[static_cast<size_t>( t ) * 32 + c] = static_cast<uint8_t>( static_cast<int>( code ) + L / 2 );
	}
}

__global__ void k_act( const float* x, float* y, const float* alpha, int C, int T ) {
	const size_t n = static_cast<size_t>( C ) * T;
	const int sc = alpha ? C / 2 : 0;
	for ( size_t i = blockIdx.x * static_cast<size_t>( blockDim.x ) + threadIdx.x; i < n; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		const int c = static_cast<int>( i / T );
		const float v = x[i];
		if ( c < sc ) {
			const float a = alpha[c], s = sinf( a * v );
			y[i] = v + s * s / ( a + 1e-9f );
		} else {
			y[i] = v > 0.0f ? v : 0.01f * v;
		}
	}
}

// cols [ci * k, n]: x[c, reflect((t0 + t) stride + j dil - pad)]
__global__ void k_cols( const float* x, float* cols, int ci, int k, int dil, int stride, int pad, int T, int t0, int n ) {
	const size_t total = static_cast<size_t>( ci ) * k * n;
	for ( size_t i = blockIdx.x * static_cast<size_t>( blockDim.x ) + threadIdx.x; i < total; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		const int t = static_cast<int>( i % n ), row = static_cast<int>( i / n ), c = row / k, j = row % k;
		int s = ( t0 + t ) * stride + j * dil - pad;
		s = s < 0 ? -s : s;
		s = s >= T ? 2 * ( T - 1 ) - s : s;
		cols[i] = x[static_cast<size_t>( c ) * T + s];
	}
}

// cols [3 * ci, n]: x[c, q0 + q + shift], shift -1 / 0 / +1, zero past the ends
__global__ void k_up_cols( const float* x, float* cols, int ci, int T, int q0, int n ) {
	const size_t total = static_cast<size_t>( 3 ) * ci * n;
	for ( size_t i = blockIdx.x * static_cast<size_t>( blockDim.x ) + threadIdx.x; i < total; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		const int q = static_cast<int>( i % n ), row = static_cast<int>( i / n ), sh = row / ci - 1, c = row % ci;
		const int s = q0 + q + sh;
		cols[i] = s >= 0 && s < T ? x[static_cast<size_t>( c ) * T + s] : 0.0f;
	}
}

// ph [s * co, n] (phase r of channel c at row r co + c) -> y[c, (q0 + q) s + r] + bias
__global__ void k_interleave( const float* ph, const float* bias, float* y, int co, int s, int T_out, int q0, int n ) {
	const size_t total = static_cast<size_t>( s ) * co * n;
	for ( size_t i = blockIdx.x * static_cast<size_t>( blockDim.x ) + threadIdx.x; i < total; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		const int q = static_cast<int>( i % n ), row = static_cast<int>( i / n ), r = row / co, c = row % co;
		y[static_cast<size_t>( c ) * T_out + static_cast<size_t>( q0 + q ) * s + r] = ph[i] + bias[c];
	}
}

__global__ void k_channel_bias( float* y, const float* b, int C, int T ) {
	const size_t n = static_cast<size_t>( C ) * T;
	for ( size_t i = blockIdx.x * static_cast<size_t>( blockDim.x ) + threadIdx.x; i < n; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		y[i] += b[i / T];
	}
}

__global__ void k_axpy( float* y, const float* x, float a, float b, size_t n ) {
	for ( size_t i = blockIdx.x * static_cast<size_t>( blockDim.x ) + threadIdx.x; i < n; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		y[i] = a * y[i] + b * x[i];
	}
}

__global__ void k_clamp( float* y, size_t n ) {
	for ( size_t i = blockIdx.x * static_cast<size_t>( blockDim.x ) + threadIdx.x; i < n; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		y[i] = fminf( 1.0f, fmaxf( -1.0f, y[i] ) );
	}
}

} // namespace

void TtsOps::gemm( const backend::half_t* A, const backend::half_t* W, void* C, int M, int N, int K, bool c32, backend::stream_t stream ) {
	const float alpha = 1.0f, beta = 0.0f;
	cublasHandle_t h = blas();
	CUBLAS_CHECK( cublasSetStream( h, stream ) );
#if defined(USE_ROCM)
	if ( c32 ) {
		// hipBLAS's FP16-in / FP32-out GEMMs take kernels without the matrix cores on RDNA3 (Tensile HSS MT64x32x8: half of a generation's GPU
		// time): the FP16-out GEMM (WMMA), then widened. The result is rounded to FP16 once, as vall_e's bf16 autocast rounds it to bf16
		const size_t n = static_cast<size_t>( M ) * N;
		int dev = 0;
		CUDA_CHECK( cudaGetDevice( &dev ) );
		if ( n > g_c16_elems[dev] ) {
			if ( g_c16[dev] ) { CUDA_CHECK( cudaFree( g_c16[dev] ) ); }
			CUDA_CHECK( cudaMalloc( &g_c16[dev], n * sizeof( half ) ) );
			g_c16_elems[dev] = n;
		}
		CUBLAS_CHECK( cublasGemmEx( h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16F, K, A, CUDA_R_16F, K, &beta, g_c16[dev], CUDA_R_16F, N,
			CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT ) );
		engine::dense::k_convert<<<static_cast<int>( std::min<size_t>( 4096, ( n + 255 ) / 256 ) ), 256, 0, stream>>>( g_c16[dev], static_cast<float*>( C ), n );
		return;
	}
#endif
	CUBLAS_CHECK( cublasGemmEx( h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16F, K, A, CUDA_R_16F, K, &beta, C, c32 ? CUDA_R_32F : CUDA_R_16F, N,
		CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT ) );
}

void TtsOps::digits_values( const uint8_t* digits, backend::half_t* vals, int n, backend::stream_t stream ) {
	k_digits_values<<<( n + 255 ) / 256, 256, 0, stream>>>( digits, vals, n );
}

void TtsOps::embed_mix( float* x, const float* a, const backend::half_t* text_emb, const int* ids, const uint8_t* kind, const uint8_t* masked, const float* mask_emb,
	const float* len_q, const float* segment, const float* mode_vec, int L, int d, backend::stream_t stream ) {
	k_embed_mix<<<L, 256, 0, stream>>>( x, a, text_emb, ids, kind, masked, mask_emb, len_q, segment, mode_vec, d );
}

void TtsOps::rope_table( float* cs, float* sn, int n, int rot, float theta, backend::stream_t stream ) {
	k_rope_table<<<256, 256, 0, stream>>>( cs, sn, n, rot, theta );
}

void TtsOps::add_rms_norm( float* x, const float* branch, const float* w, int rows, int d, float eps, backend::half_t* out16, float* out32, backend::stream_t stream ) {
	ENGINE_ASSERT( d <= 256 * engine::dense::kMaxPerThread, "TtsOps::add_rms_norm: d too large" );
	if ( out16 ) { engine::dense::k_norm_mod<half><<<rows, 256, 0, stream>>>( x, branch, nullptr, 0, w, nullptr, nullptr, 0, nullptr, d, eps, out16 ); }
	else { engine::dense::k_norm_mod<float><<<rows, 256, 0, stream>>>( x, branch, nullptr, 0, w, nullptr, nullptr, 0, nullptr, d, eps, out32 ); }
}

void TtsOps::add_bias( float* y, const float* bias, int rows, int cols, backend::stream_t stream ) {
	// as many blocks as the call sites always launched (the grid-stride loop covers any size)
	const int blocks = static_cast<size_t>( rows ) * cols <= 1024 ? 4 : 256;
	engine::dense::k_add_bias<<<blocks, 256, 0, stream>>>( y, bias, rows, cols );
}

bool TtsOps::head_major( int head_dim ) { return !fused_attention( head_dim ); }

size_t TtsOps::attention_scratch( int heads, int head_dim, int max_len ) {
	( void ) heads;
	static_assert( ATTN_BYTES >= 2560ull * 2560 * ( 4 + 2 ), "one head's scores at 2560 positions" );
	ENGINE_ASSERT( static_cast<size_t>( max_len ) * max_len * ( 4 + 2 ) <= ATTN_BYTES, "TtsOps::attention_scratch: rows too long" );
	return fused_attention( head_dim ) ? 0 : ATTN_BYTES;
}

void TtsOps::qk_norm_rope( const backend::half_t* qkv, const float* qn, const float* kn, backend::half_t* q, backend::half_t* k, backend::half_t* v, const int* pos, int L, int H,
	int hd, int rot, const float* cs, const float* sn, float eps, bool head_major, backend::stream_t stream ) {
	ENGINE_ASSERT( hd % 32 == 0 && hd <= 128, "TtsOps::qk_norm_rope: head_dim a multiple of 32 up to 128" );
	k_qk_norm_rope<<<( L * H + QK_WARPS - 1 ) / QK_WARPS, 32 * QK_WARPS, QK_WARPS * 2 * hd * sizeof( float ), stream>>>( qkv, qn, kn, q, k, v, pos, L, H, hd, rot, cs, sn,
		eps, head_major );
}

void TtsOps::attention( const backend::half_t* q, const backend::half_t* k, const backend::half_t* v, backend::half_t* o, int s, int Lr, int L, int H, int hd, float scale,
	void* scratch, size_t scratch_bytes, backend::stream_t stream ) {
	if ( fused_attention( hd ) ) {
		const size_t at = static_cast<size_t>( s ) * H * hd;   // [L, H, hd]
		ENGINE_ASSERT( engine::video::h3_attention( q + at, k + at, v + at, o + at, Lr, Lr, H, scale, false, stream, hd ), "tts: fused attention" );
		return;
	}
	ENGINE_ASSERT( scratch && scratch_bytes >= ATTN_BYTES, "TtsOps::attention: the scores' scratch" );
	cublasHandle_t blas_h = blas();
	CUBLAS_CHECK( cublasSetStream( blas_h, stream ) );
	float* S = static_cast<float*>( scratch );
	half* P = reinterpret_cast<half*>( static_cast<char*>( scratch ) + ATTN_S_BYTES );
	const float one = 1.0f, zero = 0.0f;
	const int hg = std::min( H, static_cast<int>( ATTN_BYTES / 6 / ( static_cast<size_t>( Lr ) * Lr ) ) );   // heads per batch of scores
	for ( int h0 = 0; h0 < H; h0 += hg ) {
		const int nh = std::min( hg, H - h0 );
		const size_t at = ( static_cast<size_t>( h0 ) * L + s ) * hd;   // [H, L, hd]: this row's positions of head h0, heads L * hd apart
		// S[h] = Q[h] K[h]^T, row-major [Lr(q), Lr(k)]
#if defined(USE_ROCM)
		// FP16 scores, already scaled: hipBLAS takes a kernel without the matrix cores for FP32 output (Tensile HSS MT64x32x8, a fifth of
		// a generation's GPU time with its softmax); the scaled scores of QK-normed heads are small, and FP16 holds them to ~1e-3
		half* S16 = reinterpret_cast<half*>( S );
		CUBLAS_CHECK( cublasGemmStridedBatchedEx( blas_h, CUBLAS_OP_T, CUBLAS_OP_N, Lr, Lr, hd, &scale, k + at, CUDA_R_16F, hd, static_cast<long long>( L ) * hd,
			q + at, CUDA_R_16F, hd, static_cast<long long>( L ) * hd, &zero, S16, CUDA_R_16F, Lr, static_cast<long long>( Lr ) * Lr, nh, CUBLAS_COMPUTE_32F,
			CUBLAS_GEMM_DEFAULT ) );
		k_softmax_h<<<nh * Lr, 256, 0, stream>>>( S16, P, Lr );
#else
		CUBLAS_CHECK( cublasGemmStridedBatchedEx( blas_h, CUBLAS_OP_T, CUBLAS_OP_N, Lr, Lr, hd, &one, k + at, CUDA_R_16F, hd, static_cast<long long>( L ) * hd,
			q + at, CUDA_R_16F, hd, static_cast<long long>( L ) * hd, &zero, S, CUDA_R_32F, Lr, static_cast<long long>( Lr ) * Lr, nh, CUBLAS_COMPUTE_32F,
			CUBLAS_GEMM_DEFAULT ) );
		k_softmax<<<nh * Lr, 256, 0, stream>>>( S, P, Lr, scale );
#endif
		// O[h] = P[h] V[h], row-major [Lr, hd]
		CUBLAS_CHECK( cublasGemmStridedBatchedEx( blas_h, CUBLAS_OP_N, CUBLAS_OP_N, hd, Lr, Lr, &one, v + at, CUDA_R_16F, hd, static_cast<long long>( L ) * hd,
			P, CUDA_R_16F, Lr, static_cast<long long>( Lr ) * Lr, &zero, o + at, CUDA_R_16F, hd, static_cast<long long>( L ) * hd, nh, CUBLAS_COMPUTE_32F,
			CUBLAS_GEMM_DEFAULT ) );
	}
}

void TtsOps::heads_gate( const backend::half_t* o, const backend::half_t* g, backend::half_t* y, int L, int H, int hd, bool head_major, backend::stream_t stream ) {
	k_heads_gate<<<L, 256, 0, stream>>>( o, g, y, L, H, hd, head_major );
}

void TtsOps::swiglu( const backend::half_t* up, backend::half_t* act, int rows, int F, backend::stream_t stream ) {
	k_swiglu<<<rows, 256, 0, stream>>>( up, act, F );
}

void TtsOps::gather_rows( const float* src, const int* pos, int offset, backend::half_t* dst, int n, int d, backend::stream_t stream ) {
	k_gather_half<<<n, 256, 0, stream>>>( src, pos, offset, dst, d );
}

void TtsOps::time_embedding( float t, backend::half_t* out, backend::stream_t stream ) {
	k_time_embedding<<<1, 128, 0, stream>>>( t, out );
}

void TtsOps::bias_act( const float* x, const float* bias, backend::half_t* y, int rows, int cols, bool silu, backend::stream_t stream ) {
	const int blocks = static_cast<size_t>( rows ) * cols <= 1024 ? 4 : 256;
	k_bias_act_half<<<blocks, 256, 0, stream>>>( x, bias, y, rows, cols, silu ? 1 : 0 );
}

void TtsOps::cond_silu( const float* cond, const float* temb, backend::half_t* sc, int rows, int W, backend::stream_t stream ) {
	k_cond_silu<<<256, 256, 0, stream>>>( cond, temb, sc, rows, W );
}

void TtsOps::ln_mod( const float* x, const float* mod, int mstride, backend::half_t* out, int rows, int W, backend::stream_t stream ) {
	k_ln_mod<<<rows, 256, 0, stream>>>( x, mod, mstride, out, W );
}

void TtsOps::gate_add( float* x, const float* branch, const float* bias, const float* mod, int mstride, int rows, int W, backend::stream_t stream ) {
	k_gate_add<<<256, 256, 0, stream>>>( x, branch, bias, mod, mstride, rows, W );
}

void TtsOps::replicate( const float* x, backend::half_t* y, int n, int R, backend::stream_t stream ) {
	k_replicate_half<<<256, 256, 0, stream>>>( x, y, n, R );
}

void TtsOps::guide_step( const float* v, const float* bias, float* x, float* vt, int n, int V, int R, float cfg, float cfg_prompt, float dt, backend::stream_t stream ) {
	k_guide_step<<<64, 256, 0, stream>>>( v, bias, x, vt, n, V, R, cfg, cfg_prompt, dt );
}

void TtsOps::snap( const float* x, uint8_t* out, int n, backend::stream_t stream ) {
	k_snap<<<64, 256, 0, stream>>>( x, out, n );
}

// ---- the codec: a convolution is GEMMs over chunks of time, the chunk's columns against the weight [co, ci * k]

void CodecOps::conv( const float* x, const float* w, const float* b, float* y, int ci, int co, int k, int dil, int stride, int pad, int T, int T_out, float* cols,
	size_t chunk, backend::stream_t stream ) {
	const int K = ci * k;
	const int n = static_cast<int>( std::max<size_t>( 1, std::min<size_t>( T_out, chunk / K ) ) );
	const float one = 1.0f, zero = 0.0f;
	cublasHandle_t h = blas();
	CUBLAS_CHECK( cublasSetStream( h, stream ) );
	for ( int t0 = 0; t0 < T_out; t0 += n ) {
		const int m = std::min( n, T_out - t0 );
		k_cols<<<grid( static_cast<size_t>( K ) * m ), 256, 0, stream>>>( x, cols, ci, k, dil, stride, pad, T, t0, m );
		// row-major Y[co, T_out] at column t0 = column-major [m, co] with ld T_out: cols^T [m, K] x W^T [K, co]
		CUBLAS_CHECK( cublasSgemm( h, CUBLAS_OP_N, CUBLAS_OP_N, m, co, K, &one, cols, m, w, K, &zero, y + t0, T_out ) );
	}
	k_channel_bias<<<grid( static_cast<size_t>( co ) * T_out ), 256, 0, stream>>>( y, b, co, T_out );
}

void CodecOps::up_conv( const float* x, const float* w, const float* b, float* y, int ci, int co, int s, int T_in, float* cols, float* ph, size_t chunk,
	backend::stream_t stream ) {
	const int K = 3 * ci, rows = s * co;
	const int n = static_cast<int>( std::max<size_t>( 1, std::min<size_t>( { static_cast<size_t>( T_in ), chunk / K, chunk / rows } ) ) );
	const float one = 1.0f, zero = 0.0f;
	cublasHandle_t h = blas();
	CUBLAS_CHECK( cublasSetStream( h, stream ) );
	for ( int q0 = 0; q0 < T_in; q0 += n ) {
		const int m = std::min( n, T_in - q0 );
		k_up_cols<<<grid( static_cast<size_t>( K ) * m ), 256, 0, stream>>>( x, cols, ci, T_in, q0, m );
		CUBLAS_CHECK( cublasSgemm( h, CUBLAS_OP_N, CUBLAS_OP_N, m, rows, K, &one, cols, m, w, K, &zero, ph, m ) );
		k_interleave<<<grid( static_cast<size_t>( rows ) * m ), 256, 0, stream>>>( ph, b, y, co, s, T_in * s, q0, m );
	}
}

void CodecOps::act( const float* x, float* y, const float* alpha, int C, int T, backend::stream_t stream ) {
	k_act<<<grid( static_cast<size_t>( C ) * T ), 256, 0, stream>>>( x, y, alpha, C, T );
}

void CodecOps::axpy( float* y, const float* x, float a, float b, size_t n, backend::stream_t stream ) {
	k_axpy<<<grid( n ), 256, 0, stream>>>( y, x, a, b, n );
}

void CodecOps::clamp( float* y, size_t n, backend::stream_t stream ) {
	k_clamp<<<grid( n ), 256, 0, stream>>>( y, n );
}

void CodecOps::dequant( const uint8_t* digits, float* x, int T, const int* levels, backend::stream_t stream ) {
	const int4 hl = make_int4( levels[0] / 2, levels[1] / 2, levels[2] / 2, levels[3] / 2 );
	k_dequant<<<grid( static_cast<size_t>( 32 ) * T ), 256, 0, stream>>>( digits, x, T, hl );
}

void CodecOps::quant( const float* x, uint8_t* digits, int T, const int* levels, float eps, backend::stream_t stream ) {
	const int4 lv = make_int4( levels[0], levels[1], levels[2], levels[3] );
	k_quant<<<grid( static_cast<size_t>( 32 ) * T ), 256, 0, stream>>>( x, digits, T, lv, eps );
}

} // namespace engine::modules
