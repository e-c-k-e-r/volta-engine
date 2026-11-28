#include "engine/modules/vision.h"
#include "engine/backend/cuda/reduce.cuh"
#include "trellis_context.h"

#include "engine/modules/gemm.h"
#include <cuda_fp16.h>
#include <mma.h>
#if defined(USE_ROCM)
	#include "../../../kernels/rocm/mma_emu.h"
#endif
#include "../../../kernels/cuda/mma_sm80.cuh"
#include <cmath>

namespace engine::modules {

namespace {

__device__ __forceinline__ float gelu_tanh_f( float x ) {
	const float u = 0.7978845608028654f * ( x + 0.044715f * x * x * x );
	return 0.5f * x * ( 1.0f + tanhf( u ) );
}

__device__ __forceinline__ float gelu_erf_f( float x ) {
	return 0.5f * x * ( 1.0f + erff( x * 0.7071067811865476f ) );
}

__global__ void add_bias_pos_kernel( float* out, const half* bias, const half* add, int n, size_t total ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i >= total ) return;
	out[i] += __half2float( bias[i % n] ) + ( add ? __half2float( add[i] ) : 0.0f );
}

// One block per row; block-wide mean / variance in FP32.
// (called twice in a row by the norm kernels, hence the closing barrier)
__device__ float block_sum( float v, float* ) {
	const float t = engine::math::block_sum_ordered<32>( v );
	__syncthreads();
	return t;
}

__global__ void layer_norm_kernel( const float* __restrict__ x, const half* __restrict__ w, const half* __restrict__ b,
	half* __restrict__ out, int n, int group, float eps ) {
	__shared__ float smem[32];
	const int row = blockIdx.x;
	const float* xr = x + static_cast<size_t>( row ) * n;
	float s = 0.0f;
	for ( int i = threadIdx.x; i < n; i += blockDim.x ) s += xr[i];
	const float mean = block_sum( s, smem ) / n;
	float q = 0.0f;
	for ( int i = threadIdx.x; i < n; i += blockDim.x ) {
		const float d = xr[i] - mean;
		q += d * d;
	}
	const float inv = rsqrtf( block_sum( q, smem ) / n + eps );
	// group rows are concatenated: row r lands in output row r / group at column offset ( r % group ) * n
	half* o = out + static_cast<size_t>( row / group ) * group * n + static_cast<size_t>( row % group ) * n;
	for ( int i = threadIdx.x; i < n; i += blockDim.x ) {
		o[i] = __float2half( ( xr[i] - mean ) * inv * __half2float( w[i] ) + __half2float( b[i] ) );
	}
}


__global__ void rms_norm_f32_kernel( const float* __restrict__ x, const half* __restrict__ w, half* __restrict__ out, int n, float eps ) {
	__shared__ float smem[32];
	const int row = blockIdx.x;
	const float* xr = x + static_cast<size_t>( row ) * n;
	float q = 0.0f;
	for ( int i = threadIdx.x; i < n; i += blockDim.x ) q += xr[i] * xr[i];
	const float inv = rsqrtf( block_sum( q, smem ) / n + eps );
	for ( int i = threadIdx.x; i < n; i += blockDim.x ) out[static_cast<size_t>( row ) * n + i] = __float2half( xr[i] * inv * __half2float( w[i] ) );
}

__global__ void layer_norm_half_kernel( const half* __restrict__ x, const half* __restrict__ w, const half* __restrict__ b, half* __restrict__ out, int n, float eps ) {
	__shared__ float smem[32];
	const int row = blockIdx.x;
	const half* xr = x + static_cast<size_t>( row ) * n;
	float s = 0.0f;
	for ( int i = threadIdx.x; i < n; i += blockDim.x ) s += __half2float( xr[i] );
	const float mean = block_sum( s, smem ) / n;
	float q = 0.0f;
	for ( int i = threadIdx.x; i < n; i += blockDim.x ) {
		const float d = __half2float( xr[i] ) - mean;
		q += d * d;
	}
	const float inv = rsqrtf( block_sum( q, smem ) / n + eps );
	for ( int i = threadIdx.x; i < n; i += blockDim.x ) {
		out[static_cast<size_t>( row ) * n + i] = __float2half( ( __half2float( xr[i] ) - mean ) * inv * __half2float( w[i] ) + __half2float( b[i] ) );
	}
}

// One block per row, one warp per head, lane = rotary pair (d, d + 32) of a 64-wide head
__global__ void qkv_norm_rope_kernel( half* __restrict__ q, half* __restrict__ k, half* __restrict__ v, const half* __restrict__ bq, const half* __restrict__ bk,
	const half* __restrict__ bv, const half* __restrict__ qn, const half* __restrict__ kn, const float* __restrict__ cs, const float* __restrict__ sn,
	int heads, float eps ) {
	const int row = blockIdx.x, head = threadIdx.x >> 5, d = threadIdx.x & 31;
	if ( head >= heads ) return;
	const int width = heads * 64;
	const size_t base = static_cast<size_t>( row ) * width + head * 64;
	const int i1 = head * 64 + d, i2 = i1 + 32;
	const float c1 = cs[static_cast<size_t>( row ) * 64 + d], s1 = sn[static_cast<size_t>( row ) * 64 + d];
	const float c2 = cs[static_cast<size_t>( row ) * 64 + d + 32], s2 = sn[static_cast<size_t>( row ) * 64 + d + 32];
	for ( int which = 0; which < 3; ++which ) {
		half* t = ( which == 0 ) ? q : ( which == 1 ? k : v );
		const half* bias = ( which == 0 ) ? bq : ( which == 1 ? bk : bv );
		float a = __half2float( t[base + d] ) + __half2float( bias[i1] );
		float b = __half2float( t[base + d + 32] ) + __half2float( bias[i2] );
		if ( which < 2 ) {
			const half* gain = ( which == 0 ) ? qn : kn;
			float ss = a * a + b * b;
			ss = engine::math::warp_sum( ss );
			const float inv = rsqrtf( ss / 64.0f + eps );
			a = a * inv * __half2float( gain[d] );
			b = b * inv * __half2float( gain[d + 32] );
			const float ra = a * c1 - b * s1, rb = b * c2 + a * s2;
			a = ra;
			b = rb;
		}
		t[base + d] = __float2half( a );
		t[base + d + 32] = __float2half( b );
	}
}

// One block per (row, head), 128 threads: q, k normalized with gains, v without, then q and k rotated in their x / y halves
__global__ void qkv_norm_rope2d_kernel( half* __restrict__ q, half* __restrict__ k, half* __restrict__ v, const half* __restrict__ qn,
	const half* __restrict__ kn, const int* __restrict__ pos_xy, int heads, int hd, float theta, float eps ) {
	__shared__ float s_x[128];
	const int row = blockIdx.y, head = blockIdx.x, t = threadIdx.x;
	const size_t off = ( static_cast<size_t>( row ) * heads + head ) * hd;
	const int half_d = hd / 2, quarter = hd / 4;
	for ( int which = 0; which < 3; ++which ) {
		half* x = ( which == 0 ? q : which == 1 ? k : v ) + off;
		const half* w = which == 0 ? qn : which == 1 ? kn : nullptr;
		const float val = t < hd ? __half2float( x[t] ) : 0.0f;
		const float inv = rsqrtf( engine::math::block_sum<4>( val * val ) / hd + eps );
		float y = val * inv * ( w != nullptr && t < hd ? __half2float( w[t] ) : 1.0f );
		if ( which < 2 ) {
			// pair (i, i + quarter) inside the half: s_x holds the normalized head, every thread reads its partner
			__syncthreads();
			if ( t < hd ) s_x[t] = y;
			__syncthreads();
			if ( t < hd ) {
				const int hf = t / half_d, i = t % half_d, pi = i % quarter;
				const float pos = static_cast<float>( pos_xy[2 * row + hf] );
				const float ang = pos * powf( theta, -2.0f * pi / static_cast<float>( half_d ) );
				float sn, cs;
				sincosf( ang, &sn, &cs );
				const float partner = s_x[hf * half_d + ( i < quarter ? i + quarter : i - quarter )];
				y = i < quarter ? y * cs - partner * sn : y * cs + partner * sn;
			}
		}
		if ( t < hd ) x[t] = __float2half_rn( y );
		__syncthreads();   // block_sum's shared array is reused by the next tensor
	}
}

// One block per soft token: the mean of its k x k patches, scaled and standardized
__global__ void pool_std_kernel( const float* __restrict__ x, const half* __restrict__ bias, const half* __restrict__ scale, half* __restrict__ out,
	int gw, int k, int n, float mul ) {
	const int tok = blockIdx.x, tw = gw / k;
	const int ty = tok / tw, tx = tok % tw;
	const float inv = 1.0f / static_cast<float>( k * k );
	for ( int c = threadIdx.x; c < n; c += blockDim.x ) {
		float acc = 0.0f;
		for ( int dy = 0; dy < k; ++dy )
			for ( int dx = 0; dx < k; ++dx ) acc += x[( static_cast<size_t>( ty * k + dy ) * gw + tx * k + dx ) * n + c];
		float vv = acc * inv * mul;
		if ( bias ) vv -= __half2float( bias[c] );
		if ( scale ) vv *= __half2float( scale[c] );
		out[static_cast<size_t>( tok ) * n + c] = __float2half_rn( vv );
	}
}

__global__ void merge_gather_kernel( const half* __restrict__ h, half* __restrict__ out, int n, int group ) {
	const int t = blockIdx.x;
	for ( int idx = threadIdx.x; idx < n * group; idx += blockDim.x ) {
		const int c = idx / group, j = idx % group;
		out[static_cast<size_t>( t ) * n * group + idx] = h[( static_cast<size_t>( t ) * group + j ) * n + c];
	}
}

__global__ void qkv_bias_rope_kernel( half* __restrict__ q, half* __restrict__ k, half* __restrict__ v, const half* __restrict__ bq,
	const half* __restrict__ bk, const half* __restrict__ bv, const float* __restrict__ cs, const float* __restrict__ sn,
	int heads, int head_dim ) {
	const int row = blockIdx.x;
	const int half_dim = head_dim / 2;
	const int width = heads * head_dim;
	const size_t base = static_cast<size_t>( row ) * width;
	for ( int idx = threadIdx.x; idx < heads * half_dim; idx += blockDim.x ) {
		const int h = idx / half_dim, d = idx % half_dim;
		const int i1 = h * head_dim + d, i2 = i1 + half_dim;
		const float c1 = cs[static_cast<size_t>( row ) * head_dim + d], s1 = sn[static_cast<size_t>( row ) * head_dim + d];
		const float c2 = cs[static_cast<size_t>( row ) * head_dim + d + half_dim], s2 = sn[static_cast<size_t>( row ) * head_dim + d + half_dim];
		{
			const float a = __half2float( q[base + i1] ) + __half2float( bq[i1] );
			const float b = __half2float( q[base + i2] ) + __half2float( bq[i2] );
			q[base + i1] = __float2half( a * c1 - b * s1 );
			q[base + i2] = __float2half( b * c2 + a * s2 );
		}
		{
			const float a = __half2float( k[base + i1] ) + __half2float( bk[i1] );
			const float b = __half2float( k[base + i2] ) + __half2float( bk[i2] );
			k[base + i1] = __float2half( a * c1 - b * s1 );
			k[base + i2] = __float2half( b * c2 + a * s2 );
		}
		v[base + i1] = __float2half( __half2float( v[base + i1] ) + __half2float( bv[i1] ) );
		v[base + i2] = __float2half( __half2float( v[base + i2] ) + __half2float( bv[i2] ) );
	}
}

constexpr int kAttnWarps = 8;
constexpr int kAttnTile = 64;
constexpr int kAttnMaxDim = 96;

// Flash-style attention on the CUDA cores: a block of 8 warps shares K / V tiles of 64 keys; a warp owns one query. Lanes score
// keys lane and lane + 32, then split the head dimension for the value accumulation.
__global__ void attention_kernel( const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v, half* __restrict__ out,
	int m, int heads, int head_dim, float scale ) {
	extern __shared__ float smem[];
	const int ld = head_dim + 1;
	float* ks = smem;
	float* vs = smem + kAttnTile * ld;
	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	const int head = blockIdx.y;
	const int qi = blockIdx.x * kAttnWarps + warp;
	const int width = heads * head_dim;
	const bool active = qi < m;

	float qr[kAttnMaxDim];
	if ( active ) {
		for ( int d = 0; d < head_dim; ++d ) qr[d] = __half2float( q[static_cast<size_t>( qi ) * width + head * head_dim + d] ) * scale;
	}
	float acc[3] = { 0.0f, 0.0f, 0.0f };
	float run_max = -INFINITY, run_sum = 0.0f;

	for ( int t0 = 0; t0 < m; t0 += kAttnTile ) {
		const int tile = min( kAttnTile, m - t0 );
		__syncthreads();
		for ( int idx = threadIdx.x; idx < tile * head_dim; idx += blockDim.x ) {
			const int j = idx / head_dim, d = idx % head_dim;
			const size_t src = static_cast<size_t>( t0 + j ) * width + head * head_dim + d;
			ks[j * ld + d] = __half2float( k[src] );
			vs[j * ld + d] = __half2float( v[src] );
		}
		__syncthreads();
		if ( !active ) continue;
		float s[2];
		#pragma unroll
		for ( int h = 0; h < 2; ++h ) {
			const int j = lane + h * 32;
			float dot = -INFINITY;
			if ( j < tile ) {
				dot = 0.0f;
				for ( int d = 0; d < head_dim; ++d ) dot = fmaf( qr[d], ks[j * ld + d], dot );
			}
			s[h] = dot;
		}
		float tmax = fmaxf( s[0], s[1] );
		tmax = engine::math::warp_max( tmax );
		const float new_max = fmaxf( run_max, tmax );
		const float corr = __expf( run_max - new_max );
		const float p0 = ( s[0] == -INFINITY ) ? 0.0f : __expf( s[0] - new_max );
		const float p1 = ( s[1] == -INFINITY ) ? 0.0f : __expf( s[1] - new_max );
		float psum = p0 + p1;
		psum = engine::math::warp_sum( psum );
		run_sum = run_sum * corr + psum;
		run_max = new_max;
		acc[0] *= corr;
		acc[1] *= corr;
		acc[2] *= corr;
		for ( int j = 0; j < tile; ++j ) {
			const float p = __shfl_sync( 0xffffffff, ( j < 32 ) ? p0 : p1, j & 31 );
			#pragma unroll
			for ( int c = 0; c < 3; ++c ) {
				const int d = lane + c * 32;
				if ( d < head_dim ) acc[c] = fmaf( p, vs[j * ld + d], acc[c] );
			}
		}
	}
	if ( !active ) return;
	const float inv = 1.0f / run_sum;
	#pragma unroll
	for ( int c = 0; c < 3; ++c ) {
		const int d = lane + c * 32;
		if ( d < head_dim ) out[static_cast<size_t>( qi ) * width + head * head_dim + d] = __float2half( acc[c] * inv );
	}
}


// Tensor-core flash attention (WMMA 16x16x16, FP32 accumulate) for head_dim <= 80: 4 warps share 64-key K / V tiles, each warp owns 16
// queries. The head dimension is padded to 80 with zeros in shared memory. Softmax statistics live per row in registers (a lane pair
// per row); the running output stays in shared memory so the rescale by the new row maximum is plain element-wise work.
constexpr int kTcWarps = 4;
constexpr int kTcQ = kTcWarps * 16;
constexpr int kTcKeys = 64;
constexpr int kTcDim = 80;
constexpr int kTcSLd = kTcKeys + 4;

constexpr size_t kTcSmemBytes = ( 3 * kTcQ * kTcDim ) * sizeof( half ) + kTcWarps * ( 16 * kTcSLd * sizeof( float ) + 16 * kTcKeys * sizeof( half ) + 16 * kTcDim * sizeof( float ) );

__global__ void __launch_bounds__( kTcWarps * 32 ) attention_tc_kernel( const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v,
	half* __restrict__ out, int m, int heads, int head_dim, float scale ) {
	using namespace nvcuda;
	extern __shared__ __align__( 128 ) unsigned char smem_raw[];
	half* Qs = reinterpret_cast<half*>( smem_raw );
	half* Ks = Qs + kTcQ * kTcDim;
	half* Vs = Ks + kTcKeys * kTcDim;
	float* Sall = reinterpret_cast<float*>( Vs + kTcKeys * kTcDim );
	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	float* Ss = Sall + warp * 16 * kTcSLd;
	half* Ps = reinterpret_cast<half*>( Sall + kTcWarps * 16 * kTcSLd ) + warp * 16 * kTcKeys;
	float* Os = reinterpret_cast<float*>( reinterpret_cast<half*>( Sall + kTcWarps * 16 * kTcSLd ) + kTcWarps * 16 * kTcKeys ) + warp * 16 * kTcDim;
	const int head = blockIdx.y;
	const int width = heads * head_dim;
	const int q_base = blockIdx.x * kTcQ;

	for ( int idx = threadIdx.x; idx < kTcQ * kTcDim; idx += blockDim.x ) {
		const int r = idx / kTcDim, d = idx % kTcDim;
		half val = __float2half( 0.0f );
		if ( q_base + r < m && d < head_dim ) val = q[static_cast<size_t>( q_base + r ) * width + head * head_dim + d];
		Qs[idx] = val;
	}
	for ( int idx = lane; idx < 16 * kTcDim; idx += 32 ) Os[idx] = 0.0f;
	float run_max = -INFINITY, run_sum = 0.0f;
	const int row = lane >> 1, col0 = ( lane & 1 ) * 32;

	for ( int t0 = 0; t0 < m; t0 += kTcKeys ) {
		__syncthreads();
		for ( int idx = threadIdx.x; idx < kTcKeys * kTcDim; idx += blockDim.x ) {
			const int r = idx / kTcDim, d = idx % kTcDim;
			half kv = __float2half( 0.0f ), vv = __float2half( 0.0f );
			if ( t0 + r < m && d < head_dim ) {
				const size_t src = static_cast<size_t>( t0 + r ) * width + head * head_dim + d;
				kv = k[src];
				vv = v[src];
			}
			Ks[idx] = kv;
			Vs[idx] = vv;
		}
		__syncthreads();

		#pragma unroll
		for ( int n = 0; n < kTcKeys / 16; ++n ) {
			wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
			wmma::fill_fragment( acc, 0.0f );
			#pragma unroll
			for ( int kk = 0; kk < kTcDim / 16; ++kk ) {
				wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
				wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
				wmma::load_matrix_sync( a, Qs + warp * 16 * kTcDim + kk * 16, kTcDim );
				wmma::load_matrix_sync( b, Ks + n * 16 * kTcDim + kk * 16, kTcDim );
				wmma::mma_sync( acc, a, b, acc );
			}
			wmma::store_matrix_sync( Ss + n * 16, acc, kTcSLd, wmma::mem_row_major );
		}
		__syncwarp();

		// Online softmax: this lane pair owns row `row`, 32 key columns each
		float mx = -INFINITY;
		#pragma unroll
		for ( int c = 0; c < 32; ++c ) {
			const float sv = ( t0 + col0 + c < m ) ? Ss[row * kTcSLd + col0 + c] * scale : -INFINITY;
			mx = fmaxf( mx, sv );
		}
		mx = fmaxf( mx, __shfl_xor_sync( 0xffffffff, mx, 1 ) );
		const float new_max = fmaxf( run_max, mx );
		const float corr = ( run_max == -INFINITY ) ? 0.0f : __expf( run_max - new_max );
		float psum = 0.0f;
		#pragma unroll
		for ( int c = 0; c < 32; ++c ) {
			float p = 0.0f;
			if ( t0 + col0 + c < m ) p = __expf( Ss[row * kTcSLd + col0 + c] * scale - new_max );
			psum += p;
			Ps[row * kTcKeys + col0 + c] = __float2half( p );
		}
		psum += __shfl_xor_sync( 0xffffffff, psum, 1 );
		run_sum = run_sum * corr + psum;
		run_max = new_max;
		// Rescale the running output rows (lane pair per row, 40 dims each)
		#pragma unroll
		for ( int d = ( lane & 1 ); d < kTcDim; d += 2 ) Os[row * kTcDim + d] *= corr;
		__syncwarp();

		#pragma unroll
		for ( int n = 0; n < kTcDim / 16; ++n ) {
			wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
			wmma::load_matrix_sync( acc, Os + n * 16, kTcDim, wmma::mem_row_major );
			#pragma unroll
			for ( int kk = 0; kk < kTcKeys / 16; ++kk ) {
				wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
				wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
				wmma::load_matrix_sync( a, Ps + kk * 16, kTcKeys );
				wmma::load_matrix_sync( b, Vs + kk * 16 * kTcDim + n * 16, kTcDim );
				wmma::mma_sync( acc, a, b, acc );
			}
			wmma::store_matrix_sync( Os + n * 16, acc, kTcDim, wmma::mem_row_major );
		}
		__syncwarp();
	}

	const int qi = q_base + warp * 16 + row;
	if ( qi < m ) {
		const float inv = 1.0f / run_sum;
		for ( int d = ( lane & 1 ); d < head_dim; d += 2 ) {
			out[static_cast<size_t>( qi ) * width + head * head_dim + d] = __float2half( Os[row * kTcDim + d] * inv );
		}
	}
}

// attention_mma_kernel: the register-resident Volta flash attention (vision_attn_mma.cuh, shared with the Vulkan backend's CUDA launches)
#include "vision_attn_mma.cuh"

__global__ void bias_act_kernel( half* x, const half* bias, int n, size_t total, int act ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i >= total ) return;
	float v = __half2float( x[i] ) + ( bias ? __half2float( bias[i % n] ) : 0.0f );
	if ( act == Vision::ACT_GELU_TANH ) v = gelu_tanh_f( v );
	else if ( act == Vision::ACT_GELU_ERF ) v = gelu_erf_f( v );
	else if ( act == Vision::ACT_SILU ) v = v / ( 1.0f + __expf( -v ) );
	x[i] = __float2half( v );
}

__global__ void clamp_kernel( half* x, size_t n, float lo, float hi ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i < n ) x[i] = __float2half( fminf( fmaxf( __half2float( x[i] ), lo ), hi ) );
}

// One block per output position (row, frequency), a thread per channel: the 3x3 taps over the input channels, then the channel LayerNorm and ReLU
__global__ void conv_ln_relu_kernel( const float* __restrict__ in, const float* __restrict__ w, const float* __restrict__ g, float* __restrict__ out, int T, int F,
	int C, int O, float eps ) {
	__shared__ float red[256];
	const int Fo = ( F + 1 ) / 2, pos = blockIdx.x, t = pos / Fo, f = pos % Fo, o = threadIdx.x;
	float acc = 0.0f;
	if ( o < O ) {
		for ( int ky = 0; ky < 3; ++ky ) {
			const int ti = 2 * t - 1 + ky;
			if ( ti < 0 || ti >= T ) continue;
			for ( int kx = 0; kx < 3; ++kx ) {
				const int fi = 2 * f - 1 + kx;
				if ( fi < 0 || fi >= F ) continue;
				const float* src = in + ( static_cast<size_t>( ti ) * F + fi ) * C;
				const float* wk = w + static_cast<size_t>( o ) * C * 9 + ky * 3 + kx;
				for ( int c = 0; c < C; ++c ) acc += wk[c * 9] * src[c];
			}
		}
	}
	red[o] = o < O ? acc : 0.0f;
	__syncthreads();
	for ( int s = blockDim.x / 2; s > 0; s >>= 1 ) {
		if ( o < s ) red[o] += red[o + s];
		__syncthreads();
	}
	const float mean = red[0] / O;
	__syncthreads();
	const float d = o < O ? acc - mean : 0.0f;
	red[o] = d * d;
	__syncthreads();
	for ( int s = blockDim.x / 2; s > 0; s >>= 1 ) {
		if ( o < s ) red[o] += red[o + s];
		__syncthreads();
	}
	if ( o < O ) out[static_cast<size_t>( pos ) * O + o] = fmaxf( 0.0f, d * rsqrtf( red[0] / O + eps ) * g[o] );
}

// One thread per output element: the GLU of the K rows up to t, weighted by the channel's taps
__global__ void glu_dwconv_kernel( const half* __restrict__ x, const half* __restrict__ w, half* __restrict__ out, int T, int D, int K ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i >= static_cast<size_t>( T ) * D ) return;
	const int t = static_cast<int>( i / D ), c = static_cast<int>( i % D );
	float acc = 0.0f;
	for ( int k = 0; k < K; ++k ) {
		const int r = t - K + 1 + k;
		if ( r < 0 ) continue;
		const float a = __half2float( x[static_cast<size_t>( r ) * 2 * D + c] ), g = __half2float( x[static_cast<size_t>( r ) * 2 * D + D + c] );
		acc += __half2float( w[static_cast<size_t>( c ) * K + k] ) * a / ( 1.0f + __expf( -g ) );
	}
	out[i] = __float2half( acc );
}

// One thread per (row, head): the window's scores, their softmax and the weighted values, in FP32
__global__ void local_rel_attention_kernel( const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v, const float* __restrict__ q_scale,
	float k_scale, const float* __restrict__ rel, half* __restrict__ out, int T, int heads, int hd, int window, float cap ) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx >= T * heads ) return;
	const int t = idx / heads, h = idx % heads, HD = heads * hd;
	const half* qr = q + static_cast<size_t>( t ) * HD + h * hd;
	float s[32];
	const int n = min( window, t + 1 );
	float mx = -3.0e38f;
	for ( int d = 0; d < n; ++d ) {
		const half* kr = k + static_cast<size_t>( t - d ) * HD + h * hd;
		const float* rr = rel + static_cast<size_t>( d ) * HD + h * hd;
		float ac = 0.0f, bd = 0.0f;
		for ( int e = 0; e < hd; ++e ) {
			const float qe = __half2float( qr[e] ) * q_scale[e];
			ac += qe * __half2float( kr[e] );
			bd += qe * rr[e];
		}
		const float sc = cap * tanhf( ( ac * k_scale + bd ) / cap );
		s[d] = sc;
		mx = fmaxf( mx, sc );
	}
	float sum = 0.0f;
	for ( int d = 0; d < n; ++d ) {
		s[d] = __expf( s[d] - mx );
		sum += s[d];
	}
	const float inv = 1.0f / sum;
	half* orow = out + static_cast<size_t>( t ) * HD + h * hd;
	for ( int e = 0; e < hd; ++e ) {
		float acc = 0.0f;
		for ( int d = 0; d < n; ++d ) acc += s[d] * __half2float( v[static_cast<size_t>( t - d ) * HD + h * hd + e] );
		orow[e] = __float2half( acc * inv );
	}
}

__global__ void bias_residual_kernel( float* hidden, const half* y, const half* bias, int n, size_t total ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i >= total ) return;
	hidden[i] += __half2float( y[i] ) + __half2float( bias[i % n] );
}

__global__ void f32_to_f16_kernel( const float* in, half* out, size_t n ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i < n ) out[i] = __float2half( in[i] );
}

// One block per output row (merge-block order). The weights are computed in FP32 exactly as the reference interpolation does.
__global__ void pos_interp_kernel( const half* __restrict__ table, half* __restrict__ out, int gh, int gw, int merge, int side, int dim ) {
	const int row = blockIdx.x;
	const int blocks_w = gw / merge;
	const int mm = merge * merge;
	const int blk = row / mm, within = row % mm;
	const int by = blk / blocks_w, bx = blk % blocks_w;
	const int y = by * merge + within / merge, x = bx * merge + within % merge;
	const float hy = gh == 1 ? 0.0f : static_cast<float>( y ) * ( side - 1 ) / ( gh - 1 );
	const float wx = gw == 1 ? 0.0f : static_cast<float>( x ) * ( side - 1 ) / ( gw - 1 );
	const int hf = static_cast<int>( hy ), wf = static_cast<int>( wx );
	const int hc = min( hf + 1, side - 1 ), wc = min( wf + 1, side - 1 );
	const float dh = hy - hf, dw = wx - wf;
	const half* t00 = table + ( static_cast<size_t>( hf ) * side + wf ) * dim;
	const half* t01 = table + ( static_cast<size_t>( hf ) * side + wc ) * dim;
	const half* t10 = table + ( static_cast<size_t>( hc ) * side + wf ) * dim;
	const half* t11 = table + ( static_cast<size_t>( hc ) * side + wc ) * dim;
	for ( int d = threadIdx.x; d < dim; d += blockDim.x ) {
		const float val = ( 1 - dh ) * ( 1 - dw ) * __half2float( t00[d] ) + ( 1 - dh ) * dw * __half2float( t01[d] ) +
			dh * ( 1 - dw ) * __half2float( t10[d] ) + dh * dw * __half2float( t11[d] );
		out[static_cast<size_t>( row ) * dim + d] = __float2half( val );
	}
}

} // namespace

void Vision::f32_to_f16( const float* in, backend::half_t* out, size_t n, cudaStream_t stream ) {
	f32_to_f16_kernel<<<static_cast<unsigned>( ( n + 255 ) / 256 ), 256, 0, stream>>>( in, reinterpret_cast<half*>( out ), n );
}

void Vision::pos_interp( const backend::half_t* table, backend::half_t* out, int gh, int gw, int merge, int side, int dim, cudaStream_t stream ) {
	pos_interp_kernel<<<gh * gw, 256, 0, stream>>>( reinterpret_cast<const half*>( table ), reinterpret_cast<half*>( out ), gh, gw, merge, side, dim );
}

void Vision::patch_embed( const half* patches, const half* w, const half* bias, const half* add, float* out,
	int m, int n, int k, cudaStream_t stream ) {
	// FP32 output straight from the tensor-core GEMM
	Gemm::tn( patches, k, w, k, out, n, true, m, n, k, 0.0f, stream );
	const size_t total = static_cast<size_t>( m ) * n;
	add_bias_pos_kernel<<<static_cast<unsigned>( ( total + 255 ) / 256 ), 256, 0, stream>>>( out, bias, add, n, total );
}

void Vision::rms_norm( const float* x, const half* w, half* out, int m, int n, float eps, cudaStream_t stream ) {
	rms_norm_f32_kernel<<<m, 256, 0, stream>>>( x, w, out, n, eps );
}

void Vision::layer_norm_half( const half* x, const half* w, const half* b, half* out, int m, int n, float eps, cudaStream_t stream ) {
	layer_norm_half_kernel<<<m, 256, 0, stream>>>( x, w, b, out, n, eps );
}

void Vision::qkv_bias_norm_rope( half* q, half* k, half* v, const half* bq, const half* bk, const half* bv, const half* qn, const half* kn,
	const float* cos_table, const float* sin_table, int m, int heads, int head_dim, float eps, cudaStream_t stream ) {
	ENGINE_ASSERT( head_dim == 64 && heads <= 32, "Vision::qkv_bias_norm_rope: 64-wide heads, at most 32" );
	qkv_norm_rope_kernel<<<m, heads * 32, 0, stream>>>( q, k, v, bq, bk, bv, qn, kn, cos_table, sin_table, heads, eps );
}

void Vision::qkv_norm_rope2d( half* q, half* k, half* v, const half* qn, const half* kn, const int* pos_xy, int m, int heads, int head_dim, float theta,
	float eps, cudaStream_t stream ) {
	ENGINE_ASSERT( head_dim <= 128 && head_dim % 4 == 0, "Vision::qkv_norm_rope2d: head_dim a multiple of 4, at most 128" );
	qkv_norm_rope2d_kernel<<<dim3( heads, m ), 128, 0, stream>>>( q, k, v, qn, kn, pos_xy, heads, head_dim, theta, eps );
}

void Vision::pool_std( const float* x, const half* bias, const half* scale, half* out, int gh, int gw, int k, int n, cudaStream_t stream ) {
	ENGINE_ASSERT( gh % k == 0 && gw % k == 0, "Vision::pool_std: the patch grid must be whole k x k cells" );
	pool_std_kernel<<<( gh / k ) * ( gw / k ), 256, 0, stream>>>( x, bias, scale, out, gw, k, n, sqrtf( static_cast<float>( n ) ) );
}

void Vision::merge_gather( const half* h, half* out, int tokens, int n, int group, cudaStream_t stream ) {
	merge_gather_kernel<<<tokens, 256, 0, stream>>>( h, out, n, group );
}

void Vision::dense_gemm( const half* a, const half* w, half* out, int m, int n, int k, cudaStream_t stream ) {
	Gemm::tn( a, k, w, k, out, n, false, m, n, k, 0.0f, stream );
}

void Vision::layer_norm( const float* x, const half* w, const half* b, half* out, int m, int n, float eps, cudaStream_t stream ) {
	layer_norm_kernel<<<m, 256, 0, stream>>>( x, w, b, out, n, 1, eps );
}

void Vision::layer_norm_merge( const float* x, const half* w, const half* b, half* out, int m, int n, int group, float eps, cudaStream_t stream ) {
	ENGINE_ASSERT( m % group == 0, "Vision::layer_norm_merge: rows must be a multiple of the group" );
	layer_norm_kernel<<<m, 256, 0, stream>>>( x, w, b, out, n, group, eps );
}

void Vision::qkv_bias_rope( half* q, half* k, half* v, const half* bq, const half* bk, const half* bv, const float* cos_table,
	const float* sin_table, int m, int heads, int head_dim, cudaStream_t stream ) {
	ENGINE_ASSERT( head_dim % 2 == 0, "Vision::qkv_bias_rope: even head_dim" );
	qkv_bias_rope_kernel<<<m, 256, 0, stream>>>( q, k, v, bq, bk, bv, cos_table, sin_table, heads, head_dim );
}

// Compute capability major of the current device (cached per device)
static int vision_cc_major() {
	static int major[ENGINE_NUM_GPUS] = {};
	int device = 0;
	(void) cudaGetDevice( &device );
	if ( device < 0 || device >= ENGINE_NUM_GPUS ) device = 0;
	if ( major[device] == 0 ) (void) cudaDeviceGetAttribute( &major[device], cudaDevAttrComputeCapabilityMajor, device );
	return major[device];
}

void Vision::attention( const half* q, const half* k, const half* v, half* out, int m, int heads, int head_dim, float scale, cudaStream_t stream ) {
	ENGINE_ASSERT( head_dim <= kTcDim, "Vision::attention: head_dim too large" );
	const dim3 mgrid( ( m + kMmaWarps * 16 - 1 ) / ( kMmaWarps * 16 ), heads );
	// attention_mma_kernel is built on the Volta m8n8k4; SM_80+ runs the wmma kernel, native there
	const bool volta_mma = vision_cc_major() < 8;
	if ( volta_mma && head_dim == 64 ) {
		attention_mma_kernel<64><<<mgrid, kMmaWarps * 32, 0, stream>>>( q, k, v, out, m, heads, scale );
		return;
	}
	if ( volta_mma && head_dim == 72 ) {
		attention_mma_kernel<72><<<mgrid, kMmaWarps * 32, 0, stream>>>( q, k, v, out, m, heads, scale );
		return;
	}
#if defined(USE_ROCM)
	// The WMMA kernel stages ~75 KB; RDNA grants a workgroup 64 KB
	if ( kTcSmemBytes > static_cast<size_t>( rocm_smem_optin_max() ) ) {
		attention_simt( q, k, v, out, m, heads, head_dim, scale, stream );
		return;
	}
#endif
	static bool opted_in = false;
	if ( !opted_in ) {
		(void) cudaFuncSetAttribute( attention_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>( kTcSmemBytes ) );
		opted_in = true;
	}
	const dim3 grid( ( m + kTcQ - 1 ) / kTcQ, heads );
	attention_tc_kernel<<<grid, kTcWarps * 32, kTcSmemBytes, stream>>>( q, k, v, out, m, heads, head_dim, scale );
}

void Vision::attention_simt( const half* q, const half* k, const half* v, half* out, int m, int heads, int head_dim, float scale, cudaStream_t stream ) {
	ENGINE_ASSERT( head_dim <= kAttnMaxDim, "Vision::attention_simt: head_dim too large" );
	const size_t smem = static_cast<size_t>( 2 * kAttnTile * ( head_dim + 1 ) ) * sizeof( float );
	ENGINE_ASSERT( smem <= 48 * 1024, "Vision::attention_simt: shared memory over the default limit" );
	const dim3 grid( ( m + kAttnWarps - 1 ) / kAttnWarps, heads );
	attention_kernel<<<grid, kAttnWarps * 32, smem, stream>>>( q, k, v, out, m, heads, head_dim, scale );
}

void Vision::bias_act( half* x, const half* bias, int m, int n, int act, cudaStream_t stream ) {
	const size_t total = static_cast<size_t>( m ) * n;
	bias_act_kernel<<<static_cast<unsigned>( ( total + 255 ) / 256 ), 256, 0, stream>>>( x, bias, n, total, act );
}

void Vision::bias_residual( float* hidden, const half* y, const half* bias, int m, int n, cudaStream_t stream ) {
	const size_t total = static_cast<size_t>( m ) * n;
	bias_residual_kernel<<<static_cast<unsigned>( ( total + 255 ) / 256 ), 256, 0, stream>>>( hidden, y, bias, n, total );
}

void Vision::clamp( half* x, size_t n, float lo, float hi, cudaStream_t stream ) {
	if ( n == 0 ) return;
	clamp_kernel<<<static_cast<unsigned>( ( n + 255 ) / 256 ), 256, 0, stream>>>( x, n, lo, hi );
}

void Vision::conv_ln_relu( const float* in, const float* w, const float* g, float* out, int T, int F, int C, int O, float eps, cudaStream_t stream ) {
	ENGINE_ASSERT( O >= 1 && O <= 256, "Vision::conv_ln_relu: 1 to 256 channels" );
	const int positions = ( ( T + 1 ) / 2 ) * ( ( F + 1 ) / 2 );
	if ( positions == 0 ) return;
	const int threads = O <= 32 ? 32 : O <= 64 ? 64 : O <= 128 ? 128 : 256;
	conv_ln_relu_kernel<<<positions, threads, 0, stream>>>( in, w, g, out, T, F, C, O, eps );
}

void Vision::glu_dwconv( const half* x, const half* w, half* out, int T, int D, int K, cudaStream_t stream ) {
	const size_t total = static_cast<size_t>( T ) * D;
	if ( total == 0 ) return;
	glu_dwconv_kernel<<<static_cast<unsigned>( ( total + 255 ) / 256 ), 256, 0, stream>>>( x, w, out, T, D, K );
}

void Vision::local_rel_attention( const half* q, const half* k, const half* v, const float* q_scale, float k_scale, const float* rel, half* out, int T,
	int heads, int head_dim, int window, float cap, cudaStream_t stream ) {
	ENGINE_ASSERT( window <= 32, "Vision::local_rel_attention: window above 32" );
	const int total = T * heads;
	if ( total == 0 ) return;
	local_rel_attention_kernel<<<( total + 127 ) / 128, 128, 0, stream>>>( q, k, v, q_scale, k_scale, rel, out, T, heads, head_dim, window, cap );
}

} // namespace engine::modules
