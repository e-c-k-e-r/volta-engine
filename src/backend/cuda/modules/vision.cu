#include "engine/modules/vision.h"
#include "trellis_context.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
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
__device__ float block_sum( float v, float* smem ) {
	#pragma unroll
	for ( int mask = 16; mask > 0; mask >>= 1 ) v += __shfl_xor_sync( 0xffffffff, v, mask );
	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	if ( lane == 0 ) smem[warp] = v;
	__syncthreads();
	float t = 0.0f;
	for ( int w = 0; w < static_cast<int>( blockDim.x >> 5 ); ++w ) t += smem[w];
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
			#pragma unroll
			for ( int mask = 16; mask > 0; mask >>= 1 ) ss += __shfl_xor_sync( 0xffffffff, ss, mask );
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
		#pragma unroll
		for ( int mask = 16; mask > 0; mask >>= 1 ) tmax = fmaxf( tmax, __shfl_xor_sync( 0xffffffff, tmax, mask ) );
		const float new_max = fmaxf( run_max, tmax );
		const float corr = __expf( run_max - new_max );
		const float p0 = ( s[0] == -INFINITY ) ? 0.0f : __expf( s[0] - new_max );
		const float p1 = ( s[1] == -INFINITY ) ? 0.0f : __expf( s[1] - new_max );
		float psum = p0 + p1;
		#pragma unroll
		for ( int mask = 16; mask > 0; mask >>= 1 ) psum += __shfl_xor_sync( 0xffffffff, psum, mask );
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

__global__ void bias_act_kernel( half* x, const half* bias, int n, size_t total, int act ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i >= total ) return;
	float v = __half2float( x[i] ) + ( bias ? __half2float( bias[i % n] ) : 0.0f );
	if ( act == Vision::ACT_GELU_TANH ) v = gelu_tanh_f( v );
	else if ( act == Vision::ACT_GELU_ERF ) v = gelu_erf_f( v );
	x[i] = __float2half( v );
}

__global__ void bias_residual_kernel( float* hidden, const half* y, const half* bias, int n, size_t total ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i >= total ) return;
	hidden[i] += __half2float( y[i] ) + __half2float( bias[i % n] );
}

} // namespace

void Vision::patch_embed( const half* patches, const half* w, const half* bias, const half* add, float* out,
	int m, int n, int k, cudaStream_t stream ) {
	int device = 0;
	cudaGetDevice( &device );
	// FP32 output straight from the tensor-core GEMM: C^T[n, m] = W[n, k] A^T with W and A row-major (k fastest).
	cublasHandle_t h = trellis_get_cublas( device );
	cublasSetStream( h, stream );
	const float alpha = 1.0f, beta = 0.0f;
	const cublasStatus_t st = cublasGemmEx( h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, w, CUDA_R_16F, k, patches, CUDA_R_16F, k,
		&beta, out, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP );
	ENGINE_ASSERT( st == CUBLAS_STATUS_SUCCESS, "Vision::patch_embed: cublasGemmEx failed" );
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

void Vision::merge_gather( const half* h, half* out, int tokens, int n, int group, cudaStream_t stream ) {
	merge_gather_kernel<<<tokens, 256, 0, stream>>>( h, out, n, group );
}

void Vision::dense_gemm( const half* a, const half* w, half* out, int m, int n, int k, cudaStream_t stream ) {
	int device = 0;
	cudaGetDevice( &device );
	cublasHandle_t h = trellis_get_cublas( device );
	cublasSetStream( h, stream );
	const float alpha = 1.0f, beta = 0.0f;
	const cublasStatus_t st = cublasGemmEx( h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, w, CUDA_R_16F, k, a, CUDA_R_16F, k,
		&beta, out, CUDA_R_16F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP );
	ENGINE_ASSERT( st == CUBLAS_STATUS_SUCCESS, "Vision::dense_gemm: cublasGemmEx failed" );
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

void Vision::attention( const half* q, const half* k, const half* v, half* out, int m, int heads, int head_dim, float scale, cudaStream_t stream ) {
	ENGINE_ASSERT( head_dim <= kTcDim, "Vision::attention: head_dim too large" );
	static bool opted_in = false;
	if ( !opted_in ) {
		cudaFuncSetAttribute( attention_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>( kTcSmemBytes ) );
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

} // namespace engine::modules
