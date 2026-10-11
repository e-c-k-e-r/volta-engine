// MiniMax-H3 DiT forward (single GPU). See inc/engine/video/h3_dit.h for the numeric contract.
#include "engine/video/h3_dit.h"
#include "engine/core/arena.h"
#include "engine/core/log.h"
#include "engine/loader/safetensors.h"
#include "engine/video/h3_fmha.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#define CUBLAS_CHECK( call ) \
	do { \
		cublasStatus_t st_ = ( call ); \
		if ( ENGINE_UNLIKELY( st_ != CUBLAS_STATUS_SUCCESS ) ) { \
			std::fprintf( stderr, "[CUBLAS ERROR] %s (%s:%d): %d\n", #call, __FILE__, __LINE__, static_cast<int>( st_ ) ); \
			std::_Exit( 1 ); \
		} \
	} while ( 0 )

namespace engine::video {
namespace {

constexpr int kModalities = 3;

// ---------------------------------------------------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------------------------------------------------

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

// INT8 [N, K] * row scale -> FP16 [N, K], then the 256-group Hadamard along K (x @ H, H = H4^(x4) / 16, symmetric): the checkpoint
// stores W @ H^T, so W_eff = stored @ H.
__global__ void k_dequant_convrot( const int8_t* w, const float* s, half* out, int K ) {
	__shared__ float v[256];
	const int n = blockIdx.y, g = blockIdx.x, t = threadIdx.x;
	const size_t base = static_cast<size_t>( n ) * K + static_cast<size_t>( g ) * 256;
	v[t] = static_cast<float>( w[base + t] ) * s[n];
	__syncthreads();
	for ( int st = 1; st < 256; st *= 4 ) {
		if ( t < 64 ) {
			const int off = t % st, blk = t / st;
			const int b0 = blk * 4 * st + off;
			const float a = v[b0], b = v[b0 + st], c = v[b0 + 2 * st], d = v[b0 + 3 * st];
			v[b0] = a + b + c - d;
			v[b0 + st] = a + b - c + d;
			v[b0 + 2 * st] = a - b + c + d;
			v[b0 + 3 * st] = -a + b + c + d;
		}
		__syncthreads();
	}
	out[base + t] = __float2half_rn( v[t] * 0.0625f );
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
__global__ void k_add_gated( float* x, const float* branch, const float* rowscale, const float* gate, long gstride, const int* idx, int H ) {
	const int row = blockIdx.x;
	const long ii = idx ? idx[row] : 0;
	const float rs = rowscale ? rowscale[row] : 1.0f;
	for ( int c = threadIdx.x; c < H; c += blockDim.x ) {
		const float g = gate ? gate[ii * gstride + c] : 1.0f;
		x[static_cast<size_t>( row ) * H + c] += g * ( branch[static_cast<size_t>( row ) * H + c] * rs );
	}
}

__global__ void k_comb( const int* inverse, const int* tags, int* comb, int n ) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < n ) { comb[i] = inverse[i] * kModalities + max( tags[i], 0 ); }
}

// AdaLN projection of the interpolated time coordinates: out[m, j] = t[m] . W[j] + b[j]
__global__ void k_adaln( const float* t, const float* W, const float* b, float* out, int M, int J, int D ) {
	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	if ( j >= J ) { return; }
	float wv[8];
	for ( int d = 0; d < D; ++d ) { wv[d] = W[static_cast<size_t>( j ) * D + d]; }
	const float bj = b[j];
	for ( int m = 0; m < M; ++m ) {
		float a = bj;
		for ( int d = 0; d < D; ++d ) { a += t[m * D + d] * wv[d]; }
		out[static_cast<size_t>( m ) * J + j] = a;
	}
}

__global__ void k_rope_table( const float* pos, const float* inv_freq, float* cosb, float* sinb, int U, int F ) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= U * 3 * F ) { return; }
	const int row = i / ( 3 * F ), c = i % ( 3 * F ), axis = c / F, f = c % F;
	const float ang = pos[row * 3 + axis] * inv_freq[f];
	cosb[i] = cosf( ang );
	sinb[i] = sinf( ang );
}

__global__ void k_add_bias( float* y, const float* b, int rows, int cols ) {
	const size_t n = static_cast<size_t>( rows ) * cols;
	for ( size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		y[i] += b[i % cols];
	}
}

__global__ void k_scatter_add( float* dst, const float* src, const float* bias, const int* pos, int H ) {
	const int i = blockIdx.x;
	float* d = dst + static_cast<size_t>( pos[i] ) * H;
	const float* s = src + static_cast<size_t>( i ) * H;
	for ( int c = threadIdx.x; c < H; c += blockDim.x ) {
		d[c] += s[c] + ( bias ? bias[c] : 0.0f );
	}
}

__global__ void k_gather_out( const float* all, const int* pos, const uint8_t* mask, float* out, int width ) {
	const int i = blockIdx.x;
	const float m = mask ? ( mask[i] ? 1.0f : 0.0f ) : 1.0f;
	for ( int c = threadIdx.x; c < width; c += blockDim.x ) {
		out[static_cast<size_t>( i ) * width + c] = all[static_cast<size_t>( pos[i] ) * width + c] * m;
	}
}

// Per (token, head, q|k): RMSNorm over 128, optional rotary over the first 96 dims (pairs i, i+48); writes contiguous [U, heads*128].
__global__ void k_qk_norm_rope( const half* qkv, int qkv_stride, const float* qw, const float* kw, const float* cosb, const float* sinb, half* qo, half* ko,
	int heads, float eps ) {
	__shared__ float sm[8][128];
	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	const int row = blockIdx.x;
	const int task = blockIdx.y * 8 + warp; // 0 .. 2*heads-1
	if ( task >= 2 * heads ) { return; }
	const int which = task / heads, head = task % heads;
	const half* src = qkv + static_cast<size_t>( row ) * qkv_stride + which * heads * 128 + head * 128 + lane * 4;
	const float* w = which ? kw : qw;
	float v[4];
	float ss = 0.0f;
	#pragma unroll
	for ( int i = 0; i < 4; ++i ) {
		v[i] = __half2float( src[i] );
		ss += v[i] * v[i];
	}
	for ( int d = 16; d; d >>= 1 ) { ss += __shfl_xor_sync( 0xffffffffu, ss, d ); }
	const float inv = rsqrtf( ss / 128.0f + eps );
	#pragma unroll
	for ( int i = 0; i < 4; ++i ) { sm[warp][lane * 4 + i] = v[i] * inv * w[lane * 4 + i]; }
	__syncwarp();
	half* dst = ( which ? ko : qo ) + ( static_cast<size_t>( row ) * heads + head ) * 128 + lane * 4;
	#pragma unroll
	for ( int i = 0; i < 4; ++i ) {
		const int e = lane * 4 + i;
		float o = sm[warp][e];
		if ( cosb && e < 96 ) {
			const int f = e < 48 ? e : e - 48;
			const float c = cosb[static_cast<size_t>( row ) * 48 + f], s = sinb[static_cast<size_t>( row ) * 48 + f];
			o = e < 48 ? sm[warp][e] * c - sm[warp][e + 48] * s : sm[warp][e] * c + sm[warp][e - 48] * s;
		}
		dst[i] = __float2half_rn( o );
	}
}

// Bring-up attention: one block per (query, head); scores live in shared memory. Not for production shapes.
__global__ void k_attn_simple( const half* q, const half* k, const half* v, int vstride, half* o, int U, int heads ) {
	extern __shared__ float sc[];
	__shared__ float red[32];
	__shared__ float qs[128];
	const int i = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
	qs[t] = __half2float( q[( static_cast<size_t>( i ) * heads + h ) * 128 + t] );
	__syncthreads();
	const float scale = rsqrtf( 128.0f );
	float lmax = -1e30f;
	for ( int j = t; j < U; j += 128 ) {
		const uint4* kp = reinterpret_cast<const uint4*>( k + ( static_cast<size_t>( j ) * heads + h ) * 128 );
		float a = 0.0f;
		#pragma unroll
		for ( int c = 0; c < 16; ++c ) {
			const uint4 u = kp[c];
			const half2* hh = reinterpret_cast<const half2*>( &u );
			#pragma unroll
			for ( int e = 0; e < 4; ++e ) {
				const float2 f = __half22float2( hh[e] );
				a += f.x * qs[c * 8 + e * 2] + f.y * qs[c * 8 + e * 2 + 1];
			}
		}
		a *= scale;
		sc[j] = a;
		lmax = fmaxf( lmax, a );
	}
	lmax = block_max( lmax, red );
	float lsum = 0.0f;
	for ( int j = t; j < U; j += 128 ) {
		const float p = __expf( sc[j] - lmax );
		sc[j] = p;
		lsum += p;
	}
	lsum = block_sum( lsum, red );
	__syncthreads();
	float acc = 0.0f;
	for ( int j = 0; j < U; ++j ) {
		acc += sc[j] * __half2float( v[static_cast<size_t>( j ) * vstride + h * 128 + t] );
	}
	o[( static_cast<size_t>( i ) * heads + h ) * 128 + t] = __float2half_rn( acc / lsum );
}

// silu(gate) * up with a per-row power-of-two scale that keeps the FP16 result in range (restored after the down projection).
__global__ void k_silu_mul( const half* f1, half* f2, float* rowscale, int F ) {
	__shared__ float red[32];
	const int row = blockIdx.x;
	const half* g = f1 + static_cast<size_t>( row ) * 2 * F;
	const half* u = g + F;
	float mx = 0.0f;
	for ( int c = threadIdx.x; c < F; c += blockDim.x ) {
		const float a = __half2float( g[c] ), b = __half2float( u[c] );
		mx = fmaxf( mx, fabsf( a / ( 1.0f + __expf( -a ) ) * b ) );
	}
	mx = block_max( mx, red );
	int e = 0;
	if ( isfinite( mx ) ) { frexpf( mx, &e ); }
	const float s = ldexpf( 1.0f, max( e - 11, 0 ) );
	if ( threadIdx.x == 0 ) { rowscale[row] = s; }
	for ( int c = threadIdx.x; c < F; c += blockDim.x ) {
		const float a = __half2float( g[c] ), b = __half2float( u[c] );
		f2[static_cast<size_t>( row ) * F + c] = __float2half_rn( a / ( 1.0f + __expf( -a ) ) * b / s );
	}
}

// ---------------------------------------------------------------------------------------------------------------------
// weights
// ---------------------------------------------------------------------------------------------------------------------

struct Mat {
	const int8_t* i8 = nullptr;  // quantized ConvRot matrix [n, k] (or null)
	const float* scale = nullptr;
	const half* h = nullptr;     // dense FP16 matrix [n, k] (refiner)
	int n = 0, k = 0;
};

struct BlockW {
	Mat qkv, out, fc1, fc2;
	const float *n1 = nullptr, *n2 = nullptr, *qn = nullptr, *kn = nullptr;
	const float *aw = nullptr, *ab = nullptr; // adaln linear [J, 8], bias [J] (null in the refiner)
};

} // namespace

struct H3DiT::Impl {
	H3Config cfg;
	int device = 0;
	int max_rows = 0;
	pod::SafeTensorsFile st{};
	pod::MemoryArena warena{}, sarena{};
	cublasHandle_t blas = nullptr;
	cudaStream_t stream = nullptr;

	std::vector<BlockW> blocks, refiner;
	const float *vproj_w = nullptr, *vproj_b = nullptr, *aproj_w = nullptr, *aproj_b = nullptr, *cond_w = nullptr, *cond_b = nullptr;
	const float *ref_final = nullptr, *rope_inv = nullptr, *fin_norm = nullptr, *fin_aw = nullptr, *fin_ab = nullptr;
	const float *vout_w = nullptr, *vout_b = nullptr, *aout_w = nullptr, *aout_b = nullptr;
	std::vector<float> table; // [grid, dim] adaln interpolation table (host)

	// scratch
	float *x = nullptr, *br = nullptr, *tx = nullptr, *rowscale = nullptr, *ada = nullptr, *temb = nullptr, *cosb = nullptr, *sinb = nullptr, *vall = nullptr, *aall = nullptr;
	half *h16 = nullptr, *v16 = nullptr, *qkv16 = nullptr, *q16 = nullptr, *k16 = nullptr, *o16 = nullptr, *f1 = nullptr, *f2 = nullptr, *wscratch = nullptr;
	int* comb = nullptr;
	size_t wbytes = 0;
	bool use_fmha = false;

	template <typename T>
	T* alloc( pod::MemoryArena& a, size_t n ) { return static_cast<T*>( engine::core::arena_alloc( a, n * sizeof( T ) ) ); }

	const pod::SafeTensorMeta* find( const std::string& name ) {
		const pod::SafeTensorMeta* m = engine::loader::safetensors_find( st, name.c_str() );
		if ( !m ) { LOG_ERROR( "H3: missing tensor %s", name.c_str() ); }
		return m;
	}

	// Device copy of the raw tensor bytes
	void* upload_raw( const pod::SafeTensorMeta* m ) {
		void* d = engine::core::arena_alloc( warena, m->size_bytes );
		CUDA_CHECK( cudaMemcpy( d, engine::loader::safetensors_get_data_cpu( st, m ), m->size_bytes, cudaMemcpyHostToDevice ) );
		return d;
	}

	size_t numel( const pod::SafeTensorMeta* m ) {
		size_t n = 1;
		for ( int i = 0; i < m->ndim; ++i ) { n *= static_cast<size_t>( m->shape[i] ); }
		return n;
	}

	const float* upload_f32( const std::string& name ) {
		const pod::SafeTensorMeta* m = find( name );
		if ( !m ) { return nullptr; }
		const size_t n = numel( m );
		if ( m->dtype == pod::DType::FP32 ) { return static_cast<const float*>( upload_raw( m ) ); }
		void* tmp = nullptr;
		CUDA_CHECK( cudaMalloc( &tmp, m->size_bytes ) );
		CUDA_CHECK( cudaMemcpy( tmp, engine::loader::safetensors_get_data_cpu( st, m ), m->size_bytes, cudaMemcpyHostToDevice ) );
		float* d = alloc<float>( warena, n );
		if ( m->dtype == pod::DType::BF16 ) {
			k_convert<__nv_bfloat16, float><<<1024, 256>>>( static_cast<const __nv_bfloat16*>( tmp ), d, n );
		} else if ( m->dtype == pod::DType::FP16 ) {
			k_convert<half, float><<<1024, 256>>>( static_cast<const half*>( tmp ), d, n );
		} else {
			ENGINE_ASSERT( false, "H3: unsupported dtype for an FP32 upload" );
		}
		CUDA_CHECK( cudaDeviceSynchronize() );
		CUDA_CHECK( cudaFree( tmp ) );
		return d;
	}

	const half* upload_f16( const std::string& name, int* n_out = nullptr, int* k_out = nullptr ) {
		const pod::SafeTensorMeta* m = find( name );
		if ( !m ) { return nullptr; }
		const size_t n = numel( m );
		if ( n_out ) { *n_out = m->shape[0]; }
		if ( k_out ) { *k_out = m->shape[1]; }
		if ( m->dtype == pod::DType::FP16 ) { return static_cast<const half*>( upload_raw( m ) ); }
		ENGINE_ASSERT( m->dtype == pod::DType::BF16, "H3: dense matrices are BF16 or FP16" );
		void* tmp = nullptr;
		CUDA_CHECK( cudaMalloc( &tmp, m->size_bytes ) );
		CUDA_CHECK( cudaMemcpy( tmp, engine::loader::safetensors_get_data_cpu( st, m ), m->size_bytes, cudaMemcpyHostToDevice ) );
		half* d = alloc<half>( warena, n );
		k_convert<__nv_bfloat16, half><<<1024, 256>>>( static_cast<const __nv_bfloat16*>( tmp ), d, n );
		CUDA_CHECK( cudaDeviceSynchronize() );
		CUDA_CHECK( cudaFree( tmp ) );
		return d;
	}

	bool load_mat_i8( const std::string& base, Mat& m ) {
		const pod::SafeTensorMeta* w = find( base + ".weight" );
		const pod::SafeTensorMeta* s = find( base + ".weight_scale" );
		const pod::SafeTensorMeta* q = find( base + ".comfy_quant" );
		if ( !w || !s || !q ) { return false; }
		ENGINE_ASSERT( w->dtype == pod::DType::INT8 && w->ndim == 2, "H3: expected an INT8 matrix" );
		const char* blob = static_cast<const char*>( engine::loader::safetensors_get_data_cpu( st, q ) );
		const std::string meta( blob, q->size_bytes );
		ENGINE_ASSERT( meta.find( "\"convrot\": true" ) != std::string::npos && meta.find( "256" ) != std::string::npos, "H3: only 256-group ConvRot INT8 is supported" );
		m.i8 = static_cast<const int8_t*>( upload_raw( w ) );
		m.scale = upload_f32( base + ".weight_scale" );
		m.n = w->shape[0];
		m.k = w->shape[1];
		return true;
	}

	bool load_block( int i, bool refiner_block ) {
		BlockW b;
		const std::string p = refiner_block ? "token_refiner.blocks." + std::to_string( i ) : "blocks." + std::to_string( i );
		if ( refiner_block ) {
			b.qkv.h = upload_f16( p + ".attn.qkv_proj.weight", &b.qkv.n, &b.qkv.k );
			b.out.h = upload_f16( p + ".attn.out_proj.weight", &b.out.n, &b.out.k );
			b.fc1.h = upload_f16( p + ".mlp.fc1.weight", &b.fc1.n, &b.fc1.k );
			b.fc2.h = upload_f16( p + ".mlp.fc2.weight", &b.fc2.n, &b.fc2.k );
		} else {
			if ( !load_mat_i8( p + ".attn.qkv_proj", b.qkv ) || !load_mat_i8( p + ".attn.out_proj", b.out ) || !load_mat_i8( p + ".mlp.fc1", b.fc1 ) ||
				!load_mat_i8( p + ".mlp.fc2", b.fc2 ) ) {
				return false;
			}
			b.aw = upload_f32( p + ".adaln_proj.linear.weight" );
			b.ab = upload_f32( p + ".adaln_proj.linear.bias" );
		}
		b.n1 = upload_f32( p + ".norm1.weight" );
		b.n2 = upload_f32( p + ".norm2.weight" );
		b.qn = upload_f32( p + ".attn.q_norm.weight" );
		b.kn = upload_f32( p + ".attn.k_norm.weight" );
		( refiner_block ? refiner : blocks ).push_back( b );
		return true;
	}

	// -----------------------------------------------------------------------------------------------------------------
	// compute helpers
	// -----------------------------------------------------------------------------------------------------------------
	void gemm_h( const half* A, const half* W, void* C, int M, int N, int K, bool c32 ) {
		const float alpha = 1.0f, beta = 0.0f;
		CUBLAS_CHECK( cublasGemmEx( blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16F, K, A, CUDA_R_16F, K, &beta, C,
			c32 ? CUDA_R_32F : CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP ) );
	}

	void gemm_f( const float* A, const float* W, float* C, int M, int N, int K ) {
		const float alpha = 1.0f, beta = 0.0f;
		CUBLAS_CHECK( cublasSgemm( blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, K, A, K, &beta, C, N ) );
	}

	const half* matrix( const Mat& m ) {
		if ( m.h ) { return m.h; }
		dim3 grid( m.k / 256, m.n );
		k_dequant_convrot<<<grid, 256, 0, stream>>>( m.i8, m.scale, wscratch, m.k );
		return wscratch;
	}

	// One pre-norm transformer block over `rows` packed rows of the FP32 stream `xs`. ada != null selects the AdaLN variant
	// (6 chunks per modality row, rows indexed by comb) with rotary tables; otherwise the plain refiner block.
	void run_block( const BlockW& b, float* xs, int rows, const float* adab, const int* idx, bool rope ) {
		const int H = cfg.hidden, nh = cfg.heads, qd = nh * cfg.head_dim, F = cfg.ffn;
		const long st6 = 6L * H;
		const float *sh_a = nullptr, *sc_a = nullptr, *g_a = nullptr, *sh_m = nullptr, *sc_m = nullptr, *g_m = nullptr;
		if ( adab ) {
			sh_a = adab; sc_a = adab + H; g_a = adab + 2 * H; sh_m = adab + 3 * H; sc_m = adab + 4 * H; g_m = adab + 5 * H;
		}
		k_norm_mod<half><<<rows, 256, 0, stream>>>( xs, nullptr, nullptr, 0, b.n1, sh_a, sc_a, st6, idx, H, cfg.eps, h16 );
		gemm_h( h16, matrix( b.qkv ), qkv16, rows, 3 * qd, H, false );
		k_qk_norm_rope<<<dim3( rows, ( 2 * nh + 7 ) / 8 ), 256, 0, stream>>>( qkv16, 3 * qd, b.qn, b.kn, rope ? cosb : nullptr, rope ? sinb : nullptr, q16, k16, nh, cfg.qk_eps );
		if ( use_fmha ) {
			CUDA_CHECK( cudaMemcpy2DAsync( v16, qd * sizeof( half ), qkv16 + 2 * qd, 3 * qd * sizeof( half ), qd * sizeof( half ), rows, cudaMemcpyDeviceToDevice, stream ) );
			h3_flash_attention( q16, k16, v16, o16, rows, nh, 1.0f / std::sqrt( static_cast<float>( cfg.head_dim ) ), 0, rows >= 1024 ? 128 : 64, stream );
		} else {
			const size_t smem = static_cast<size_t>( rows + 8 ) * sizeof( float );
			k_attn_simple<<<dim3( rows, nh ), 128, smem, stream>>>( q16, k16, qkv16 + 2 * qd, 3 * qd, o16, rows, nh );
		}
		gemm_h( o16, matrix( b.out ), br, rows, H, qd, true );
		k_norm_mod<half><<<rows, 256, 0, stream>>>( xs, br, g_a, st6, b.n2, sh_m, sc_m, st6, idx, H, cfg.eps, h16 );
		gemm_h( h16, matrix( b.fc1 ), f1, rows, 2 * F, H, false );
		k_silu_mul<<<rows, 256, 0, stream>>>( f1, f2, rowscale, F );
		gemm_h( f2, matrix( b.fc2 ), br, rows, H, F, true );
		k_add_gated<<<rows, 256, 0, stream>>>( xs, br, rowscale, g_m, st6, idx, H );
	}
};

H3DiT::H3DiT() : p_( new Impl ) {}

H3DiT::~H3DiT() {
	if ( p_ ) {
		if ( p_->blas ) { cublasDestroy( p_->blas ); }
		if ( p_->warena.base_ptr ) { engine::core::arena_destroy( p_->warena ); }
		if ( p_->sarena.base_ptr ) { engine::core::arena_destroy( p_->sarena ); }
		engine::loader::safetensors_close( p_->st );
		delete p_;
	}
}

size_t H3DiT::weight_bytes() const { return p_->warena.offset_bytes; }

bool H3DiT::load( const char* checkpoint, int device, int max_rows, const H3Config& cfg ) {
	Impl& s = *p_;
	s.cfg = cfg;
	s.device = device;
	s.max_rows = max_rows;
	CUDA_CHECK( cudaSetDevice( device ) );
	CUDA_CHECK( cudaStreamCreateWithFlags( &s.stream, cudaStreamNonBlocking ) );
	CUBLAS_CHECK( cublasCreate( &s.blas ) );
	CUBLAS_CHECK( cublasSetStream( s.blas, s.stream ) );
	CUBLAS_CHECK( cublasSetMathMode( s.blas, CUBLAS_DEFAULT_MATH ) );
	if ( !engine::loader::safetensors_open( s.st, checkpoint ) ) { return false; }
	if ( !engine::core::arena_create( s.warena, device, s.st.file_size + ( 1ULL << 30 ) ) ) { return false; }

	const int H = cfg.hidden;
	for ( int i = 0; i < cfg.refiner_layers; ++i ) {
		if ( !s.load_block( i, true ) ) { return false; }
	}
	for ( int i = 0; i < cfg.layers; ++i ) {
		if ( !s.load_block( i, false ) ) { return false; }
	}
	s.vproj_w = s.upload_f32( "video_patch_proj.weight" );
	s.vproj_b = s.upload_f32( "video_patch_proj.bias" );
	s.aproj_w = s.upload_f32( "audio_patch_proj.weight" );
	s.aproj_b = s.upload_f32( "audio_patch_proj.bias" );
	s.cond_w = s.upload_f32( "condition_proj.weight" );
	s.cond_b = s.upload_f32( "condition_proj.bias" );
	s.ref_final = s.upload_f32( "token_refiner.final_norm.weight" );
	s.rope_inv = s.upload_f32( "rope.inv_freq" );
	s.fin_norm = s.upload_f32( "final_layer.norm.weight" );
	s.fin_aw = s.upload_f32( "final_layer.adaln_proj.linear.weight" );
	s.fin_ab = s.upload_f32( "final_layer.adaln_proj.linear.bias" );
	s.vout_w = s.upload_f32( "final_layer.video_out.weight" );
	s.vout_b = s.upload_f32( "final_layer.video_out.bias" );
	s.aout_w = s.upload_f32( "final_layer.audio_out.weight" );
	s.aout_b = s.upload_f32( "final_layer.audio_out.bias" );
	const pod::SafeTensorMeta* tm = s.find( "adaln_t_table" );
	if ( !tm || !s.cond_b || !s.aout_b ) { return false; }
	s.table.assign( static_cast<const float*>( engine::loader::safetensors_get_data_cpu( s.st, tm ) ), static_cast<const float*>( engine::loader::safetensors_get_data_cpu( s.st, tm ) ) + s.numel( tm ) );
	ENGINE_ASSERT( tm->shape[0] == cfg.adaln_grid && tm->shape[1] == cfg.adaln_dim, "H3: adaln table shape" );
	s.wbytes = s.warena.offset_bytes;
	s.use_fmha = h3_fmha_available() && !cfg.simple_attention;
	LOG_INFO( "H3: attention kernel %s", s.use_fmha ? "CUTLASS FMHA" : "simple (slow)" );
	CUDA_CHECK( cudaFuncSetAttribute( k_attn_simple, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304 - 1024 ) );

	// scratch sized for max_rows
	const size_t U = static_cast<size_t>( max_rows ), qd = static_cast<size_t>( cfg.heads ) * cfg.head_dim, F = cfg.ffn;
	size_t need = U * H * 4 * 3 + U * 4 * 2 + U * 3 * qd * 2 + U * qd * 2 * 4 + U * 2 * F * 2 + U * F * 2 + U * H * 2 + U * 48 * 4 * 2 + U * 128 * 4 + 64ull * 3 * 6 * H * 4 +
		static_cast<size_t>( 28672 ) * 5376 * 2 + ( 64ull << 20 );
	need += 4096ull * H * 4; // refiner stream (text rows)
	if ( !engine::core::arena_create( s.sarena, device, need ) ) { return false; }
	s.x = s.alloc<float>( s.sarena, U * H );
	s.br = s.alloc<float>( s.sarena, U * H );
	s.tx = s.alloc<float>( s.sarena, 4096ull * H );
	s.rowscale = s.alloc<float>( s.sarena, U );
	s.ada = s.alloc<float>( s.sarena, 16ull * kModalities * 6 * H );
	s.temb = s.alloc<float>( s.sarena, 16 * 8 );
	s.cosb = s.alloc<float>( s.sarena, U * 48 );
	s.sinb = s.alloc<float>( s.sarena, U * 48 );
	s.vall = s.alloc<float>( s.sarena, U * 96 );
	s.aall = s.alloc<float>( s.sarena, U * 32 );
	s.h16 = s.alloc<half>( s.sarena, U * H );
	s.v16 = s.alloc<half>( s.sarena, U * qd );
	s.qkv16 = s.alloc<half>( s.sarena, U * 3 * qd );
	s.q16 = s.alloc<half>( s.sarena, U * qd );
	s.k16 = s.alloc<half>( s.sarena, U * qd );
	s.o16 = s.alloc<half>( s.sarena, U * qd );
	s.f1 = s.alloc<half>( s.sarena, U * 2 * F );
	s.f2 = s.alloc<half>( s.sarena, U * F );
	s.wscratch = s.alloc<half>( s.sarena, 28672ull * 5376 );
	s.comb = s.alloc<int>( s.sarena, U );
	return true;
}

void H3DiT::forward( const H3Request& r, float* video_out, float* audio_out, void* stream_ptr ) {
	Impl& s = *p_;
	(void)stream_ptr;
	const H3Config& c = s.cfg;
	const int H = c.hidden, U = r.used;
	ENGINE_ASSERT( U <= s.max_rows && r.text_len <= 4096, "H3: request larger than the allocated scratch" );
	cudaStream_t st = s.stream;
	const int M = static_cast<int>( r.unique_t.size() );
	ENGINE_ASSERT( M >= 1 && M <= 16, "H3: unique timestep count" );

	// time coordinates (host lerp of the interpolation table), then their rows
	std::vector<float> tc( static_cast<size_t>( M ) * c.adaln_dim );
	for ( int m = 0; m < M; ++m ) {
		const float pos = std::fmin( std::fmax( r.unique_t[m], 0.0f ), 1.0f ) * static_cast<float>( c.adaln_grid - 1 );
		int lo = static_cast<int>( std::floor( pos ) );
		lo = std::min( lo, c.adaln_grid - 2 );
		const float fr = pos - static_cast<float>( lo );
		for ( int d = 0; d < c.adaln_dim; ++d ) {
			const float a = s.table[static_cast<size_t>( lo ) * c.adaln_dim + d], b = s.table[static_cast<size_t>( lo + 1 ) * c.adaln_dim + d];
			tc[static_cast<size_t>( m ) * c.adaln_dim + d] = a + fr * ( b - a );
		}
	}
	CUDA_CHECK( cudaMemcpyAsync( s.temb, tc.data(), tc.size() * sizeof( float ), cudaMemcpyHostToDevice, st ) );
	CUDA_CHECK( cudaStreamSynchronize( st ) );

	k_comb<<<( U + 255 ) / 256, 256, 0, st>>>( r.inverse, r.tags, s.comb, U );
	k_rope_table<<<( U * 48 + 255 ) / 256, 256, 0, st>>>( r.position_ids, s.rope_inv, s.cosb, s.sinb, U, c.rope_freqs );

	// embedding
	CUDA_CHECK( cudaMemsetAsync( s.x, 0, static_cast<size_t>( U ) * H * sizeof( float ), st ) );
	// text: condition projection (+bias) -> refiner -> scatter
	s.gemm_f( r.prompt, s.cond_w, s.tx, r.text_len, H, c.text_dim );
	k_add_bias<<<1024, 256, 0, st>>>( s.tx, s.cond_b, r.text_len, H );
	for ( const BlockW& b : s.refiner ) { s.run_block( b, s.tx, r.text_len, nullptr, nullptr, false ); }
	k_norm_mod<float><<<r.text_len, 256, 0, st>>>( s.tx, nullptr, nullptr, 0, s.ref_final, nullptr, nullptr, 0, nullptr, H, c.final_eps, s.br );
	k_scatter_add<<<r.text_len, 256, 0, st>>>( s.x, s.br, nullptr, r.text_pos, H );
	// video and audio rows
	s.gemm_f( r.video_rows, s.vproj_w, s.br, r.n_img, H, c.latents );
	k_scatter_add<<<r.n_img, 256, 0, st>>>( s.x, s.br, s.vproj_b, r.img_pos, H );
	s.gemm_f( r.audio_rows, s.aproj_w, s.br, r.n_audio, H, c.audio_latents );
	k_scatter_add<<<r.n_audio, 256, 0, st>>>( s.x, s.br, s.aproj_b, r.audio_pos, H );
	if ( probe ) { CUDA_CHECK( cudaStreamSynchronize( st ) ); probe( "embed", s.x, U ); }

	for ( int i = 0; i < c.layers; ++i ) {
		const BlockW& b = s.blocks[i];
		k_adaln<<<( 6 * kModalities * H + 255 ) / 256, 256, 0, st>>>( s.temb, b.aw, b.ab, s.ada, M, 6 * kModalities * H, c.adaln_dim );
		s.run_block( b, s.x, U, s.ada, s.comb, true );
		if ( probe ) {
			for ( int pb : probe_blocks ) {
				if ( pb == i ) {
					CUDA_CHECK( cudaStreamSynchronize( st ) );
					char tag[32];
					std::snprintf( tag, sizeof( tag ), "block%d", i );
					probe( tag, s.x, U );
				}
			}
		}
	}

	// final layer
	k_adaln<<<( 2 * H + 255 ) / 256, 256, 0, st>>>( s.temb, s.fin_aw, s.fin_ab, s.ada, M, 2 * H, c.adaln_dim );
	k_norm_mod<float><<<U, 256, 0, st>>>( s.x, nullptr, nullptr, 0, s.fin_norm, s.ada, s.ada + H, 2L * H, r.inverse, H, c.final_eps, s.br );
	if ( probe ) { CUDA_CHECK( cudaStreamSynchronize( st ) ); probe( "final_hidden", s.br, U ); }
	s.gemm_f( s.br, s.vout_w, s.vall, U, c.latents, H );
	k_add_bias<<<1024, 256, 0, st>>>( s.vall, s.vout_b, U, c.latents );
	s.gemm_f( s.br, s.aout_w, s.aall, U, c.audio_latents, H );
	k_add_bias<<<1024, 256, 0, st>>>( s.aall, s.aout_b, U, c.audio_latents );
	k_gather_out<<<r.n_img, 96, 0, st>>>( s.vall, r.img_pos, r.update_mask, video_out, c.latents );
	k_gather_out<<<r.n_audio, 32, 0, st>>>( s.aall, r.audio_pos, nullptr, audio_out, c.audio_latents );
	CUDA_CHECK( cudaStreamSynchronize( st ) );
	CUDA_CHECK( cudaGetLastError() );
}

} // namespace engine::video
