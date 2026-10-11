// MiniMax-H3 DiT forward (single GPU). See inc/engine/video/h3_dit.h for the numeric contract.
#include "engine/video/h3_dit.h"
#include "engine/core/arena.h"
#include "engine/core/log.h"
#include "engine/loader/safetensors.h"
#include "engine/video/h3_attn.h"
#include "h3_kernels.cuh"
#include "h3_util.h"
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

using namespace kern;

constexpr int kModalities = 3;

// ---------------------------------------------------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------------------------------------------------

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

	// Staged LoRA adapters (PEFT A/B pairs, kept apart from the INT8 base): parts of each linear, [layer * 4 + slot], layers 0..49 DiT, 50..51 refiner
	struct LoraPart {
		const half* a = nullptr;
		const half* b = nullptr;
		int offset = 0, width = 0, rank = 0;
		float alpha_scale = 0.0625f; // alpha / rank of this tensor pair
	};
	struct Adapter {
		std::string name;
		std::vector<std::vector<LoraPart>> parts;
		float alpha_over_rank = 0.0625f;
		float scale = 0.0f; // request multiplier; 0 = bypassed
		pod::MemoryArena arena{};
	};
	std::vector<Adapter*> adapters;
	half* lora_t = nullptr;
	// sequence parallelism
	int sp_rank = 0, sp_n = 1;
	std::function<void( int, void* )> gather;
	half *kfull = nullptr, *vfull = nullptr;
	int total_rows = 0, max_local = 0;

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

	// C[:, offset : offset + N] (+)= alpha * A @ W^T with ldc = full width; beta 1 accumulates into an existing result
	void gemm_acc( const half* A, const half* W, void* C, int M, int N, int K, int ldc, int offset, float alpha, bool c32 ) {
		const float beta = 1.0f;
		void* cp = c32 ? static_cast<void*>( static_cast<float*>( C ) + offset ) : static_cast<void*>( static_cast<half*>( C ) + offset );
		CUBLAS_CHECK( cublasGemmEx( blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16F, K, A, CUDA_R_16F, K, &beta, cp, c32 ? CUDA_R_32F : CUDA_R_16F, ldc,
			CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP ) );
	}

	// add every active adapter's delta of linear (layer, slot) to `out`; in: the FP16 input of the base GEMM [rows, K]
	void lora_apply( int layer, int slot, const half* in, void* out, int rows, int K, int ldc, bool c32 ) {
		for ( Adapter* ad : adapters ) {
			if ( ad->scale == 0.0f || layer < 0 ) { continue; }
			for ( const LoraPart& part : ad->parts[static_cast<size_t>( layer ) * 4 + slot] ) {
				gemm_h( in, part.a, lora_t, rows, part.rank, K, false );
				gemm_acc( lora_t, part.b, out, rows, part.width, part.rank, ldc, part.offset, ad->scale * part.alpha_scale, c32 );
			}
		}
	}

	const half* matrix( const Mat& m ) {
		if ( m.h ) { return m.h; }
		dim3 grid( m.k / 256, m.n );
		k_dequant_convrot<<<grid, 256, 0, stream>>>( m.i8, m.scale, wscratch, m.k );
		return wscratch;
	}

	// One pre-norm transformer block over `rows` packed rows of the FP32 stream `xs`. ada != null selects the AdaLN variant
	// (6 chunks per modality row, rows indexed by comb) with rotary tables; otherwise the plain refiner block.
	void run_block( const BlockW& b, float* xs, int rows, const float* adab, const int* idx, bool rope, int lora_layer, int row0 = 0 ) {
		const int H = cfg.hidden, nh = cfg.heads, qd = nh * cfg.head_dim, F = cfg.ffn;
		const long st6 = 6L * H;
		const float *sh_a = nullptr, *sc_a = nullptr, *g_a = nullptr, *sh_m = nullptr, *sc_m = nullptr, *g_m = nullptr;
		if ( adab ) {
			sh_a = adab; sc_a = adab + H; g_a = adab + 2 * H; sh_m = adab + 3 * H; sc_m = adab + 4 * H; g_m = adab + 5 * H;
		}
		k_norm_mod<half><<<rows, 256, 0, stream>>>( xs, nullptr, nullptr, 0, b.n1, sh_a, sc_a, st6, idx, H, cfg.eps, h16 );
		gemm_h( h16, matrix( b.qkv ), qkv16, rows, 3 * qd, H, false );
		lora_apply( lora_layer, 0, h16, qkv16, rows, H, 3 * qd, false );
		const bool sp = sp_n > 1 && rope;
		k_qk_norm_rope<<<dim3( rows, ( 2 * nh + 7 ) / 8 ), 256, 0, stream>>>( qkv16, 3 * qd, b.qn, b.kn, rope ? cosb + static_cast<size_t>( row0 ) * 48 : nullptr,
			rope ? sinb + static_cast<size_t>( row0 ) * 48 : nullptr, q16, sp ? kfull + static_cast<size_t>( row0 ) * qd : k16, nh, cfg.qk_eps );
		if ( sp ) {
			CUDA_CHECK( cudaMemcpy2DAsync( vfull + static_cast<size_t>( row0 ) * qd, qd * sizeof( half ), qkv16 + 2 * qd, 3 * qd * sizeof( half ), qd * sizeof( half ), rows, cudaMemcpyDeviceToDevice, stream ) );
			gather( total_rows, stream );
			h3_attention( q16, kfull, vfull, o16, rows, total_rows, nh, 1.0f / std::sqrt( static_cast<float>( cfg.head_dim ) ), false, stream );
		} else if ( use_fmha ) {
			CUDA_CHECK( cudaMemcpy2DAsync( v16, qd * sizeof( half ), qkv16 + 2 * qd, 3 * qd * sizeof( half ), qd * sizeof( half ), rows, cudaMemcpyDeviceToDevice, stream ) );
			h3_attention( q16, k16, v16, o16, rows, rows, nh, 1.0f / std::sqrt( static_cast<float>( cfg.head_dim ) ), false, stream );
		} else {
			const size_t smem = static_cast<size_t>( rows + 8 ) * sizeof( float );
			k_attn_simple<<<dim3( rows, nh ), 128, smem, stream>>>( q16, k16, qkv16 + 2 * qd, 3 * qd, o16, rows, nh );
		}
		gemm_h( o16, matrix( b.out ), br, rows, H, qd, true );
		lora_apply( lora_layer, 1, o16, br, rows, qd, H, true );
		k_norm_mod<half><<<rows, 256, 0, stream>>>( xs, br, g_a, st6, b.n2, sh_m, sc_m, st6, idx, H, cfg.eps, h16 );
		gemm_h( h16, matrix( b.fc1 ), f1, rows, 2 * F, H, false );
		lora_apply( lora_layer, 2, h16, f1, rows, H, 2 * F, false );
		k_silu_mul<<<rows, 256, 0, stream>>>( f1, f2, rowscale, F );
		gemm_h( f2, matrix( b.fc2 ), br, rows, H, F, true );
		lora_apply( lora_layer, 3, f2, br, rows, F, H, true );
		k_add_gated<<<rows, 256, 0, stream>>>( xs, br, rowscale, g_m, st6, idx, H );
	}
};

H3DiT::H3DiT() : p_( new Impl ) {}

H3DiT::~H3DiT() {
	if ( p_ ) {
		if ( p_->blas ) { cublasDestroy( p_->blas ); }
		if ( p_->warena.base_ptr ) { engine::core::arena_destroy( p_->warena ); }
		if ( p_->sarena.base_ptr ) { engine::core::arena_destroy( p_->sarena ); }
		for ( Impl::Adapter* a : p_->adapters ) {
			if ( a->arena.base_ptr ) { engine::core::arena_destroy( a->arena ); }
			delete a;
		}
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
	s.use_fmha = !cfg.simple_attention;
	LOG_INFO( "H3: attention kernel %s", s.use_fmha ? "H3 WMMA attention" : "simple (slow)" );
	CUDA_CHECK( cudaFuncSetAttribute( k_attn_simple, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304 - 1024 ) );

	CUDA_CHECK( cudaDeviceSynchronize() ); // loading ran on the default stream; the models run on non-blocking streams, which it does not order against
	return reserve();
}

// Activation scratch sized for max_rows (allocated by load; a resident transformer frees it after a job with trim() and takes it back with reserve())
bool H3DiT::reserve() {
	Impl& s = *p_;
	if ( s.sarena.base_ptr ) { return true; }
	CUDA_CHECK( cudaSetDevice( s.device ) );
	const int H = s.cfg.hidden;
	// scratch sized for max_rows
	const size_t U = static_cast<size_t>( s.max_rows ), qd = static_cast<size_t>( s.cfg.heads ) * s.cfg.head_dim, F = s.cfg.ffn;
	// rows owned by one rank (row-local activations); the residual stream, rope tables and the K/V all-gather buffers cover the whole sequence
	const size_t UL = s.cfg.sp_ranks > 1 ? std::max<size_t>( ( U + s.cfg.sp_ranks - 1 ) / s.cfg.sp_ranks, 8192 ) : U;
	size_t need = U * H * 4 + UL * H * 4 + U * 4 + UL * 4 + UL * 3 * qd * 2 + UL * qd * 2 * 4 + UL * 2 * F * 2 + UL * F * 2 + UL * H * 2 + U * 48 * 4 * 2 + UL * 128 * 4 + 64ull * 3 * 6 * H * 4 +
		static_cast<size_t>( 28672 ) * 5376 * 2 + U * qd * 2 * 2 + UL * 256 * 2 + ( 64ull << 20 );
	need += 16384ull * H * 4; // refiner stream (text rows)
	if ( !h3_scratch_arena_create( s.sarena, s.device, need ) ) { return false; }
	s.x = s.alloc<float>( s.sarena, U * H );
	s.br = s.alloc<float>( s.sarena, UL * H );
	s.tx = s.alloc<float>( s.sarena, 16384ull * H );
	s.rowscale = s.alloc<float>( s.sarena, UL );
	s.ada = s.alloc<float>( s.sarena, 16ull * kModalities * 6 * H );
	s.temb = s.alloc<float>( s.sarena, 16 * 8 );
	s.cosb = s.alloc<float>( s.sarena, U * 48 );
	s.sinb = s.alloc<float>( s.sarena, U * 48 );
	s.vall = s.alloc<float>( s.sarena, UL * 96 );
	s.aall = s.alloc<float>( s.sarena, UL * 32 );
	s.h16 = s.alloc<half>( s.sarena, UL * H );
	s.v16 = s.alloc<half>( s.sarena, UL * qd );
	s.qkv16 = s.alloc<half>( s.sarena, UL * 3 * qd );
	s.q16 = s.alloc<half>( s.sarena, UL * qd );
	s.k16 = s.alloc<half>( s.sarena, UL * qd );
	s.o16 = s.alloc<half>( s.sarena, UL * qd );
	s.f1 = s.alloc<half>( s.sarena, UL * 2 * F );
	s.f2 = s.alloc<half>( s.sarena, UL * F );
	s.wscratch = s.alloc<half>( s.sarena, 28672ull * 5376 );
	s.comb = s.alloc<int>( s.sarena, U );
	s.max_local = static_cast<int>( UL );
	s.lora_t = s.alloc<half>( s.sarena, UL * 256 );
	s.kfull = s.alloc<half>( s.sarena, U * qd );
	s.vfull = s.alloc<half>( s.sarena, U * qd );
	return true;
}

void H3DiT::trim() {
	Impl& s = *p_;
	if ( !s.sarena.base_ptr ) { return; }
	CUDA_CHECK( cudaSetDevice( s.device ) );
	CUDA_CHECK( cudaDeviceSynchronize() );
	engine::core::arena_destroy( s.sarena );
	s.sarena = pod::MemoryArena{};
}

int H3DiT::add_lora( const char* path, float scale ) {
	Impl& s = *p_;
	CUDA_CHECK( cudaSetDevice( s.device ) );
	pod::SafeTensorsFile f{};
	if ( !engine::loader::safetensors_open( f, path ) ) { return -1; }
	auto* ad = new Impl::Adapter();
	ad->name = path;
	ad->scale = scale;
	// alpha from the file's __metadata__ ("alpha":"8"); default 8
	float alpha = 8.0f;
	{
		FILE* fp = std::fopen( path, "rb" );
		uint64_t len = 0;
		if ( fp && std::fread( &len, 8, 1, fp ) == 1 && len < ( 64ull << 20 ) ) {
			std::string hdr( len, '\0' );
			if ( std::fread( hdr.data(), 1, len, fp ) == len ) {
				const size_t pos = hdr.find( "\"alpha\"" );
				if ( pos != std::string::npos ) {
					const size_t q = hdr.find( ':', pos );
					if ( q != std::string::npos ) { alpha = static_cast<float>( std::atof( hdr.c_str() + q + 2 ) ); }
				}
			}
		}
		if ( fp ) { std::fclose( fp ); }
	}
	size_t bytes = 0;
	for ( int i = 0; i < f.tensor_count; ++i ) { bytes += f.tensors[i].size_bytes; }
	if ( !engine::core::arena_create( ad->arena, s.device, bytes + ( 64ull << 20 ) ) ) { engine::loader::safetensors_close( f ); delete ad; return -1; }
	ad->parts.assign( static_cast<size_t>( s.cfg.layers + s.cfg.refiner_layers ) * 4, {} );
	int rank = 0;
	auto up = [&]( const std::string& name ) -> const half* {
		const pod::SafeTensorMeta* m = engine::loader::safetensors_find( f, name.c_str() );
		if ( !m ) { return nullptr; }
		const size_t n = s.numel( m );
		half* d = static_cast<half*>( engine::core::arena_alloc( ad->arena, n * sizeof( half ) ) );
		void* tmp = nullptr;
		CUDA_CHECK( cudaMalloc( &tmp, m->size_bytes ) );
		CUDA_CHECK( cudaMemcpy( tmp, engine::loader::safetensors_get_data_cpu( f, m ), m->size_bytes, cudaMemcpyHostToDevice ) );
		if ( m->dtype == pod::DType::BF16 ) {
			k_convert<__nv_bfloat16, half><<<1024, 256>>>( static_cast<const __nv_bfloat16*>( tmp ), d, n );
		} else if ( m->dtype == pod::DType::FP16 ) {
			CUDA_CHECK( cudaMemcpy( d, tmp, m->size_bytes, cudaMemcpyDeviceToDevice ) );
		} else if ( m->dtype == pod::DType::FP32 ) {
			k_convert<float, half><<<1024, 256>>>( static_cast<const float*>( tmp ), d, n );
		} else {
			d = nullptr;
		}
		CUDA_CHECK( cudaDeviceSynchronize() );
		CUDA_CHECK( cudaFree( tmp ) );
		return d;
	};
	// Three key layouts: diffusers (turbo: separate q/k/v, alpha in the file metadata), kohya (lora_unet_blocks_N_attn_qkv_proj.lora_down / lora_up / alpha) and
	// ai-toolkit (diffusion_model.blocks.N.attn.qkv_proj.lora_A / lora_B, alpha = rank); the last two address the fused native projections directly.
	struct Target { const char* suffix; int slot; int offset; };
	const Target dtargets[] = { { "attn.to_q", 0, 0 }, { "attn.to_k", 0, 7168 }, { "attn.to_v", 0, 14336 }, { "attn.to_out.0", 1, 0 }, { "ff.net.0.proj", 2, 0 }, { "ff.net.2", 3, 0 } };
	const char* native[4] = { "attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2" };
	auto scalar = [&]( const std::string& name, float def ) -> float {
		const pod::SafeTensorMeta* m = engine::loader::safetensors_find( f, name.c_str() );
		if ( !m || m->size_bytes == 0 ) { return def; }
		const void* d = engine::loader::safetensors_get_data_cpu( f, m );
		if ( m->dtype == pod::DType::FP32 ) { return *static_cast<const float*>( d ); }
		if ( m->dtype == pod::DType::FP16 ) { return __half2float( *static_cast<const half*>( d ) ); }
		if ( m->dtype == pod::DType::BF16 ) { const uint32_t bits = static_cast<uint32_t>( *static_cast<const uint16_t*>( d ) ) << 16; float v; std::memcpy( &v, &bits, 4 ); return v; }
		return def;
	};
	int found = 0;
	auto add_part = [&]( int L, int slot, const std::string& an, const std::string& bn, int offset, float alpha_scale_in, bool alpha_from_rank ) -> bool {
		const pod::SafeTensorMeta* am = engine::loader::safetensors_find( f, an.c_str() );
		const pod::SafeTensorMeta* bm = engine::loader::safetensors_find( f, bn.c_str() );
		if ( !am || !bm ) { return true; }
		Impl::LoraPart part;
		part.a = up( an );
		part.b = up( bn );
		part.rank = am->shape[0];
		part.width = bm->shape[0];
		part.offset = offset;
		part.alpha_scale = alpha_from_rank ? 1.0f : alpha_scale_in / static_cast<float>( part.rank );
		rank = std::max( rank, part.rank );
		if ( !part.a || !part.b || bm->shape[1] != am->shape[0] || am->shape[0] > 256 ) {
			LOG_ERROR( "H3: LoRA tensor pair %s / %s unusable (A %dx%d, B %dx%d)", an.c_str(), bn.c_str(), am->shape[0], am->shape[1], bm->shape[0], bm->shape[1] );
			return false;
		}
		ad->parts[static_cast<size_t>( L ) * 4 + slot].push_back( part );
		++found;
		return true;
	};
	for ( int L = 0; L < s.cfg.layers + s.cfg.refiner_layers; ++L ) {
		const bool refiner = L >= s.cfg.layers;
		const std::string dprefix = refiner ? "token_refiner.refiner_blocks." + std::to_string( L - s.cfg.layers ) + "." : "transformer_blocks." + std::to_string( L ) + ".";
		const std::string nprefix = refiner ? "token_refiner.blocks." + std::to_string( L - s.cfg.layers ) + "." : "blocks." + std::to_string( L ) + ".";
		for ( const Target& t : dtargets ) {
			if ( !add_part( L, t.slot, dprefix + t.suffix + ".lora_A.default.weight", dprefix + t.suffix + ".lora_B.default.weight", t.offset, alpha, false ) ) { engine::loader::safetensors_close( f ); engine::core::arena_destroy( ad->arena ); delete ad; return -1; }
		}
		for ( int slot = 0; slot < 4; ++slot ) {
			std::string kohya = std::string( "lora_unet_" ) + nprefix + native[slot];
			for ( char& ch : kohya ) { if ( ch == '.' ) { ch = '_'; } }
			const float ka = scalar( kohya + ".alpha", -1.0f );
			if ( !add_part( L, slot, kohya + ".lora_down.weight", kohya + ".lora_up.weight", 0, ka > 0 ? ka : 0.0f, ka <= 0 ) ||
				!add_part( L, slot, "diffusion_model." + nprefix + native[slot] + ".lora_A.weight", "diffusion_model." + nprefix + native[slot] + ".lora_B.weight", 0, 0.0f, true ) ) {
				engine::loader::safetensors_close( f ); engine::core::arena_destroy( ad->arena ); delete ad; return -1;
			}
		}
	}
	engine::loader::safetensors_close( f );
	if ( found == 0 || rank <= 0 || rank > 256 ) {
		LOG_ERROR( "H3: LoRA %s: no usable targets (found %d, rank %d)", path, found, rank ); engine::core::arena_destroy( ad->arena ); delete ad; return -1; }
	s.adapters.push_back( ad );
	LOG_INFO( "H3: LoRA %s: %d targets, rank %d, alpha %g, scale %g", path, found, rank, alpha, scale );
	CUDA_CHECK( cudaDeviceSynchronize() ); // staging ran on the default stream; the forward runs on a non-blocking one
	return static_cast<int>( s.adapters.size() ) - 1;
}

void H3DiT::clear_loras() {
	Impl& s = *p_;
	CUDA_CHECK( cudaSetDevice( s.device ) );
	CUDA_CHECK( cudaDeviceSynchronize() );
	for ( Impl::Adapter* a : s.adapters ) {
		if ( a->arena.base_ptr ) { engine::core::arena_destroy( a->arena ); }
		delete a;
	}
	s.adapters.clear();
}

void H3DiT::set_lora_scale( int id, float scale ) {
	if ( id >= 0 && id < static_cast<int>( p_->adapters.size() ) ) { p_->adapters[static_cast<size_t>( id )]->scale = scale; }
}

void H3DiT::set_sp( int rank, int n, std::function<void( int, void* )> gather ) {
	p_->sp_rank = rank;
	p_->sp_n = n;
	p_->gather = std::move( gather );
}

half_ptr_pair H3DiT::kv_buffers() const { return { p_->kfull, p_->vfull }; }
int H3DiT::device() const { return p_->device; }

void H3DiT::forward_sp( const H3Request& r, float* host_vall, float* host_aall ) { run_forward( r, nullptr, nullptr, host_vall, host_aall ); }

void H3DiT::forward( const H3Request& r, float* video_out, float* audio_out, void* stream_ptr ) {
	( void )stream_ptr;
	run_forward( r, video_out, audio_out, nullptr, nullptr );
}

void H3DiT::run_forward( const H3Request& r, float* video_out, float* audio_out, float* host_vall, float* host_aall ) {
	Impl& s = *p_;
	CUDA_CHECK( cudaSetDevice( s.device ) );
	const H3Config& c = s.cfg;
	const int H = c.hidden, U = r.used;
	ENGINE_ASSERT( U <= s.max_rows && r.text_len <= s.max_local && r.text_len <= 16384, "H3: request larger than the allocated scratch" );
	s.total_rows = U;
	// this rank's rows (sequence parallel); one rank owns everything
	const int chunk = ( U + s.sp_n - 1 ) / s.sp_n;
	const int row0 = std::min( U, s.sp_rank * chunk ), row1 = std::min( U, row0 + chunk ), lrows = row1 - row0;
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
	for ( size_t ri = 0; ri < s.refiner.size(); ++ri ) { s.run_block( s.refiner[ri], s.tx, r.text_len, nullptr, nullptr, false, s.cfg.layers + static_cast<int>( ri ) ); }
	k_norm_mod<float><<<r.text_len, 256, 0, st>>>( s.tx, nullptr, nullptr, 0, s.ref_final, nullptr, nullptr, 0, nullptr, H, c.final_eps, s.br );
	k_scatter_add<<<r.text_len, 256, 0, st>>>( s.x, s.br, nullptr, r.text_pos, H );
	// video and audio rows
	for ( int off = 0; off < r.n_img; off += s.max_local ) {
		const int n = std::min( s.max_local, r.n_img - off );
		s.gemm_f( r.video_rows + static_cast<size_t>( off ) * c.latents, s.vproj_w, s.br, n, H, c.latents );
		k_scatter_add<<<n, 256, 0, st>>>( s.x, s.br, s.vproj_b, r.img_pos + off, H );
	}
	for ( int off = 0; off < r.n_audio; off += s.max_local ) {
		const int n = std::min( s.max_local, r.n_audio - off );
		s.gemm_f( r.audio_rows + static_cast<size_t>( off ) * c.audio_latents, s.aproj_w, s.br, n, H, c.audio_latents );
		k_scatter_add<<<n, 256, 0, st>>>( s.x, s.br, s.aproj_b, r.audio_pos + off, H );
	}
	if ( probe ) { CUDA_CHECK( cudaStreamSynchronize( st ) ); probe( "embed", s.x, U ); }

	for ( int i = 0; i < c.layers; ++i ) {
		const BlockW& b = s.blocks[i];
		k_adaln<<<( 6 * kModalities * H + 255 ) / 256, 256, 0, st>>>( s.temb, b.aw, b.ab, s.ada, M, 6 * kModalities * H, c.adaln_dim );
		if ( lrows > 0 ) { s.run_block( b, s.x + static_cast<size_t>( row0 ) * H, lrows, s.ada, s.comb + row0, true, i, row0 ); }
		else if ( s.sp_n > 1 ) { s.gather( U, st ); }
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
	if ( lrows > 0 ) { k_norm_mod<float><<<lrows, 256, 0, st>>>( s.x + static_cast<size_t>( row0 ) * H, nullptr, nullptr, 0, s.fin_norm, s.ada, s.ada + H, 2L * H, r.inverse + row0, H, c.final_eps, s.br ); }
	if ( probe ) { CUDA_CHECK( cudaStreamSynchronize( st ) ); probe( "final_hidden", s.br, U ); }
	if ( lrows > 0 ) {
		s.gemm_f( s.br, s.vout_w, s.vall, lrows, c.latents, H );
		k_add_bias<<<1024, 256, 0, st>>>( s.vall, s.vout_b, lrows, c.latents );
		s.gemm_f( s.br, s.aout_w, s.aall, lrows, c.audio_latents, H );
		k_add_bias<<<1024, 256, 0, st>>>( s.aall, s.aout_b, lrows, c.audio_latents );
	}
	if ( s.sp_n == 1 ) {
		k_gather_out<<<r.n_img, 96, 0, st>>>( s.vall, r.img_pos, r.update_mask, video_out, c.latents );
		k_gather_out<<<r.n_audio, 32, 0, st>>>( s.aall, r.audio_pos, nullptr, audio_out, c.audio_latents );
	} else if ( lrows > 0 ) {
		CUDA_CHECK( cudaMemcpyAsync( host_vall + static_cast<size_t>( row0 ) * c.latents, s.vall, static_cast<size_t>( lrows ) * c.latents * sizeof( float ), cudaMemcpyDeviceToHost, st ) );
		CUDA_CHECK( cudaMemcpyAsync( host_aall + static_cast<size_t>( row0 ) * c.audio_latents, s.aall, static_cast<size_t>( lrows ) * c.audio_latents * sizeof( float ), cudaMemcpyDeviceToHost, st ) );
	}
	CUDA_CHECK( cudaStreamSynchronize( st ) );
	CUDA_CHECK( cudaGetLastError() );
}

} // namespace engine::video
