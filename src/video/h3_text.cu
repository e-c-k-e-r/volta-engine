// Streaming Qwen3 text tower for the H3 conditioning (see h3_text.h).
#include "engine/video/h3_text.h"
#include "engine/video/h3_fmha.h"
#include "h3_util.h"
#include "engine/core/arena.h"
#include "engine/core/log.h"
#include "engine/loader/safetensors.h"
#include "h3_kernels.cuh"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>
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

constexpr int kH = 5120, kNh = 64, kNkv = 8, kHd = 128, kFfn = 25600;
constexpr float kEps = 1e-6f;
constexpr float kTheta = 5.0e6f;

// per (token, head) RMSNorm with a 128-weight, then neox rotary over all 128 dims (pairs i, i+64); q heads then k heads
__global__ void k_qwen_qk( half* q, half* k, const float* qw, const float* kw, const int* pos, int S ) {
	__shared__ float sm[8][128];
	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	const int s = blockIdx.x, task = blockIdx.y * 8 + warp;
	if ( task >= kNh + kNkv ) { return; }
	half* p = task < kNh ? q + ( static_cast<size_t>( s ) * kNh + task ) * kHd : k + ( static_cast<size_t>( s ) * kNkv + ( task - kNh ) ) * kHd;
	const float* w = task < kNh ? qw : kw;
	float v[4];
	float ss = 0.0f;
	#pragma unroll
	for ( int i = 0; i < 4; ++i ) {
		v[i] = __half2float( p[lane * 4 + i] );
		ss += v[i] * v[i];
	}
	for ( int d = 16; d; d >>= 1 ) { ss += __shfl_xor_sync( 0xffffffffu, ss, d ); }
	const float inv = rsqrtf( ss / kHd + kEps );
	#pragma unroll
	for ( int i = 0; i < 4; ++i ) { sm[warp][lane * 4 + i] = v[i] * inv * w[lane * 4 + i]; }
	__syncwarp();
	#pragma unroll
	for ( int i = 0; i < 4; ++i ) {
		const int e = lane * 4 + i, f = e & 63;
		// interleaved mRoPE: frequency f takes the height position when f % 3 == 1 and f < 60, the width position when f % 3 == 2 and f < 60, else time
		const int axis = f < 60 && f % 3 == 1 ? 1 : ( f < 60 && f % 3 == 2 ? 2 : 0 );
		const float ang = static_cast<float>( pos[axis * S + s] ) * powf( kTheta, -static_cast<float>( f ) / 64.0f );
		const float c = cosf( ang ), sn = sinf( ang );
		const float o = e < 64 ? sm[warp][e] * c - sm[warp][e + 64] * sn : sm[warp][e] * c + sm[warp][e - 64] * sn;
		p[e] = __float2half_rn( o );
	}
}

// causal GQA attention, one block per (query, head); fine for short prompts
// GQA: [S, kNkv, kHd] K and V -> [S, kNh, kHd] with every kv head repeated for its group of query heads (the FMHA kernel has equal head counts)
__global__ void k_expand_kv( const half* k, const half* v, half* kx, half* vx, int S ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>( S ) * kNh * kHd;
	if ( i >= total ) { return; }
	const int d = static_cast<int>( i % kHd );
	const size_t sh = i / kHd;
	const int h = static_cast<int>( sh % kNh );
	const size_t s = sh / kNh;
	const size_t src = ( s * kNkv + h / ( kNh / kNkv ) ) * kHd + d;
	kx[i] = k[src];
	vx[i] = v[src];
}

__global__ void k_attn_causal( const half* q, const half* k, const half* v, half* o ) {
	extern __shared__ float sc[];
	__shared__ float red[32];
	__shared__ float qs[kHd];
	const int i = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
	const int kvh = h / ( kNh / kNkv );
	qs[t] = __half2float( q[( static_cast<size_t>( i ) * kNh + h ) * kHd + t] );
	__syncthreads();
	const float scale = rsqrtf( static_cast<float>( kHd ) );
	float lmax = -1e30f;
	for ( int j = t; j <= i; j += 128 ) {
		const uint4* kp = reinterpret_cast<const uint4*>( k + ( static_cast<size_t>( j ) * kNkv + kvh ) * kHd );
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
	for ( int j = t; j <= i; j += 128 ) {
		const float p = __expf( sc[j] - lmax );
		sc[j] = p;
		lsum += p;
	}
	lsum = block_sum( lsum, red );
	__syncthreads();
	float acc = 0.0f;
	for ( int j = 0; j <= i; ++j ) { acc += sc[j] * __half2float( v[( static_cast<size_t>( j ) * kNkv + kvh ) * kHd + t] ); }
	o[( static_cast<size_t>( i ) * kNh + h ) * kHd + t] = __float2half_rn( acc / lsum );
}

// silu(gate) * up from separate [S, F] gate / up tensors, with the per-row power-of-two scale
__global__ void k_silu_mul2( const half* g, const half* u, half* out, float* rowscale, int F ) {
	__shared__ float red[32];
	const int row = blockIdx.x;
	const half* gr = g + static_cast<size_t>( row ) * F;
	const half* ur = u + static_cast<size_t>( row ) * F;
	float mx = 0.0f;
	for ( int c = threadIdx.x; c < F; c += blockDim.x ) {
		const float a = __half2float( gr[c] ), b = __half2float( ur[c] );
		mx = fmaxf( mx, fabsf( a / ( 1.0f + __expf( -a ) ) * b ) );
	}
	mx = block_max( mx, red );
	int e = 0;
	if ( isfinite( mx ) ) { frexpf( mx, &e ); }
	const float s = ldexpf( 1.0f, max( e - 11, 0 ) );
	if ( threadIdx.x == 0 ) { rowscale[row] = s; }
	for ( int c = threadIdx.x; c < F; c += blockDim.x ) {
		const float a = __half2float( gr[c] ), b = __half2float( ur[c] );
		out[static_cast<size_t>( row ) * F + c] = __float2half_rn( a / ( 1.0f + __expf( -a ) ) * b / s );
	}
}

// x[rows[i]] += feat[i] (DeepStack injection)
static __global__ void k_add_rows( float* x, const float* feat, const int* rows, int H ) {
	const int i = blockIdx.x;
	for ( int c = threadIdx.x; c < H; c += blockDim.x ) { x[static_cast<size_t>( rows[i] ) * H + c] += feat[static_cast<size_t>( i ) * H + c]; }
}

struct LayerW {
	half *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr, *gate = nullptr, *up = nullptr, *down = nullptr;
	float *in_norm = nullptr, *post_norm = nullptr, *qn = nullptr, *kn = nullptr;
};

} // namespace

struct H3TextEncoder::Impl {
	engine::loader::SafeTensorsRepo repo;
	int device = 0, max_tokens = 0, layers = 0;
	pod::MemoryArena slot_arena[2]{}, sarena{};
	LayerW slot[2];
	void* staging = nullptr;
	size_t staging_bytes = 0;
	cublasHandle_t blas = nullptr;
	cudaStream_t cs = nullptr, ps = nullptr; // compute, prefetch
	cudaEvent_t loaded[2], consumed[2];
	static constexpr size_t kChunk = size_t( 16 ) << 20;
	static constexpr int kPinned = 4;
	void* pinned[kPinned] = {};  // pinned host chunks (the page-cache copy of a tensor is staged through them)
	cudaEvent_t chunk_free[kPinned];
	float *x = nullptr, *br = nullptr, *rowscale = nullptr;
	int* d_pos = nullptr; // [3, max_tokens]
	half *h16 = nullptr, *q16 = nullptr, *k16 = nullptr, *v16 = nullptr, *o16 = nullptr, *g16 = nullptr, *u16 = nullptr, *f16 = nullptr;
	half *kx16 = nullptr, *vx16 = nullptr; // K / V repeated to all query heads for the FMHA kernel
	bool use_fmha = false;
	std::string embed_name = "model.language_model.embed_tokens.weight";

	template <typename T>
	T* alloc( pod::MemoryArena& a, size_t n ) { return static_cast<T*>( engine::core::arena_alloc( a, n * sizeof( T ) ) ); }

	// Copies `bytes` of (page cache) host memory to the device staging buffer through pinned chunks: the CPU copy of one chunk overlaps the DMA of the previous ones.
	void stage_bytes( const void* src, size_t bytes ) {
		const char* p = static_cast<const char*>( src );
		for ( size_t off = 0, k = 0; off < bytes; off += kChunk, ++k ) {
			const int slot = static_cast<int>( k % kPinned );
			const size_t n = std::min( kChunk, bytes - off );
			CUDA_CHECK( cudaEventSynchronize( chunk_free[slot] ) ); // the DMA that last read this pinned chunk is done
			{ // the page-cache copy (page faults included) runs on several threads: one thread reaches only ~5 GB/s
				constexpr int kCopyThreads = 4;
				const size_t part = ( n + kCopyThreads - 1 ) / kCopyThreads;
				std::thread helpers[kCopyThreads - 1];
				for ( int t = 1; t < kCopyThreads; ++t ) {
					const size_t b = std::min( n, part * t ), e = std::min( n, part * ( t + 1 ) );
					helpers[t - 1] = std::thread( [this, slot, p, off, b, e] { std::memcpy( static_cast<char*>( pinned[slot] ) + b, p + off + b, e - b ); } );
				}
				std::memcpy( pinned[slot], p + off, std::min( n, part ) );
				for ( std::thread& h : helpers ) { h.join(); }
			}
			CUDA_CHECK( cudaMemcpyAsync( static_cast<char*>( staging ) + off, pinned[slot], n, cudaMemcpyHostToDevice, ps ) );
			CUDA_CHECK( cudaEventRecord( chunk_free[slot], ps ) );
		}
	}

	// BF16 tensor -> device FP16 matrix through the staging buffer (prefetch stream)
	half* upload_mat( int s, const std::string& name ) {
		const pod::SafeTensorMeta* m = repo.find( name.c_str() );
		ENGINE_ASSERT( m && m->dtype == pod::DType::BF16, "H3 text: missing or non-BF16 matrix" );
		const size_t n = m->size_bytes / 2;
		half* d = alloc<half>( slot_arena[s], n );
		stage_bytes( repo.cpu_data( name.c_str() ), m->size_bytes );
		k_convert<__nv_bfloat16, half><<<2048, 256, 0, ps>>>( static_cast<const __nv_bfloat16*>( staging ), d, n );
		CUDA_CHECK( cudaGetLastError() );
		CUDA_CHECK( cudaStreamSynchronize( ps ) ); // the staging buffer is reused by the next tensor
		return d;
	}

	float* upload_vec( int s, const std::string& name ) {
		const pod::SafeTensorMeta* m = repo.find( name.c_str() );
		ENGINE_ASSERT( m && m->dtype == pod::DType::BF16, "H3 text: missing or non-BF16 vector" );
		const size_t n = m->size_bytes / 2;
		float* d = alloc<float>( slot_arena[s], n );
		stage_bytes( repo.cpu_data( name.c_str() ), m->size_bytes );
		k_convert<__nv_bfloat16, float><<<64, 256, 0, ps>>>( static_cast<const __nv_bfloat16*>( staging ), d, n );
		CUDA_CHECK( cudaStreamSynchronize( ps ) );
		return d;
	}

	void load_layer( int s, int i ) {
		engine::core::arena_reset( slot_arena[s] );
		const std::string p = "model.language_model.layers." + std::to_string( i ) + ".";
		LayerW& w = slot[s];
		w.q = upload_mat( s, p + "self_attn.q_proj.weight" );
		w.k = upload_mat( s, p + "self_attn.k_proj.weight" );
		w.v = upload_mat( s, p + "self_attn.v_proj.weight" );
		w.o = upload_mat( s, p + "self_attn.o_proj.weight" );
		w.gate = upload_mat( s, p + "mlp.gate_proj.weight" );
		w.up = upload_mat( s, p + "mlp.up_proj.weight" );
		w.down = upload_mat( s, p + "mlp.down_proj.weight" );
		w.in_norm = upload_vec( s, p + "input_layernorm.weight" );
		w.post_norm = upload_vec( s, p + "post_attention_layernorm.weight" );
		w.qn = upload_vec( s, p + "self_attn.q_norm.weight" );
		w.kn = upload_vec( s, p + "self_attn.k_norm.weight" );
	}

	void gemm( const half* A, const half* W, void* C, int M, int N, int K, bool c32 ) {
		const float alpha = 1.0f, beta = 0.0f;
		CUBLAS_CHECK( cublasGemmEx( blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16F, K, A, CUDA_R_16F, K, &beta, C, c32 ? CUDA_R_32F : CUDA_R_16F, N,
			CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP ) );
	}

	void run_layer( const LayerW& w, int S ) {
		cudaStream_t s = cs;
		k_norm_mod<half><<<S, 256, 0, s>>>( x, nullptr, nullptr, 0, w.in_norm, nullptr, nullptr, 0, nullptr, kH, kEps, h16 );
		gemm( h16, w.q, q16, S, kNh * kHd, kH, false );
		gemm( h16, w.k, k16, S, kNkv * kHd, kH, false );
		gemm( h16, w.v, v16, S, kNkv * kHd, kH, false );
		k_qwen_qk<<<dim3( S, ( kNh + kNkv + 7 ) / 8 ), 256, 0, s>>>( q16, k16, w.qn, w.kn, d_pos, S );
		if ( use_fmha ) {
			k_expand_kv<<<( static_cast<size_t>( S ) * kNh * kHd + 255 ) / 256, 256, 0, s>>>( k16, v16, kx16, vx16, S );
			h3_flash_attention( q16, kx16, vx16, o16, S, kNh, 0.08838834764831845f, 0, S >= 1024 ? 128 : 64, s, true );
		} else {
			k_attn_causal<<<dim3( S, kNh ), 128, static_cast<size_t>( S + 8 ) * sizeof( float ), s>>>( q16, k16, v16, o16 );
		}
		gemm( o16, w.o, br, S, kH, kNh * kHd, true );
		k_add_gated<<<S, 256, 0, s>>>( x, br, nullptr, nullptr, 0, nullptr, kH );
		k_norm_mod<half><<<S, 256, 0, s>>>( x, nullptr, nullptr, 0, w.post_norm, nullptr, nullptr, 0, nullptr, kH, kEps, h16 );
		gemm( h16, w.gate, g16, S, kFfn, kH, false );
		gemm( h16, w.up, u16, S, kFfn, kH, false );
		k_silu_mul2<<<S, 256, 0, s>>>( g16, u16, f16, rowscale, kFfn );
		gemm( f16, w.down, br, S, kH, kFfn, true );
		k_add_gated<<<S, 256, 0, s>>>( x, br, rowscale, nullptr, 0, nullptr, kH );
	}
};

H3TextEncoder::H3TextEncoder() : p_( new Impl ) {}

H3TextEncoder::~H3TextEncoder() {
	if ( p_ ) {
		if ( p_->blas ) { cublasDestroy( p_->blas ); }
		for ( int i = 0; i < 2; ++i ) {
			if ( p_->slot_arena[i].base_ptr ) { engine::core::arena_destroy( p_->slot_arena[i] ); }
		}
		if ( p_->sarena.base_ptr ) { engine::core::arena_destroy( p_->sarena ); }
		for ( int i = 0; i < Impl::kPinned; ++i ) { if ( p_->pinned[i] ) { cudaFreeHost( p_->pinned[i] ); } }
		if ( p_->staging ) { cudaFree( p_->staging ); }
		delete p_;
	}
}

bool H3TextEncoder::load( const char* dir, int device, int max_tokens, int layers ) {
	Impl& s = *p_;
	s.device = device;
	s.max_tokens = max_tokens;
	s.layers = layers;
	CUDA_CHECK( cudaSetDevice( device ) );
	CUDA_CHECK( cudaStreamCreateWithFlags( &s.cs, cudaStreamNonBlocking ) );
	CUDA_CHECK( cudaStreamCreateWithFlags( &s.ps, cudaStreamNonBlocking ) );
	for ( int i = 0; i < 2; ++i ) {
		CUDA_CHECK( cudaEventCreateWithFlags( &s.loaded[i], cudaEventDisableTiming ) );
		CUDA_CHECK( cudaEventCreateWithFlags( &s.consumed[i], cudaEventDisableTiming ) );
	}
	CUBLAS_CHECK( cublasCreate( &s.blas ) );
	CUBLAS_CHECK( cublasSetStream( s.blas, s.cs ) );
	CUBLAS_CHECK( cublasSetMathMode( s.blas, CUBLAS_DEFAULT_MATH ) );
	if ( !s.repo.open( dir ) ) { return false; }
	const size_t layer_bytes = ( static_cast<size_t>( kNh * kHd ) * kH * 2 + static_cast<size_t>( kNkv * kHd ) * kH * 2 + static_cast<size_t>( kFfn ) * kH * 3 ) * sizeof( half ) + ( 4ULL << 20 );
	for ( int i = 0; i < 2; ++i ) {
		if ( !h3_scratch_arena_create( s.slot_arena[i], device, layer_bytes ) ) { return false; }
	}
	s.staging_bytes = static_cast<size_t>( kFfn ) * kH * 2;
	CUDA_CHECK( cudaMalloc( &s.staging, s.staging_bytes ) );
	for ( int i = 0; i < Impl::kPinned; ++i ) {
		CUDA_CHECK( cudaHostAlloc( &s.pinned[i], Impl::kChunk, cudaHostAllocDefault ) );
		CUDA_CHECK( cudaEventCreateWithFlags( &s.chunk_free[i], cudaEventDisableTiming ) );
		CUDA_CHECK( cudaEventRecord( s.chunk_free[i], s.ps ) ); // recorded, so the first wait returns at once
	}
	CUDA_CHECK( cudaMemset( s.staging, 0, s.staging_bytes ) );
	CUDA_CHECK( cudaStreamSynchronize( nullptr ) );
	const size_t S = static_cast<size_t>( max_tokens );
	const size_t need = S * kH * 4 * 2 + S * 4 * 4 + S * kH * 2 + S * kNh * kHd * 2 * 2 + S * kNkv * kHd * 2 * 2 + S * kNh * kHd * 2 * 2 + S * kFfn * 2 * 3 + ( 16ULL << 20 );
	if ( !h3_scratch_arena_create( s.sarena, device, need ) ) { return false; }
	s.x = s.alloc<float>( s.sarena, S * kH );
	s.br = s.alloc<float>( s.sarena, S * kH );
	s.rowscale = s.alloc<float>( s.sarena, S );
	s.d_pos = s.alloc<int>( s.sarena, 3 * S );
	s.h16 = s.alloc<half>( s.sarena, S * kH );
	s.q16 = s.alloc<half>( s.sarena, S * kNh * kHd );
	s.o16 = s.alloc<half>( s.sarena, S * kNh * kHd );
	s.k16 = s.alloc<half>( s.sarena, S * kNkv * kHd );
	s.v16 = s.alloc<half>( s.sarena, S * kNkv * kHd );
	s.kx16 = s.alloc<half>( s.sarena, S * kNh * kHd );
	s.vx16 = s.alloc<half>( s.sarena, S * kNh * kHd );
	s.use_fmha = h3_fmha_available();
	s.g16 = s.alloc<half>( s.sarena, S * kFfn );
	s.u16 = s.alloc<half>( s.sarena, S * kFfn );
	s.f16 = s.alloc<half>( s.sarena, S * kFfn );
	CUDA_CHECK( cudaFuncSetAttribute( k_attn_causal, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304 - 1024 ) );
	CUDA_CHECK( cudaDeviceSynchronize() ); // loading ran on the default stream; the models run on non-blocking streams, which it does not order against
	return s.repo.find( s.embed_name.c_str() ) != nullptr;
}

bool H3TextEncoder::encode( const std::vector<int>& ids, float* d_out ) { return encode_mm( ids, -1, {}, d_out ); }

bool H3TextEncoder::encode_mm( const std::vector<int>& ids, int image_token_id, const std::vector<H3ImageFeatures>& images, float* d_out ) {
	Impl& s = *p_;
	const int S = static_cast<int>( ids.size() );
	if ( S <= 0 || S > s.max_tokens ) { return false; }
	// embedding rows (BF16 table on disk) -> FP32; image placeholders take the vision features
	const pod::SafeTensorMeta* em = s.repo.find( s.embed_name.c_str() );
	const uint16_t* table = static_cast<const uint16_t*>( s.repo.cpu_data( s.embed_name.c_str() ) );
	std::vector<float> rows( static_cast<size_t>( S ) * kH );
	std::vector<int> pad_rows;
	std::vector<int> mm( S, 0 );
	for ( int i = 0; i < S; ++i ) {
		if ( ids[i] < 0 || ids[i] >= em->shape[0] ) { return false; }
		if ( ids[i] == image_token_id ) { mm[i] = 1; pad_rows.push_back( i ); }
		for ( int c = 0; c < kH; ++c ) {
			const uint32_t bits = static_cast<uint32_t>( table[static_cast<size_t>( ids[i] ) * kH + c] ) << 16;
			std::memcpy( &rows[static_cast<size_t>( i ) * kH + c], &bits, 4 );
		}
	}
	size_t total_tokens = 0;
	for ( const H3ImageFeatures& f : images ) { total_tokens += f.merged.size() / kH; }
	if ( total_tokens != pad_rows.size() ) {
		LOG_ERROR( "H3 text: %zu image placeholders but %zu image tokens of features", pad_rows.size(), total_tokens );
		return false;
	}
	std::vector<float> ds[3];
	{
		size_t at = 0;
		for ( const H3ImageFeatures& f : images ) {
			const size_t n = f.merged.size() / kH;
			for ( size_t t = 0; t < n; ++t ) { std::memcpy( &rows[static_cast<size_t>( pad_rows[at + t] ) * kH], &f.merged[t * kH], kH * sizeof( float ) ); }
			for ( int d = 0; d < 3; ++d ) { ds[d].insert( ds[d].end(), f.deepstack[d].begin(), f.deepstack[d].end() ); }
			at += n;
		}
	}
	// 3-D positions (Qwen3-VL get_rope_index): text runs advance all axes, an image run lays its merged grid out from the current position
	std::vector<int> pos( static_cast<size_t>( 3 ) * S, 0 );
	{
		int cur = 0, i = 0;
		size_t img = 0;
		while ( i < S ) {
			int j = i;
			while ( j < S && mm[j] == mm[i] ) { ++j; }
			if ( mm[i] == 0 ) {
				for ( int k = i; k < j; ++k ) { for ( int a = 0; a < 3; ++a ) { pos[static_cast<size_t>( a ) * S + k] = cur + ( k - i ); } }
				cur += j - i;
			} else {
				const H3ImageFeatures& f = images[std::min( img, images.size() - 1 )];
				const int gh = f.grid_h / 2, gw = f.grid_w / 2;
				for ( int k = i; k < j; ++k ) {
					const int n = k - i;
					pos[static_cast<size_t>( 0 ) * S + k] = cur;
					pos[static_cast<size_t>( 1 ) * S + k] = cur + n / gw;
					pos[static_cast<size_t>( 2 ) * S + k] = cur + n % gw;
				}
				cur += std::max( gh, gw );
				++img;
			}
			i = j;
		}
	}
	CUDA_CHECK( cudaMemcpyAsync( s.d_pos, pos.data(), pos.size() * sizeof( int ), cudaMemcpyHostToDevice, s.cs ) );
	CUDA_CHECK( cudaMemcpyAsync( s.x, rows.data(), rows.size() * sizeof( float ), cudaMemcpyHostToDevice, s.cs ) );
	float* d_feat = nullptr;
	int* d_rows = nullptr;
	if ( !pad_rows.empty() ) {
		CUDA_CHECK( cudaMalloc( &d_feat, pad_rows.size() * kH * sizeof( float ) ) );
		CUDA_CHECK( cudaMalloc( &d_rows, pad_rows.size() * sizeof( int ) ) );
		CUDA_CHECK( cudaMemcpyAsync( d_rows, pad_rows.data(), pad_rows.size() * sizeof( int ), cudaMemcpyHostToDevice, s.cs ) );
	}
	CUDA_CHECK( cudaStreamSynchronize( s.cs ) );
	// layer pipeline: slot (i % 2) holds layer i; a thread uploads layer i + 1 while layer i computes (slot (i + 1) % 2 held layer i - 1, whose compute ended in the
	// previous iteration)
	s.load_layer( 0, 0 );
	for ( int i = 0; i < s.layers; ++i ) {
		const int cur = i % 2, nxt = 1 - cur;
		std::thread prefetch;
		if ( i + 1 < s.layers ) {
			prefetch = std::thread( [&s, nxt, i] {
				CUDA_CHECK( cudaSetDevice( s.device ) );
				s.load_layer( nxt, i + 1 );
			} );
		}
		s.run_layer( s.slot[cur], S );
		if ( d_feat && i < 3 ) {
			CUDA_CHECK( cudaMemcpyAsync( d_feat, ds[i].data(), pad_rows.size() * kH * sizeof( float ), cudaMemcpyHostToDevice, s.cs ) );
			CUDA_CHECK( cudaStreamSynchronize( s.cs ) );
			k_add_rows<<<static_cast<int>( pad_rows.size() ), 256, 0, s.cs>>>( s.x, d_feat, d_rows, kH );
		}
		CUDA_CHECK( cudaStreamSynchronize( s.cs ) );
		if ( prefetch.joinable() ) { prefetch.join(); }
		if ( probe ) { probe( i, s.x, S ); }
	}
	CUDA_CHECK( cudaMemcpyAsync( d_out, s.x, static_cast<size_t>( S ) * kH * sizeof( float ), cudaMemcpyDeviceToDevice, s.cs ) );
	CUDA_CHECK( cudaStreamSynchronize( s.cs ) );
	if ( d_feat ) { cudaFree( d_feat ); cudaFree( d_rows ); }
	CUDA_CHECK( cudaGetLastError() );
	return true;
}

} // namespace engine::video
