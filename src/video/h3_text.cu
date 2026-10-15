// Streaming Qwen3 text tower for the H3 conditioning (see h3_text.h).
#include "engine/video/h3_text.h"
#include "engine/backend/cuda/activation.cuh"
#include "engine/video/h3_attn.h"
#include "h3_util.h"
#include "engine/core/arena.h"
#include "engine/core/log.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/linear.h"
#include "engine/modules/attention.h"
#include "engine/runtime/kv_cache.h"
#include "engine/core/batch.h"
#include "../kernels/cuda/trellis/trellis_context.h"
#include "h3_kernels.cuh"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <initializer_list>
#include <numeric>
#include <random>
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
__global__ void k_qwen_qk( half* q, half* k, const float* qw, const float* kw, const int* pos, int S, int nh, int nkv ) {
	__shared__ float sm[8][128];
	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	const int s = blockIdx.x, task = blockIdx.y * 8 + warp;
	if ( task >= nh + nkv ) { return; }
	half* p = task < nh ? q + ( static_cast<size_t>( s ) * nh + task ) * kHd : k + ( static_cast<size_t>( s ) * nkv + ( task - nh ) ) * kHd;
	const float* w = task < nh ? qw : kw;
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
// GQA: [S, kNkv, kHd] K and V -> [S, kNh, kHd] with every kv head repeated for its group of query heads (the attention kernel has equal head counts)
__global__ void k_expand_kv( const half* k, const half* v, half* kx, half* vx, int S, int nh, int nkv ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>( S ) * nh * kHd;
	if ( i >= total ) { return; }
	const int d = static_cast<int>( i % kHd );
	const size_t sh = i / kHd;
	const int h = static_cast<int>( sh % nh );
	const size_t s = sh / nh;
	const size_t src = ( s * nkv + h / ( nh / nkv ) ) * kHd + d;
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
		mx = fmaxf( mx, fabsf( engine::math::silu( a ) * b ) );
	}
	mx = block_max( mx, red );
	int e = 0;
	if ( isfinite( mx ) ) { frexpf( mx, &e ); }
	const float s = ldexpf( 1.0f, max( e - 11, 0 ) );
	if ( threadIdx.x == 0 ) { rowscale[row] = s; }
	for ( int c = threadIdx.x; c < F; c += blockDim.x ) {
		const float a = __half2float( gr[c] ), b = __half2float( ur[c] );
		out[static_cast<size_t>( row ) * F + c] = __float2half_rn( engine::math::silu( a ) * b / s );
	}
}

// tensor-parallel version of the above: the row scale comes from the whole F, so the max over this rank's slice goes to amax[row] (reduced over the ranks by the caller)
// and k_silu2_apply derives the scale from the global value
__global__ void k_silu2_amax( const half* g, const half* u, float* amax, int F ) {
	__shared__ float red[32];
	const int row = blockIdx.x;
	float mx = 0.0f;
	for ( int c = threadIdx.x; c < F; c += blockDim.x ) {
		const float a = __half2float( g[static_cast<size_t>( row ) * F + c] ), b = __half2float( u[static_cast<size_t>( row ) * F + c] );
		mx = fmaxf( mx, fabsf( engine::math::silu( a ) * b ) );
	}
	mx = block_max( mx, red );
	if ( threadIdx.x == 0 ) { amax[row] = mx; }
}

__global__ void k_silu2_apply( const half* g, const half* u, half* out, float* rowscale, int F ) {
	const int row = blockIdx.x;
	const float mx = rowscale[row];
	int e = 0;
	if ( isfinite( mx ) ) { frexpf( mx, &e ); }
	const float s = ldexpf( 1.0f, max( e - 11, 0 ) );
	__syncthreads(); // every thread has read the maximum before it is replaced by the scale
	if ( threadIdx.x == 0 ) { rowscale[row] = s; }
	for ( int c = threadIdx.x; c < F; c += blockDim.x ) {
		const float a = __half2float( g[static_cast<size_t>( row ) * F + c] ), b = __half2float( u[static_cast<size_t>( row ) * F + c] );
		out[static_cast<size_t>( row ) * F + c] = __float2half_rn( engine::math::silu( a ) * b / s );
	}
}

// x[rows[i]] += feat[i] (DeepStack injection)
static __global__ void k_add_rows( float* x, const float* feat, const int* rows, int H ) {
	const int i = blockIdx.x;
	for ( int c = threadIdx.x; c < H; c += blockDim.x ) { x[static_cast<size_t>( rows[i] ) * H + c] += feat[static_cast<size_t>( i ) * H + c]; }
}

// FP16 branch output [S, H] added to the FP32 stream, optionally scaled per row (the EXL3 projections produce FP16)
__global__ void k_add_half( float* x, const half* branch, const float* rowscale, int H ) {
	const int row = blockIdx.x;
	const float rs = rowscale ? rowscale[row] : 1.0f;
	for ( int c = threadIdx.x; c < H; c += blockDim.x ) { x[static_cast<size_t>( row ) * H + c] += __half2float( branch[static_cast<size_t>( row ) * H + c] ) * rs; }
}

struct LayerW {
	half *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr, *gate = nullptr, *up = nullptr, *down = nullptr;
	float *in_norm = nullptr, *post_norm = nullptr, *qn = nullptr, *kn = nullptr;
};

// The same layer with EXL3 (trellis) projections, resident on the device
struct ExlLayer {
	pod::LinearWeight q, k, v, o, gate, up, down;
	float *in_norm = nullptr, *post_norm = nullptr, *qn = nullptr, *kn = nullptr;
};

} // namespace

struct H3TextEncoder::Impl {
	engine::loader::SafeTensorsRepo repo;
	int device = 0, max_tokens = 0, layers = 0;
	pod::MemoryArena slot_arena[2]{}, sarena{};
	LayerW slot[2];
	bool exl3 = false;          // resident quantized projections (EXL3 trellis or NVFP4) instead of streamed BF16 matrices
	bool nvfp4 = false;         // ... of the NVFP4 kind
	pod::MemoryArena warena{};  // EXL3 weights
	std::vector<ExlLayer> xl;
	half* br16 = nullptr;       // FP16 branch output of the EXL3 projections
	int tp_rank = 0, tp_n = 1;  // tensor parallelism of the resident EXL3 mode
	int nh_l = kNh, nkv_l = kNkv, ffn_l = kFfn; // this rank's query heads, KV heads, MLP width
	H3TpComm tp_comm;
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
	half *kx16 = nullptr, *vx16 = nullptr; // K / V repeated to all query heads for the attention kernel
	bool use_fmha = false;
	// captioner (load(captioner = true)): all decoder layers, the LM head on rank 0, and a KV cache the decode steps attend to
	bool cap = false;
	int total_layers = 0;       // layers loaded (layers = depth of the hidden states encode() returns)
	int kv_max_seq = 0;
	engine::runtime::KVCachePool kv;
	pod::MemoryArena karena{};
	pod::BatchRows* d_brows = nullptr; // device: one BatchRows per 512-row chunk of the prompt, then the decode row
	int n_chunks = 0;
	pod::LinearWeight head;
	float* final_norm = nullptr;
	half* logits16 = nullptr;   // [vocab]
	float* d_tok = nullptr;     // sampled token broadcast to the ranks
	int vocab = 0;
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

	// small norm vector -> FP32 on the device (resident EXL3 mode), from BF16 / FP16 / FP32 storage
	float* upload_vec_resident( const std::string& name ) {
		const pod::SafeTensorMeta* m = repo.find( name.c_str() );
		ENGINE_ASSERT( m, "H3 text: missing norm vector" );
		const void* src = repo.cpu_data( name.c_str() );
		size_t n = 1;
		for ( int i = 0; i < m->ndim; ++i ) { n *= static_cast<size_t>( m->shape[i] ); }
		std::vector<float> host( n );
		for ( size_t i = 0; i < n; ++i ) {
			if ( m->dtype == pod::DType::FP32 ) { host[i] = static_cast<const float*>( src )[i]; }
			else if ( m->dtype == pod::DType::FP16 ) { host[i] = __half2float( static_cast<const half*>( src )[i] ); }
			else {
				const uint32_t bits = static_cast<uint32_t>( static_cast<const uint16_t*>( src )[i] ) << 16;
				std::memcpy( &host[i], &bits, 4 );
			}
		}
		float* d = alloc<float>( warena, n );
		CUDA_CHECK( cudaMemcpy( d, host.data(), n * sizeof( float ), cudaMemcpyHostToDevice ) );
		return d;
	}

	// x [S, K] FP16 -> out [S, N] FP16 through the trellis kernels (fused for few rows, dequantize + HGEMM for many)
	void lin( const pod::LinearWeight& w, const half* in, half* out, int S, int K, int N ) {
		pod::TensorView xv = pod::make_tensor_2d( const_cast<half*>( in ), pod::DType::FP16, S, K );
		pod::TensorView ov = pod::make_tensor_2d( out, pod::DType::FP16, S, N );
		engine::modules::Linear::forward( xv, w, ov, cs );
	}

	// mode 0: plain; 1: prompt rows, K/V also stored in the cache (layer `layer`); 2: one decode row that attends to the cache
	void run_layer_x( const ExlLayer& w, int S, int layer = 0, int mode = 0 ) {
		cudaStream_t s = cs;
		const int qd = nh_l * kHd, kd = nkv_l * kHd;
		k_norm_mod<half><<<S, 256, 0, s>>>( x, nullptr, nullptr, 0, w.in_norm, nullptr, nullptr, 0, nullptr, kH, kEps, h16 );
		lin( w.q, h16, q16, S, kH, qd );
		lin( w.k, h16, k16, S, kH, kd );
		lin( w.v, h16, v16, S, kH, kd );
		k_qwen_qk<<<dim3( S, ( nh_l + nkv_l + 7 ) / 8 ), 256, 0, s>>>( q16, k16, w.qn, w.kn, d_pos, S, nh_l, nkv_l );
		if ( mode == 2 ) {
			pod::TensorView kt = pod::make_tensor_2d( k16, pod::DType::FP16, 1, kd ), vt = pod::make_tensor_2d( v16, pod::DType::FP16, 1, kd );
			pod::TensorView qt = pod::make_tensor_2d( q16, pod::DType::FP16, 1, qd ), ot = pod::make_tensor_2d( o16, pod::DType::FP16, 1, qd ), nogate{};
			engine::modules::Attention::store_kv_cache( kv.ref(), kt, vt, kv.slot_stride_elems, layer, d_brows + n_chunks, 1, nkv_l, kHd, kv_max_seq, s );
			engine::modules::Attention::forward_decode( kv.ref(), qt, nogate, ot, kv.slot_stride_elems, layer, d_brows + n_chunks, 1, nh_l, nkv_l, kHd, kv_max_seq, s );
		} else {
			if ( mode == 1 ) {
				for ( int c = 0; c * pod::MAX_ROWS < S; ++c ) {
					const int r0 = c * pod::MAX_ROWS, n = std::min( pod::MAX_ROWS, S - r0 );
					pod::TensorView kt = pod::make_tensor_2d( k16 + static_cast<size_t>( r0 ) * kd, pod::DType::FP16, n, kd );
					pod::TensorView vt = pod::make_tensor_2d( v16 + static_cast<size_t>( r0 ) * kd, pod::DType::FP16, n, kd );
					engine::modules::Attention::store_kv_cache( kv.ref(), kt, vt, kv.slot_stride_elems, layer, d_brows + c, n, nkv_l, kHd, kv_max_seq, s );
				}
			}
			k_expand_kv<<<( static_cast<size_t>( S ) * qd + 255 ) / 256, 256, 0, s>>>( k16, v16, kx16, vx16, S, nh_l, nkv_l );
			h3_attention( q16, kx16, vx16, o16, S, S, nh_l, 0.08838834764831845f, true, s );
		}
		lin( w.o, o16, br16, S, qd, kH );
		if ( tp_n > 1 ) { tp_comm.sum_f16( br16, static_cast<size_t>( S ) * kH, s ); } // partial products of the rank's heads
		k_add_half<<<S, 256, 0, s>>>( x, br16, nullptr, kH );
		k_norm_mod<half><<<S, 256, 0, s>>>( x, nullptr, nullptr, 0, w.post_norm, nullptr, nullptr, 0, nullptr, kH, kEps, h16 );
		lin( w.gate, h16, g16, S, kH, ffn_l );
		lin( w.up, h16, u16, S, kH, ffn_l );
		if ( tp_n > 1 ) {
			k_silu2_amax<<<S, 256, 0, s>>>( g16, u16, rowscale, ffn_l );
			tp_comm.max_f32( rowscale, S, s );
			k_silu2_apply<<<S, 256, 0, s>>>( g16, u16, f16, rowscale, ffn_l );
		} else {
			k_silu_mul2<<<S, 256, 0, s>>>( g16, u16, f16, rowscale, ffn_l );
		}
		lin( w.down, f16, br16, S, ffn_l, kH );
		if ( tp_n > 1 ) { tp_comm.sum_f16( br16, static_cast<size_t>( S ) * kH, s ); }
		k_add_half<<<S, 256, 0, s>>>( x, br16, rowscale, kH );
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
		k_qwen_qk<<<dim3( S, ( kNh + kNkv + 7 ) / 8 ), 256, 0, s>>>( q16, k16, w.qn, w.kn, d_pos, S, kNh, kNkv );
		if ( use_fmha ) {
			k_expand_kv<<<( static_cast<size_t>( S ) * kNh * kHd + 255 ) / 256, 256, 0, s>>>( k16, v16, kx16, vx16, S, kNh, kNkv );
			h3_attention( q16, kx16, vx16, o16, S, S, kNh, 0.08838834764831845f, true, s );
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
		if ( p_->warena.base_ptr ) { engine::core::arena_destroy( p_->warena ); }
		for ( int i = 0; i < Impl::kPinned; ++i ) { if ( p_->pinned[i] ) { (void) cudaFreeHost( p_->pinned[i] ); } }
		if ( p_->staging ) { (void) cudaFree( p_->staging ); }
		delete p_;
	}
}

void H3TextEncoder::set_tp( H3TpComm comm ) { p_->tp_comm = std::move( comm ); }

bool H3TextEncoder::load( const char* dir, int device, int max_tokens, int layers, int tp_rank, int tp_ranks, bool captioner ) {
	Impl& s = *p_;
	s.tp_rank = tp_rank;
	s.tp_n = std::max( 1, tp_ranks );
	s.device = device;
	s.max_tokens = max_tokens;
	s.layers = layers;
	s.cap = captioner;
	s.total_layers = captioner ? std::max( layers, 64 ) : layers;
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
	s.nvfp4 = s.repo.find( "model.language_model.layers.0.self_attn.q_proj.weight_packed" ) != nullptr; // compressed-tensors NVFP4
	s.exl3 = s.nvfp4 || s.repo.find( "model.language_model.layers.0.self_attn.q_proj.trellis" ) != nullptr; // resident quantized projections (EXL3 trellis or NVFP4)
	if ( captioner && ( !s.exl3 || s.nvfp4 ) ) { LOG_ERROR( "H3 text: the captioner needs an EXL3 text_encoder" ); return false; }
	if ( s.tp_n > 1 ) {
		ENGINE_ASSERT( s.exl3, "H3 text: tensor parallelism needs an EXL3 checkpoint" );
		ENGINE_ASSERT( kNh % s.tp_n == 0 && kNkv % s.tp_n == 0 && ( kFfn / s.tp_n ) % 128 == 0, "H3 text: the shards must keep whole 128-column Hadamard blocks" );
		s.nh_l = kNh / s.tp_n;
		s.nkv_l = kNkv / s.tp_n;
		s.ffn_l = kFfn / s.tp_n;
	}
	if ( s.exl3 ) {
		// resident EXL3 projections: 7 trellis linears and 4 norm vectors per layer
		const char* proj[7] = { "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" };
		size_t bytes = 64ULL << 20;
		for ( int i = 0; i < s.total_layers; ++i ) {
			const std::string p = "model.language_model.layers." + std::to_string( i ) + ".";
			for ( const char* pr : proj ) {
				static const char* const kNvfp4Parts[] = { ".weight_packed", ".weight_scale" };
				static const char* const kExl3Parts[] = { ".trellis", ".suh", ".svh" };
				for ( const char* part : s.nvfp4 ? std::vector<const char*>( std::begin( kNvfp4Parts ), std::end( kNvfp4Parts ) ) : std::vector<const char*>( std::begin( kExl3Parts ), std::end( kExl3Parts ) ) ) {
					const pod::SafeTensorMeta* m = s.repo.find( ( p + pr + part ).c_str() );
					if ( !m ) { LOG_ERROR( "H3 text: quantized tensor %s%s%s missing", p.c_str(), pr, part ); return false; }
					bytes += m->size_bytes + 512 + ( s.nvfp4 ? m->size_bytes / 8 : 0 ); // NVFP4 pads the rows to a multiple of 32 and prepacks
				}
			}
			bytes += 4ULL * kH * 4 + 4096;
		}
		if ( s.tp_n > 1 ) { bytes = bytes / s.tp_n + ( 256ULL << 20 ); }
		if ( captioner && tp_rank == 0 ) { // LM head (rank 0 only) and the final norm
			const pod::SafeTensorMeta* hm = s.repo.find( "lm_head.trellis" );
			if ( !hm ) { LOG_ERROR( "H3 text: the checkpoint has no lm_head (captioner)" ); return false; }
			bytes += hm->size_bytes + ( 64ULL << 20 );
		}
		if ( !h3_scratch_arena_create( s.warena, device, bytes ) ) { return false; }
		s.xl.assign( s.total_layers, ExlLayer() );
		for ( int i = 0; i < s.total_layers; ++i ) {
			const std::string p = "model.language_model.layers." + std::to_string( i ) + ".";
			ExlLayer& w = s.xl[i];
			pod::LinearWeight* dst[7] = { &w.q, &w.k, &w.v, &w.o, &w.gate, &w.up, &w.down };
			for ( int j = 0; j < 7; ++j ) {
				// q, k, v, gate, up split their outputs (column parallel); o and down split their inputs (row parallel)
				const bool row_parallel = j == 3 || j == 6;
				if ( s.nvfp4 ) { *dst[j] = s.repo.load_nvfp4_slice( ( p + proj[j] ).c_str(), s.warena, !row_parallel, s.tp_rank, s.tp_n, s.cs ); continue; }
				*dst[j] = s.tp_n > 1 ? s.repo.load_linear_slice( ( p + proj[j] ).c_str(), s.warena, 6, row_parallel ? 0 : 1, s.tp_rank, s.tp_n, s.cs ) : s.repo.load_linear( ( p + proj[j] ).c_str(), s.warena, 6, s.cs );
			}
			w.in_norm = s.upload_vec_resident( p + "input_layernorm.weight" );
			w.post_norm = s.upload_vec_resident( p + "post_attention_layernorm.weight" );
			w.qn = s.upload_vec_resident( p + "self_attn.q_norm.weight" );
			w.kn = s.upload_vec_resident( p + "self_attn.k_norm.weight" );
		}
		if ( captioner ) {
			s.final_norm = s.upload_vec_resident( "model.language_model.norm.weight" );
			if ( tp_rank == 0 ) { s.head = s.repo.load_linear( "lm_head", s.warena, 6, s.cs ); }
		}
		// The quantized kernels grow their per-device workspaces lazily, under a global lock, with cudaFree / cudaMalloc. Doing that on one rank in the middle of a forward
		// deadlocks the all-reduces (the other ranks spin in NCCL, the cudaFree waits for their kernels), so every workspace is sized here, before any collective.
		engine::modules::trellis_context_init_all();
		engine::modules::trellis_get_recon_ws( device, static_cast<size_t>( std::max( kH * s.ffn_l, kH * s.nh_l * kHd ) ) * sizeof( half ) );
		engine::modules::trellis_get_cublas( device );
		CUDA_CHECK( cudaStreamSynchronize( s.cs ) );
		LOG_INFO( "H3 text: %d %s layers resident, %.2f GB", s.total_layers, s.nvfp4 ? "NVFP4" : "EXL3", s.warena.offset_bytes / 1e9 );
	} else {
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

	}
	const size_t S = static_cast<size_t>( max_tokens );
	const size_t need = S * kH * 4 * 2 + S * 4 * 4 + S * kH * 2 + S * s.nh_l * kHd * 2 * 2 + S * s.nkv_l * kHd * 2 * 2 + S * s.nh_l * kHd * 2 * 2 + S * s.ffn_l * 2 * 3 + S * kH * 2 + ( 16ULL << 20 );
	if ( !h3_scratch_arena_create( s.sarena, device, need ) ) { return false; }
	s.x = s.alloc<float>( s.sarena, S * kH );
	s.br = s.alloc<float>( s.sarena, S * kH );
	s.rowscale = s.alloc<float>( s.sarena, S );
	s.d_pos = s.alloc<int>( s.sarena, 3 * S );
	s.h16 = s.alloc<half>( s.sarena, S * kH );
	s.q16 = s.alloc<half>( s.sarena, S * s.nh_l * kHd );
	s.o16 = s.alloc<half>( s.sarena, S * s.nh_l * kHd );
	s.k16 = s.alloc<half>( s.sarena, S * s.nkv_l * kHd );
	s.v16 = s.alloc<half>( s.sarena, S * s.nkv_l * kHd );
	s.kx16 = s.alloc<half>( s.sarena, S * s.nh_l * kHd );
	s.vx16 = s.alloc<half>( s.sarena, S * s.nh_l * kHd );
	s.use_fmha = true;
	s.g16 = s.alloc<half>( s.sarena, S * s.ffn_l );
	s.u16 = s.alloc<half>( s.sarena, S * s.ffn_l );
	s.f16 = s.alloc<half>( s.sarena, S * s.ffn_l );
	s.br16 = s.alloc<half>( s.sarena, S * kH );
	if ( captioner ) {
		const pod::SafeTensorMeta* em = s.repo.find( s.embed_name.c_str() );
		s.vocab = em ? em->shape[0] : 151936;
		s.kv_max_seq = max_tokens + 1024; // the prompt and the answer
		const size_t kv_bytes = 2 * static_cast<size_t>( s.total_layers ) * s.kv_max_seq * s.nkv_l * kHd * sizeof( half ) + ( 8ULL << 20 );
		if ( !h3_scratch_arena_create( s.karena, device, kv_bytes ) ) { return false; }
		if ( !s.kv.init( device, s.karena, 1, s.kv_max_seq, s.total_layers, s.nkv_l, kHd ) ) { return false; }
		s.n_chunks = ( max_tokens + pod::MAX_ROWS - 1 ) / pod::MAX_ROWS;
		CUDA_CHECK( cudaMalloc( &s.d_brows, static_cast<size_t>( s.n_chunks + 1 ) * sizeof( pod::BatchRows ) ) );
		CUDA_CHECK( cudaMalloc( &s.d_tok, sizeof( float ) ) );
		if ( tp_rank == 0 ) { CUDA_CHECK( cudaMalloc( &s.logits16, static_cast<size_t>( s.vocab ) * sizeof( half ) ) ); }
		// every lazily grown workspace of the M = 1 paths is created here, before the first collective (see the note above): one local pass through the projections
		if ( s.exl3 ) {
			const ExlLayer& w = s.xl[0];
			s.lin( w.q, s.h16, s.q16, 1, kH, s.nh_l * kHd );
			s.lin( w.k, s.h16, s.k16, 1, kH, s.nkv_l * kHd );
			s.lin( w.o, s.q16, s.br16, 1, s.nh_l * kHd, kH );
			s.lin( w.gate, s.h16, s.g16, 1, kH, s.ffn_l );
			s.lin( w.down, s.g16, s.br16, 1, s.ffn_l, kH );
			if ( tp_rank == 0 ) { s.lin( s.head, s.h16, s.logits16, 1, kH, s.vocab ); }
			CUDA_CHECK( cudaStreamSynchronize( s.cs ) );
		}
	}
	CUDA_CHECK( cudaFuncSetAttribute( k_attn_causal, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304 - 1024 ) );
	CUDA_CHECK( cudaDeviceSynchronize() ); // loading ran on the default stream; the models run on non-blocking streams, which it does not order against
	return s.repo.find( s.embed_name.c_str() ) != nullptr;
}

bool H3TextEncoder::encode( const std::vector<int>& ids, float* d_out ) { return encode_mm( ids, -1, {}, d_out ); }

bool H3TextEncoder::encode_mm( const std::vector<int>& ids, int image_token_id, const std::vector<H3ImageFeatures>& images, float* d_out ) {
	return run_mm( ids, image_token_id, images, p_->layers, false, nullptr, d_out );
}

// The decoder stack over the prompt rows. d_out (optional) receives the stream after `nlayers` layers; store_kv fills the KV cache (needs the captioner); next_pos
// receives the rotary position of the token that follows the prompt.
bool H3TextEncoder::run_mm( const std::vector<int>& ids, int image_token_id, const std::vector<H3ImageFeatures>& images, int nlayers, bool store_kv, int* next_pos, float* d_out ) {
	Impl& s = *p_;
	const int S = static_cast<int>( ids.size() );
	if ( S <= 0 || S > s.max_tokens ) { return false; }
	// embedding rows (BF16 table on disk) -> FP32; image placeholders take the vision features
	const pod::SafeTensorMeta* em = s.repo.find( s.embed_name.c_str() );
	const uint16_t* table = static_cast<const uint16_t*>( s.repo.cpu_data( s.embed_name.c_str() ) );
	const bool table_fp16 = em->dtype == pod::DType::FP16; // EXL3 conversions keep the embedding in FP16, the source in BF16
	std::vector<float> rows( static_cast<size_t>( S ) * kH );
	std::vector<int> pad_rows;
	std::vector<int> mm( S, 0 );
	for ( int i = 0; i < S; ++i ) {
		if ( ids[i] < 0 || ids[i] >= em->shape[0] ) { return false; }
		if ( ids[i] == image_token_id ) { mm[i] = 1; pad_rows.push_back( i ); }
		for ( int c = 0; c < kH; ++c ) {
			const uint16_t raw = table[static_cast<size_t>( ids[i] ) * kH + c];
			if ( table_fp16 ) {
				half hv;
				std::memcpy( &hv, &raw, 2 );
				rows[static_cast<size_t>( i ) * kH + c] = __half2float( hv );
			} else {
				const uint32_t bits = static_cast<uint32_t>( raw ) << 16;
				std::memcpy( &rows[static_cast<size_t>( i ) * kH + c], &bits, 4 );
			}
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
	int rope_cur = 0;
	{
		int i = 0;
		size_t img = 0;
		while ( i < S ) {
			int j = i;
			while ( j < S && mm[j] == mm[i] ) { ++j; }
			if ( mm[i] == 0 ) {
				for ( int k = i; k < j; ++k ) { for ( int a = 0; a < 3; ++a ) { pos[static_cast<size_t>( a ) * S + k] = rope_cur + ( k - i ); } }
				rope_cur += j - i;
			} else {
				const H3ImageFeatures& f = images[std::min( img, images.size() - 1 )];
				const int gh = f.grid_h / 2, gw = f.grid_w / 2;
				for ( int k = i; k < j; ++k ) {
					const int n = k - i;
					pos[static_cast<size_t>( 0 ) * S + k] = rope_cur;
					pos[static_cast<size_t>( 1 ) * S + k] = rope_cur + n / gw;
					pos[static_cast<size_t>( 2 ) * S + k] = rope_cur + n % gw;
				}
				rope_cur += std::max( gh, gw );
				++img;
			}
			i = j;
		}
	}
	if ( next_pos ) { *next_pos = rope_cur; }
	if ( store_kv ) {
		ENGINE_ASSERT( s.cap, "H3 text: the KV cache needs load(captioner = true)" );
		std::vector<pod::BatchRows> chunks;
		for ( int r0 = 0; r0 < S; r0 += pod::MAX_ROWS ) { chunks.push_back( pod::BatchRows::causal( std::min( pod::MAX_ROWS, S - r0 ), r0, 0 ) ); }
		CUDA_CHECK( cudaMemcpyAsync( s.d_brows, chunks.data(), chunks.size() * sizeof( pod::BatchRows ), cudaMemcpyHostToDevice, s.cs ) );
		CUDA_CHECK( cudaStreamSynchronize( s.cs ) ); // `chunks` goes out of scope
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
	if ( !s.exl3 ) { s.load_layer( 0, 0 ); }
	for ( int i = 0; i < nlayers; ++i ) {
		const int cur = i % 2, nxt = 1 - cur;
		std::thread prefetch;
		if ( !s.exl3 && i + 1 < nlayers ) {
			prefetch = std::thread( [&s, nxt, i] {
				CUDA_CHECK( cudaSetDevice( s.device ) );
				s.load_layer( nxt, i + 1 );
			} );
		}
		if ( s.exl3 ) { s.run_layer_x( s.xl[i], S, i, store_kv ? 1 : 0 ); } else { s.run_layer( s.slot[cur], S ); }
		if ( d_feat && i < 3 ) {
			CUDA_CHECK( cudaMemcpyAsync( d_feat, ds[i].data(), pad_rows.size() * kH * sizeof( float ), cudaMemcpyHostToDevice, s.cs ) );
			CUDA_CHECK( cudaStreamSynchronize( s.cs ) );
			k_add_rows<<<static_cast<int>( pad_rows.size() ), 256, 0, s.cs>>>( s.x, d_feat, d_rows, kH );
		}
		CUDA_CHECK( cudaStreamSynchronize( s.cs ) );
		if ( prefetch.joinable() ) { prefetch.join(); }
		if ( probe ) { probe( i, s.x, S ); }
	}
	if ( d_out ) { CUDA_CHECK( cudaMemcpyAsync( d_out, s.x, static_cast<size_t>( S ) * kH * sizeof( float ), cudaMemcpyDeviceToDevice, s.cs ) ); }
	CUDA_CHECK( cudaStreamSynchronize( s.cs ) );
	if ( d_feat ) { (void) cudaFree( d_feat ); (void) cudaFree( d_rows ); }
	CUDA_CHECK( cudaGetLastError() );
	return true;
}

namespace {

// Qwen3's special tokens end at <|im_end|> (151645); <|endoftext|> (151643) and <|im_end|> end an answer, the other specials never appear in one
constexpr int kEndOfText = 151643, kImEnd = 151645;

int sample_logits( const std::vector<half>& h, const H3CaptionOptions& o, std::mt19937& rng ) {
	std::vector<std::pair<float, int>> c;
	c.reserve( 160000 );
	for ( int i = 0; i < static_cast<int>( h.size() ); ++i ) {
		if ( i < kEndOfText || i == kEndOfText || i == kImEnd ) { c.emplace_back( __half2float( h[i] ), i ); }
	}
	auto top = []( const std::pair<float, int>& a, const std::pair<float, int>& b ) { return a.first > b.first; };
	if ( o.temperature <= 0.0f ) { return std::min_element( c.begin(), c.end(), top )->second; }
	const size_t k = std::min<size_t>( c.size(), o.top_k > 0 ? static_cast<size_t>( o.top_k ) : c.size() );
	std::partial_sort( c.begin(), c.begin() + k, c.end(), top );
	std::vector<double> p( k );
	double sum = 0.0;
	for ( size_t i = 0; i < k; ++i ) {
		p[i] = std::exp( ( c[i].first - c[0].first ) / o.temperature );
		sum += p[i];
	}
	size_t keep = k;
	if ( o.top_p > 0.0f && o.top_p < 1.0f ) { // smallest prefix whose probability reaches top_p
		double acc = 0.0;
		for ( size_t i = 0; i < k; ++i ) {
			acc += p[i] / sum;
			if ( acc >= o.top_p ) { keep = i + 1; break; }
		}
	}
	double kept = 0.0;
	for ( size_t i = 0; i < keep; ++i ) { kept += p[i]; }
	double r = std::uniform_real_distribution<double>( 0.0, kept )( rng );
	for ( size_t i = 0; i < keep; ++i ) {
		r -= p[i];
		if ( r <= 0.0 ) { return c[i].second; }
	}
	return c[keep - 1].second;
}

} // namespace

bool H3TextEncoder::generate( const std::vector<int>& ids, int image_token_id, const std::vector<H3ImageFeatures>& images, const H3CaptionOptions& opt, std::vector<int>* out_tokens ) {
	Impl& s = *p_;
	if ( !s.cap || !out_tokens ) { return false; }
	CUDA_CHECK( cudaSetDevice( s.device ) );
	out_tokens->clear();
	const int S = static_cast<int>( ids.size() );
	if ( S <= 0 || S + opt.max_new_tokens > s.kv_max_seq ) { return false; }
	int next_pos = 0;
	if ( !run_mm( ids, image_token_id, images, s.total_layers, true, &next_pos, nullptr ) ) { return false; }
	const pod::SafeTensorMeta* em = s.repo.find( s.embed_name.c_str() );
	const uint16_t* table = static_cast<const uint16_t*>( s.repo.cpu_data( s.embed_name.c_str() ) );
	std::mt19937 rng( opt.seed );
	std::vector<half> logits;
	std::vector<float> row( kH );
	int kv_pos = S;
	const float* last = s.x + static_cast<size_t>( S - 1 ) * kH;
	for ( int step = 0; step < opt.max_new_tokens; ++step ) {
		float tok = 0.0f;
		if ( s.tp_rank == 0 ) { // the head and the sampling live on rank 0; the others only learn the token
			k_norm_mod<half><<<1, 256, 0, s.cs>>>( const_cast<float*>( last ), nullptr, nullptr, 0, s.final_norm, nullptr, nullptr, 0, nullptr, kH, kEps, s.h16 );
			s.lin( s.head, s.h16, s.logits16, 1, kH, s.vocab );
			logits.resize( s.vocab );
			CUDA_CHECK( cudaMemcpyAsync( logits.data(), s.logits16, logits.size() * sizeof( half ), cudaMemcpyDeviceToHost, s.cs ) );
			CUDA_CHECK( cudaStreamSynchronize( s.cs ) );
			tok = static_cast<float>( sample_logits( logits, opt, rng ) );
		}
		CUDA_CHECK( cudaMemcpyAsync( s.d_tok, &tok, sizeof( float ), cudaMemcpyHostToDevice, s.cs ) );
		if ( s.tp_n > 1 ) { s.tp_comm.sum_f32( s.d_tok, 1, s.cs ); }
		CUDA_CHECK( cudaMemcpyAsync( &tok, s.d_tok, sizeof( float ), cudaMemcpyDeviceToHost, s.cs ) );
		CUDA_CHECK( cudaStreamSynchronize( s.cs ) );
		const int id = static_cast<int>( std::lround( tok ) );
		if ( id == kImEnd || id == kEndOfText ) { break; }
		out_tokens->push_back( id );
		if ( step + 1 == opt.max_new_tokens ) { break; }
		// the new token's embedding is the one row of the stream; its rotary position is the next text position on all three axes
		for ( int c = 0; c < kH; ++c ) {
			const uint16_t raw = table[static_cast<size_t>( id ) * kH + c];
			if ( em->dtype == pod::DType::FP16 ) {
				half hv;
				std::memcpy( &hv, &raw, 2 );
				row[c] = __half2float( hv );
			} else {
				const uint32_t bits = static_cast<uint32_t>( raw ) << 16;
				std::memcpy( &row[c], &bits, 4 );
			}
		}
		const int pos3[3] = { next_pos, next_pos, next_pos };
		pod::BatchRows br = pod::BatchRows::causal( 1, kv_pos, 0 );
		CUDA_CHECK( cudaMemcpyAsync( s.x, row.data(), kH * sizeof( float ), cudaMemcpyHostToDevice, s.cs ) );
		CUDA_CHECK( cudaMemcpyAsync( s.d_pos, pos3, sizeof( pos3 ), cudaMemcpyHostToDevice, s.cs ) );
		CUDA_CHECK( cudaMemcpyAsync( s.d_brows + s.n_chunks, &br, br.upload_bytes(), cudaMemcpyHostToDevice, s.cs ) );
		CUDA_CHECK( cudaStreamSynchronize( s.cs ) ); // the host buffers are reused next step
		for ( int i = 0; i < s.total_layers; ++i ) { s.run_layer_x( s.xl[i], 1, i, 2 ); }
		last = s.x;
		++kv_pos;
		++next_pos;
	}
	CUDA_CHECK( cudaStreamSynchronize( s.cs ) );
	CUDA_CHECK( cudaGetLastError() );
	return true;
}

} // namespace engine::video
