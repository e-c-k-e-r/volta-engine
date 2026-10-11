// H3 vision tower (see h3_vision.h).
#include "engine/video/h3_vision.h"
#include "engine/core/arena.h"
#include "engine/core/log.h"
#include "engine/loader/safetensors.h"
#include "engine/models/vision_tower.h"
#include "engine/modules/vision.h"
#include "h3_kernels.cuh"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
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

constexpr int kDepth = 27, kH = 1152, kHeads = 16, kHd = 72, kMlp = 4304, kOut = 5120, kMerge = 4, kPatchDim = 1536, kSide = 48;
constexpr float kEps = 1e-6f;
const int kDeepstack[3] = { 8, 16, 24 };

struct Blk {
	const half *qkv = nullptr, *proj = nullptr, *fc1 = nullptr, *fc2 = nullptr;
	const half *qkv_b = nullptr, *proj_b = nullptr, *fc1_b = nullptr, *fc2_b = nullptr, *n1w = nullptr, *n1b = nullptr, *n2w = nullptr, *n2b = nullptr;
};

struct Merger {
	const half *norm_w = nullptr, *norm_b = nullptr, *fc1 = nullptr, *fc1_b = nullptr, *fc2 = nullptr, *fc2_b = nullptr;
};

static __global__ void k_f32_to_f32_dummy() {}

} // namespace

struct H3VisionTower::Impl {
	engine::loader::SafeTensorsRepo repo;
	pod::MemoryArena warena{}, sarena{};
	cublasHandle_t blas = nullptr;
	cudaStream_t stream = nullptr;
	int max_patches = 0;
	std::vector<Blk> blocks;
	Merger merger, deep[3];
	const half *patch_w = nullptr, *patch_b = nullptr, *pos_table = nullptr;
	void* staging = nullptr;
	size_t staging_bytes = 0;
	float *d_hidden = nullptr, *d_cos = nullptr, *d_sin = nullptr, *d_patch32 = nullptr;
	half *d_patches = nullptr, *d_pos = nullptr, *d_ln = nullptr, *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_attn = nullptr, *d_tmp = nullptr, *d_mlp = nullptr, *d_merged = nullptr, *d_f1 = nullptr, *d_o = nullptr;

	template <typename T>
	T* alloc( pod::MemoryArena& a, size_t n ) { return static_cast<T*>( engine::core::arena_alloc( a, n * sizeof( T ) ) ); }

	const half* fp16( const std::string& name, const size_t slice_off = 0, const size_t slice_n = 0 ) {
		const pod::SafeTensorMeta* m = repo.find( name.c_str() );
		if ( !m ) { LOG_ERROR( "H3 vision: missing %s", name.c_str() ); return nullptr; }
		const size_t n = m->size_bytes / 2;
		ENGINE_ASSERT( m->dtype == pod::DType::BF16 || m->dtype == pod::DType::FP16, "H3 vision: BF16 or FP16 expected" );
		half* d = alloc<half>( warena, n );
		if ( n * 2 > staging_bytes ) {
			if ( staging ) { cudaFree( staging ); }
			staging_bytes = n * 2;
			CUDA_CHECK( cudaMalloc( &staging, staging_bytes ) );
		}
		CUDA_CHECK( cudaMemcpy( staging, repo.cpu_data( name.c_str() ), n * 2, cudaMemcpyHostToDevice ) );
		if ( m->dtype == pod::DType::BF16 ) {
			kern::k_convert<__nv_bfloat16, half><<<1024, 256>>>( static_cast<const __nv_bfloat16*>( staging ), d, n );
		} else {
			CUDA_CHECK( cudaMemcpy( d, staging, n * 2, cudaMemcpyDeviceToDevice ) );
		}
		CUDA_CHECK( cudaDeviceSynchronize() );
		( void )slice_off; ( void )slice_n;
		return d;
	}

	Merger load_merger( const std::string& p ) {
		Merger m;
		m.norm_w = fp16( p + ".norm.weight" ); m.norm_b = fp16( p + ".norm.bias" );
		m.fc1 = fp16( p + ".linear_fc1.weight" ); m.fc1_b = fp16( p + ".linear_fc1.bias" );
		m.fc2 = fp16( p + ".linear_fc2.weight" ); m.fc2_b = fp16( p + ".linear_fc2.bias" );
		return m;
	}

	void gemm( const half* A, const half* W, half* C, int M, int N, int K ) {
		const float alpha = 1.0f, beta = 0.0f;
		CUBLAS_CHECK( cublasGemmEx( blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16F, K, A, CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
			CUBLAS_GEMM_DEFAULT_TENSOR_OP ) );
	}
};

H3VisionTower::H3VisionTower() : p_( new Impl ) {}

H3VisionTower::~H3VisionTower() {
	if ( p_ ) {
		if ( p_->blas ) { cublasDestroy( p_->blas ); }
		if ( p_->warena.base_ptr ) { engine::core::arena_destroy( p_->warena ); }
		if ( p_->sarena.base_ptr ) { engine::core::arena_destroy( p_->sarena ); }
		if ( p_->staging ) { cudaFree( p_->staging ); }
		delete p_;
	}
}

int H3VisionTower::max_patches() const { return p_->max_patches; }

bool H3VisionTower::load( const char* dir, int device, int max_patches ) {
	Impl& s = *p_;
	s.max_patches = max_patches;
	CUDA_CHECK( cudaSetDevice( device ) );
	CUDA_CHECK( cudaStreamCreateWithFlags( &s.stream, cudaStreamNonBlocking ) );
	CUBLAS_CHECK( cublasCreate( &s.blas ) );
	CUBLAS_CHECK( cublasSetStream( s.blas, s.stream ) );
	CUBLAS_CHECK( cublasSetMathMode( s.blas, CUBLAS_DEFAULT_MATH ) );
	if ( !s.repo.open( dir ) ) { return false; }
	if ( !engine::core::arena_create( s.warena, device, 2ULL << 30 ) ) { return false; }
	s.patch_w = s.fp16( "model.visual.patch_embed.proj.weight" );
	s.patch_b = s.fp16( "model.visual.patch_embed.proj.bias" );
	s.pos_table = s.fp16( "model.visual.pos_embed.weight" );
	for ( int i = 0; i < kDepth; ++i ) {
		const std::string p = "model.visual.blocks." + std::to_string( i ) + ".";
		Blk b;
		b.n1w = s.fp16( p + "norm1.weight" ); b.n1b = s.fp16( p + "norm1.bias" );
		b.n2w = s.fp16( p + "norm2.weight" ); b.n2b = s.fp16( p + "norm2.bias" );
		b.qkv = s.fp16( p + "attn.qkv.weight" ); b.qkv_b = s.fp16( p + "attn.qkv.bias" );
		b.proj = s.fp16( p + "attn.proj.weight" ); b.proj_b = s.fp16( p + "attn.proj.bias" );
		b.fc1 = s.fp16( p + "mlp.linear_fc1.weight" ); b.fc1_b = s.fp16( p + "mlp.linear_fc1.bias" );
		b.fc2 = s.fp16( p + "mlp.linear_fc2.weight" ); b.fc2_b = s.fp16( p + "mlp.linear_fc2.bias" );
		if ( !b.qkv || !b.fc2_b ) { return false; }
		s.blocks.push_back( b );
	}
	s.merger = s.load_merger( "model.visual.merger" );
	for ( int i = 0; i < 3; ++i ) { s.deep[i] = s.load_merger( "model.visual.deepstack_merger_list." + std::to_string( i ) ); }
	if ( !s.deep[2].fc2_b ) { return false; }
	const size_t M = static_cast<size_t>( max_patches ), T = M / kMerge;
	const size_t need = M * kPatchDim * 4 + M * kPatchDim * 2 + M * kH * 4 + M * kHd * 4 * 2 + M * kH * 2 * 8 + M * kMlp * 2 + T * kH * kMerge * 2 * 2 + T * kOut * 2 + ( 32ULL << 20 );
	if ( !engine::core::arena_create( s.sarena, device, need ) ) { return false; }
	s.d_patch32 = s.alloc<float>( s.sarena, M * kPatchDim );
	s.d_patches = s.alloc<half>( s.sarena, M * kPatchDim );
	s.d_pos = s.alloc<half>( s.sarena, M * kH );
	s.d_hidden = s.alloc<float>( s.sarena, M * kH );
	s.d_cos = s.alloc<float>( s.sarena, M * kHd );
	s.d_sin = s.alloc<float>( s.sarena, M * kHd );
	s.d_ln = s.alloc<half>( s.sarena, M * kH );
	s.d_q = s.alloc<half>( s.sarena, M * kH );
	s.d_k = s.alloc<half>( s.sarena, M * kH );
	s.d_v = s.alloc<half>( s.sarena, M * kH );
	s.d_attn = s.alloc<half>( s.sarena, M * kH );
	s.d_tmp = s.alloc<half>( s.sarena, M * kH );
	s.d_mlp = s.alloc<half>( s.sarena, M * kMlp );
	s.d_merged = s.alloc<half>( s.sarena, T * kH * kMerge );
	s.d_f1 = s.alloc<half>( s.sarena, T * kH * kMerge );
	s.d_o = s.alloc<half>( s.sarena, T * kOut );
	return true;
}

bool H3VisionTower::encode( const std::vector<float>& patches, int gh, int gw, std::vector<float>* merged, std::vector<float>* deepstack ) {
	Impl& s = *p_;
	using engine::modules::Vision;
	const int m = gh * gw;
	if ( m < kMerge || m > s.max_patches || patches.size() != static_cast<size_t>( m ) * kPatchDim ) { return false; }
	cudaStream_t st = s.stream;
	std::vector<float> cs, sn;
	engine::models::vision_rotary_tables( gh, gw, 2, kHd, &cs, &sn );
	CUDA_CHECK( cudaMemcpyAsync( s.d_patch32, patches.data(), patches.size() * sizeof( float ), cudaMemcpyHostToDevice, st ) );
	CUDA_CHECK( cudaMemcpyAsync( s.d_cos, cs.data(), cs.size() * sizeof( float ), cudaMemcpyHostToDevice, st ) );
	CUDA_CHECK( cudaMemcpyAsync( s.d_sin, sn.data(), sn.size() * sizeof( float ), cudaMemcpyHostToDevice, st ) );
	Vision::f32_to_f16( s.d_patch32, s.d_patches, patches.size(), st );
	Vision::pos_interp( s.pos_table, s.d_pos, gh, gw, 2, kSide, kH, st );
	Vision::patch_embed( s.d_patches, s.patch_w, s.patch_b, s.d_pos, s.d_hidden, m, kH, kPatchDim, st );
	const float scale = 1.0f / std::sqrt( static_cast<float>( kHd ) );
	const int tokens = m / kMerge;
	std::vector<float> ds[3];
	auto run_merger = [&]( const Merger& g, bool postshuffle, half* out16 ) {
		if ( postshuffle ) {
			Vision::layer_norm( s.d_hidden, g.norm_w, g.norm_b, s.d_merged, tokens, kH * kMerge, kEps, st );
		} else {
			Vision::layer_norm_merge( s.d_hidden, g.norm_w, g.norm_b, s.d_merged, m, kH, kMerge, kEps, st );
		}
		s.gemm( s.d_merged, g.fc1, s.d_f1, tokens, kH * kMerge, kH * kMerge );
		Vision::bias_act( s.d_f1, g.fc1_b, tokens, kH * kMerge, Vision::ACT_GELU_ERF, st );
		s.gemm( s.d_f1, g.fc2, out16, tokens, kOut, kH * kMerge );
		Vision::bias_act( out16, g.fc2_b, tokens, kOut, Vision::ACT_NONE, st );
	};
	auto fetch = [&]( std::vector<float>* dst ) {
		std::vector<half> h( static_cast<size_t>( tokens ) * kOut );
		CUDA_CHECK( cudaMemcpyAsync( h.data(), s.d_o, h.size() * sizeof( half ), cudaMemcpyDeviceToHost, st ) );
		CUDA_CHECK( cudaStreamSynchronize( st ) );
		dst->resize( h.size() );
		for ( size_t i = 0; i < h.size(); ++i ) { ( *dst )[i] = __half2float( h[i] ); }
	};
	for ( int i = 0; i < kDepth; ++i ) {
		const Blk& b = s.blocks[i];
		Vision::layer_norm( s.d_hidden, b.n1w, b.n1b, s.d_ln, m, kH, kEps, st );
		const size_t ws = static_cast<size_t>( kH ) * kH;
		s.gemm( s.d_ln, b.qkv, s.d_q, m, kH, kH );
		s.gemm( s.d_ln, b.qkv + ws, s.d_k, m, kH, kH );
		s.gemm( s.d_ln, b.qkv + 2 * ws, s.d_v, m, kH, kH );
		Vision::qkv_bias_rope( s.d_q, s.d_k, s.d_v, b.qkv_b, b.qkv_b + kH, b.qkv_b + 2 * kH, s.d_cos, s.d_sin, m, kHeads, kHd, st );
		Vision::attention( s.d_q, s.d_k, s.d_v, s.d_attn, m, kHeads, kHd, scale, st );
		s.gemm( s.d_attn, b.proj, s.d_tmp, m, kH, kH );
		Vision::bias_residual( s.d_hidden, s.d_tmp, b.proj_b, m, kH, st );
		Vision::layer_norm( s.d_hidden, b.n2w, b.n2b, s.d_ln, m, kH, kEps, st );
		s.gemm( s.d_ln, b.fc1, s.d_mlp, m, kMlp, kH );
		Vision::bias_act( s.d_mlp, b.fc1_b, m, kMlp, Vision::ACT_GELU_TANH, st );
		s.gemm( s.d_mlp, b.fc2, s.d_tmp, m, kH, kMlp );
		Vision::bias_residual( s.d_hidden, s.d_tmp, b.fc2_b, m, kH, st );
		for ( int d = 0; d < 3; ++d ) {
			if ( i == kDeepstack[d] ) {
				run_merger( s.deep[d], true, s.d_o );
				fetch( &deepstack[d] );
			}
		}
	}
	run_merger( s.merger, false, s.d_o );
	fetch( merged );
	CUDA_CHECK( cudaGetLastError() );
	return true;
}

} // namespace engine::video
