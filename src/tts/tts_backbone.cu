// The TTS backbone (vall_e/model.py V3.hidden) for one unpadded row: embeddings, pre-norm blocks of bidirectional
// gated attention (QK RMS norm, RoPE) and a SwiGLU MLP, then the final RMS norm. Attention is the plain version for now:
// scores by batched GEMM, a row softmax, a second batched GEMM (rows are at most MAX_LEN long).
#include "engine/tts/tts_model.h"
#include "engine/core/blas_check.h"
#include "engine/core/log.h"
#include "engine/kernels/dense.cuh"
#include "tts_kernels.cuh"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace engine::tts {

// nn.RMSNorm's default eps is the input dtype's machine epsilon: the residual stream is FP32 (1.19e-7), while under the
// bf16 autocast the model was trained and is served with, the q / k norms see bf16 projections (2^-7)
static constexpr float EPS_FP32 = 1.1920929e-7f;
static constexpr float EPS_QK = 0.0078125f;

int TtsRow::length() const {
	return static_cast<int>( text.size() ) + static_cast<int>( prompt.size() / 64 ) + ( len_query ? 1 : static_cast<int>( masked.size() ) );
}

void TtsModel::alloc_workspace() {
	const size_t L = MAX_LEN, d = cfg_.d_model, H = cfg_.heads, F = cfg_.ffn(), V = cfg_.values();
	const size_t bytes = L * d * 4 * 2 + H * L * L * ( 4 + 2 ) + L * d * 2 * 9 + L * 3 * d * 2 + L * 2 * F * 2 + L * F * 2 + L * V * 3 + L * 16 + ( 64 << 20 )
		+ 3 * L * ( d * 6 + cfg_.flow_width * 4 * 12 + V * 16 ) + ( 16 << 20 );  // + the flow head's (tts_flow.cu)
	ENGINE_ASSERT( engine::core::arena_create( ws_arena_, 0, bytes ), "tts: can't allocate the workspace" );
	auto A = [&]( size_t n ) { return engine::core::arena_alloc( ws_arena_, n ); };
	ws_.x = static_cast<float*>( A( L * d * 4 ) );
	ws_.a = static_cast<float*>( A( L * d * 4 ) );
	ws_.S = static_cast<float*>( A( H * L * L * 4 ) );
	ws_.P = static_cast<half*>( A( H * L * L * 2 ) );
	ws_.xn = static_cast<half*>( A( L * d * 2 ) );
	ws_.qkv = static_cast<half*>( A( L * 3 * d * 2 ) );
	ws_.q = static_cast<half*>( A( L * d * 2 ) );
	ws_.k = static_cast<half*>( A( L * d * 2 ) );
	ws_.v = static_cast<half*>( A( L * d * 2 ) );
	ws_.o = static_cast<half*>( A( L * d * 2 ) );
	ws_.g = static_cast<half*>( A( L * d * 2 ) );
	ws_.y = static_cast<half*>( A( L * d * 2 ) );
	ws_.up = static_cast<half*>( A( L * 2 * F * 2 ) );
	ws_.act = static_cast<half*>( A( L * F * 2 ) );
	ws_.vals = static_cast<half*>( A( L * V * 2 ) );
	ws_.digits = static_cast<uint8_t*>( A( L * V ) );
	ws_.ids = static_cast<int*>( A( L * 4 ) );
	ws_.kind = static_cast<uint8_t*>( A( L ) );
	ws_.masked = static_cast<uint8_t*>( A( L ) );
	alloc_flow_workspace();
	cublasHandle_t h;
	CUBLAS_CHECK( cublasCreate( &h ) );
	blas_ = h;
}

// C [M, N] = A [M, K] @ W [N, K]^T (row-major; FP32 or FP16 out, FP32 accumulation)
namespace {
__global__ void k_half_to_float( const half* x, float* y, size_t n ) {
	for ( size_t i = blockIdx.x * static_cast<size_t>( blockDim.x ) + threadIdx.x; i < n; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) { y[i] = __half2float( x[i] ); }
}
}  // namespace

void TtsModel::gemm( const half* A, const half* W, void* C, int M, int N, int K, bool c32 ) {
	const float alpha = 1.0f, beta = 0.0f;
#if defined(USE_ROCM)
	if ( c32 ) {
		// hipBLAS's FP16-in / FP32-out GEMMs take kernels without the matrix cores on RDNA3 (Tensile HSS MT64x32x8: half of a generation's GPU
		// time): the FP16-out GEMM (WMMA), then widened. The result is rounded to FP16 once, as vall_e's bf16 autocast rounds it to bf16
		const size_t n = static_cast<size_t>( M ) * N;
		if ( n > c16_elems_ ) {
			if ( c16_ ) { cudaFree( c16_ ); }
			CUDA_CHECK( cudaMalloc( &c16_, n * sizeof( half ) ) );
			c16_elems_ = n;
		}
		CUBLAS_CHECK( cublasGemmEx( static_cast<cublasHandle_t>( blas_ ), CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16F, K, A, CUDA_R_16F, K, &beta, c16_,
			CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT ) );
		k_half_to_float<<<static_cast<int>( std::min<size_t>( 4096, ( n + 255 ) / 256 ) ), 256>>>( c16_, static_cast<float*>( C ), n );
		return;
	}
#endif
	CUBLAS_CHECK( cublasGemmEx( static_cast<cublasHandle_t>( blas_ ), CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16F, K, A, CUDA_R_16F, K, &beta, C,
		c32 ? CUDA_R_32F : CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT ) );
}

void TtsModel::hidden( const TtsRow& row, float* out, std::vector<std::vector<float>>* taps ) {
	const int d = cfg_.d_model, H = cfg_.heads, hd = cfg_.head_dim(), F = cfg_.ffn(), V = cfg_.values();
	const int Lt = static_cast<int>( row.text.size() ), Lp = static_cast<int>( row.prompt.size() / V );
	const int La = row.len_query ? 1 : static_cast<int>( row.masked.size() );
	const int L = Lt + Lp + La;
	ENGINE_ASSERT( L <= MAX_LEN, "tts: row longer than MAX_LEN" );
	cublasHandle_t blas = static_cast<cublasHandle_t>( blas_ );

	// host-side layout of the row
	std::vector<int> ids( L, 0 );
	std::vector<uint8_t> kind( L ), masked( L, 0 ), digits( static_cast<size_t>( L ) * V, 0 );
	for ( int i = 0; i < Lt; ++i ) { ids[i] = row.text[i]; kind[i] = 1; }
	for ( int i = 0; i < Lp; ++i ) { kind[Lt + i] = 2; }
	std::copy( row.prompt.begin(), row.prompt.end(), digits.begin() + static_cast<size_t>( Lt ) * V );
	if ( row.len_query ) {
		kind[Lt + Lp] = 5;
	} else {
		for ( int i = 0; i < La; ++i ) { kind[Lt + Lp + i] = 3; masked[Lt + Lp + i] = row.masked[i]; }
		std::copy( row.target.begin(), row.target.end(), digits.begin() + static_cast<size_t>( Lt + Lp ) * V );
	}
	CUDA_CHECK( cudaMemcpy( ws_.ids, ids.data(), L * 4, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( ws_.kind, kind.data(), L, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( ws_.masked, masked.data(), L, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( ws_.digits, digits.data(), digits.size(), cudaMemcpyHostToDevice ) );

	// embeddings: audio_in(values) + bias everywhere (the text / mask / query positions are overwritten), then the mix
	kern::k_digits_values<<<( L * V + 255 ) / 256, 256>>>( ws_.digits, ws_.vals, L * V );
	gemm( ws_.vals, w_.audio_in, ws_.a, L, d, V, true );
	engine::dense::k_add_bias<<<256, 256>>>( ws_.a, w_.audio_in_b, L, d );
	kern::k_embed_mix<<<L, 256>>>( ws_.x, ws_.a, w_.text, ws_.ids, ws_.kind, ws_.masked, w_.mask_emb, w_.len_q, w_.segment,
		w_.mode + static_cast<size_t>( row.mode ) * d, d );

	auto tap = [&]( const float* src ) {
		if ( !taps ) { return; }
		taps->emplace_back( static_cast<size_t>( L ) * d );
		CUDA_CHECK( cudaMemcpy( taps->back().data(), src, static_cast<size_t>( L ) * d * 4, cudaMemcpyDeviceToHost ) );
	};
	const int rot = static_cast<int>( hd * cfg_.rot_frac ) / 2 * 2;
	const float scale = 1.0f / std::sqrt( static_cast<float>( hd ) );
	const float one = 1.0f, zero = 0.0f;
	for ( const TtsBlock& b : w_.blocks ) {
		tap( ws_.x );
		// attention
		engine::dense::k_norm_mod<half><<<L, 256>>>( ws_.x, nullptr, nullptr, 0, b.n1, nullptr, nullptr, 0, nullptr, d, EPS_FP32, ws_.xn );
		gemm( ws_.xn, b.qkv, ws_.qkv, L, 3 * d, d, false );
		kern::k_qk_norm_rope<<<L * H, hd, 2 * hd * sizeof( float )>>>( ws_.qkv, b.q_norm, b.k_norm, ws_.q, ws_.k, ws_.v, L, H, hd, rot, cfg_.rope_theta, EPS_QK );
		// S[h] = Q[h] K[h]^T, row-major [L(q), L(k)]
		CUBLAS_CHECK( cublasGemmStridedBatchedEx( blas, CUBLAS_OP_T, CUBLAS_OP_N, L, L, hd, &one, ws_.k, CUDA_R_16F, hd, static_cast<long long>( L ) * hd,
			ws_.q, CUDA_R_16F, hd, static_cast<long long>( L ) * hd, &zero, ws_.S, CUDA_R_32F, L, static_cast<long long>( L ) * L, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT ) );
		kern::k_softmax<<<H * L, 256>>>( ws_.S, ws_.P, L, scale );
		// O[h] = P[h] V[h], row-major [L, hd]
		CUBLAS_CHECK( cublasGemmStridedBatchedEx( blas, CUBLAS_OP_N, CUBLAS_OP_N, hd, L, L, &one, ws_.v, CUDA_R_16F, hd, static_cast<long long>( L ) * hd,
			ws_.P, CUDA_R_16F, L, static_cast<long long>( L ) * L, &zero, ws_.o, CUDA_R_16F, hd, static_cast<long long>( L ) * hd, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT ) );
		if ( b.gate ) { gemm( ws_.xn, b.gate, ws_.g, L, d, d, false ); }
		kern::k_heads_gate<<<L, 256>>>( ws_.o, b.gate ? ws_.g : nullptr, ws_.y, L, H, hd );
		gemm( ws_.y, b.out, ws_.a, L, d, d, true );
		engine::dense::k_add_gated<<<L, 256>>>( ws_.x, ws_.a, nullptr, nullptr, 0, nullptr, d );
		// MLP
		engine::dense::k_norm_mod<half><<<L, 256>>>( ws_.x, nullptr, nullptr, 0, b.n2, nullptr, nullptr, 0, nullptr, d, EPS_FP32, ws_.xn );
		gemm( ws_.xn, b.up, ws_.up, L, 2 * F, d, false );
		kern::k_swiglu<<<L, 256>>>( ws_.up, ws_.act, F );
		gemm( ws_.act, b.down, ws_.a, L, d, F, true );
		engine::dense::k_add_gated<<<L, 256>>>( ws_.x, ws_.a, nullptr, nullptr, 0, nullptr, d );
	}
	engine::dense::k_norm_mod<float><<<L, 256>>>( ws_.x, nullptr, nullptr, 0, w_.norm, nullptr, nullptr, 0, nullptr, d, EPS_FP32, out );
	CUDA_CHECK( cudaGetLastError() );
	tap( out );
}

int TtsModel::predict_length( const std::vector<int>& text, const std::vector<uint8_t>& prompt, float* log_patches ) {
	ENGINE_ASSERT( w_.length_w, "tts: this export has no length head" );
	TtsRow row;
	row.text = text;
	row.prompt = prompt;
	row.mode = 2;
	row.len_query = true;
	hidden( row, fw_.h, nullptr );
	// one d-long dot product: the query's hidden state and the head's weights on the host, in double as the reference's FP32 sum
	const int d = cfg_.d_model;
	std::vector<float> h( d ), w( d );
	float bias = 0.0f;
	CUDA_CHECK( cudaMemcpy( h.data(), fw_.h + static_cast<size_t>( row.length() - 1 ) * d, d * 4, cudaMemcpyDeviceToHost ) );
	CUDA_CHECK( cudaMemcpy( w.data(), w_.length_w, d * 4, cudaMemcpyDeviceToHost ) );
	CUDA_CHECK( cudaMemcpy( &bias, w_.length_b, 4, cudaMemcpyDeviceToHost ) );
	double s = bias;
	for ( int i = 0; i < d; ++i ) { s += static_cast<double>( h[i] ) * w[i]; }
	if ( log_patches ) { *log_patches = static_cast<float>( s ); }
	return std::max( 1, static_cast<int>( std::nearbyint( std::exp( s ) * cfg_.patch ) ) );
}

} // namespace engine::tts
