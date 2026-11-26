// The TTS backbone (vall_e/model.py V3.hidden) for one unpadded row: embeddings, pre-norm blocks of bidirectional
// gated attention (QK RMS norm, RoPE) and a SwiGLU MLP, then the final RMS norm. Attention is the fused WMMA kernel on CUDA
// (h3_attention: head_dim 64 or 128); on ROCm, whose WMMA accumulator layout that kernel doesn't follow, scores by batched GEMM,
// a row softmax and a second batched GEMM (rows are at most MAX_LEN long).
#include "engine/tts/tts_model.h"
#include "engine/core/blas_check.h"
#include "engine/core/log.h"
#include "engine/kernels/dense.cuh"
#include "engine/video/h3_attn.h"
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
// The attention scores (FP32) and probabilities (FP16) of the heads taken at once: all 24 of v2.0's for rows up to ~960 positions, 3 at
// MAX_LEN (one head needs 39 MiB there)
static constexpr size_t ATTN_BYTES = 128ull << 20;

static bool fused_attention( int head_dim ) {
#if defined(USE_ROCM)
	( void ) head_dim;
	return false;
#else
	return head_dim == 64 || head_dim == 128;
#endif
}

int TtsRow::length() const {
	return static_cast<int>( text.size() ) + static_cast<int>( prompt.size() / 64 ) + ( len_query ? 1 : static_cast<int>( masked.size() ) );
}

void TtsModel::alloc_workspace() {
	const size_t L = MAX_LEN, d = cfg_.d_model, F = cfg_.ffn(), V = cfg_.values();
	static_assert( ATTN_BYTES >= static_cast<size_t>( MAX_LEN ) * MAX_LEN * ( 4 + 2 ), "one head's scores at MAX_LEN" );
	const bool scores = !fused_attention( cfg_.head_dim() );
	const size_t bytes = L * d * 4 * 2 + ( scores ? ATTN_BYTES : 0 ) + L * d * 2 * 9 + L * 3 * d * 2 + L * 2 * F * 2 + L * F * 2 + L * V * 3 + L * 16 + ( 64 << 20 )
		+ 3 * L * ( d * 6 + cfg_.flow_width * 4 * 12 + V * 16 ) + ( 16 << 20 );  // + the flow head's (tts_flow.cu)
	ENGINE_ASSERT( engine::core::arena_create( ws_arena_, device_, bytes ), "tts: can't allocate the workspace" );
	auto A = [&]( size_t n ) { return engine::core::arena_alloc( ws_arena_, n ); };
	ws_.x = static_cast<float*>( A( L * d * 4 ) );
	ws_.a = static_cast<float*>( A( L * d * 4 ) );
	if ( scores ) {
		ws_.S = static_cast<float*>( A( ATTN_BYTES / 6 * 4 ) );
		ws_.P = static_cast<half*>( A( ATTN_BYTES / 6 * 2 ) );
	}
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
	ws_.pos = static_cast<int*>( A( L * 4 ) );
	ws_.kind = static_cast<uint8_t*>( A( L ) );
	ws_.masked = static_cast<uint8_t*>( A( L ) );
	const int rot = static_cast<int>( cfg_.head_dim() * cfg_.rot_frac ) / 2 * 2;
	ws_.rope_cs = static_cast<float*>( A( L * rot / 2 * 4 ) );
	ws_.rope_sn = static_cast<float*>( A( L * rot / 2 * 4 ) );
	if ( rot ) { kern::k_rope_table<<<256, 256>>>( ws_.rope_cs, ws_.rope_sn, MAX_LEN, rot, cfg_.rope_theta ); }
	alloc_flow_workspace();
	cublasHandle_t h;
	CUBLAS_CHECK( cublasCreate( &h ) );
	blas_ = h;
}

// C [M, N] = A [M, K] @ W [N, K]^T (row-major; FP32 or FP16 out, FP32 accumulation)
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
		engine::dense::k_convert<<<static_cast<int>( std::min<size_t>( 4096, ( n + 255 ) / 256 ) ), 256>>>( c16_, static_cast<float*>( C ), n );
		return;
	}
#endif
	CUBLAS_CHECK( cublasGemmEx( static_cast<cublasHandle_t>( blas_ ), CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16F, K, A, CUDA_R_16F, K, &beta, C,
		c32 ? CUDA_R_32F : CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT ) );
}

void TtsModel::hidden( const TtsRow& row, float* out, std::vector<std::vector<float>>* taps ) {
	const TtsRow* r = &row;
	hidden( &r, 1, out, taps );
}

void TtsModel::hidden( const TtsRow* const* rows, int n, float* out, std::vector<std::vector<float>>* taps ) {
	const int d = cfg_.d_model, H = cfg_.heads, hd = cfg_.head_dim(), F = cfg_.ffn(), V = cfg_.values();
	// the rows one after another: the linears and norms see them as one block of positions, attention stays within each row
	std::vector<int> start( n + 1, 0 );
	for ( int r = 0; r < n; ++r ) {
		start[r + 1] = start[r] + rows[r]->length();
		ENGINE_ASSERT( rows[r]->mode == rows[0]->mode, "tts: rows of one pass share their mode" );
	}
	const int L = start[n];
	ENGINE_ASSERT( L <= MAX_LEN, "tts: rows longer than MAX_LEN" );
	cublasHandle_t blas = static_cast<cublasHandle_t>( blas_ );

	// host-side layout of the rows
	std::vector<int> ids( L, 0 ), pos( L );
	std::vector<uint8_t> kind( L ), masked( L, 0 ), digits( static_cast<size_t>( L ) * V, 0 );
	for ( int r = 0; r < n; ++r ) {
		const TtsRow& row = *rows[r];
		const int s = start[r], Lt = static_cast<int>( row.text.size() ), Lp = static_cast<int>( row.prompt.size() / V );
		const int La = row.len_query ? 1 : static_cast<int>( row.masked.size() );
		for ( int i = 0; i < Lt + Lp + La; ++i ) { pos[s + i] = i; }
		for ( int i = 0; i < Lt; ++i ) { ids[s + i] = row.text[i]; kind[s + i] = 1; }
		for ( int i = 0; i < Lp; ++i ) { kind[s + Lt + i] = 2; }
		std::copy( row.prompt.begin(), row.prompt.end(), digits.begin() + static_cast<size_t>( s + Lt ) * V );
		if ( row.len_query ) {
			kind[s + Lt + Lp] = 5;
		} else {
			for ( int i = 0; i < La; ++i ) { kind[s + Lt + Lp + i] = 3; masked[s + Lt + Lp + i] = row.masked[i]; }
			std::copy( row.target.begin(), row.target.end(), digits.begin() + static_cast<size_t>( s + Lt + Lp ) * V );
		}
	}
	CUDA_CHECK( cudaMemcpy( ws_.ids, ids.data(), L * 4, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( ws_.pos, pos.data(), L * 4, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( ws_.kind, kind.data(), L, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( ws_.masked, masked.data(), L, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( ws_.digits, digits.data(), digits.size(), cudaMemcpyHostToDevice ) );

	// embeddings: audio_in(values) + bias everywhere (the text / mask / query positions are overwritten), then the mix
	kern::k_digits_values<<<( L * V + 255 ) / 256, 256>>>( ws_.digits, ws_.vals, L * V );
	gemm( ws_.vals, w_.audio_in, ws_.a, L, d, V, true );
	engine::dense::k_add_bias<<<256, 256>>>( ws_.a, w_.audio_in_b, L, d );
	kern::k_embed_mix<<<L, 256>>>( ws_.x, ws_.a, w_.text, ws_.ids, ws_.kind, ws_.masked, w_.mask_emb, w_.len_q, w_.segment,
		w_.mode + static_cast<size_t>( rows[0]->mode ) * d, d );

	auto tap = [&]( const float* src ) {
		if ( !taps ) { return; }
		taps->emplace_back( static_cast<size_t>( L ) * d );
		CUDA_CHECK( cudaMemcpy( taps->back().data(), src, static_cast<size_t>( L ) * d * 4, cudaMemcpyDeviceToHost ) );
	};
	const int rot = static_cast<int>( hd * cfg_.rot_frac ) / 2 * 2;
	const float scale = 1.0f / std::sqrt( static_cast<float>( hd ) );
	const float one = 1.0f, zero = 0.0f;
	const bool fused = fused_attention( hd );
	const float* branch = nullptr;   // the MLP's output, added into the residual stream by the next norm (k_norm_mod adds it first)
	for ( const TtsBlock& b : w_.blocks ) {
		// attention
		engine::dense::k_norm_mod<half><<<L, 256>>>( ws_.x, branch, nullptr, 0, b.n1, nullptr, nullptr, 0, nullptr, d, EPS_FP32, ws_.xn );
		tap( ws_.x );
		gemm( ws_.xn, b.qkv, ws_.qkv, L, 3 * d, d, false );
		kern::k_qk_norm_rope<<<( L * H + kern::QK_WARPS - 1 ) / kern::QK_WARPS, 32 * kern::QK_WARPS, kern::QK_WARPS * 2 * hd * sizeof( float )>>>( ws_.qkv,
			b.q_norm, b.k_norm, ws_.q, ws_.k, ws_.v, ws_.pos, L, H, hd, rot, ws_.rope_cs, ws_.rope_sn, EPS_QK, !fused );
		for ( int r = 0; r < n; ++r ) {
			const int s = start[r], Lr = start[r + 1] - start[r];
			if ( fused ) {
				const size_t at = static_cast<size_t>( s ) * d;   // [L, H, hd]
				ENGINE_ASSERT( engine::video::h3_attention( ws_.q + at, ws_.k + at, ws_.v + at, ws_.o + at, Lr, Lr, H, scale, false, nullptr, hd ), "tts: fused attention" );
				continue;
			}
			const int hg = std::min( H, static_cast<int>( ATTN_BYTES / 6 / ( static_cast<size_t>( Lr ) * Lr ) ) );   // heads per batch of scores
			for ( int h0 = 0; h0 < H; h0 += hg ) {
				const int nh = std::min( hg, H - h0 );
				const size_t at = ( static_cast<size_t>( h0 ) * L + s ) * hd;   // [H, L, hd]: this row's positions of head h0, heads L * hd apart
				// S[h] = Q[h] K[h]^T, row-major [Lr(q), Lr(k)]
#if defined(USE_ROCM)
				// FP16 scores, already scaled: hipBLAS takes a kernel without the matrix cores for FP32 output (Tensile HSS MT64x32x8, a fifth of
				// a generation's GPU time with its softmax); the scaled scores of QK-normed heads are small, and FP16 holds them to ~1e-3
				half* S16 = reinterpret_cast<half*>( ws_.S );
				CUBLAS_CHECK( cublasGemmStridedBatchedEx( blas, CUBLAS_OP_T, CUBLAS_OP_N, Lr, Lr, hd, &scale, ws_.k + at, CUDA_R_16F, hd, static_cast<long long>( L ) * hd,
					ws_.q + at, CUDA_R_16F, hd, static_cast<long long>( L ) * hd, &zero, S16, CUDA_R_16F, Lr, static_cast<long long>( Lr ) * Lr, nh, CUBLAS_COMPUTE_32F,
					CUBLAS_GEMM_DEFAULT ) );
				kern::k_softmax_h<<<nh * Lr, 256>>>( S16, ws_.P, Lr );
#else
				CUBLAS_CHECK( cublasGemmStridedBatchedEx( blas, CUBLAS_OP_T, CUBLAS_OP_N, Lr, Lr, hd, &one, ws_.k + at, CUDA_R_16F, hd, static_cast<long long>( L ) * hd,
					ws_.q + at, CUDA_R_16F, hd, static_cast<long long>( L ) * hd, &zero, ws_.S, CUDA_R_32F, Lr, static_cast<long long>( Lr ) * Lr, nh, CUBLAS_COMPUTE_32F,
					CUBLAS_GEMM_DEFAULT ) );
				kern::k_softmax<<<nh * Lr, 256>>>( ws_.S, ws_.P, Lr, scale );
#endif
				// O[h] = P[h] V[h], row-major [Lr, hd]
				CUBLAS_CHECK( cublasGemmStridedBatchedEx( blas, CUBLAS_OP_N, CUBLAS_OP_N, hd, Lr, Lr, &one, ws_.v + at, CUDA_R_16F, hd, static_cast<long long>( L ) * hd,
					ws_.P, CUDA_R_16F, Lr, static_cast<long long>( Lr ) * Lr, &zero, ws_.o + at, CUDA_R_16F, hd, static_cast<long long>( L ) * hd, nh, CUBLAS_COMPUTE_32F,
					CUBLAS_GEMM_DEFAULT ) );
			}
		}
		if ( b.gate ) { gemm( ws_.xn, b.gate, ws_.g, L, d, d, false ); }
		kern::k_heads_gate<<<L, 256>>>( ws_.o, b.gate ? ws_.g : nullptr, ws_.y, L, H, hd, !fused );
		gemm( ws_.y, b.out, ws_.a, L, d, d, true );
		// MLP (the attention's output added first)
		engine::dense::k_norm_mod<half><<<L, 256>>>( ws_.x, ws_.a, nullptr, 0, b.n2, nullptr, nullptr, 0, nullptr, d, EPS_FP32, ws_.xn );
		gemm( ws_.xn, b.up, ws_.up, L, 2 * F, d, false );
		kern::k_swiglu<<<L, 256>>>( ws_.up, ws_.act, F );
		gemm( ws_.act, b.down, ws_.a, L, d, F, true );
		branch = ws_.a;
	}
	engine::dense::k_norm_mod<float><<<L, 256>>>( ws_.x, branch, nullptr, 0, w_.norm, nullptr, nullptr, 0, nullptr, d, EPS_FP32, out );
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
