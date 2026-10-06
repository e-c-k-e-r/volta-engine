#include "engine/core/arena.h"
#include "engine/modules/attention.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>

static float half_to_float( half h ) {
	return __half2float( h );
}

static half float_to_half( float f ) {
	return __float2half_rn( f );
}

int main() {
	std::printf( "========================================================\n" );
	std::printf( "   volta-engine Tier 2 Parity: Attention Module         \n" );
	std::printf( "========================================================\n" );

	CUDA_CHECK( cudaSetDevice( 0 ) );
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );

	pod::MemoryArena arena;
	bool ok = engine::core::arena_create( arena, 0, 128ULL * 1024 * 1024 );
	ENGINE_ASSERT( ok, "Failed to create arena" );

	// Test configurations: (q_heads, kv_heads, head_dim)
	struct AttnConfig {
		int q_heads;
		int kv_heads;
		int head_dim;
		int max_seq;
	} configs[] = {
		{ 6, 1, 256, 1024 }, // Qwen TP=4 shape
		{ 24, 4, 256, 1024 }, // Qwen single-GPU shape
		{ 16, 16, 128, 1024 }, // GLM head shape
	};

	for ( size_t c_idx = 0; c_idx < sizeof( configs ) / sizeof( configs[0] ); ++c_idx ) {
		const auto& cfg = configs[c_idx];
		std::printf( "[*] Testing Config %zu: q_heads=%d, kv_heads=%d, head_dim=%d...\n",
			c_idx + 1, cfg.q_heads, cfg.kv_heads, cfg.head_dim );

		int q_shape[2] = { 1, cfg.q_heads * cfg.head_dim };
		int k_shape[2] = { 1, cfg.kv_heads * cfg.head_dim };
		int v_shape[2] = { 1, cfg.kv_heads * cfg.head_dim };
		int out_shape[2] = { 1, cfg.q_heads * cfg.head_dim };

		pod::TensorView d_q = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );
		pod::TensorView d_k = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, k_shape, 2 );
		pod::TensorView d_v = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, v_shape, 2 );
		pod::TensorView d_gate = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );
		pod::TensorView d_out = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, out_shape, 2 );

		size_t cache_elems = static_cast<size_t>( cfg.max_seq ) * cfg.kv_heads * cfg.head_dim;
		half* d_k_cache = static_cast<half*>( engine::core::arena_alloc( arena, cache_elems * sizeof( half ) ) );
		half* d_v_cache = static_cast<half*>( engine::core::arena_alloc( arena, cache_elems * sizeof( half ) ) );
		CUDA_CHECK( cudaMemset( d_k_cache, 0, cache_elems * sizeof( half ) ) );
		CUDA_CHECK( cudaMemset( d_v_cache, 0, cache_elems * sizeof( half ) ) );

		std::vector<float> cpu_k_cache( cache_elems, 0.0f );
		std::vector<float> cpu_v_cache( cache_elems, 0.0f );

		// Simulate 32 sequential decoding steps
		for ( int pos = 0; pos < 32; ++pos ) {
			std::vector<half> h_q( cfg.q_heads * cfg.head_dim );
			std::vector<half> h_k( cfg.kv_heads * cfg.head_dim );
			std::vector<half> h_v( cfg.kv_heads * cfg.head_dim );
			std::vector<half> h_gate( cfg.q_heads * cfg.head_dim );

			for ( size_t i = 0; i < h_q.size(); ++i ) {
				h_q[i] = float_to_half( std::sin( static_cast<float>( pos * 1000 + i ) * 0.02f ) );
			}
			for ( size_t i = 0; i < h_k.size(); ++i ) {
				float val = std::cos( static_cast<float>( pos * 700 + i ) * 0.03f );
				h_k[i] = float_to_half( val );
				cpu_k_cache[pos * ( cfg.kv_heads * cfg.head_dim ) + i] = val;
			}
			for ( size_t i = 0; i < h_v.size(); ++i ) {
				float val = std::sin( static_cast<float>( pos * 500 + i ) * 0.04f );
				h_v[i] = float_to_half( val );
				cpu_v_cache[pos * ( cfg.kv_heads * cfg.head_dim ) + i] = val;
			}
			for ( size_t i = 0; i < h_gate.size(); ++i ) {
				h_gate[i] = float_to_half( 0.2f * std::cos( static_cast<float>( i ) * 0.05f ) );
			}

			CUDA_CHECK( cudaMemcpyAsync( d_q.data, h_q.data(), h_q.size() * sizeof( half ), cudaMemcpyHostToDevice, stream ) );
			CUDA_CHECK( cudaMemcpyAsync( d_k.data, h_k.data(), h_k.size() * sizeof( half ), cudaMemcpyHostToDevice, stream ) );
			CUDA_CHECK( cudaMemcpyAsync( d_v.data, h_v.data(), h_v.size() * sizeof( half ), cudaMemcpyHostToDevice, stream ) );
			CUDA_CHECK( cudaMemcpyAsync( d_gate.data, h_gate.data(), h_gate.size() * sizeof( half ), cudaMemcpyHostToDevice, stream ) );

			// Store K, V in cache
			const auto rows = pod::BatchRows::causal( 1, pos, 0 );
			engine::modules::Attention::store_kv_cache(
				d_k_cache, d_v_cache, d_k, d_v, 0, 0, rows, cfg.kv_heads, cfg.head_dim, cfg.max_seq, stream
			);

			// Forward decode
			engine::modules::Attention::forward_decode(
				d_q, d_k_cache, d_v_cache, d_gate, d_out, 0, 0, rows, cfg.q_heads, cfg.kv_heads, cfg.head_dim, cfg.max_seq, stream
			);
			CUDA_CHECK( cudaStreamSynchronize( stream ) );

			// Compute exact CPU reference
			std::vector<float> h_ref_out( cfg.q_heads * cfg.head_dim, 0.0f );
			float scale = 1.0f / std::sqrt( static_cast<float>( cfg.head_dim ) );
			int heads_per_kv = cfg.q_heads / cfg.kv_heads;

			for ( int qh = 0; qh < cfg.q_heads; ++qh ) {
				int kvh = qh / heads_per_kv;
				std::vector<float> scores( pos + 1 );
				float max_s = -1e30f;

				for ( int t = 0; t <= pos; ++t ) {
					double dot = 0.0;
					for ( int d = 0; d < cfg.head_dim; ++d ) {
						float q_val = half_to_float( h_q[qh * cfg.head_dim + d] );
						float k_val = cpu_k_cache[t * ( cfg.kv_heads * cfg.head_dim ) + kvh * cfg.head_dim + d];
						dot += q_val * k_val;
					}
					float s = static_cast<float>( dot ) * scale;
					scores[t] = s;
					if ( s > max_s ) max_s = s;
				}

				double sum_exp = 0.0;
				for ( int t = 0; t <= pos; ++t ) {
					scores[t] = std::exp( scores[t] - max_s );
					sum_exp += scores[t];
				}
				for ( int t = 0; t <= pos; ++t ) {
					scores[t] /= static_cast<float>( sum_exp );
				}

				for ( int d = 0; d < cfg.head_dim; ++d ) {
					double acc = 0.0;
					for ( int t = 0; t <= pos; ++t ) {
						float v_val = cpu_v_cache[t * ( cfg.kv_heads * cfg.head_dim ) + kvh * cfg.head_dim + d];
						acc += scores[t] * v_val;
					}
					float g_val = half_to_float( h_gate[qh * cfg.head_dim + d] );
					float sig_g = 1.0f / ( 1.0f + std::exp( -g_val ) );
					h_ref_out[qh * cfg.head_dim + d] = static_cast<float>( acc ) * sig_g;
				}
			}

			// Read back GPU output
			std::vector<half> h_gpu_out( cfg.q_heads * cfg.head_dim );
			CUDA_CHECK( cudaMemcpy( h_gpu_out.data(), d_out.data, h_gpu_out.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );

			double dot = 0.0, norm_ref = 0.0, norm_gpu = 0.0;
			for ( size_t i = 0; i < h_ref_out.size(); ++i ) {
				float r = h_ref_out[i];
				float g = half_to_float( h_gpu_out[i] );
				dot += r * g;
				norm_ref += r * r;
				norm_gpu += g * g;
			}
			double cos_sim = dot / ( std::sqrt( norm_ref ) * std::sqrt( norm_gpu ) + 1e-9 );
			ENGINE_ASSERT( cos_sim >= 0.9999, "Attention decode failed cosine similarity" );
		}
		std::printf( "    -> Config %zu 32/32 steps PASS (cos_sim >= 0.9999).\n", c_idx + 1 );
		engine::core::arena_reset( arena );
	}

	CUDA_CHECK( cudaStreamDestroy( stream ) );
	engine::core::arena_destroy( arena );
	CUDA_CHECK( cudaDeviceReset() );

	std::printf( "[+] ALL ATTENTION PARITY TESTS PASSED.\n" );
	return 0;
}
