#include "engine/core/arena.h"
#include "engine/modules/mla.h"
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
	std::printf( "   volta-engine Tier 2 Parity: MLA Module               \n" );
	std::printf( "========================================================\n" );

	CUDA_CHECK( cudaSetDevice( 0 ) );
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );

	pod::MemoryArena arena;
	bool ok = engine::core::arena_create( arena, 0, 128ULL * 1024 * 1024 );
	ENGINE_ASSERT( ok, "Failed to create arena" );

	const int num_heads = 16;
	const int head_dim = 256;
	const int max_seq = 1024;
	const int total_elems = num_heads * head_dim;

	int q_shape[2] = { 1, total_elems };
	int k_shape[2] = { 1, total_elems };
	int v_shape[2] = { 1, total_elems };
	int out_shape[2] = { 1, total_elems };

	pod::TensorView d_q = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, q_shape, 2 );
	pod::TensorView d_k = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, k_shape, 2 );
	pod::TensorView d_v = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, v_shape, 2 );
	pod::TensorView d_out = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, out_shape, 2 );

	size_t cache_elems = static_cast<size_t>( max_seq ) * total_elems;
	half* d_k_cache = static_cast<half*>( engine::core::arena_alloc( arena, cache_elems * sizeof( half ) ) );
	half* d_v_cache = static_cast<half*>( engine::core::arena_alloc( arena, cache_elems * sizeof( half ) ) );
	CUDA_CHECK( cudaMemset( d_k_cache, 0, cache_elems * sizeof( half ) ) );
	CUDA_CHECK( cudaMemset( d_v_cache, 0, cache_elems * sizeof( half ) ) );

	std::vector<float> cpu_k_cache( cache_elems, 0.0f );
	std::vector<float> cpu_v_cache( cache_elems, 0.0f );

	std::printf( "[*] Simulating 32 sequential MLA decode steps...\n" );
	for ( int pos = 0; pos < 32; ++pos ) {
		std::vector<half> h_q( total_elems );
		std::vector<half> h_k( total_elems );
		std::vector<half> h_v( total_elems );

		for ( int i = 0; i < total_elems; ++i ) {
			h_q[i] = float_to_half( std::sin( static_cast<float>( pos * 1000 + i ) * 0.02f ) );
			float k_val = std::cos( static_cast<float>( pos * 700 + i ) * 0.03f );
			h_k[i] = float_to_half( k_val );
			cpu_k_cache[pos * total_elems + i] = k_val;
			float v_val = std::sin( static_cast<float>( pos * 500 + i ) * 0.04f );
			h_v[i] = float_to_half( v_val );
			cpu_v_cache[pos * total_elems + i] = v_val;
		}

		CUDA_CHECK( cudaMemcpyAsync( d_q.data, h_q.data(), total_elems * sizeof( half ), cudaMemcpyHostToDevice, stream ) );
		CUDA_CHECK( cudaMemcpyAsync( d_k.data, h_k.data(), total_elems * sizeof( half ), cudaMemcpyHostToDevice, stream ) );
		CUDA_CHECK( cudaMemcpyAsync( d_v.data, h_v.data(), total_elems * sizeof( half ), cudaMemcpyHostToDevice, stream ) );

		// Store K, V to MLA cache
		engine::modules::MLA::store_kv_cache(
			d_k_cache, d_v_cache, d_k, d_v, 0, pos, total_elems, max_seq, stream
		);

		// Forward decode
		engine::modules::MLA::forward_decode(
			d_q, d_k_cache, d_v_cache, d_out, 0, pos, num_heads, head_dim, max_seq, stream
		);
		CUDA_CHECK( cudaStreamSynchronize( stream ) );

		// Compute CPU reference
		std::vector<float> h_ref_out( total_elems, 0.0f );
		float scale = 1.0f / std::sqrt( static_cast<float>( head_dim ) );

		for ( int h = 0; h < num_heads; ++h ) {
			std::vector<float> scores( pos + 1 );
			float max_s = -1e30f;

			for ( int t = 0; t <= pos; ++t ) {
				double dot = 0.0;
				for ( int d = 0; d < head_dim; ++d ) {
					float q_val = half_to_float( h_q[h * head_dim + d] );
					float k_val = cpu_k_cache[t * total_elems + h * head_dim + d];
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

			for ( int d = 0; d < head_dim; ++d ) {
				double acc = 0.0;
				for ( int t = 0; t <= pos; ++t ) {
					float v_val = cpu_v_cache[t * total_elems + h * head_dim + d];
					acc += scores[t] * v_val;
				}
				h_ref_out[h * head_dim + d] = static_cast<float>( acc );
			}
		}

		// Read back GPU output
		std::vector<half> h_gpu_out( total_elems );
		CUDA_CHECK( cudaMemcpy( h_gpu_out.data(), d_out.data, total_elems * sizeof( half ), cudaMemcpyDeviceToHost ) );

		double dot = 0.0, norm_ref = 0.0, norm_gpu = 0.0;
		for ( size_t i = 0; i < total_elems; ++i ) {
			float r = h_ref_out[i];
			float g = half_to_float( h_gpu_out[i] );
			dot += r * g;
			norm_ref += r * r;
			norm_gpu += g * g;
		}
		double cos_sim = dot / ( std::sqrt( norm_ref ) * std::sqrt( norm_gpu ) + 1e-9 );
		ENGINE_ASSERT( cos_sim >= 0.9999, "MLA decode step failed cosine similarity" );
	}
	std::printf( "    -> MLA 32/32 steps PASS (cos_sim >= 0.9999).\n" );

	CUDA_CHECK( cudaStreamDestroy( stream ) );
	engine::core::arena_destroy( arena );
	CUDA_CHECK( cudaDeviceReset() );

	std::printf( "[+] ALL MLA PARITY TESTS PASSED.\n" );
	return 0;
}
