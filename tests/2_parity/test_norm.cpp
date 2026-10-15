#include "engine/core/arena.h"
#include "engine/modules/norm.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <cmath>

static float half_to_float( half h ) {
	return __half2float( h );
}

static half float_to_half( float f ) {
	return __float2half_rn( f );
}

int main() {
	std::printf( "========================================================\n" );
	std::printf( "   volta-engine Tier 2 Parity: Norm & Elementwise       \n" );
	std::printf( "========================================================\n" );

	CUDA_CHECK( cudaSetDevice( 0 ) );
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );

	pod::MemoryArena arena;
	bool ok = engine::core::arena_create( arena, 0, 64ULL * 1024 * 1024 );
	ENGINE_ASSERT( ok, "Failed to create arena" );

	const int hidden_size = 5120;
	int shape_1d[1] = { hidden_size };
	int shape_2d[2] = { 1, hidden_size };

	pod::TensorView d_x = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape_2d, 2 );
	pod::TensorView d_w = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape_1d, 1 );
	pod::TensorView d_out = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape_2d, 2 );

	// Test 1: RMSNorm::forward with bias = 0.0f and bias = 1.0f
	std::printf( "[*] Test 1: RMSNorm::forward with bias=0.0f and bias=1.0f...\n" );
	for ( float bias : { 0.0f, 1.0f } ) {
		std::vector<half> h_x( hidden_size );
		std::vector<half> h_w( hidden_size );
		std::vector<float> cpu_out( hidden_size );

		double sum_sq = 0.0;
		for ( int i = 0; i < hidden_size; ++i ) {
			float val = std::sin( static_cast<float>( i ) * 0.02f );
			h_x[i] = float_to_half( val );
			sum_sq += val * val;
			h_w[i] = float_to_half( 0.6f + 0.3f * std::cos( static_cast<float>( i ) * 0.05f ) );
		}
		float inv_rms = 1.0f / std::sqrt( static_cast<float>( sum_sq / hidden_size ) + 1e-6f );
		for ( int i = 0; i < hidden_size; ++i ) {
			float val = half_to_float( h_x[i] );
			float w_val = half_to_float( h_w[i] ) + bias;
			cpu_out[i] = val * w_val * inv_rms;
		}

		CUDA_CHECK( cudaMemcpyAsync( d_x.data, h_x.data(), hidden_size * sizeof( half ), cudaMemcpyHostToDevice, stream ) );
		CUDA_CHECK( cudaMemcpyAsync( d_w.data, h_w.data(), hidden_size * sizeof( half ), cudaMemcpyHostToDevice, stream ) );

		engine::modules::RMSNorm::forward( d_x, d_w, d_out, 1e-6f, bias, stream );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );

		std::vector<half> h_gpu_out( hidden_size );
		CUDA_CHECK( cudaMemcpy( h_gpu_out.data(), d_out.data, hidden_size * sizeof( half ), cudaMemcpyDeviceToHost ) );

		double dot = 0.0, norm_ref = 0.0, norm_gpu = 0.0;
		for ( size_t i = 0; i < hidden_size; ++i ) {
			float r = cpu_out[i];
			float g = half_to_float( h_gpu_out[i] );
			dot += r * g;
			norm_ref += r * r;
			norm_gpu += g * g;
		}
		double cos_sim = dot / ( std::sqrt( norm_ref ) * std::sqrt( norm_gpu ) + 1e-9 );
		ENGINE_ASSERT( cos_sim >= 0.9999, "RMSNorm forward failed cosine similarity" );
	}
	std::printf( "    -> RMSNorm::forward PASS.\n" );

	// Test 2: Elementwise::argmax and batched_argmax
	std::printf( "[*] Test 2: Elementwise argmax...\n" );
	const int vocab_size = 248320;
	int vocab_shape[2] = { 8, vocab_size };
	pod::TensorView d_logits = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, vocab_shape, 2 );

	int* d_argmax_out = static_cast<int*>( engine::core::arena_alloc( arena, 8 * sizeof( int ) ) );

	for ( int count = 1; count <= 8; count += 3 ) {
		std::vector<half> h_logits( count * vocab_size );
		std::vector<int> expected_argmax( count );

		for ( int b = 0; b < count; ++b ) {
			int target_idx = ( b * 31415 + 42 ) % vocab_size;
			expected_argmax[b] = target_idx;
			for ( int v = 0; v < vocab_size; ++v ) {
				h_logits[b * vocab_size + v] = float_to_half( -10.0f + 0.001f * ( v % 100 ) );
			}
			h_logits[b * vocab_size + target_idx] = float_to_half( 15.0f );
		}

		CUDA_CHECK( cudaMemcpyAsync( d_logits.data, h_logits.data(), count * vocab_size * sizeof( half ), cudaMemcpyHostToDevice, stream ) );
		engine::modules::Elementwise::batched_argmax( d_logits, d_argmax_out, vocab_size, count, stream );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );

		std::vector<int> h_gpu_argmax( count );
		CUDA_CHECK( cudaMemcpy( h_gpu_argmax.data(), d_argmax_out, count * sizeof( int ), cudaMemcpyDeviceToHost ) );

		for ( int b = 0; b < count; ++b ) {
			ENGINE_ASSERT( h_gpu_argmax[b] == expected_argmax[b], "Batched argmax mismatch" );
		}
	}
	// Ties: the same maximum at several positions resolves to the lowest index, whatever the thread layout
	for ( int count = 1; count <= 8; count += 7 ) {
		std::vector<half> h_logits( count * vocab_size, float_to_half( -3.0f ) );
		std::vector<int> expected_argmax( count );
		for ( int b = 0; b < count; ++b ) {
			const int lowest = ( b * 977 + 1234 ) % 100000;
			expected_argmax[b] = lowest;
			for ( int idx : { lowest, lowest + 31, lowest + 512, lowest + 7777, vocab_size - 5 - b, 150000 + b } )
				h_logits[b * vocab_size + idx] = float_to_half( 9.0f );
		}
		CUDA_CHECK( cudaMemcpyAsync( d_logits.data, h_logits.data(), count * vocab_size * sizeof( half ), cudaMemcpyHostToDevice, stream ) );
		engine::modules::Elementwise::batched_argmax( d_logits, d_argmax_out, vocab_size, count, stream );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		std::vector<int> h_gpu_argmax( count );
		CUDA_CHECK( cudaMemcpy( h_gpu_argmax.data(), d_argmax_out, count * sizeof( int ), cudaMemcpyDeviceToHost ) );
		for ( int b = 0; b < count; ++b ) ENGINE_ASSERT( h_gpu_argmax[b] == expected_argmax[b], "Batched argmax tie is not the lowest index" );
	}
	std::printf( "    -> Elementwise::argmax and batched_argmax PASS.\n" );

	CUDA_CHECK( cudaStreamDestroy( stream ) );
	engine::core::arena_destroy( arena );
	CUDA_CHECK( cudaDeviceReset() );

	std::printf( "[+] ALL NORM & ELEMENTWISE PARITY TESTS PASSED.\n" );
	return 0;
}
