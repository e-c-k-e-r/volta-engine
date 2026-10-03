#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/modules/nvfp4.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <chrono>

int main( int argc, char** argv ) {
	(void) argc;
	(void) argv;

	std::printf( "========================================================\n" );
	std::printf( "   volta-engine Phase 1: 1Cat Line-Rate NVFP4 Benchmark  \n" );
	std::printf( "========================================================\n" );

	int dev = 0;
	CUDA_CHECK( cudaSetDevice( dev ) );

	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );

	pod::MemoryArena arena;
	size_t arena_bytes = 512 * 1024 * 1024;
	bool ok = engine::core::arena_create( arena, dev, arena_bytes );
	ENGINE_ASSERT( ok, "Failed to create static arena" );

	const int k_dim = 4096;
	const int n_dim = 65536;
	const float gscale = 1.0f;

	const size_t codes_bytes = static_cast<size_t>( n_dim ) * ( k_dim >> 1 );
	const size_t scales_bytes = static_cast<size_t>( n_dim ) * ( k_dim >> 4 );

	int codes_shape[2] = { n_dim, k_dim / 2 };
	pod::TensorView codes_tv = engine::core::arena_alloc_tensor( arena, pod::DType::UINT8, codes_shape, 2 );

	int scales_shape[2] = { n_dim, k_dim / 16 };
	pod::TensorView scales_tv = engine::core::arena_alloc_tensor( arena, pod::DType::UINT8, scales_shape, 2 );

	CUDA_CHECK( cudaMemsetAsync( codes_tv.data, 0x44, codes_bytes, stream ) );
	CUDA_CHECK( cudaMemsetAsync( scales_tv.data, 0x38, scales_bytes, stream ) );

	double peak_bw = 0.0;
	int m_values[] = { 1, 4, 8 };

	for ( int m : m_values ) {
		int x_shape[2] = { m, k_dim };
		pod::TensorView x_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, x_shape, 2 );

		int out_shape[2] = { m, n_dim };
		pod::TensorView out_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, out_shape, 2 );

		CUDA_CHECK( cudaMemsetAsync( x_tv.data, 0x3c, static_cast<size_t>( m ) * k_dim * sizeof( half ), stream ) );

		// Warm-up to trigger GPU boost clock (1530 MHz)
		for ( int iter = 0; iter < 100; ++iter ) {
			engine::modules::LinearNVFP4::forward( x_tv, codes_tv, scales_tv, gscale, out_tv, stream );
		}
		CUDA_CHECK( cudaStreamSynchronize( stream ) );

		const int iters = 200;
		cudaEvent_t ev_start, ev_stop;
		CUDA_CHECK( cudaEventCreate( &ev_start ) );
		CUDA_CHECK( cudaEventCreate( &ev_stop ) );

		CUDA_CHECK( cudaEventRecord( ev_start, stream ) );
		for ( int iter = 0; iter < iters; ++iter ) {
			engine::modules::LinearNVFP4::forward( x_tv, codes_tv, scales_tv, gscale, out_tv, stream );
		}
		CUDA_CHECK( cudaEventRecord( ev_stop, stream ) );
		CUDA_CHECK( cudaEventSynchronize( ev_stop ) );

		float ms = 0.0f;
		CUDA_CHECK( cudaEventElapsedTime( &ms, ev_start, ev_stop ) );
		double s_per_iter = ( ms / 1000.0 ) / iters;

		size_t total_bytes = codes_bytes + scales_bytes
			+ static_cast<size_t>( m ) * ( k_dim + n_dim ) * sizeof( half );
		double bw_gbs = ( total_bytes / s_per_iter ) / 1e9;

		std::printf( "[+] NVFP4 Kernel (M=%d, K=%d, N=%d): %.2f GB/s (%.3f ms)\n",
			m, k_dim, n_dim, bw_gbs, s_per_iter * 1000.0 );
		std::fflush( stdout );

		if ( bw_gbs > peak_bw ) {
			peak_bw = bw_gbs;
		}

		CUDA_CHECK( cudaEventDestroy( ev_start ) );
		CUDA_CHECK( cudaEventDestroy( ev_stop ) );
	}

	CUDA_CHECK( cudaStreamDestroy( stream ) );
	engine::core::arena_destroy( arena );

	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		cudaSetDevice( d );
		cudaDeviceReset();
	}

	std::printf( "--------------------------------------------------------\n" );
	std::printf( "[+] Peak NVFP4 Bandwidth: %.2f GB/s (target >= 750.00 GB/s)\n", peak_bw );
	std::fflush( stdout );
	ENGINE_ASSERT( peak_bw >= 750.0, "NVFP4 bandwidth below 750 GB/s exit gate" );
	std::printf( "[+] NVFP4 Line-Rate Gate PASS.\n" );

	return 0;
}
