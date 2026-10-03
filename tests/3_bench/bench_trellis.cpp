#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/modules/trellis.h"
#include "trellis_context.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <chrono>

int main( int argc, char** argv ) {
	(void) argc;
	(void) argv;

	std::printf( "========================================================\n" );
	std::printf( "   volta-engine Phase 1: Trellis Shape 4 GEMM Benchmark \n" );
	std::printf( "========================================================\n" );

	int dev = 0;
	CUDA_CHECK( cudaSetDevice( dev ) );

	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );

	pod::MemoryArena arena;
	size_t arena_bytes = 512 * 1024 * 1024;
	bool ok = engine::core::arena_create( arena, dev, arena_bytes );
	ENGINE_ASSERT( ok, "Failed to create static arena" );

	const int k_dim = 5120;
	const int n_dim = 10240;
	const int bits = 4;
	const bool mcg = false;

	const size_t trellis_words = static_cast<size_t>( k_dim / 16 ) * ( n_dim / 16 ) * ( 16 * bits );
	const size_t weight_bytes = trellis_words * sizeof( uint16_t );

	int trellis_shape[3] = { k_dim / 16, n_dim / 16, 16 * bits };
	pod::TensorView trellis_tv = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, trellis_shape, 3 );

	int suh_shape[1] = { k_dim };
	pod::TensorView suh_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, suh_shape, 1 );

	int svh_shape[1] = { n_dim };
	pod::TensorView svh_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, svh_shape, 1 );

	CUDA_CHECK( cudaMemsetAsync( trellis_tv.data, 0x5a, weight_bytes, stream ) );
	CUDA_CHECK( cudaMemsetAsync( suh_tv.data, 0x3c, k_dim * sizeof( half ), stream ) );
	CUDA_CHECK( cudaMemsetAsync( svh_tv.data, 0x3c, n_dim * sizeof( half ), stream ) );

	double peak_bw = 0.0;
	int m_values[] = { 1, 2, 4, 8 };

	for ( int m : m_values ) {
		int x_shape[2] = { m, k_dim };
		pod::TensorView x_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, x_shape, 2 );

		int out_shape[2] = { m, n_dim };
		pod::TensorView out_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, out_shape, 2 );

		CUDA_CHECK( cudaMemsetAsync( x_tv.data, 0x3c, static_cast<size_t>( m ) * k_dim * sizeof( half ), stream ) );

		// Warm-up
		for ( int iter = 0; iter < 50; ++iter ) {
			engine::modules::LinearTrellis::forward( x_tv, trellis_tv, suh_tv, svh_tv, out_tv, bits, mcg, stream );
		}
		CUDA_CHECK( cudaStreamSynchronize( stream ) );

		const int iters = 200;
		cudaEvent_t ev_start, ev_stop;
		CUDA_CHECK( cudaEventCreate( &ev_start ) );
		CUDA_CHECK( cudaEventCreate( &ev_stop ) );

		CUDA_CHECK( cudaEventRecord( ev_start, stream ) );
		for ( int iter = 0; iter < iters; ++iter ) {
			engine::modules::LinearTrellis::forward( x_tv, trellis_tv, suh_tv, svh_tv, out_tv, bits, mcg, stream );
		}
		CUDA_CHECK( cudaEventRecord( ev_stop, stream ) );
		CUDA_CHECK( cudaEventSynchronize( ev_stop ) );

		float ms = 0.0f;
		CUDA_CHECK( cudaEventElapsedTime( &ms, ev_start, ev_stop ) );
		double s_per_iter = ( ms / 1000.0 ) / iters;

		size_t effective_bytes = static_cast<size_t>( k_dim ) * n_dim * sizeof( half )
			+ static_cast<size_t>( m ) * ( k_dim + n_dim ) * sizeof( half );
		double bw_gbs = ( effective_bytes / s_per_iter ) / 1e9;

		std::printf( "[+] Shape 4 GEMM (M=%d, K=%d, N=%d, %d-bpw): %.2f GB/s effective (%.3f ms)\n",
			m, k_dim, n_dim, bits, bw_gbs, s_per_iter * 1000.0 );
		std::fflush( stdout );

		if ( bw_gbs > peak_bw ) {
			peak_bw = bw_gbs;
		}

		CUDA_CHECK( cudaEventDestroy( ev_start ) );
		CUDA_CHECK( cudaEventDestroy( ev_stop ) );
	}

	CUDA_CHECK( cudaStreamDestroy( stream ) );
	engine::core::arena_destroy( arena );
	engine::modules::trellis_context_destroy();

	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		cudaSetDevice( d );
		cudaDeviceReset();
	}

	std::printf( "--------------------------------------------------------\n" );
	std::printf( "[+] Peak Trellis Shape 4 Bandwidth: %.2f GB/s (target >= 520.00 GB/s)\n", peak_bw );
	std::fflush( stdout );
	ENGINE_ASSERT( peak_bw >= 520.0, "Trellis Shape 4 bandwidth below 520 GB/s exit gate" );
	std::printf( "[+] Trellis Shape 4 Bandwidth Gate PASS.\n" );

	return 0;
}
