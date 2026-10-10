// NVFP4 skinny GEMV at expert / dense decode shapes: time per call over distinct weight sets (cold L2) in graph replay, and the GB/s of packed weights + scales.
//   bench_nvfp4_shapes [K N ...]   default: the Flash-Next expert shapes (gate / up 2560 -> 640, down 640 -> 2560) and two dense ones
#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/modules/nvfp4.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main( int argc, char** argv ) {
	CUDA_CHECK( cudaSetDevice( 0 ) );
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );
	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 3ULL * 1024 * 1024 * 1024 ), "arena" );
	std::vector<std::pair<int, int>> shapes;
	for ( int i = 1; i + 1 < argc; i += 2 ) shapes.push_back( { std::atoi( argv[i] ), std::atoi( argv[i + 1] ) } );
	if ( shapes.empty() ) shapes = { { 2560, 640 }, { 640, 2560 }, { 2560, 2560 }, { 2560, 16384 } };
	const int SETS = 64;
	cudaEvent_t e0, e1;
	CUDA_CHECK( cudaEventCreate( &e0 ) );
	CUDA_CHECK( cudaEventCreate( &e1 ) );
	for ( auto [K, N] : shapes ) {
		std::vector<pod::TensorView> codes( SETS ), scales( SETS );
		int cs[2] = { N, K / 2 }, ss[2] = { N, K / 16 };
		for ( int s = 0; s < SETS; ++s ) {
			codes[s] = engine::core::arena_alloc_tensor( arena, pod::DType::UINT8, cs, 2 );
			scales[s] = engine::core::arena_alloc_tensor( arena, pod::DType::UINT8, ss, 2 );
			CUDA_CHECK( cudaMemsetAsync( codes[s].data, 0x44, static_cast<size_t>( N ) * K / 2, stream ) );
			CUDA_CHECK( cudaMemsetAsync( scales[s].data, 0x38, static_cast<size_t>( N ) * K / 16, stream ) );
		}
		for ( int m : { 1, 4, 8 } ) {
			int xs[2] = { m, K }, os[2] = { m, N };
			pod::TensorView x = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
			pod::TensorView o = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, os, 2 );
			CUDA_CHECK( cudaMemsetAsync( x.data, 0x3c, static_cast<size_t>( m ) * K * 2, stream ) );
			auto run = [&]() { for ( int s = 0; s < SETS; ++s ) engine::modules::LinearNVFP4::forward( x, codes[s], scales[s], 1.0f, o, stream ); };
			run();
			CUDA_CHECK( cudaStreamSynchronize( stream ) );
			cudaGraph_t g;
			cudaGraphExec_t ex;
			CUDA_CHECK( cudaStreamBeginCapture( stream, cudaStreamCaptureModeGlobal ) );
			run();
			CUDA_CHECK( cudaStreamEndCapture( stream, &g ) );
			CUDA_CHECK( cudaGraphInstantiate( &ex, g, nullptr, nullptr, 0 ) );
			CUDA_CHECK( cudaGraphLaunch( ex, stream ) );
			CUDA_CHECK( cudaStreamSynchronize( stream ) );
			CUDA_CHECK( cudaEventRecord( e0, stream ) );
			const int iters = 20;
			for ( int i = 0; i < iters; ++i ) CUDA_CHECK( cudaGraphLaunch( ex, stream ) );
			CUDA_CHECK( cudaEventRecord( e1, stream ) );
			CUDA_CHECK( cudaEventSynchronize( e1 ) );
			float ms = 0.0f;
			CUDA_CHECK( cudaEventElapsedTime( &ms, e0, e1 ) );
			const double us = 1000.0 * ms / iters / SETS;
			const double bytes = static_cast<double>( N ) * K / 2 + static_cast<double>( N ) * K / 16;
			std::printf( "K %5d N %5d M %d: %6.2f us per call, %6.1f GB/s (packed %.2f MB)\n", K, N, m, us, bytes / ( us * 1e-6 ) / 1e9, bytes / 1e6 );
		}
	}
	return 0;
}
