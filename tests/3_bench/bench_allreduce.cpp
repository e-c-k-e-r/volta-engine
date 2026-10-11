// All-reduce of an activation-sized FP16 tensor over the GPUs (NCCL, the reference for what tensor-parallel sharding of the H3 transformer would pay per layer) next to the
// ring-less alternative of the sequence-parallel design (each rank pulling the K/V of its peers).   bench_allreduce [rows=34472] [cols=5376] [gpus=4] [repeats=20]
#include "engine/config.h"
#include <nccl.h>
#include <cuda_fp16.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define NCCL_CHECK( x ) do { ncclResult_t r_ = ( x ); if ( r_ != ncclSuccess ) { std::fprintf( stderr, "NCCL error %s at %s:%d\n", ncclGetErrorString( r_ ), __FILE__, __LINE__ ); std::exit( 1 ); } } while ( 0 )

int main( int argc, char** argv ) {
	const size_t rows = argc > 1 ? std::atoi( argv[1] ) : 34472, cols = argc > 2 ? std::atoi( argv[2] ) : 5376;
	const int n = argc > 3 ? std::atoi( argv[3] ) : 4, reps = argc > 4 ? std::atoi( argv[4] ) : 20;
	const size_t count = rows * cols, bytes = count * sizeof( half );
	std::vector<int> devs( n );
	for ( int i = 0; i < n; ++i ) { devs[i] = i; }
	std::vector<ncclComm_t> comms( n );
	NCCL_CHECK( ncclCommInitAll( comms.data(), n, devs.data() ) );
	std::vector<half*> buf( n );
	std::vector<cudaStream_t> st( n );
	for ( int i = 0; i < n; ++i ) {
		CUDA_CHECK( cudaSetDevice( i ) );
		CUDA_CHECK( cudaMalloc( &buf[i], bytes ) );
		CUDA_CHECK( cudaMemset( buf[i], 0, bytes ) );
		CUDA_CHECK( cudaStreamCreateWithFlags( &st[i], cudaStreamNonBlocking ) );
	}
	auto run = [&]() {
		NCCL_CHECK( ncclGroupStart() );
		for ( int i = 0; i < n; ++i ) { NCCL_CHECK( ncclAllReduce( buf[i], buf[i], count, ncclFloat16, ncclSum, comms[i], st[i] ) ); }
		NCCL_CHECK( ncclGroupEnd() );
	};
	auto sync = [&]() { for ( int i = 0; i < n; ++i ) { CUDA_CHECK( cudaSetDevice( i ) ); CUDA_CHECK( cudaStreamSynchronize( st[i] ) ); } };
	run();
	sync();
	const auto t0 = std::chrono::steady_clock::now();
	for ( int r = 0; r < reps; ++r ) { run(); }
	sync();
	const double s = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() / reps;
	std::printf( "all-reduce of %zu x %zu FP16 (%.0f MB) over %d GPUs: %.1f ms, bus bandwidth %.1f GB/s\n", rows, cols, bytes / 1e6, n, s * 1e3, 2.0 * ( n - 1 ) / n * bytes / s / 1e9 );
	std::printf( "a layer pays two (attention out, MLP down): %.1f ms, 50 layers: %.2f s per forward\n", 2 * s * 1e3, 100 * s );
	for ( int i = 0; i < n; ++i ) { ncclCommDestroy( comms[i] ); }
	return 0;
}
