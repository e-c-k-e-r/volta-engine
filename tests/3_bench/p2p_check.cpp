// Peer-to-peer copy integrity and bandwidth between every GPU pair.   p2p_check [MB=256] [repeats=20]
#include "engine/config.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main( int argc, char** argv ) {
	const size_t bytes = static_cast<size_t>( argc > 1 ? std::atoi( argv[1] ) : 256 ) << 20;
	const int reps = argc > 2 ? std::atoi( argv[2] ) : 20;
	int n = 0;
	CUDA_CHECK( cudaGetDeviceCount( &n ) );
	std::vector<void*> buf( n );
	for ( int i = 0; i < n; ++i ) {
		CUDA_CHECK( cudaSetDevice( i ) );
		CUDA_CHECK( cudaMalloc( &buf[i], bytes ) );
		for ( int j = 0; j < n; ++j ) { if ( i != j ) { cudaDeviceEnablePeerAccess( j, 0 ); cudaGetLastError(); } }
	}
	std::vector<unsigned char> pat( bytes ), back( bytes );
	for ( size_t i = 0; i < bytes; ++i ) { pat[i] = static_cast<unsigned char>( ( i * 2654435761u ) >> 24 ); }
	int bad = 0;
	for ( int a = 0; a < n; ++a ) {
		for ( int b = 0; b < n; ++b ) {
			if ( a == b ) { continue; }
			int errors = 0;
			double secs = 0;
			for ( int r = 0; r < reps; ++r ) {
				CUDA_CHECK( cudaSetDevice( a ) );
				CUDA_CHECK( cudaMemcpy( buf[a], pat.data(), bytes, cudaMemcpyHostToDevice ) );
				CUDA_CHECK( cudaSetDevice( b ) );
				CUDA_CHECK( cudaMemset( buf[b], 0, bytes ) );
				CUDA_CHECK( cudaSetDevice( a ) );
				cudaStream_t s;
				CUDA_CHECK( cudaStreamCreateWithFlags( &s, cudaStreamNonBlocking ) );
				auto t0 = std::chrono::steady_clock::now();
				CUDA_CHECK( cudaMemcpyPeerAsync( buf[b], b, buf[a], a, bytes, s ) );
				CUDA_CHECK( cudaStreamSynchronize( s ) );
				secs += std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
				cudaStreamDestroy( s );
				CUDA_CHECK( cudaSetDevice( b ) );
				CUDA_CHECK( cudaMemcpy( back.data(), buf[b], bytes, cudaMemcpyDeviceToHost ) );
				for ( size_t i = 0; i < bytes; ++i ) { if ( back[i] != pat[i] ) { ++errors; } }
			}
			std::printf( "GPU %d -> %d: %.1f GB/s, %d corrupted bytes over %d copies\n", a, b, bytes * reps / secs / 1e9, errors, reps );
			bad += errors > 0;
		}
	}
	std::printf( bad ? "FAIL\n" : "ok\n" );
	return bad ? 1 : 0;
}
