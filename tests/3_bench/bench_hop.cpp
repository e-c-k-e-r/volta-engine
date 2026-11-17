// The hop between two GPUs by message size: a direct peer copy, and a bounce through pinned host memory (the path when the two groups of a split rig share
// no peer link), each as latency of one synchronized copy (median, microseconds) and as pipelined bandwidth.
//   bench_hop [src gpu = 0] [dst gpu = 1] [repeats = 200]
// A pipeline stage boundary moves [rows, hidden] FP16 once per step, a few KB to a few hundred KB (GLM and DeepSeek hidden 4096 -> 8 KB a row), so the
// latency columns of the small sizes are the cost; the large sizes show the link (prefill activations, snapshots).
#include "engine/config.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double micros( Clock::time_point t0 ) {
	return std::chrono::duration<double, std::micro>( Clock::now() - t0 ).count();
}

double median( std::vector<double> v ) {
	std::sort( v.begin(), v.end() );
	return v[v.size() / 2];
}

} // namespace

int main( int argc, char** argv ) {
	const int src = argc > 1 ? std::atoi( argv[1] ) : 0, dst = argc > 2 ? std::atoi( argv[2] ) : 1, reps = argc > 3 ? std::atoi( argv[3] ) : 200;
	int n = 0, can = 0;
	CUDA_CHECK( cudaGetDeviceCount( &n ) );
	if ( src >= n || dst >= n || src == dst ) { std::fprintf( stderr, "need two different GPUs below %d\n", n ); return 2; }
	CUDA_CHECK( cudaDeviceCanAccessPeer( &can, src, dst ) );
	const size_t max_bytes = 64u << 20;
	void *a = nullptr, *b = nullptr, *host = nullptr;
	cudaStream_t sa, sb;
	CUDA_CHECK( cudaSetDevice( src ) );
	CUDA_CHECK( cudaMalloc( &a, max_bytes ) );
	CUDA_CHECK( cudaStreamCreateWithFlags( &sa, cudaStreamNonBlocking ) );
	if ( can ) { (void) cudaDeviceEnablePeerAccess( dst, 0 ); (void) cudaGetLastError(); }
	CUDA_CHECK( cudaSetDevice( dst ) );
	CUDA_CHECK( cudaMalloc( &b, max_bytes ) );
	CUDA_CHECK( cudaStreamCreateWithFlags( &sb, cudaStreamNonBlocking ) );
	if ( can ) { (void) cudaDeviceEnablePeerAccess( src, 0 ); (void) cudaGetLastError(); }
	CUDA_CHECK( cudaHostAlloc( &host, max_bytes, cudaHostAllocPortable ) );
	CUDA_CHECK( cudaSetDevice( src ) );
	CUDA_CHECK( cudaMemset( a, 1, max_bytes ) );
	CUDA_CHECK( cudaDeviceSynchronize() );

	std::printf( "GPU %d -> %d, peer access %s\n", src, dst, can ? "yes" : "no" );
	std::printf( "%10s | %12s %10s | %12s %10s\n", "bytes", "peer us", "peer GB/s", "bounce us", "bounce GB/s" );
	for ( size_t bytes : { 4096ul, 16384ul, 65536ul, 131072ul, 262144ul, 1048576ul, 4194304ul, 16777216ul, 67108864ul } ) {
		// latency: one synchronized copy
		std::vector<double> peer, bounce;
		for ( int r = 0; r < reps + 10; ++r ) {
			auto t0 = Clock::now();
			CUDA_CHECK( cudaMemcpyPeerAsync( b, dst, a, src, bytes, sa ) );
			CUDA_CHECK( cudaStreamSynchronize( sa ) );
			if ( r >= 10 ) peer.push_back( micros( t0 ) );
			t0 = Clock::now();
			CUDA_CHECK( cudaSetDevice( src ) );
			CUDA_CHECK( cudaMemcpyAsync( host, a, bytes, cudaMemcpyDeviceToHost, sa ) );
			CUDA_CHECK( cudaStreamSynchronize( sa ) );
			CUDA_CHECK( cudaSetDevice( dst ) );
			CUDA_CHECK( cudaMemcpyAsync( b, host, bytes, cudaMemcpyHostToDevice, sb ) );
			CUDA_CHECK( cudaStreamSynchronize( sb ) );
			if ( r >= 10 ) bounce.push_back( micros( t0 ) );
			CUDA_CHECK( cudaSetDevice( src ) );
		}
		// bandwidth: copies back to back on one stream
		const int burst = bytes >= ( 4u << 20 ) ? 20 : 100;
		auto t0 = Clock::now();
		for ( int r = 0; r < burst; ++r ) CUDA_CHECK( cudaMemcpyPeerAsync( b, dst, a, src, bytes, sa ) );
		CUDA_CHECK( cudaStreamSynchronize( sa ) );
		const double peer_bw = bytes * burst / ( micros( t0 ) * 1e-6 ) / 1e9;
		t0 = Clock::now();
		for ( int r = 0; r < burst; ++r ) {
			CUDA_CHECK( cudaMemcpyAsync( host, a, bytes, cudaMemcpyDeviceToHost, sa ) );
			CUDA_CHECK( cudaStreamSynchronize( sa ) );
			CUDA_CHECK( cudaSetDevice( dst ) );
			CUDA_CHECK( cudaMemcpyAsync( b, host, bytes, cudaMemcpyHostToDevice, sb ) );
			CUDA_CHECK( cudaStreamSynchronize( sb ) );
			CUDA_CHECK( cudaSetDevice( src ) );
		}
		const double bounce_bw = bytes * burst / ( micros( t0 ) * 1e-6 ) / 1e9;
		std::printf( "%10zu | %12.1f %10.1f | %12.1f %10.1f\n", bytes, median( peer ), peer_bw, median( bounce ), bounce_bw );
	}
	return 0;
}
