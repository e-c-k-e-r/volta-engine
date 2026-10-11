// Throughput of the H3 attention kernel at the DiT's sequence-parallel shape.   bench_h3_fmha [q_rows=8618] [k_rows=34471] [heads=56] [causal=0]
#include "engine/config.h"
#include "engine/video/h3_attn.h"
#include <cuda_fp16.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main( int argc, char** argv ) {
	const int qr = argc > 1 ? std::atoi( argv[1] ) : 8618, kr = argc > 2 ? std::atoi( argv[2] ) : 34471, heads = argc > 3 ? std::atoi( argv[3] ) : 56;
	const bool causal = argc > 4 && std::atoi( argv[4] ) != 0;
	half *dq, *dk, *dv, *dout;
	CUDA_CHECK( cudaMalloc( &dq, static_cast<size_t>( qr ) * heads * 128 * 2 ) );
	CUDA_CHECK( cudaMalloc( &dk, static_cast<size_t>( kr ) * heads * 128 * 2 ) );
	CUDA_CHECK( cudaMalloc( &dv, static_cast<size_t>( kr ) * heads * 128 * 2 ) );
	CUDA_CHECK( cudaMalloc( &dout, static_cast<size_t>( qr ) * heads * 128 * 2 ) );
	CUDA_CHECK( cudaMemset( dq, 0x2a, static_cast<size_t>( qr ) * heads * 128 * 2 ) );
	CUDA_CHECK( cudaMemset( dk, 0x2a, static_cast<size_t>( kr ) * heads * 128 * 2 ) );
	CUDA_CHECK( cudaMemset( dv, 0x2a, static_cast<size_t>( kr ) * heads * 128 * 2 ) );
	const double flops = 4.0 * qr * static_cast<double>( kr ) * 128 * heads * ( causal ? 0.5 : 1.0 );
	{
		engine::video::h3_attention( dq, dk, dv, dout, qr, kr, heads, 0.088f, causal, nullptr );
		CUDA_CHECK( cudaDeviceSynchronize() );
		const auto t0 = std::chrono::steady_clock::now();
		const int reps = 3;
		for ( int r = 0; r < reps; ++r ) { engine::video::h3_attention( dq, dk, dv, dout, qr, kr, heads, 0.088f, causal, nullptr ); }
		CUDA_CHECK( cudaDeviceSynchronize() );
		const double s = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() / reps;
		std::printf( "q %d k %d heads %d H3 WMMA attention: %.1f ms, %.1f TFLOP/s\n", qr, kr, heads, s * 1e3, flops / s / 1e12 );
	}
	return 0;
}
