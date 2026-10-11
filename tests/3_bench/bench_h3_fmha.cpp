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
		const int reps = argc > 5 ? std::atoi( argv[5] ) : 3;
		for ( int r = 0; r < reps; ++r ) { engine::video::h3_attention( dq, dk, dv, dout, qr, kr, heads, 0.088f, causal, nullptr ); }
		CUDA_CHECK( cudaDeviceSynchronize() );
		const double s = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() / reps;
		std::printf( "q %d k %d heads %d H3 WMMA attention: %.1f ms, %.1f TFLOP/s\n", qr, kr, heads, s * 1e3, flops / s / 1e12 );
	}
	if ( !causal ) { // the same keys as four ranges with the running state in between (the sequence-parallel order)
		float* so = nullptr;
		float2* sml = nullptr;
		CUDA_CHECK( cudaMalloc( &so, static_cast<size_t>( qr ) * heads * 128 * 4 ) );
		CUDA_CHECK( cudaMalloc( &sml, static_cast<size_t>( qr ) * heads * 8 ) );
		const int chunks = 4, per = ( kr + chunks - 1 ) / chunks;
		auto run = [&]() {
			for ( int c = 0; c < chunks; ++c ) {
				const int k0 = c * per, k1 = std::min( kr, k0 + per );
				const size_t off = static_cast<size_t>( k0 ) * heads * 128;
				engine::video::h3_attention_partial( dq, dk + off, dv + off, dout, so, sml, qr, k1 - k0, heads, 0.088f, c == 0, c == chunks - 1, nullptr );
			}
		};
		run();
		CUDA_CHECK( cudaDeviceSynchronize() );
		const auto t0 = std::chrono::steady_clock::now();
		for ( int r = 0; r < 3; ++r ) { run(); }
		CUDA_CHECK( cudaDeviceSynchronize() );
		const double s = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() / 3;
		std::printf( "q %d k %d heads %d in %d key ranges: %.1f ms, %.1f TFLOP/s\n", qr, kr, heads, chunks, s * 1e3, flops / s / 1e12 );
	}
	return 0;
}
