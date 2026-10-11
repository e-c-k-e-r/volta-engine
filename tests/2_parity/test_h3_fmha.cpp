// H3 FMHA with different query and key lengths against a CPU softmax-attention reference.   test_h3_fmha [q_rows=263] [k_rows=526] [heads=2]
#include "engine/config.h"
#include "engine/video/h3_fmha.h"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main( int argc, char** argv ) {
	const int qr = argc > 1 ? std::atoi( argv[1] ) : 263, kr = argc > 2 ? std::atoi( argv[2] ) : 526, heads = argc > 3 ? std::atoi( argv[3] ) : 2;
	std::vector<half> q( static_cast<size_t>( qr ) * heads * 128 ), k( static_cast<size_t>( kr ) * heads * 128 ), v( k.size() ), o( q.size() );
	uint32_t s = 5;
	auto rnd = [&]() { s = s * 1664525u + 1013904223u; return ( ( s >> 8 ) / 16777216.0f - 0.5f ); };
	for ( auto& x : q ) { x = __float2half( rnd() * 2 ); }
	for ( auto& x : k ) { x = __float2half( rnd() * 2 ); }
	for ( auto& x : v ) { x = __float2half( rnd() * 2 ); }
	half *dq, *dk, *dv, *dout;
	CUDA_CHECK( cudaMalloc( &dq, q.size() * 2 ) ); CUDA_CHECK( cudaMalloc( &dk, k.size() * 2 ) ); CUDA_CHECK( cudaMalloc( &dv, v.size() * 2 ) ); CUDA_CHECK( cudaMalloc( &dout, q.size() * 2 ) );
	CUDA_CHECK( cudaMemcpy( dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice ) ); CUDA_CHECK( cudaMemcpy( dk, k.data(), k.size() * 2, cudaMemcpyHostToDevice ) ); CUDA_CHECK( cudaMemcpy( dv, v.data(), v.size() * 2, cudaMemcpyHostToDevice ) );
	const float scale = 1.0f / std::sqrt( 128.0f );
	int bad = 0;
	for ( int qt : { 64, 128 } ) {
		for ( int kt : { 64, 128 } ) {
			CUDA_CHECK( cudaMemset( dout, 0, q.size() * 2 ) );
			engine::video::h3_flash_attention_qk( dq, dk, dv, dout, qr, kr, heads, scale, kt, qt, nullptr );
			CUDA_CHECK( cudaDeviceSynchronize() );
			CUDA_CHECK( cudaMemcpy( o.data(), dout, o.size() * 2, cudaMemcpyDeviceToHost ) );
			double worst = 0;
			for ( int h = 0; h < heads; ++h ) {
				for ( int i = 0; i < qr; i += 7 ) {
					std::vector<double> sc( kr );
					double mx = -1e30;
					for ( int j = 0; j < kr; ++j ) {
						double a = 0;
						for ( int d = 0; d < 128; ++d ) { a += static_cast<double>( __half2float( q[( static_cast<size_t>( i ) * heads + h ) * 128 + d] ) ) * __half2float( k[( static_cast<size_t>( j ) * heads + h ) * 128 + d] ); }
						sc[j] = a * scale;
						mx = std::fmax( mx, sc[j] );
					}
					double sum = 0;
					for ( int j = 0; j < kr; ++j ) { sc[j] = std::exp( sc[j] - mx ); sum += sc[j]; }
					for ( int d = 0; d < 128; ++d ) {
						double a = 0;
						for ( int j = 0; j < kr; ++j ) { a += sc[j] / sum * __half2float( v[( static_cast<size_t>( j ) * heads + h ) * 128 + d] ); }
						worst = std::fmax( worst, std::fabs( a - __half2float( o[( static_cast<size_t>( i ) * heads + h ) * 128 + d] ) ) );
					}
				}
			}
			std::printf( "q %d k %d heads %d, tiles q%d k%d: max abs error %.4g\n", qr, kr, heads, qt, kt, worst );
			if ( worst > 0.01 ) { ++bad; }
		}
	}
	std::printf( bad ? "FAIL\n" : "ok\n" );
	return bad ? 1 : 0;
}
