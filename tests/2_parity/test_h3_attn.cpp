// H3 attention kernel against a CPU softmax-attention reference (FP32/double), several shapes with tails, full and causal.   test_h3_attn
#include "engine/config.h"
#include "engine/video/h3_attn.h"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <vector>

static int check( int qr, int kr, int heads, bool causal ) {
	std::vector<half> q( static_cast<size_t>( qr ) * heads * 128 ), k( static_cast<size_t>( kr ) * heads * 128 ), v( k.size() ), o( q.size() );
	uint32_t s = 7 + qr + kr;
	auto rnd = [&]() { s = s * 1664525u + 1013904223u; return ( ( s >> 8 ) / 16777216.0f - 0.5f ); };
	for ( auto& x : q ) { x = __float2half( rnd() * 2 ); }
	for ( auto& x : k ) { x = __float2half( rnd() * 2 ); }
	for ( auto& x : v ) { x = __float2half( rnd() * 2 ); }
	half *dq, *dk, *dv, *dout;
	CUDA_CHECK( cudaMalloc( &dq, q.size() * 2 ) ); CUDA_CHECK( cudaMalloc( &dk, k.size() * 2 ) ); CUDA_CHECK( cudaMalloc( &dv, v.size() * 2 ) ); CUDA_CHECK( cudaMalloc( &dout, q.size() * 2 ) );
	CUDA_CHECK( cudaMemcpy( dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice ) ); CUDA_CHECK( cudaMemcpy( dk, k.data(), k.size() * 2, cudaMemcpyHostToDevice ) ); CUDA_CHECK( cudaMemcpy( dv, v.data(), v.size() * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemset( dout, 0, q.size() * 2 ) );
	const float scale = 1.0f / std::sqrt( 128.0f );
	engine::video::h3_attention( dq, dk, dv, dout, qr, kr, heads, scale, causal, nullptr );
	CUDA_CHECK( cudaDeviceSynchronize() );
	CUDA_CHECK( cudaMemcpy( o.data(), dout, o.size() * 2, cudaMemcpyDeviceToHost ) );
	double worst = 0;
	for ( int h = 0; h < heads; ++h ) {
		for ( int i = 0; i < qr; i += 5 ) {
			const int last = causal ? std::min( kr - 1, i ) : kr - 1;
			std::vector<double> sc( last + 1 );
			double mx = -1e30;
			for ( int j = 0; j <= last; ++j ) {
				double a = 0;
				for ( int d = 0; d < 128; ++d ) { a += static_cast<double>( __half2float( q[( static_cast<size_t>( i ) * heads + h ) * 128 + d] ) ) * __half2float( k[( static_cast<size_t>( j ) * heads + h ) * 128 + d] ); }
				sc[j] = a * scale;
				mx = std::fmax( mx, sc[j] );
			}
			double sum = 0;
			for ( int j = 0; j <= last; ++j ) { sc[j] = std::exp( sc[j] - mx ); sum += sc[j]; }
			for ( int d = 0; d < 128; ++d ) {
				double a = 0;
				for ( int j = 0; j <= last; ++j ) { a += sc[j] / sum * __half2float( v[( static_cast<size_t>( j ) * heads + h ) * 128 + d] ); }
				worst = std::fmax( worst, std::fabs( a - __half2float( o[( static_cast<size_t>( i ) * heads + h ) * 128 + d] ) ) );
			}
		}
	}
	std::printf( "q %5d k %5d heads %d %-7s: max abs error %.4g\n", qr, kr, heads, causal ? "causal" : "full", worst );
	cudaFree( dq ); cudaFree( dk ); cudaFree( dv ); cudaFree( dout );
	return worst > 0.01 ? 1 : 0;
}

int main() {
	int bad = 0;
	bad += check( 128, 64, 1, false );
	bad += check( 263, 526, 2, false );
	bad += check( 100, 1000, 3, false );
	bad += check( 700, 91, 2, false );
	bad += check( 300, 300, 2, true );
	bad += check( 1000, 1000, 1, true );
	std::printf( bad ? "FAIL\n" : "ok\n" );
	return bad ? 1 : 0;
}
