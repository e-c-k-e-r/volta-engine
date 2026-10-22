// DeepSeek-V4.1 FP8 projections (DSV4::linear_fp8): random E4M3 bytes with E8M0 block scales against a double-precision product, for the row
// counts that take the direct kernel (1 to 16) and those that dequantize into a scratch and use the dense GEMM (above 16), at the shapes of the
// attention and shared-expert projections.
#include "engine/config.h"
#include "engine/modules/dsv4.h"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using engine::modules::DSV4;

static uint32_t g_seed = 4321u;
static uint32_t rnd32() { g_seed = g_seed * 1664525u + 1013904223u; return g_seed >> 8; }

static double e4m3( uint8_t b ) {
	const int s = b >> 7, e = ( b >> 3 ) & 15, m = b & 7;
	const double v = e == 0 ? std::ldexp( m / 8.0, -6 ) : std::ldexp( 1.0 + m / 8.0, e - 7 );
	return s ? -v : v;
}

int main() {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	CUDA_CHECK( cudaSetDevice( 0 ) );
	struct Shape { int n, k; };
	const Shape shapes[] = { { 1280, 5120 }, { 512, 5120 }, { 1024, 4096 }, { 5120, 2048 }, { 8192, 1280 }, { 2304, 5120 }, { 5120, 2304 } };
	const int ms[] = { 1, 4, 5, 16, 17, 40, 300 };
	bool pass = true;
	for ( const Shape& sh : shapes ) {
		const int n = sh.n, k = sh.k;
		std::vector<uint8_t> w( static_cast<size_t>( n ) * k ), sc( static_cast<size_t>( n / 32 ) * ( k / 32 ) );
		for ( auto& b : w ) { do b = static_cast<uint8_t>( rnd32() ); while ( ( b & 0x7f ) == 0x7f ); }
		for ( auto& e : sc ) e = static_cast<uint8_t>( 112 + rnd32() % 8 );   // 2^-15 .. 2^-8
		std::vector<double> wd( w.size() );
		for ( int r = 0; r < n; ++r )
			for ( int c = 0; c < k; ++c ) wd[static_cast<size_t>( r ) * k + c] = e4m3( w[static_cast<size_t>( r ) * k + c] ) * std::ldexp( 1.0, static_cast<int>( sc[static_cast<size_t>( r / 32 ) * ( k / 32 ) + c / 32] ) - 127 );
		uint8_t *d_w, *d_s;
		half *d_x, *d_o, *d_scr;
		CUDA_CHECK( cudaMalloc( &d_w, w.size() ) );
		CUDA_CHECK( cudaMalloc( &d_s, sc.size() ) );
		CUDA_CHECK( cudaMalloc( &d_x, 300ull * k * sizeof( half ) ) );
		CUDA_CHECK( cudaMalloc( &d_o, 300ull * n * sizeof( half ) ) );
		CUDA_CHECK( cudaMalloc( &d_scr, static_cast<size_t>( n ) * k * sizeof( half ) ) );
		CUDA_CHECK( cudaMemcpy( d_w, w.data(), w.size(), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( d_s, sc.data(), sc.size(), cudaMemcpyHostToDevice ) );
		for ( int m : ms ) {
			std::vector<half> x( static_cast<size_t>( m ) * k ), o( static_cast<size_t>( m ) * n );
			for ( auto& v : x ) v = __float2half( ( static_cast<int>( rnd32() % 2001 ) - 1000 ) / 500.0f );
			CUDA_CHECK( cudaMemcpy( d_x, x.data(), x.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
			CUDA_CHECK( cudaMemset( d_o, 0, o.size() * sizeof( half ) ) );
			DSV4::linear_fp8( d_x, d_w, d_s, d_o, m, n, k, d_scr, nullptr );
			CUDA_CHECK( cudaDeviceSynchronize() );
			CUDA_CHECK( cudaMemcpy( o.data(), d_o, o.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
			double num = 0, den = 0, maxd = 0;
			for ( int r = 0; r < m; r += ( m > 40 ? 37 : 1 ) )
				for ( int c = 0; c < n; c += 7 ) {
					double ref = 0;
					for ( int j = 0; j < k; ++j ) ref += wd[static_cast<size_t>( c ) * k + j] * __half2float( x[static_cast<size_t>( r ) * k + j] );
					const double got = __half2float( o[static_cast<size_t>( r ) * n + c] );
					num += ( ref - got ) * ( ref - got );
					den += ref * ref;
					maxd = std::fmax( maxd, std::fabs( ref - got ) );
				}
			const double rel = std::sqrt( num / ( den + 1e-30 ) );
			const bool ok = rel < 2e-3;
			pass = pass && ok;
			std::printf( "[%c] n=%5d k=%5d m=%3d: relative error %.2e (max abs %.3g)\n", ok ? '+' : '-', n, k, m, rel, maxd );
		}
		cudaFree( d_w ); cudaFree( d_s ); cudaFree( d_x ); cudaFree( d_o ); cudaFree( d_scr );
	}
	std::printf( "[%c] FP8 LINEAR %s\n", pass ? '+' : '-', pass ? "PASS" : "FAIL" );
	return pass ? 0 : 1;
}
