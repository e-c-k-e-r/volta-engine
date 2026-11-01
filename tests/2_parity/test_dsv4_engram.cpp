// DeepSeek-V4.1 Engram gate (DSV4::engram_gate): random hyper-connection streams, looked-up key / value rows and weights against a
// double-precision version of the reference's Engram.forward, with and without a skip mask (rows inside an image span, whose gate the
// reference shuts: those rows must come back untouched, bit for bit).
#include "engine/config.h"
#include "engine/modules/dsv4.h"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using engine::modules::DSV4;

static uint32_t g_seed = 9137u;
static float rndf() { g_seed = g_seed * 1664525u + 1013904223u; return ( ( g_seed >> 8 ) / 16777216.0f ) * 2.0f - 1.0f; }

int main() {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	CUDA_CHECK( cudaSetDevice( 0 ) );
	const int H = 4096, rows = 7;
	const float eps = 1e-6f;
	std::vector<float> s( static_cast<size_t>( rows ) * 4 * H ), w( 4 * H );
	std::vector<half> kv( static_cast<size_t>( rows ) * 5 * H );
	for ( auto& x : s ) x = rndf() * 2.0f;
	for ( auto& x : w ) x = 0.5f + rndf() * 0.5f;
	for ( size_t i = 0; i < kv.size(); ++i ) kv[i] = __float2half( rndf() * ( i % ( 5 * H ) < 4u * H ? 1.0f : 0.25f ) );
	// half the rows correlate key with stream so the gate covers more than the middle of the sigmoid
	for ( int r = 0; r < rows; r += 2 )
		for ( int h = 0; h < 4; ++h )
			for ( int d = 0; d < H; ++d ) kv[static_cast<size_t>( r ) * 5 * H + h * H + d] = __float2half( ( r % 4 ? -0.6f : 0.6f ) * s[( static_cast<size_t>( r ) * 4 + h ) * H + d] );
	std::vector<uint8_t> skip( rows, 0 );
	skip[1] = skip[4] = 1;

	float* d_s; half* d_kv; float* d_w; uint8_t* d_skip;
	CUDA_CHECK( cudaMalloc( &d_s, s.size() * sizeof( float ) ) );
	CUDA_CHECK( cudaMalloc( &d_kv, kv.size() * sizeof( half ) ) );
	CUDA_CHECK( cudaMalloc( &d_w, w.size() * sizeof( float ) ) );
	CUDA_CHECK( cudaMalloc( &d_skip, rows ) );
	CUDA_CHECK( cudaMemcpy( d_kv, kv.data(), kv.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_w, w.data(), w.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_skip, skip.data(), rows, cudaMemcpyHostToDevice ) );

	bool pass = true;
	for ( int masked = 0; masked < 2; ++masked ) {
		CUDA_CHECK( cudaMemcpy( d_s, s.data(), s.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
		DSV4::engram_gate( d_s, d_kv, d_w, masked ? d_skip : nullptr, H, eps, rows, nullptr );
		CUDA_CHECK( cudaDeviceSynchronize() );
		std::vector<float> out( s.size() );
		CUDA_CHECK( cudaMemcpy( out.data(), d_s, out.size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
		double worst = 0.0, gmin = 1.0, gmax = 0.0;
		for ( int r = 0; r < rows; ++r )
			for ( int h = 0; h < 4; ++h ) {
				const float* x = &s[( static_cast<size_t>( r ) * 4 + h ) * H];
				const float* o = &out[( static_cast<size_t>( r ) * 4 + h ) * H];
				if ( masked && skip[r] ) {
					if ( std::memcmp( x, o, H * sizeof( float ) ) != 0 ) { std::printf( "  FAIL: skipped row %d stream %d changed\n", r, h ); pass = false; }
					continue;
				}
				double s2 = 0, k2 = 0, dot = 0;
				for ( int d = 0; d < H; ++d ) {
					const double k = __half2float( kv[static_cast<size_t>( r ) * 5 * H + h * H + d] );
					s2 += double( x[d] ) * x[d]; k2 += k * k; dot += double( x[d] ) * w[h * H + d] * k;
				}
				const double g_in = dot / std::sqrt( s2 / H + eps ) / std::sqrt( k2 / H + eps ) / std::sqrt( double( H ) );
				const double g = 1.0 / ( 1.0 + std::exp( -std::copysign( std::sqrt( std::max( std::fabs( g_in ), 1e-6 ) ), g_in ) ) );
				gmin = std::min( gmin, g ); gmax = std::max( gmax, g );
				for ( int d = 0; d < H; ++d ) {
					const double ref = x[d] + g * __half2float( kv[static_cast<size_t>( r ) * 5 * H + 4 * H + d] );
					worst = std::max( worst, std::fabs( ref - o[d] ) );
				}
			}
		const bool ok = worst < 1e-4;
		pass &= ok;
		std::printf( "  %s: worst |diff| %.2e over the gated rows (gates %.3f .. %.3f)%s\n", masked ? "skip mask" : "no mask  ", worst, gmin, gmax, ok ? "" : "  FAIL" );
	}
	std::printf( "test_dsv4_engram: %s\n", pass ? "OK" : "FAIL" );
	return pass ? 0 : 1;
}
