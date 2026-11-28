// The NVFP4 quantizer (engine/quant/nvfp4_quantize.h) on a synthetic weight and Hessian: round to nearest against a host restatement of the
// block quantizer (codes and scales identical), the packed tensors decoded on the host give the reported proxy errors, and the error feedback
// lowers the proxy error below round to nearest
#include "engine/quant/nvfp4_quantize.h"
#include "engine/config.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

int failures = 0;

void check( bool ok, const char* what ) {
	std::printf( "%s %s\n", ok ? "PASS" : "FAIL", what );
	failures += ok ? 0 : 1;
}

float e4m3_value( int c ) {
	const int e = c >> 3, m = c & 7;
	return e == 0 ? m * 0.001953125f : ( 1.0f + m * 0.125f ) * std::exp2( static_cast<float>( e - 7 ) );
}

const float GRID[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };

int e2m1_index( float a ) {
	const float th[7] = { 0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f };
	int i = 0;
	for ( float t : th ) i += a > t;
	return i;
}

// One column's 16 values: the scale code and the 4-bit codes (sign in bit 3)
int quantize16( const float* v, float gs, uint8_t* q ) {
	float amax = 0.0f;
	for ( int r = 0; r < 16; ++r ) amax = std::max( amax, std::fabs( v[r] ) );
	const float ideal = amax / ( 6.0f * gs );
	int c0 = 0;
	while ( c0 < 126 && e4m3_value( c0 ) < ideal ) ++c0;
	const int below = c0 > 0 ? c0 - 1 : 0;
	if ( ideal - e4m3_value( below ) < e4m3_value( c0 ) - ideal ) c0 = below;
	c0 = std::min( std::max( c0, 1 ), 126 );
	float best = 0.0f;
	int best_c = c0;
	for ( int d = -2; d <= 2; ++d ) {
		const int c = std::min( std::max( c0 + d, 1 ), 126 );
		const float s = e4m3_value( c ) * gs;
		float e = 0.0f;
		for ( int r = 0; r < 16; ++r ) {
			const float x = v[r] / s;
			const float dq = ( x < 0.0f ? -1.0f : 1.0f ) * GRID[e2m1_index( std::fabs( x ) )] * s;
			e += ( dq - v[r] ) * ( dq - v[r] );
		}
		if ( d == -2 || e < best ) {
			best = e;
			best_c = c;
		}
	}
	const float s = e4m3_value( best_c ) * gs;
	for ( int r = 0; r < 16; ++r ) {
		const float x = v[r] / s;
		q[r] = static_cast<uint8_t>( e2m1_index( std::fabs( x ) ) | ( x < 0.0f ? 8 : 0 ) );
	}
	return best_c;
}

// The packed tensor decoded to k x n (input-major)
std::vector<float> decode( const engine::quant::Nvfp4Tensor& t ) {
	std::vector<float> w( static_cast<size_t>( t.k ) * t.n );
	for ( int col = 0; col < t.n; ++col )
		for ( int i = 0; i < t.k; ++i ) {
			const uint8_t b = t.codes[static_cast<size_t>( col ) * ( t.k / 2 ) + i / 2];
			const int c = ( i & 1 ) ? b >> 4 : b & 15;
			const float s = e4m3_value( t.scales[static_cast<size_t>( col ) * ( t.k / 16 ) + i / 16] ) * t.global_scale;
			w[static_cast<size_t>( i ) * t.n + col] = ( c & 8 ? -1.0f : 1.0f ) * GRID[c & 7] * s;
		}
	return w;
}

// tr( X^T H X ) for X k x n (H k x k, mean)
double metric( const std::vector<float>& X, const std::vector<float>& H, int k, int n ) {
	double s = 0.0;
	for ( int col = 0; col < n; ++col )
		for ( int i = 0; i < k; ++i ) {
			double hx = 0.0;
			for ( int j = 0; j < k; ++j ) hx += static_cast<double>( H[static_cast<size_t>( i ) * k + j] ) * X[static_cast<size_t>( j ) * n + col];
			s += hx * X[static_cast<size_t>( i ) * n + col];
		}
	return s;
}

}   // namespace

int main() {
	const int k = 256, n = 96, rows = 2048;
	std::mt19937 rng( 7 );
	std::normal_distribution<float> nd( 0.0f, 1.0f );
	// inputs with correlated channels (a random mixing of fewer factors) and uneven scales, weights with a few large values
	std::vector<float> X( static_cast<size_t>( rows ) * k ), mix( static_cast<size_t>( 64 ) * k ), W( static_cast<size_t>( k ) * n );
	for ( auto& v : mix ) v = nd( rng );
	for ( int r = 0; r < rows; ++r ) {
		float f[64];
		for ( auto& v : f ) v = nd( rng );
		for ( int i = 0; i < k; ++i ) {
			float s = 0.2f * nd( rng );
			for ( int j = 0; j < 64; ++j ) s += f[j] * mix[static_cast<size_t>( j ) * k + i] * 0.125f;
			X[static_cast<size_t>( r ) * k + i] = s * ( 1.0f + 2.0f * ( i % 7 == 0 ) );
		}
	}
	for ( auto& v : W ) v = nd( rng ) * 0.02f * ( rng() % 97 == 0 ? 8.0f : 1.0f );
	std::vector<float> Hs( static_cast<size_t>( k ) * k, 0.0f ), Hm( Hs.size() );
	for ( int r = 0; r < rows; ++r )
		for ( int i = 0; i < k; ++i )
			for ( int j = 0; j < k; ++j ) Hs[static_cast<size_t>( i ) * k + j] += X[static_cast<size_t>( r ) * k + i] * X[static_cast<size_t>( r ) * k + j];
	for ( size_t i = 0; i < Hs.size(); ++i ) Hm[i] = Hs[i] / rows;

	cudaStream_t st = nullptr;
	CUDA_CHECK( cudaStreamCreate( &st ) );
	float *d_w = nullptr, *d_h = nullptr;
	CUDA_CHECK( cudaMalloc( &d_w, W.size() * 4 ) );
	CUDA_CHECK( cudaMalloc( &d_h, Hs.size() * 4 ) );
	CUDA_CHECK( cudaMemcpy( d_w, W.data(), W.size() * 4, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_h, Hs.data(), Hs.size() * 4, cudaMemcpyHostToDevice ) );

	engine::quant::Nvfp4QuantOptions o;
	engine::quant::Nvfp4Tensor rtn, cal;
	const bool ok_r = engine::quant::nvfp4_quantize( d_w, k, n, nullptr, 0, 0.0f, o, rtn, st );
	const bool ok_c = engine::quant::nvfp4_quantize( d_w, k, n, d_h, rows, 0.0f, o, cal, st );
	check( ok_r && ok_c && !cal.fallback, "quantized, the Hessian factored" );

	float amax = 0.0f;
	for ( float v : W ) amax = std::max( amax, std::fabs( v ) );
	const float gs = amax / ( 6.0f * 448.0f );
	check( rtn.global_scale == gs, "global scale amax / (6 * 448)" );

	// round to nearest: every column block as the host restatement quantizes it
	bool same = true;
	for ( int col = 0; col < n && same; ++col )
		for ( int b = 0; b < k / 16 && same; ++b ) {
			float v[16];
			uint8_t qh[16];
			for ( int r = 0; r < 16; ++r ) v[r] = W[static_cast<size_t>( b * 16 + r ) * n + col];
			const int c = quantize16( v, gs, qh );
			same = rtn.scales[static_cast<size_t>( col ) * ( k / 16 ) + b] == c;
			for ( int r = 0; r < 16 && same; ++r ) {
				const uint8_t byte = rtn.codes[static_cast<size_t>( col ) * ( k / 2 ) + ( b * 16 + r ) / 2];
				same = ( ( r & 1 ) ? byte >> 4 : byte & 15 ) == qh[r];
			}
		}
	check( same, "round to nearest: codes and scales as the host quantizer's" );

	// the reported proxy errors from the decoded tensors
	const double ref = metric( W, Hm, k, n );
	auto proxy_of = [&]( const engine::quant::Nvfp4Tensor& t ) {
		const std::vector<float> q = decode( t );
		std::vector<float> e( W.size() );
		for ( size_t i = 0; i < e.size(); ++i ) e[i] = W[i] - q[i];
		return metric( e, Hm, k, n ) / ref;
	};
	const double pr = proxy_of( rtn ), pc = proxy_of( cal );
	char what[200];
	std::snprintf( what, sizeof( what ), "decoded proxy errors: round to nearest %.5f (reported %.5f), feedback %.5f (reported %.5f, its rtn %.5f)", pr, cal.proxy_rtn, pc,
		cal.proxy_err, cal.proxy_rtn );
	check( std::fabs( pc - cal.proxy_err ) < 1e-3 * pc && std::fabs( pr - cal.proxy_rtn ) < 1e-3 * pr, what );
	check( pc < 0.8 * pr, "the error feedback lowers the proxy error by more than 20%" );

	CUDA_CHECK( cudaFree( d_w ) );
	CUDA_CHECK( cudaFree( d_h ) );
	std::printf( failures ? "test_nvfp4_ldlq: FAIL\n" : "test_nvfp4_ldlq: PASS\n" );
	return failures ? 1 : 0;
}
