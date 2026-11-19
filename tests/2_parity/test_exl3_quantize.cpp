// Tier 2: the EXL3 quantizer (engine/quant/exl3_quantize.h) on synthetic weights and a correlated Hessian with outlier input channels.
//   (a) the proxy error it reports equals the one of the stored tensors (trellis, suh, svh read back by reconstruct_had_slice) against the
//       mean Hessian (the reported error takes the damping out, which leaves exactly the mean)
//   (b) the error falls with the rate (2 to 5 bits, about 4x per bit) and 2.5 bpw lands between 2 and 3
//   (c) the error feedback beats quantizing without a Hessian, in the same metric
//   (d) a group of two weights gives each the error it gets alone
//   (e) (a) in the stock tile order as well
//   (f) (a) with the input's shared factor instead of the scaled per-weight ones (and both errors printed)
#include "engine/config.h"
#include "engine/quant/exl3_quantize.h"
#include "engine/modules/trellis.h"
#include "engine/modules/trellis_bits.h"
#include "reconstruct.cuh"
#include "trellis_context.h"
#include "exl3_vl.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int K_IN = 512;
constexpr int ROWS = 2048;

struct Problem {
	std::vector<double> H;                  // mean Hessian, k x k
	std::vector<std::vector<float>> W;      // k x n each
	std::vector<int> ns;
	float* d_hsum = nullptr;
	std::vector<float*> d_w;
};

Problem make_problem() {
	Problem p;
	p.ns = { 256, 384 };
	std::mt19937 rng( 99u );
	std::normal_distribution<float> g( 0.0f, 1.0f );
	// Inputs: each channel mixed with one other, eight outlier channels
	std::vector<float> x( static_cast<size_t>( ROWS ) * K_IN );
	std::vector<float> z( K_IN );
	for ( int r = 0; r < ROWS; ++r ) {
		for ( auto& v : z ) v = g( rng );
		for ( int j = 0; j < K_IN; ++j ) {
			float v = z[j] + 0.5f * z[( j * 7 + 3 ) % K_IN];
			if ( j < 8 ) v *= 10.0f;
			x[static_cast<size_t>( r ) * K_IN + j] = v;
		}
	}
	std::vector<float> hsum( static_cast<size_t>( K_IN ) * K_IN );
	p.H.assign( hsum.size(), 0.0 );
	for ( int a = 0; a < K_IN; ++a ) {
		for ( int b = a; b < K_IN; ++b ) {
			double s = 0.0;
			for ( int r = 0; r < ROWS; ++r ) s += static_cast<double>( x[static_cast<size_t>( r ) * K_IN + a] ) * x[static_cast<size_t>( r ) * K_IN + b];
			hsum[static_cast<size_t>( a ) * K_IN + b] = hsum[static_cast<size_t>( b ) * K_IN + a] = static_cast<float>( s );
			p.H[static_cast<size_t>( a ) * K_IN + b] = p.H[static_cast<size_t>( b ) * K_IN + a] = s / ROWS;
		}
	}
	CUDA_CHECK( cudaMalloc( &p.d_hsum, hsum.size() * sizeof( float ) ) );
	CUDA_CHECK( cudaMemcpy( p.d_hsum, hsum.data(), hsum.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
	for ( int n : p.ns ) {
		std::vector<float> w( static_cast<size_t>( K_IN ) * n );
		for ( int r = 0; r < K_IN; ++r ) {
			for ( int c = 0; c < n; ++c ) {
				const float row_scale = r % 7 == 0 ? 3.0f : 1.0f;
				const float col_scale = 0.5f + 0.25f * ( c % 5 );
				w[static_cast<size_t>( r ) * n + c] = 0.02f * g( rng ) * row_scale * col_scale;
			}
		}
		float* d = nullptr;
		CUDA_CHECK( cudaMalloc( &d, w.size() * sizeof( float ) ) );
		CUDA_CHECK( cudaMemcpy( d, w.data(), w.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
		p.W.push_back( std::move( w ) );
		p.d_w.push_back( d );
	}
	return p;
}

// tr(E^T H E) / tr(W^T H W) of the stored tensor against the mean Hessian, from its trellis and scales read back on the GPU
double stored_proxy( const Problem& p, int idx, const engine::quant::Exl3Tensor& t, int code, int cb, bool volta ) {
	const int k = t.k, n = t.n;
	uint16_t *d_trellis = nullptr, *d_suh = nullptr, *d_svh = nullptr;
	half* d_w = nullptr;
	CUDA_CHECK( cudaMalloc( &d_trellis, t.trellis.size() * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_suh, k * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_svh, n * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_w, static_cast<size_t>( k ) * n * 2 ) );
	CUDA_CHECK( cudaMemcpy( d_trellis, t.trellis.data(), t.trellis.size() * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_suh, t.suh.data(), k * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_svh, t.svh.data(), n * 2, cudaMemcpyHostToDevice ) );
	EXL3_TILES_CALL( volta, reconstruct_had_slice, d_w, d_trellis, reinterpret_cast<const half*>( d_suh ), reinterpret_cast<const half*>( d_svh ),
		code, cb == 1, cb == 2, k, n, 0, nullptr );
	std::vector<half> wq( static_cast<size_t>( k ) * n );
	CUDA_CHECK( cudaMemcpy( wq.data(), d_w, wq.size() * 2, cudaMemcpyDeviceToHost ) );
	CUDA_CHECK( cudaFree( d_trellis ) );
	CUDA_CHECK( cudaFree( d_suh ) );
	CUDA_CHECK( cudaFree( d_svh ) );
	CUDA_CHECK( cudaFree( d_w ) );
	const auto& W = p.W[idx];
	std::vector<double> E( W.size() );
	for ( size_t i = 0; i < W.size(); ++i ) E[i] = static_cast<double>( W[i] ) - __half2float( wq[i] );
	double num = 0.0, den = 0.0;
	std::vector<double> he( n ), hw( n );
	for ( int a = 0; a < k; ++a ) {
		std::fill( he.begin(), he.end(), 0.0 );
		std::fill( hw.begin(), hw.end(), 0.0 );
		for ( int b = 0; b < k; ++b ) {
			const double h = p.H[static_cast<size_t>( a ) * k + b];
			const double* e = &E[static_cast<size_t>( b ) * n];
			const float* w = &W[static_cast<size_t>( b ) * n];
			for ( int c = 0; c < n; ++c ) {
				he[c] += h * e[c];
				hw[c] += h * w[c];
			}
		}
		for ( int c = 0; c < n; ++c ) {
			num += E[static_cast<size_t>( a ) * n + c] * he[c];
			den += W[static_cast<size_t>( a ) * n + c] * hw[c];
		}
	}
	return num / den;
}

bool close( double a, double b, double tol ) { return std::fabs( a - b ) <= tol * std::fabs( b ); }

} // namespace

int main() {
	engine::modules::trellis_context_init_all();
#if ENGINE_EXL3_VL
	const bool volta_ok = engine::modules::trellis_volta_tiles_supported();
#else
	const bool volta_ok = false;
#endif
	std::printf( "[*] test_exl3_quantize: k %d, weights 512 x 256 and 512 x 384, %d calibration rows\n", K_IN, ROWS );
	Problem p = make_problem();
	int fails = 0;
	auto check = [&]( bool ok, const char* what ) {
		std::printf( "  %-78s %s\n", what, ok ? "OK" : "FAIL" );
		fails += !ok;
	};
	char line[256];

	engine::quant::Exl3InputFactor f, f_none;
	ENGINE_ASSERT( engine::quant::exl3_input_factor( f, p.d_hsum, K_IN, ROWS, 0.025f, 7, nullptr ), "factorization failed" );
	ENGINE_ASSERT( !f.fallback, "the synthetic Hessian should factor" );
	ENGINE_ASSERT( engine::quant::exl3_input_factor( f_none, nullptr, K_IN, 0, 0.025f, 7, nullptr ), "fallback factor failed" );

	const std::vector<const float*> both = { p.d_w[0], p.d_w[1] };
	const std::vector<uint64_t> seeds = { 11, 12 };
	const bool volta = volta_ok;

	// (a), (b): the rate ladder, reported against stored
	double proxy[6] = {};
	for ( int code = 2; code <= 5; ++code ) {
		engine::quant::Exl3QuantOptions o;
		o.code = code;
		o.volta = volta;
		std::vector<engine::quant::Exl3Tensor> out;
		ENGINE_ASSERT( engine::quant::exl3_quantize_group( f, both, p.ns, seeds, o, out, nullptr ), "quantize failed" );
		for ( int i = 0; i < 2; ++i ) {
			const double stored = stored_proxy( p, i, out[i], code, o.cb, volta );
			std::snprintf( line, sizeof( line ), "(a) %d bits, weight %d: reported %.6f, stored %.6f, g %.3f", code, i, out[i].proxy_err, stored,
				out[i].g_scale );
			check( close( out[i].proxy_err, stored, 0.03 ), line );
		}
		proxy[code] = out[0].proxy_err;
	}
	for ( int code = 2; code < 5; ++code ) {
		const double r = proxy[code] / proxy[code + 1];
		std::snprintf( line, sizeof( line ), "(b) %d -> %d bits: error ratio %.2f (2.5 .. 8)", code, code + 1, r );
		check( r >= 2.5 && r <= 8.0, line );
	}
	{
		engine::quant::Exl3QuantOptions o;
		o.code = 10;
		o.volta = volta;
		std::vector<engine::quant::Exl3Tensor> out;
		ENGINE_ASSERT( engine::quant::exl3_quantize_group( f, { p.d_w[0] }, { p.ns[0] }, { seeds[0] }, o, out, nullptr ), "quantize failed" );
		const double stored = stored_proxy( p, 0, out[0], 10, o.cb, volta );
		std::snprintf( line, sizeof( line ), "(b) 2.5 bpw: reported %.6f, stored %.6f, between 3 bits %.6f and 2 bits %.6f", out[0].proxy_err, stored,
			proxy[3], proxy[2] );
		check( close( out[0].proxy_err, stored, 0.03 ) && out[0].proxy_err < proxy[2] && out[0].proxy_err > proxy[3], line );
	}

	// (c) feedback against none, at 3 bits, in the Hessian metric
	{
		engine::quant::Exl3QuantOptions o;
		o.code = 3;
		o.volta = volta;
		std::vector<engine::quant::Exl3Tensor> out;
		ENGINE_ASSERT( engine::quant::exl3_quantize_group( f_none, { p.d_w[0] }, { p.ns[0] }, { seeds[0] }, o, out, nullptr ), "quantize failed" );
		const double plain = stored_proxy( p, 0, out[0], 3, o.cb, volta );
		std::snprintf( line, sizeof( line ), "(c) 3 bits: with the Hessian %.6f, without %.6f (at most 0.9x)", proxy[3], plain );
		check( proxy[3] <= 0.9 * plain, line );
	}

	// (d) weight 0 alone against its error in the group
	{
		engine::quant::Exl3QuantOptions o;
		o.code = 3;
		o.volta = volta;
		std::vector<engine::quant::Exl3Tensor> out;
		ENGINE_ASSERT( engine::quant::exl3_quantize_group( f, { p.d_w[0] }, { p.ns[0] }, { seeds[0] }, o, out, nullptr ), "quantize failed" );
		std::snprintf( line, sizeof( line ), "(d) 3 bits, weight 0 alone %.6f, in the group %.6f", out[0].proxy_err, proxy[3] );
		check( close( out[0].proxy_err, proxy[3], 0.03 ), line );
	}

	// (f) the input's shared factor (scaled_hessian off, exllamav3 before 1.6.7): stored against reported, and its error next to the scaled one's
	{
		engine::quant::Exl3InputFactor f_shared;
		ENGINE_ASSERT( engine::quant::exl3_input_factor( f_shared, p.d_hsum, K_IN, ROWS, 0.025f, 7, nullptr, true ), "shared factorization failed" );
		engine::quant::Exl3QuantOptions o;
		o.code = 3;
		o.volta = volta;
		o.scaled_hessian = false;
		std::vector<engine::quant::Exl3Tensor> out;
		ENGINE_ASSERT( engine::quant::exl3_quantize_group( f_shared, both, p.ns, seeds, o, out, nullptr ), "quantize failed" );
		const double stored = stored_proxy( p, 0, out[0], 3, o.cb, volta );
		std::snprintf( line, sizeof( line ), "(f) 3 bits, shared factor: reported %.6f, stored %.6f", out[0].proxy_err, stored );
		check( close( out[0].proxy_err, stored, 0.03 ), line );
		std::printf( "  (f) 3 bits, weight 0: scaled factors %.6f, shared factor %.6f (%+.1f%%)\n", proxy[3], out[0].proxy_err,
			100.0 * ( proxy[3] / out[0].proxy_err - 1.0 ) );
		engine::quant::exl3_input_factor_free( f_shared );
	}

	// (e) the stock order
	if ( volta ) {
		engine::quant::Exl3QuantOptions o;
		o.code = 3;
		o.volta = false;
		std::vector<engine::quant::Exl3Tensor> out;
		ENGINE_ASSERT( engine::quant::exl3_quantize_group( f, { p.d_w[0] }, { p.ns[0] }, { seeds[0] }, o, out, nullptr ), "quantize failed" );
		const double stored = stored_proxy( p, 0, out[0], 3, o.cb, false );
		std::snprintf( line, sizeof( line ), "(e) 3 bits, stock order: reported %.6f, stored %.6f", out[0].proxy_err, stored );
		check( close( out[0].proxy_err, stored, 0.03 ), line );
	}

	engine::quant::exl3_input_factor_free( f );
	engine::quant::exl3_input_factor_free( f_none );
	std::printf( fails ? "=== EXL3 QUANTIZE: %d FAILURE(S) ===\n" : "=== EXL3 QUANTIZE: PASS ===\n", fails );
	return fails ? 1 : 0;
}
