// Tier 2: the routed experts' Hessians of the quantizer tools (src/tools/quant/expert_hessian.h) against a double-precision host reference on
// random rows (more than one activation chunk), FP16 rows and weights as the device uses them.
//   (a) the down Hessian: sum over the rows of a a^T, a = silu( x W_gate ) * ( x W_up ), and with GLM's limit
//   (b) the affinity-weighted down Hessian: eps (S / N) sum a a^T + sum p a a^T (p: the expert's routing weight, 0 on the rows not routed to it)
//   (c) the affinity-weighted gate / up Hessian: eps (S / N) H_in + sum over the routed rows of p x x^T / N
#include "engine/config.h"
#include "quant/expert_hessian.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <random>
#include <vector>

namespace {

constexpr int ROWS = 9000, HIDDEN = 256, INTER = 128;
int fails = 0;

void check( double err, double tol, const char* what ) {
	const bool ok = err <= tol;
	std::printf( "  %-58s relative error %.2e (max %.0e)  %s\n", what, err, tol, ok ? "OK" : "FAIL" );
	fails += !ok;
}

double rel( const std::vector<float>& got, const std::vector<double>& want ) {
	double e = 0.0, n = 0.0;
	for ( size_t i = 0; i < want.size(); ++i ) {
		e += ( got[i] - want[i] ) * ( got[i] - want[i] );
		n += want[i] * want[i];
	}
	return std::sqrt( e / std::max( n, 1e-300 ) );
}

float h16( float v ) { return __half2float( __float2half_rn( v ) ); }

} // namespace

int main() {
	std::mt19937 rng( 1234 );
	std::normal_distribution<float> g( 0.0f, 1.0f );
	std::uniform_real_distribution<float> uni( 0.0f, 1.0f );
	std::vector<uint16_t> x16( static_cast<size_t>( ROWS ) * HIDDEN );
	std::vector<float> x( x16.size() ), wg( static_cast<size_t>( HIDDEN ) * INTER ), wu( wg.size() );
	for ( size_t i = 0; i < x.size(); ++i ) {
		const half h = __float2half_rn( g( rng ) );
		std::memcpy( &x16[i], &h, 2 );
		x[i] = __half2float( h );
	}
	const float ws = 1.0f / std::sqrt( static_cast<float>( HIDDEN ) );
	for ( auto& v : wg ) v = ws * g( rng );
	for ( auto& v : wu ) v = ws * g( rng );
	// Routing: about one row in eight to the expert, weights in (0, 1)
	std::vector<float> p( ROWS, 0.0f );
	std::vector<int> routed;
	std::vector<float> routed_w;
	double S = 0.0;
	for ( int r = 0; r < ROWS; ++r ) {
		if ( uni( rng ) < 0.125f ) {
			p[r] = 0.05f + 0.9f * uni( rng );
			routed.push_back( r );
			routed_w.push_back( p[r] );
			S += p[r];
		}
	}
	const float eps = 0.01f;

	void* d_x = nullptr;
	float *d_g = nullptr, *d_u = nullptr, *d_h = nullptr, *d_p = nullptr, *d_hin = nullptr, *d_rw = nullptr;
	int* d_idx = nullptr;
	CUDA_CHECK( cudaMalloc( &d_x, x16.size() * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_g, wg.size() * 4 ) );
	CUDA_CHECK( cudaMalloc( &d_u, wu.size() * 4 ) );
	CUDA_CHECK( cudaMalloc( &d_h, static_cast<size_t>( HIDDEN ) * HIDDEN * 4 ) );
	CUDA_CHECK( cudaMalloc( &d_p, p.size() * 4 ) );
	CUDA_CHECK( cudaMalloc( &d_hin, static_cast<size_t>( HIDDEN ) * HIDDEN * 4 ) );
	CUDA_CHECK( cudaMalloc( &d_rw, routed_w.size() * 4 ) );
	CUDA_CHECK( cudaMalloc( &d_idx, routed.size() * 4 ) );
	CUDA_CHECK( cudaMemcpy( d_x, x16.data(), x16.size() * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_g, wg.data(), wg.size() * 4, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_u, wu.data(), wu.size() * 4, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_p, p.data(), p.size() * 4, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_rw, routed_w.data(), routed_w.size() * 4, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_idx, routed.data(), routed.size() * 4, cudaMemcpyHostToDevice ) );
	std::printf( "[*] test_expert_hessian: %d rows (%zu routed), hidden %d, intermediate %d\n", ROWS, routed.size(), HIDDEN, INTER );

	engine::tools::quant::ExpertHessian eh;
	std::vector<float> got( static_cast<size_t>( HIDDEN ) * HIDDEN );
	for ( float limit : { 0.0f, 1.5f } ) {
		// The activations as the device forms them: FP16 rows and weights, FP32 products
		std::vector<double> a( static_cast<size_t>( ROWS ) * INTER );
		for ( int r = 0; r < ROWS; ++r ) {
			for ( int j = 0; j < INTER; ++j ) {
				double gg = 0.0, uu = 0.0;
				for ( int i = 0; i < HIDDEN; ++i ) {
					gg += static_cast<double>( x[static_cast<size_t>( r ) * HIDDEN + i] ) * h16( wg[static_cast<size_t>( i ) * INTER + j] );
					uu += static_cast<double>( x[static_cast<size_t>( r ) * HIDDEN + i] ) * h16( wu[static_cast<size_t>( i ) * INTER + j] );
				}
				if ( limit > 0.0f ) {
					gg = std::min( gg, static_cast<double>( limit ) );
					uu = std::min( std::max( uu, -static_cast<double>( limit ) ), static_cast<double>( limit ) );
				}
				a[static_cast<size_t>( r ) * INTER + j] = gg / ( 1.0 + std::exp( -gg ) ) * uu;
			}
		}
		std::vector<double> plain( static_cast<size_t>( INTER ) * INTER, 0.0 ), aff( plain.size(), 0.0 );
		for ( int r = 0; r < ROWS; ++r ) {
			const double* ar = &a[static_cast<size_t>( r ) * INTER];
			for ( int i = 0; i < INTER; ++i ) {
				for ( int j = 0; j < INTER; ++j ) {
					plain[static_cast<size_t>( i ) * INTER + j] += ar[i] * ar[j];
					aff[static_cast<size_t>( i ) * INTER + j] += p[r] * ar[i] * ar[j];
				}
			}
		}
		for ( size_t i = 0; i < aff.size(); ++i ) aff[i] += eps * S / ROWS * plain[i];
		got.resize( plain.size() );
		ENGINE_ASSERT( eh.compute( d_x, ROWS, HIDDEN, d_g, d_u, INTER, limit, d_h, nullptr ), "compute failed" );
		CUDA_CHECK( cudaMemcpy( got.data(), d_h, got.size() * 4, cudaMemcpyDeviceToHost ) );
		char what[96];
		std::snprintf( what, sizeof( what ), "(a) down Hessian, limit %.1f", limit );
		check( rel( got, plain ), 1e-5, what );
		ENGINE_ASSERT( eh.compute( d_x, ROWS, HIDDEN, d_g, d_u, INTER, limit, d_h, nullptr, d_p, S, eps ), "compute failed" );
		CUDA_CHECK( cudaMemcpy( got.data(), d_h, got.size() * 4, cudaMemcpyDeviceToHost ) );
		std::snprintf( what, sizeof( what ), "(b) affinity down Hessian, limit %.1f, eps %.2f", limit, eps );
		check( rel( got, aff ), 1e-5, what );
	}

	// (c) gate / up: H_in = the rows' mean x x^T
	std::vector<double> hin( static_cast<size_t>( HIDDEN ) * HIDDEN, 0.0 ), want( hin.size(), 0.0 );
	for ( int r = 0; r < ROWS; ++r ) {
		const float* xr = &x[static_cast<size_t>( r ) * HIDDEN];
		for ( int i = 0; i < HIDDEN; ++i ) {
			for ( int j = 0; j < HIDDEN; ++j ) {
				const double v = static_cast<double>( xr[i] ) * xr[j];
				hin[static_cast<size_t>( i ) * HIDDEN + j] += v / ROWS;
				want[static_cast<size_t>( i ) * HIDDEN + j] += p[r] * v / ROWS;
			}
		}
	}
	std::vector<float> hin_f( hin.begin(), hin.end() );
	for ( size_t i = 0; i < want.size(); ++i ) want[i] += eps * S / ROWS * hin[i];
	CUDA_CHECK( cudaMemcpy( d_hin, hin_f.data(), hin_f.size() * 4, cudaMemcpyHostToDevice ) );
	got.resize( want.size() );
	ENGINE_ASSERT( eh.affinity_input( d_x, ROWS, HIDDEN, d_idx, d_rw, static_cast<int>( routed.size() ), S, eps, d_hin, d_h, nullptr ),
		"affinity_input failed" );
	CUDA_CHECK( cudaDeviceSynchronize() );
	CUDA_CHECK( cudaMemcpy( got.data(), d_h, got.size() * 4, cudaMemcpyDeviceToHost ) );
	check( rel( got, want ), 1e-5, "(c) affinity gate / up Hessian, eps 0.01" );

	for ( void* q : { d_x, (void*) d_g, (void*) d_u, (void*) d_h, (void*) d_p, (void*) d_hin, (void*) d_rw, (void*) d_idx } ) CUDA_CHECK( cudaFree( q ) );
	std::printf( fails ? "=== EXPERT HESSIAN: %d FAILURE(S) ===\n" : "=== EXPERT HESSIAN: PASS ===\n", fails );
	return fails ? 1 : 0;
}
