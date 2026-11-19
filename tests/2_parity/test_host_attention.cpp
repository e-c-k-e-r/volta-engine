// The host vision towers' attention (src/models/host_vision_ops.cpp, tiled through the dense product) against a plain double-precision reference, on random
// FP16 inputs: the rows of a few queries per case, for the head widths of the towers (64: GLM and DeepSeek, 72: Qwen), a padded key count (m not a multiple
// of 16) and the largest image of a tower (4480 patches); the time of each call is printed.
//   test_host_attention [cpus (default plan)]
#include "../../src/models/host_vision_ops.h"
#include "engine/kernels/cpu/cpu_util.h"
#include "engine/runtime/cpu_expert_attach.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace engine;
using namespace engine::models::hv;

namespace {

struct Case {
	int m, heads, hd;
	bool slow_reference;   // the reference rows cost m * hd each: a few are enough
};

bool run_case( Par& par, const Case& c ) {
	const int width = c.heads * c.hd;
	std::mt19937 rng( 1234 + c.m );
	std::normal_distribution<float> nd( 0.0f, 1.0f );
	auto rand_half = [&]( size_t n, float mul ) {
		std::vector<uint16_t> v( n );
		for ( auto& x : v ) x = f2h( nd( rng ) * mul );
		return v;
	};
	const std::vector<uint16_t> q = rand_half( static_cast<size_t>( c.m ) * width, 1.0f ), k = rand_half( static_cast<size_t>( c.m ) * width, 1.0f ), v = rand_half( static_cast<size_t>( c.m ) * width, 1.0f );
	std::vector<uint16_t> out( static_cast<size_t>( c.m ) * width, 0 );
	Scratch s;
	const float scale = 1.0f / std::sqrt( static_cast<float>( c.hd ) );
	const auto t0 = std::chrono::steady_clock::now();
	attention( par, s, q.data(), k.data(), v.data(), out.data(), c.m, c.heads, scale, c.hd );
	const double secs = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();

	// reference rows: the first, the last and a few in between, every head
	std::vector<int> rows = { 0, c.m - 1, c.m / 2, 7 % c.m, c.m / 3 };
	double worst = 0.0, worst_rel = 0.0;
	std::vector<double> sc( c.m );
	for ( int r : rows ) {
		for ( int h = 0; h < c.heads; ++h ) {
			double mx = -1e30;
			for ( int j = 0; j < c.m; ++j ) {
				double dot = 0.0;
				for ( int d = 0; d < c.hd; ++d ) dot += static_cast<double>( h2f( q[static_cast<size_t>( r ) * width + h * c.hd + d] ) ) * h2f( k[static_cast<size_t>( j ) * width + h * c.hd + d] );
				sc[j] = dot * scale;
				mx = std::max( mx, sc[j] );
			}
			double sum = 0.0;
			for ( int j = 0; j < c.m; ++j ) {
				sc[j] = std::exp( sc[j] - mx );
				sum += sc[j];
			}
			for ( int d = 0; d < c.hd; ++d ) {
				double acc = 0.0;
				for ( int j = 0; j < c.m; ++j ) acc += sc[j] * h2f( v[static_cast<size_t>( j ) * width + h * c.hd + d] );
				acc /= sum;
				const double got = h2f( out[static_cast<size_t>( r ) * width + h * c.hd + d] );
				// the output is rounded to FP16 (11 bits): the error to compare with is one unit in its last place of the larger of the two
				const double err = std::fabs( got - acc ), ulp = std::max( std::fabs( acc ), 1e-3 ) * 9.8e-4;
				worst = std::max( worst, err );
				worst_rel = std::max( worst_rel, err / ulp );
			}
		}
	}
	const bool ok = worst_rel < 4.0;
	std::printf( "m %5d heads %2d hd %2d: %7.3f s, worst |diff| %.2e (%.2f half-precision ulps) %s\n", c.m, c.heads, c.hd, secs, worst, worst_rel, ok ? "ok" : "FAIL" );
	return ok;
}

} // namespace

int main( int argc, char** argv ) {
	const std::vector<int> cpus = cpu_moe::host_worker_cpus( argc > 1 ? argv[1] : "" );
	Par par( cpus );
	Par::Pin pin( par );
	bool ok = true;
	for ( const Case& c : { Case{ 37, 3, 64, false }, Case{ 300, 4, 72, false }, Case{ 1200, 16, 64, false }, Case{ 2304, 16, 72, false }, Case{ 4480, 16, 64, true } } ) ok = run_case( par, c ) && ok;
	std::printf( "test_host_attention: %s\n", ok ? "PASS" : "FAIL" );
	return ok ? 0 : 1;
}
