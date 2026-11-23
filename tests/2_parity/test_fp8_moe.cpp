// FP8 routed experts (FP8MoE::forward) against a host reference: pseudo-random E4M3 weights with F32 128 x 128 block scales, routes to a rank
// holding experts [first, first + local) of `global`, at a decode row, a few rows, a prefill chunk and a chunk over two passes. The reference
// dequantizes the weights exactly and sums in double, rounding the gate and up products and the activation to FP16 where the kernels store
// them; the remaining differences are the FP32 accumulation orders.
//   test_fp8_moe
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/modules/fp8_moe.h"
#include "engine/quant/fp4_fp8.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

using namespace engine;
using backend::half_t;

namespace {

struct Expert {
	std::vector<uint8_t> q[3];   // gate [inter, hidden], up [inter, hidden], down [hidden, inter]
	std::vector<float> s[3];     // [rows / 128, cols / 128]
};

float h16( double v ) { return backend::to_float( backend::from_float_half( static_cast<float>( v ) ) ); }

float e4m3_lut[256];

// row r of a projection . x (k values), the weight dequantized exactly
double dot( const Expert& ex, int p, int r, int k, const float* x ) {
	double acc = 0.0;
	const uint8_t* q = ex.q[p].data() + static_cast<size_t>( r ) * k;
	const float* s = ex.s[p].data() + static_cast<size_t>( r / 128 ) * ( k / 128 );
	for ( int c0 = 0; c0 < k; c0 += 128 ) {
		double blk = 0.0;
		for ( int c = c0; c < c0 + 128; ++c ) blk += static_cast<double>( e4m3_lut[q[c]] ) * x[c];
		acc += blk * s[c0 / 128];
	}
	return acc;
}

bool run_case( int hidden, int inter, int global, int local, int first, int top_k, int rows, std::mt19937& rng ) {
	std::uniform_int_distribution<int> byte( 0, 255 );
	std::uniform_real_distribution<float> unit( -1.0f, 1.0f ), scale( 0.5e-3f, 2e-3f ), wdist( 0.05f, 0.3f );
	std::vector<Expert> ex( local );
	for ( Expert& e : ex ) {
		for ( int p = 0; p < 3; ++p ) {
			const int n = p < 2 ? inter : hidden, k = p < 2 ? hidden : inter;
			e.q[p].resize( static_cast<size_t>( n ) * k );
			for ( auto& b : e.q[p] ) { int v; do v = byte( rng ); while ( ( v & 0x7f ) == 0x7f ); b = static_cast<uint8_t>( v ); }
			e.s[p].resize( static_cast<size_t>( n / 128 ) * ( k / 128 ) );
			for ( auto& v : e.s[p] ) v = scale( rng );
		}
	}
	std::vector<half_t> x( static_cast<size_t>( rows ) * hidden );
	for ( auto& v : x ) v = backend::from_float_half( unit( rng ) );
	std::vector<int> topk( static_cast<size_t>( rows ) * top_k );
	std::vector<half_t> tw( topk.size() );
	for ( int r = 0; r < rows; ++r ) {
		std::set<int> pick;
		while ( static_cast<int>( pick.size() ) < top_k ) pick.insert( static_cast<int>( rng() % global ) );
		std::vector<int> order( pick.begin(), pick.end() );
		std::shuffle( order.begin(), order.end(), rng );
		for ( int j = 0; j < top_k; ++j ) {
			topk[static_cast<size_t>( r ) * top_k + j] = order[j];
			tw[static_cast<size_t>( r ) * top_k + j] = backend::from_float_half( wdist( rng ) );
		}
	}
	// device copies and the pointer table (gate codes / scales, up, down)
	std::vector<void*> owned;
	auto put = [&]( const void* h, size_t bytes ) {
		void* d = nullptr;
		CUDA_CHECK( cudaMalloc( &d, bytes ) );
		CUDA_CHECK( cudaMemcpy( d, h, bytes, cudaMemcpyHostToDevice ) );
		owned.push_back( d );
		return d;
	};
	std::vector<const void*> table( 6 * static_cast<size_t>( local ) );
	for ( int e = 0; e < local; ++e ) {
		for ( int p = 0; p < 3; ++p ) {
			table[( 2 * p ) * local + e] = put( ex[e].q[p].data(), ex[e].q[p].size() );
			table[( 2 * p + 1 ) * local + e] = put( ex[e].s[p].data(), ex[e].s[p].size() * sizeof( float ) );
		}
	}
	const void* const* d_table = static_cast<const void* const*>( put( table.data(), table.size() * sizeof( void* ) ) );
	engine::modules::FP8ExpertTable t;
	t.gate_c = reinterpret_cast<const uint8_t* const*>( d_table + 0 * local ); t.gate_s = reinterpret_cast<const float* const*>( d_table + 1 * local );
	t.up_c = reinterpret_cast<const uint8_t* const*>( d_table + 2 * local );   t.up_s = reinterpret_cast<const float* const*>( d_table + 3 * local );
	t.down_c = reinterpret_cast<const uint8_t* const*>( d_table + 4 * local ); t.down_s = reinterpret_cast<const float* const*>( d_table + 5 * local );
	t.num_local = local;
	t.min_expert = first;
	t.num_global = global;
	t.hidden = hidden;
	t.inter = inter;
	const half_t* d_x = static_cast<const half_t*>( put( x.data(), x.size() * sizeof( half_t ) ) );
	const int* d_topk = static_cast<const int*>( put( topk.data(), topk.size() * sizeof( int ) ) );
	const half_t* d_w = static_cast<const half_t*>( put( tw.data(), tw.size() * sizeof( half_t ) ) );
	void* scratch = nullptr;
	CUDA_CHECK( cudaMalloc( &scratch, engine::modules::FP8MoE::scratch_bytes( top_k, hidden, inter ) ) );
	owned.push_back( scratch );
	half_t* d_out = nullptr;
	CUDA_CHECK( cudaMalloc( &d_out, x.size() * sizeof( half_t ) ) );
	owned.push_back( d_out );
	uint32_t* d_heat = nullptr;
	CUDA_CHECK( cudaMalloc( &d_heat, local * sizeof( uint32_t ) ) );
	CUDA_CHECK( cudaMemset( d_heat, 0, local * sizeof( uint32_t ) ) );
	owned.push_back( d_heat );
	engine::modules::FP8MoE::forward( t, d_x, rows, d_topk, d_w, top_k, d_out, scratch, d_heat, 0.0f, nullptr );
	CUDA_CHECK( cudaDeviceSynchronize() );
	std::vector<half_t> out( x.size() );
	std::vector<uint32_t> heat( local );
	CUDA_CHECK( cudaMemcpy( out.data(), d_out, out.size() * sizeof( half_t ), cudaMemcpyDeviceToHost ) );
	CUDA_CHECK( cudaMemcpy( heat.data(), d_heat, heat.size() * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
	for ( void* p : owned ) CUDA_CHECK( cudaFree( p ) );

	// reference
	double e2 = 0.0, r2 = 0.0, worst = 0.0;
	std::vector<uint32_t> heat_ref( local, 0 );
	std::vector<float> xf( hidden ), a( inter );
	std::vector<double> ref( hidden );
	for ( int r = 0; r < rows; ++r ) {
		for ( int c = 0; c < hidden; ++c ) xf[c] = backend::to_float( x[static_cast<size_t>( r ) * hidden + c] );
		std::fill( ref.begin(), ref.end(), 0.0 );
		for ( int j = 0; j < top_k; ++j ) {
			const int e = topk[static_cast<size_t>( r ) * top_k + j] - first;
			if ( e < 0 || e >= local ) continue;
			++heat_ref[e];
			for ( int i = 0; i < inter; ++i ) {
				const double g = h16( dot( ex[e], 0, i, hidden, xf.data() ) ), u = h16( dot( ex[e], 1, i, hidden, xf.data() ) );
				a[i] = h16( g / ( 1.0 + std::exp( -g ) ) * u );
			}
			const double w = backend::to_float( tw[static_cast<size_t>( r ) * top_k + j] );
			for ( int c = 0; c < hidden; ++c ) ref[c] += w * dot( ex[e], 2, c, inter, a.data() );
		}
		double rr = 0.0, ee = 0.0;
		for ( int c = 0; c < hidden; ++c ) {
			const double got = backend::to_float( out[static_cast<size_t>( r ) * hidden + c] ), d = got - ref[c];
			ee += d * d;
			rr += ref[c] * ref[c];
		}
		e2 += ee;
		r2 += rr;
		if ( rr > 0.0 ) worst = std::max( worst, std::sqrt( ee / rr ) );
	}
	const double rel = std::sqrt( e2 / std::max( r2, 1e-30 ) );
	const bool heat_ok = heat == heat_ref;
	const bool ok = rel < 2e-3 && worst < 5e-3 && heat_ok;
	std::printf( "  hidden %d inter %d, experts %d..%d of %d, top %d, %4d rows: rel rms %.2e (worst row %.2e), heat %s  %s\n", hidden, inter, first, first + local - 1, global,
		top_k, rows, rel, worst, heat_ok ? "ok" : "WRONG", ok ? "PASS" : "FAIL" );
	return ok;
}

}   // namespace

int main() {
	CUDA_CHECK( cudaSetDevice( 0 ) );
	for ( int b = 0; b < 256; ++b ) e4m3_lut[b] = quant::e4m3_to_float( static_cast<unsigned>( b ) );
	std::mt19937 rng( 5 );
	bool ok = true;
	// GLM-5.3-Flash's expert widths (fewer experts, for the host reference)
	for ( int rows : { 1, 5, 40 } ) ok = run_case( 4096, 2048, 16, 8, 4, 8, rows, rng ) && ok;
	// a chunk over two passes of 1024 rows (32-route items), narrower
	ok = run_case( 1024, 512, 32, 8, 8, 8, 1100, rng ) && ok;
	std::printf( "%s\n", ok ? "PASS" : "FAIL" );
	return ok ? 0 : 1;
}
