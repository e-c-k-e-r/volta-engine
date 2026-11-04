// Router steering in MoERouter::route_topk (GLM's sigmoid router): a per-slot bias on the selection score. A zero table selects exactly what the
// router selects without one; a -1 bias on a chosen expert removes it from that slot's rows only; a +1 bias on an unchosen expert brings it in; the
// weights stay the raw scores renormalised over the selection.
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/core/arena.h"
#include "engine/core/batch.h"
#include "engine/modules/moe.h"
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace engine;
using backend::half_t;

namespace {

uint32_t g_seed = 11;
float rnd() { g_seed = g_seed * 1664525u + 1013904223u; return ( static_cast<float>( g_seed >> 8 ) / 16777216.0f ) - 0.5f; }

}   // namespace

int main() {
	std::printf( "=== router steering (per-slot selection bias) ===\n" );
	backend::set_device( 0 );
	const int H = 1024, E = 288, K = 8, M = 6, S = 2;
	pod::MemoryArena arena{};
	ENGINE_ASSERT( core::arena_create( arena, 0, size_t( 64 ) << 20 ), "arena" );
	auto upload = [&]( const std::vector<float>& v, std::initializer_list<int> shape ) {
		std::vector<half_t> h( v.size() );
		for ( size_t i = 0; i < v.size(); ++i ) h[i] = backend::from_float_half( v[i] );
		int dims[2]; int nd = 0;
		for ( int d : shape ) dims[nd++] = d;
		pod::TensorView t = core::arena_alloc_tensor( arena, pod::DType::FP16, dims, nd );
		backend::memcpy_sync( t.data, h.data(), h.size() * sizeof( half_t ), backend::Copy::H2D );
		return t;
	};
	std::vector<float> xv( static_cast<size_t>( M ) * H ), wv( static_cast<size_t>( E ) * H ), bv( E );
	for ( float& v : xv ) v = rnd();
	for ( float& v : wv ) v = rnd() * 0.1f;
	for ( float& v : bv ) v = rnd() * 0.02f;
	const pod::TensorView x = upload( xv, { M, H } ), w = upload( wv, { H, E } ), b = upload( bv, { E } );   // router weights are [hidden][experts]
	int* d_idx = static_cast<int*>( core::arena_alloc( arena, static_cast<size_t>( M ) * K * sizeof( int ) ) );
	half_t* d_w = static_cast<half_t*>( core::arena_alloc( arena, static_cast<size_t>( M ) * K * sizeof( half_t ) ) );
	float* d_bias = static_cast<float*>( core::arena_alloc( arena, static_cast<size_t>( S ) * E * sizeof( float ) ) );
	pod::BatchRows rows;
	rows.count = M;
	for ( int r = 0; r < M; ++r ) { rows.slot[r] = r % S; rows.pos[r] = r; }   // rows alternate between slot 0 and slot 1
	pod::BatchRows* d_rows = static_cast<pod::BatchRows*>( core::arena_alloc( arena, sizeof( pod::BatchRows ) ) );
	backend::memcpy_sync( d_rows, &rows, sizeof( rows ), backend::Copy::H2D );

	auto run = [&]( const std::vector<float>* bias, std::vector<int>* idx, std::vector<float>* wt ) {
		if ( bias ) backend::memcpy_sync( d_bias, bias->data(), bias->size() * sizeof( float ), backend::Copy::H2D );
		modules::MoERouter::route_topk( x, w, b, d_idx, d_w, H, E, K, 2.5f, nullptr, M, bias ? d_bias : nullptr, bias ? d_rows : nullptr, E );
		backend::device_sync();
		idx->resize( static_cast<size_t>( M ) * K );
		std::vector<half_t> hw( idx->size() );
		backend::memcpy_sync( idx->data(), d_idx, idx->size() * sizeof( int ), backend::Copy::D2H );
		backend::memcpy_sync( hw.data(), d_w, hw.size() * sizeof( half_t ), backend::Copy::D2H );
		wt->resize( hw.size() );
		for ( size_t i = 0; i < hw.size(); ++i ) ( *wt )[i] = backend::to_float( hw[i] );
	};
	int fails = 0;
	auto expect = [&]( bool c, const char* what ) { std::printf( "  %-62s %s\n", what, c ? "PASS" : "FAIL" ); fails += c ? 0 : 1; };

	std::vector<int> i0, i1;
	std::vector<float> w0, w1;
	run( nullptr, &i0, &w0 );
	std::vector<float> zero( static_cast<size_t>( S ) * E, 0.0f );
	run( &zero, &i1, &w1 );
	expect( i0 == i1 && w0 == w1, "a zero table selects and weighs exactly as no table" );

	// slot 0: remove its rows' first choice of row 0 (-1), bring in an expert row 0 does not pick (+1); slot 1 untouched
	const int removed = i0[0];
	int added = 0;
	while ( std::find( i0.begin(), i0.begin() + K, added ) != i0.begin() + K ) ++added;
	std::vector<float> bias( zero );
	bias[removed] = -1.0f;
	bias[added] = 1.0f;
	run( &bias, &i1, &w1 );
	bool slot0_ok = true, slot1_same = true, weights_ok = true;
	for ( int r = 0; r < M; ++r ) {
		const int* sel = &i1[static_cast<size_t>( r ) * K];
		const bool has_removed = std::find( sel, sel + K, removed ) != sel + K, has_added = std::find( sel, sel + K, added ) != sel + K;
		if ( r % S == 0 ) slot0_ok = slot0_ok && !has_removed && has_added;
		else slot1_same = slot1_same && std::equal( sel, sel + K, &i0[static_cast<size_t>( r ) * K] );
		// weights: raw sigmoid scores of the selection, renormalised to 2.5
		double sum = 0.0;
		std::vector<double> raw( K );
		for ( int k = 0; k < K; ++k ) {
			double logit = 0.0;
			for ( int h = 0; h < H; ++h ) logit += static_cast<double>( backend::to_float( backend::from_float_half( xv[static_cast<size_t>( r ) * H + h] ) ) ) * backend::to_float( backend::from_float_half( wv[static_cast<size_t>( h ) * E + sel[k]] ) );
			raw[k] = 1.0 / ( 1.0 + std::exp( -logit ) );
			sum += raw[k];
		}
		for ( int k = 0; k < K; ++k ) weights_ok = weights_ok && std::fabs( w1[static_cast<size_t>( r ) * K + k] - 2.5 * raw[k] / sum ) < 2e-3;
	}
	expect( slot0_ok, "slot 0's rows drop the -1 expert and take the +1 expert" );
	expect( slot1_same, "slot 1's rows select as before" );
	expect( weights_ok, "weights are the raw scores of the selection, renormalised" );
	std::printf( "%s\n", fails ? "[-] router steering FAIL" : "[+] router steering PASS" );
	return fails ? 1 : 0;
}
