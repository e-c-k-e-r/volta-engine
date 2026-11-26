// EXL3 trellis linear (Linear::forward, CUDA and Vulkan builds) at a model's projection shapes: time per call over several weight copies
// (cold L2) and the GB/s of the packed tiles. Random words (a valid trellis at any width; timing only).
//   bench_trellis_shapes [--code C] [--volta] [--rows 1,4,8,40,512] [--reps N] [-- K N ...]   default Gemma 4 31B's (4 bits, stock order)
#include "engine/core/arena.h"
#include "engine/backend/numeric.h"
#include "engine/backend/runtime.h"
#include "engine/modules/linear.h"
#include "engine/modules/trellis_bits.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace engine;
using backend::half_t;

int main( int argc, char** argv ) {
	int code = 4, reps = 5;
	bool volta = false;
	std::vector<int> rows = { 1, 4, 8, 40, 512 };
	std::vector<std::pair<int, int>> shapes;
	for ( int i = 1; i < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--code" ) && i + 1 < argc ) code = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--volta" ) ) volta = true;
		else if ( !std::strcmp( argv[i], "--reps" ) && i + 1 < argc ) reps = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--rows" ) && i + 1 < argc ) {
			rows.clear();
			for ( const char* s = argv[++i]; *s; ) { rows.push_back( std::atoi( s ) ); while ( *s && *s != ',' ) ++s; if ( *s ) ++s; }
		}
		else if ( !std::strcmp( argv[i], "--" ) ) {
			for ( ++i; i + 1 < argc; i += 2 ) shapes.push_back( { std::atoi( argv[i] ), std::atoi( argv[i + 1] ) } );
		}
	}
	if ( shapes.empty() ) shapes = { { 5376, 8192 }, { 5376, 21504 }, { 21504, 5376 }, { 8192, 5376 } };   // q, gate / up, down, o
	backend::set_device( 0 );
	pod::MemoryArena arena;
	ENGINE_ASSERT( core::arena_create( arena, 0, 6ULL << 30 ), "arena" );
	std::mt19937 rng( 1234 );
	std::printf( "code %d (%s), %s order\n", code, modules::trellis_is_half( code ) ? "half-integer rate" : "whole bits", volta ? "Volta" : "stock" );
	for ( auto [k, n] : shapes ) {
		const size_t tile_bytes = static_cast<size_t>( modules::trellis_tile_u16( code ) ) * 2;
		const size_t wbytes = static_cast<size_t>( k / 16 ) * ( n / 16 ) * tile_bytes;
		const int copies = static_cast<int>( std::max<size_t>( 2, std::min<size_t>( 8, ( 256ull << 20 ) / wbytes + 1 ) ) );
		std::vector<pod::LinearWeight> ws;
		std::vector<uint16_t> host( wbytes / 2 );
		for ( int c = 0; c < copies; ++c ) {
			for ( auto& v : host ) v = static_cast<uint16_t>( rng() );
			int cs[3] = { k / 16, n / 16, modules::trellis_tile_u16( code ) };
			pod::TensorView codes = core::arena_alloc_tensor( arena, pod::DType::UINT16, cs, 3 );
			backend::memcpy_sync( codes.data, host.data(), wbytes, backend::Copy::H2D );
			std::vector<half_t> sk( k ), sn( n );
			for ( auto& v : sk ) v = backend::from_float_half( ( rng() & 1 ) ? 1.0f : -1.0f );
			for ( auto& v : sn ) v = backend::from_float_half( ( rng() & 1 ) ? 1.0f : -1.0f );
			int ks[1] = { k }, ns[1] = { n };
			pod::TensorView suh = core::arena_alloc_tensor( arena, pod::DType::FP16, ks, 1 );
			pod::TensorView svh = core::arena_alloc_tensor( arena, pod::DType::FP16, ns, 1 );
			backend::memcpy_sync( suh.data, sk.data(), k * 2, backend::Copy::H2D );
			backend::memcpy_sync( svh.data, sn.data(), n * 2, backend::Copy::H2D );
			pod::LinearWeight lw = pod::LinearWeight::make_trellis( codes, suh, svh, code, true, false );
			lw.volta_tiles = volta;
			ws.push_back( lw );
		}
		for ( int m : rows ) {
			int xs[2] = { m, k }, os[2] = { m, n };
			pod::TensorView x = core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
			pod::TensorView out = core::arena_alloc_tensor( arena, pod::DType::FP16, os, 2 );
			std::vector<half_t> hx( static_cast<size_t>( m ) * k );
			for ( auto& v : hx ) v = backend::from_float_half( static_cast<float>( rng() % 2001 ) / 1000.0f - 1.0f );
			backend::memcpy_sync( x.data, hx.data(), hx.size() * 2, backend::Copy::H2D );
			for ( auto& w : ws ) modules::Linear::forward( x, w, out );   // warm-up (pipelines, scratch)
			backend::device_sync();
			double best = 1e30;
			for ( int r = 0; r < reps; ++r ) {
				const auto t0 = std::chrono::steady_clock::now();
				for ( auto& w : ws ) modules::Linear::forward( x, w, out );
				backend::device_sync();
				best = std::min( best, std::chrono::duration<double, std::micro>( std::chrono::steady_clock::now() - t0 ).count() / copies );
			}
			std::printf( "  %5d x %5d  rows %4d: %9.1f us  %7.1f GB/s of tiles\n", k, n, m, best, wbytes / best * 1e-3 );
		}
	}
	core::arena_destroy( arena );
	return 0;
}
