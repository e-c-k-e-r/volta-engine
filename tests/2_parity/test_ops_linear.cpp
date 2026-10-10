// Op-level golden test for the Linear module (trellis at 3/4/5/6 bits with MUL1 and MCG, rows 1/4/40, plain FP16, NVFP4).
// ONE source for both configurations through the module API:
//   --dump <dir>   run in the CUDA build: computes with the real kernels, writes golden outputs
//   --check <dir>  recompute with the built backend and compare (tier 2) against the golden files
// Inputs come from GoldenRng with fixed seeds; the packed trellis words are random, which is a valid trellis for any bit width.
#include "golden_io.h"
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/backend/tiers.h"
#include "engine/core/arena.h"
#include "engine/modules/linear.h"
#include <vector>

using namespace engine;
using backend::half_t;

static pod::MemoryArena g_arena{};

static pod::TensorView upload_half( const std::vector<float>& v, std::initializer_list<int> shape ) {
	int s[4] = {};
	int nd = 0;
	for ( int d : shape ) s[nd++] = d;
	pod::TensorView t = core::arena_alloc_tensor( g_arena, pod::DType::FP16, s, nd );
	std::vector<half_t> h( v.size() );
	for ( size_t i = 0; i < v.size(); ++i ) h[i] = backend::from_float_half( v[i] );
	backend::memcpy_sync( t.data, h.data(), h.size() * sizeof( half_t ), backend::Copy::H2D );
	return t;
}

static pod::TensorView upload_codes( const std::vector<uint16_t>& v, int k, int n, int bits ) {
	int s[3] = { k / 16, n / 16, 16 * bits };
	pod::TensorView t = core::arena_alloc_tensor( g_arena, pod::DType::UINT16, s, 3 );
	backend::memcpy_sync( t.data, v.data(), v.size() * sizeof( uint16_t ), backend::Copy::H2D );
	return t;
}

static std::vector<float> download_half( const pod::TensorView& t ) {
	std::vector<half_t> h( t.numel() );
	backend::device_sync();
	backend::memcpy_sync( h.data(), t.data, h.size() * sizeof( half_t ), backend::Copy::D2H );
	std::vector<float> f( h.size() );
	for ( size_t i = 0; i < h.size(); ++i ) f[i] = backend::to_float( h[i] );
	return f;
}

static std::vector<float> uniform_vec( GoldenRng& rng, size_t n, float scale ) {
	std::vector<float> v( n );
	for ( auto& x : v ) x = rng.uniform( scale );
	return v;
}

static std::vector<float> sign_vec( GoldenRng& rng, size_t n ) {
	std::vector<float> v( n );
	for ( auto& x : v ) x = ( rng.next() >> 4 ) & 1 ? 1.0f : -1.0f;
	return v;
}

static std::vector<uint16_t> code_vec( GoldenRng& rng, int k, int n, int bits ) {
	std::vector<uint16_t> c( static_cast<size_t>( k / 16 ) * ( n / 16 ) * 16 * bits );
	for ( auto& v : c ) v = static_cast<uint16_t>( rng.next() );
	return c;
}

struct Case {
	std::string name;
	std::vector<float> out;
};

int main( int argc, char** argv ) {
	GoldenArgs args( argc, argv );
	if ( !args.dump && !args.check ) args.dump = args.check = true;
	ENGINE_ASSERT( core::arena_create( g_arena, 0, 256ull << 20 ), "arena" );
	std::printf( "=== op-level Linear (backend %s) ===\n", ENGINE_BACKEND_NAME );

	const int k = 256, n = 384;
	const int row_list[] = { 1, 4, 40 };
	const int bit_list[] = { 3, 4, 5, 6 };
	std::vector<Case> cases;
	for ( int mcg = 0; mcg < 2; ++mcg ) {
		for ( int bits : bit_list ) {
			GoldenRng rng( 1000u + 17u * static_cast<uint32_t>( bits ) + static_cast<uint32_t>( mcg ) );
			auto codes = upload_codes( code_vec( rng, k, n, bits ), k, n, bits );
			auto suh = upload_half( sign_vec( rng, k ), { k } );
			auto svh = upload_half( sign_vec( rng, n ), { n } );
			const pod::LinearWeight lw = pod::LinearWeight::make_trellis( codes, suh, svh, bits, !mcg, mcg );
			for ( int m : row_list ) {
				auto x = upload_half( uniform_vec( rng, static_cast<size_t>( m ) * k, 1.0f ), { m, k } );
				int os[2] = { m, n };
				pod::TensorView out = core::arena_alloc_tensor( g_arena, pod::DType::FP16, os, 2 );
				modules::Linear::forward( x, lw, out );
				char nm[64];
				std::snprintf( nm, sizeof( nm ), "trellis_%s_b%d_m%d", mcg ? "mcg" : "mul1", bits, m );
				cases.push_back( { nm, download_half( out ) } );
			}
		}
	}
	{ // plain FP16 [k, n]
		GoldenRng rng( 7777u );
		auto w = upload_half( uniform_vec( rng, static_cast<size_t>( k ) * n, 0.25f ), { k, n } );
		const pod::LinearWeight lw = pod::LinearWeight::make_fp16( w );
		for ( int m : row_list ) {
			auto x = upload_half( uniform_vec( rng, static_cast<size_t>( m ) * k, 1.0f ), { m, k } );
			int os[2] = { m, n };
			pod::TensorView out = core::arena_alloc_tensor( g_arena, pod::DType::FP16, os, 2 );
			modules::Linear::forward( x, lw, out );
			char nm[64];
			std::snprintf( nm, sizeof( nm ), "fp16_m%d", m );
			cases.push_back( { nm, download_half( out ) } );
		}
	}

	{ // NVFP4 W4A16 on the prepacked fragment layout: random code bytes, E4M3 block scales with exponents 5 - 9 (no NaN), a global scale
		GoldenRng rng( 9191u );
		const int G = k / 16, tiles = n / 32;
		std::vector<uint8_t> hc( static_cast<size_t>( tiles ) * G * 32 * 8 ), hs( static_cast<size_t>( tiles ) * G * 32 );
		for ( auto& v : hc ) v = static_cast<uint8_t>( rng.next() );
		for ( auto& v : hs ) v = static_cast<uint8_t>( ( ( rng.next() & 1 ) << 7 ) | ( ( 5 + rng.next() % 5 ) << 3 ) | ( rng.next() & 7 ) );
		int cs[3] = { tiles, G, 256 }, ss[3] = { tiles, G, 32 };
		pod::TensorView codes = core::arena_alloc_tensor( g_arena, pod::DType::UINT8, cs, 3 );
		pod::TensorView scales = core::arena_alloc_tensor( g_arena, pod::DType::UINT8, ss, 3 );
		backend::memcpy_sync( codes.data, hc.data(), hc.size(), backend::Copy::H2D );
		backend::memcpy_sync( scales.data, hs.data(), hs.size(), backend::Copy::H2D );
		const pod::LinearWeight lw = pod::LinearWeight::make_nvfp4( codes, scales, 0.0625f );
		for ( int m : row_list ) {
			auto x = upload_half( uniform_vec( rng, static_cast<size_t>( m ) * k, 1.0f ), { m, k } );
			int os[2] = { m, n };
			pod::TensorView out = core::arena_alloc_tensor( g_arena, pod::DType::FP16, os, 2 );
			modules::Linear::forward( x, lw, out );
			char nm[64];
			std::snprintf( nm, sizeof( nm ), "nvfp4_m%d", m );
			cases.push_back( { nm, download_half( out ) } );
		}
	}

	bool pass = true;
	if ( args.dump ) {
		for ( const auto& c : cases ) pass = golden_write( args.dir, "linear_" + c.name, c.out.data(), c.out.size() * sizeof( float ) ) && pass;
		std::printf( "  dumped %zu cases to %s\n", cases.size(), args.dir.c_str() );
	}
	if ( args.check ) {
		for ( const auto& c : cases ) {
			std::vector<uint8_t> raw;
			if ( !golden_read( args.dir, "linear_" + c.name, raw ) ) {
				std::printf( "  %-28s golden missing  FAIL\n", c.name.c_str() );
				pass = false;
				continue;
			}
			std::vector<float> ref( raw.size() / sizeof( float ) );
			std::memcpy( ref.data(), raw.data(), ref.size() * sizeof( float ) );
			pass = golden_compare( c.name.c_str(), ref, c.out ) && pass;
		}
	}
	core::arena_destroy( g_arena );
	std::printf( "%s\n", pass ? "ALL PASS" : "FAILED" );
	return pass ? 0 : 1;
}
