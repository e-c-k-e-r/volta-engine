// Vision tower smoke test: encodes the golden images on one GPU and checks the output is finite, deterministic and on the scale
// of the text embeddings. The semantic check (the model describing the image) is tests/4_server/api_checks.py.
#include "engine/models/qwen3_vision.h"
#include <cstring>
#include "trellis_context.h"
#include "engine/backend/numeric.h"
#include "engine/backend/runtime.h"
#include "engine/runtime/test_driver.h"
#include "engine/modules/vision.h"
#include <random>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>

using namespace engine;
using engine::runtime::test::QWEN_DIR;

int main( int argc, char** argv ) {
	const char* dir = ( argc > 1 ) ? argv[1] : QWEN_DIR;
	const char* image = ( argc > 2 ) ? argv[2] : TEST_DATA_DIR "/image_wide.png";
	backend::set_device( 0 );
	engine::modules::trellis_context_init_all();
	pod::MemoryArena arena;
	engine::core::arena_create( arena, 0, 2ULL << 30 );
	auto stream = backend::stream_create();
	// Tensor-core attention against the CUDA-core reference on random data (m not a multiple of the tile)
	{
		const int m = 1000, H = 16, D = 72, n = m * H * D;
		std::mt19937 rng( 7 );
		std::normal_distribution<float> nd( 0.0f, 1.5f );
		std::vector<backend::half_t> hq( n ), hk( n ), hv( n ), o1( n ), o2( n );
		for ( int i = 0; i < n; ++i ) {
			hq[i] = backend::from_float_half( nd( rng ) );
			hk[i] = backend::from_float_half( nd( rng ) );
			hv[i] = backend::from_float_half( nd( rng ) );
		}
		auto* dq = static_cast<backend::half_t*>( engine::core::arena_alloc( arena, n * 2 ) );
		auto* dk = static_cast<backend::half_t*>( engine::core::arena_alloc( arena, n * 2 ) );
		auto* dv = static_cast<backend::half_t*>( engine::core::arena_alloc( arena, n * 2 ) );
		auto* da = static_cast<backend::half_t*>( engine::core::arena_alloc( arena, n * 2 ) );
		auto* db = static_cast<backend::half_t*>( engine::core::arena_alloc( arena, n * 2 ) );
		backend::memcpy_sync( dq, hq.data(), n * 2, backend::Copy::H2D );
		backend::memcpy_sync( dk, hk.data(), n * 2, backend::Copy::H2D );
		backend::memcpy_sync( dv, hv.data(), n * 2, backend::Copy::H2D );
		const float sc = 1.0f / std::sqrt( 72.0f );
		engine::modules::Vision::attention( dq, dk, dv, da, m, H, D, sc, stream );
		engine::modules::Vision::attention_simt( dq, dk, dv, db, m, H, D, sc, stream );
		backend::stream_sync( stream );
		backend::memcpy_sync( o1.data(), da, n * 2, backend::Copy::D2H );
		backend::memcpy_sync( o2.data(), db, n * 2, backend::Copy::D2H );
		double worst = 0.0, ref = 0.0;
		for ( int i = 0; i < n; ++i ) {
			worst = std::fmax( worst, std::fabs( backend::to_float( o1[i] ) - backend::to_float( o2[i] ) ) );
			ref = std::fmax( ref, std::fabs( backend::to_float( o2[i] ) ) );
		}
		std::printf( "attention tensor-core vs reference: max abs diff %.5f (max |out| %.3f)\n", worst, ref );
		if ( worst > 0.01 * ref + 0.005 ) {
			std::printf( "test_vision: FAIL (attention)\n" );
			return 1;
		}
	}
	models::Qwen3VisionTower tower;
	if ( !tower.load( dir, arena, 4096, stream ) ) {
		std::printf( "FAIL: tower did not load from %s\n", dir );
		return 1;
	}
	std::printf( "arena used %.1f MiB\n", arena.offset_bytes / 1048576.0 );
	std::ifstream f( image, std::ios::binary );
	const std::string bytes( ( std::istreambuf_iterator<char>( f ) ), std::istreambuf_iterator<char>() );
	loader::DecodedImage dec;
	loader::PreparedImage prep;
	std::string err;
	if ( !loader::decode_image( reinterpret_cast<const uint8_t*>( bytes.data() ), bytes.size(), &dec, &err ) || !loader::prepare_image( dec, tower.params(), &prep, &err ) ) {
		std::printf( "FAIL: %s\n", err.c_str() );
		return 1;
	}
	std::vector<backend::half_t> a, b;
	const auto t0 = std::chrono::steady_clock::now();
	if ( !tower.encode( prep, &a, stream ) || !tower.encode( prep, &b, stream ) ) {
		std::printf( "FAIL: encode\n" );
		return 1;
	}
	const double ms = std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count() / 2;
	const int tokens = prep.tokens( 2 ), dim = tower.config().out_hidden;
	double sq = 0.0, maxabs = 0.0;
	bool finite = true, same = true;
	for ( size_t i = 0; i < a.size(); ++i ) {
		const float v = backend::to_float( a[i] );
		if ( !std::isfinite( v ) ) finite = false;
		sq += static_cast<double>( v ) * v;
		maxabs = std::fmax( maxabs, std::fabs( v ) );
		if ( std::memcmp( &a[i], &b[i], sizeof( a[i] ) ) != 0 ) same = false;
	}
	std::printf( "%dx%d -> grid %dx%d, %d tokens x %d: rms %.4f max %.2f, %.1f ms/image, %s %s\n", dec.width, dec.height, prep.grid_h, prep.grid_w, tokens, dim,
		std::sqrt( sq / a.size() ), maxabs, ms, finite ? "finite" : "NONFINITE", same ? "deterministic" : "NONDETERMINISTIC" );
	std::printf( ( finite && same ) ? "test_vision: PASS\n" : "test_vision: FAIL\n" );
	return ( finite && same ) ? 0 : 1;
}
