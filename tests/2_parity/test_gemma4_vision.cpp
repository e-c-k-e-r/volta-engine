// Gemma 4 vision tower against exllamav3 (a script of the form of tests/tools/dsv41/vision_golden.py writes the goldens: the reference's
// pixel values in 0..1 as raster patches [y][x][channel], and its image embeddings):
//   test_gemma4_vision <model dir> <golden dir> <image>
// 1. the engine's preprocessing gives the reference's patch grid and patches (the engine stores 2 v - 1, the embedder's input),
// 2. the tower on the reference's patches matches the reference embeddings (row cosines), 3. on the engine's patches too, on average.
#include "engine/models/gemma4_vision.h"
#include "engine/loader/image.h"
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "trellis_context.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace engine;

namespace {

std::vector<float> read_f32( const std::string& path ) {
	std::ifstream f( path, std::ios::binary );
	std::vector<char> b( ( std::istreambuf_iterator<char>( f ) ), std::istreambuf_iterator<char>() );
	std::vector<float> v( b.size() / sizeof( float ) );
	std::memcpy( v.data(), b.data(), v.size() * sizeof( float ) );
	return v;
}

} // namespace

int main( int argc, char** argv ) {
	if ( argc < 4 ) {
		std::fprintf( stderr, "usage: test_gemma4_vision <model dir> <golden dir> <image>\n" );
		return 2;
	}
	const std::string model = argv[1], gold = argv[2], image = argv[3];
	int w = 0, h = 0, patches = 0, soft = 0, rows = 0, hidden = 0;
	{
		std::ifstream m( gold + "/meta.txt" );
		m >> w >> h >> patches >> soft >> rows >> hidden;
	}
	const std::vector<float> ref_patches = read_f32( gold + "/patches.f32" ), ref_emb = read_f32( gold + "/emb.f32" );
	if ( ref_patches.size() != static_cast<size_t>( patches ) * 768 || ref_emb.size() != static_cast<size_t>( rows ) * hidden ) {
		std::printf( "test_gemma4_vision: FAIL (golden sizes)\n" );
		return 1;
	}

	backend::set_device( 0 );
	engine::modules::trellis_context_init_all();
	pod::MemoryArena arena;
	engine::core::arena_create( arena, 0, 3ULL << 30 );
	auto stream = backend::stream_create();
	models::Gemma4VisionTower tower;
	if ( !tower.load( model.c_str(), arena, 280, stream ) ) { std::printf( "test_gemma4_vision: FAIL (load)\n" ); return 1; }

	// 1. preprocessing
	std::ifstream pf( image, std::ios::binary );
	const std::vector<uint8_t> bytes( ( std::istreambuf_iterator<char>( pf ) ), std::istreambuf_iterator<char>() );
	loader::DecodedImage dec;
	std::string err;
	if ( !loader::decode_image( bytes.data(), bytes.size(), &dec, &err ) ) { std::printf( "test_gemma4_vision: FAIL (decode: %s)\n", err.c_str() ); return 1; }
	loader::PreparedImage mine;
	if ( !loader::prepare_image( dec, tower.params(), &mine, &err ) ) { std::printf( "test_gemma4_vision: FAIL (prepare: %s)\n", err.c_str() ); return 1; }
	std::printf( "preprocessing: %dx%d pixels, %d patches (reference %dx%d, %d)\n", mine.grid_w * 16, mine.grid_h * 16, mine.num_patches(), w, h, patches );
	bool ok = mine.grid_w * 16 == w && mine.grid_h * 16 == h && mine.num_patches() == patches;
	std::vector<float> ref_in( ref_patches.size() );
	for ( size_t i = 0; i < ref_patches.size(); ++i ) ref_in[i] = 2.0f * ref_patches[i] - 1.0f;
	if ( ok ) {
		double maxd = 0, sum = 0;
		for ( size_t i = 0; i < ref_in.size(); ++i ) {
			const double d = std::fabs( mine.patches[i] - ref_in[i] );
			maxd = std::fmax( maxd, d );
			sum += d;
		}
		std::printf( "preprocessing: patches max |diff| %.4f, mean %.6f (one 8-bit step is 0.0078)\n", maxd, sum / ref_in.size() );
		ok = maxd < 0.02;
	}
	if ( !ok ) { std::printf( "test_gemma4_vision: FAIL (preprocessing)\n" ); return 1; }

	// 2. the tower on the reference's patches, 3. on the engine's
	for ( int pass = 0; pass < 2; ++pass ) {
		loader::PreparedImage img = mine;
		if ( pass == 0 ) img.patches = ref_in;
		std::vector<backend::half_t> emb;
		if ( !tower.encode( img, &emb, stream ) || emb.size() != ref_emb.size() ) { std::printf( "test_gemma4_vision: FAIL (encode: %zu values, expected %zu)\n", emb.size(), ref_emb.size() ); return 1; }
		double worst = 1.0, mean = 0.0;
		int worst_row = 0;
		for ( int r = 0; r < rows; ++r ) {
			double dot = 0, na = 0, nb = 0;
			for ( int c = 0; c < hidden; ++c ) {
				const double a = ref_emb[static_cast<size_t>( r ) * hidden + c], b = backend::to_float( emb[static_cast<size_t>( r ) * hidden + c] );
				dot += a * b; na += a * a; nb += b * b;
			}
			const double cs = dot / ( std::sqrt( na ) * std::sqrt( nb ) + 1e-30 );
			if ( cs < worst ) { worst = cs; worst_row = r; }
			mean += cs / rows;
		}
		std::printf( "tower on the %s patches: row cosine worst %.5f (soft token %d), mean %.6f\n", pass == 0 ? "reference's" : "engine's", worst, worst_row, mean );
		// the same patches: every row close; the engine's patches differ from the reference's by one 8-bit step in a few values, which the
		// model itself amplifies on some tokens (the reference under random one-step noise on 5% of the values: worst rows 0.35 - 0.65, mean
		// 0.977), so that pass is held to the mean
		ok = ok && ( pass == 0 ? worst > 0.99 && mean > 0.9999 : mean > 0.998 );
	}
	std::printf( "test_gemma4_vision: %s\n", ok ? "PASS" : "FAIL" );
	return ok ? 0 : 1;
}
