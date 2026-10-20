// DeepSeek-V4.1 vision tower against the checkpoint reference (tests/tools/dsv41/vision_golden.py writes the goldens):
//   test_dsv41_vision <model dir> <golden dir>
// 1. the engine's preprocessing of image.png gives the reference's patch grid and (up to the reference's BF16 rounding) its patches,
// 2. the tower on the reference's own patches matches the reference embeddings evaluated in FP32 (row cosines),
// 3. the tower on the engine's patches matches too, and the span carries the learned delimiter rows in the right places.
#include "engine/models/deepseek_v41_vision.h"
#include "engine/backend/numeric.h"
#include "engine/backend/runtime.h"
#include "engine/core/arena.h"
#include "engine/loader/image.h"
#include "trellis_context.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
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

// cosine, and the largest |a - b| over the row's norm / sqrt(n) (an error per element in units of the row's rms)
void compare_rows( const std::vector<float>& ref, const std::vector<backend::half_t>& got, size_t first_got_row, size_t rows, size_t cols, double* worst_cos, double* worst_rel ) {
	*worst_cos = 1.0;
	*worst_rel = 0.0;
	for ( size_t r = 0; r < rows; ++r ) {
		double dot = 0, na = 0, nb = 0, maxd = 0;
		for ( size_t c = 0; c < cols; ++c ) {
			const double a = ref[r * cols + c], b = backend::to_float( got[( first_got_row + r ) * cols + c] );
			dot += a * b;
			na += a * a;
			nb += b * b;
			maxd = std::fmax( maxd, std::fabs( a - b ) );
		}
		*worst_cos = std::fmin( *worst_cos, dot / ( std::sqrt( na ) * std::sqrt( nb ) + 1e-30 ) );
		*worst_rel = std::fmax( *worst_rel, maxd / ( std::sqrt( na / cols ) + 1e-30 ) );
	}
}

} // namespace

int main( int argc, char** argv ) {
	if ( argc < 3 ) {
		std::fprintf( stderr, "usage: test_dsv41_vision <model dir> <golden dir>\n" );
		return 2;
	}
	const std::string model = argv[1], gold = argv[2];
	int n_h = 0, n_w = 0, llm_h = 0, llm_w = 0, hidden = 0;
	{
		std::ifstream m( gold + "/meta.txt" );
		m >> n_h >> n_w >> llm_h >> llm_w >> hidden;
	}
	// the FP32 evaluation of the reference is the truth: the reference model itself runs in BF16 and is noisy on a few rows
	const std::vector<float> ref_patches = read_f32( gold + "/patches.f32" ), ref_emb = read_f32( gold + "/emb32.f32" );
	bool ok = ref_patches.size() == static_cast<size_t>( n_h ) * n_w * 588 && ref_emb.size() == static_cast<size_t>( llm_h ) * llm_w * hidden;
	if ( !ok ) { std::printf( "test_dsv41_vision: FAIL (golden sizes)\n" ); return 1; }

	backend::set_device( 0 );
	engine::modules::trellis_context_init_all();
	pod::MemoryArena arena;
	engine::core::arena_create( arena, 0, 3ULL << 30 );
	auto stream = backend::stream_create();
	models::DeepseekV41VisionTower tower;
	if ( !tower.load( model.c_str(), arena, 1024, stream ) ) { std::printf( "test_dsv41_vision: FAIL (load)\n" ); return 1; }

	// 1. preprocessing
	std::ifstream pf( gold + "/image.png", std::ios::binary );
	const std::vector<uint8_t> png( ( std::istreambuf_iterator<char>( pf ) ), std::istreambuf_iterator<char>() );
	loader::DecodedImage dec;
	std::string err;
	if ( !loader::decode_image( png.data(), png.size(), &dec, &err ) ) { std::printf( "test_dsv41_vision: FAIL (decode: %s)\n", err.c_str() ); return 1; }
	loader::PreparedImage mine;
	if ( !loader::prepare_image( dec, tower.params(), &mine, &err ) ) { std::printf( "test_dsv41_vision: FAIL (prepare: %s)\n", err.c_str() ); return 1; }
	std::printf( "preprocessing: patch grid %dx%d (reference %dx%d), token grid %dx%d (reference %dx%d)\n", mine.grid_h, mine.grid_w, n_h, n_w, mine.llm_h, mine.llm_w, llm_h, llm_w );
	ok = mine.grid_h == n_h && mine.grid_w == n_w && mine.llm_h == llm_h && mine.llm_w == llm_w && mine.patches.size() == ref_patches.size();
	if ( ok ) {
		double maxd = 0, sum = 0;
		for ( size_t i = 0; i < ref_patches.size(); ++i ) {
			const double d = std::fabs( mine.patches[i] - ref_patches[i] );
			maxd = std::fmax( maxd, d );
			sum += d;
		}
		std::printf( "preprocessing: patches max |diff| %.4f, mean %.5f (the reference rounds to BF16: up to 0.008)\n", maxd, sum / ref_patches.size() );
		size_t big = 0, shown = 0;
		for ( size_t i = 0; i < ref_patches.size(); ++i ) {
			if ( std::fabs( mine.patches[i] - ref_patches[i] ) <= 0.02 ) continue;
			++big;
			if ( shown++ < 6 ) std::printf( "  patch %zu channel %zu y %zu x %zu: engine %.4f reference %.4f\n", i / 588, ( i % 588 ) / 196, ( i % 196 ) / 14, i % 14, mine.patches[i], ref_patches[i] );
		}
		std::printf( "preprocessing: %zu of %zu values differ by more than 0.02\n", big, ref_patches.size() );
		ok = maxd < 0.02;
	}
	if ( !ok ) { std::printf( "test_dsv41_vision: FAIL (preprocessing)\n" ); return 1; }

	// 2. the tower on the reference's patches, 3. on the engine's
	for ( int pass = 0; pass < 2; ++pass ) {
		loader::PreparedImage img = mine;
		if ( pass == 0 ) img.patches = ref_patches;
		std::vector<backend::half_t> span;
		if ( !tower.encode( img, &span, stream ) ) { std::printf( "test_dsv41_vision: FAIL (encode)\n" ); return 1; }
		const size_t expect = static_cast<size_t>( img.span_tokens() ) * hidden;
		if ( span.size() != expect ) { std::printf( "test_dsv41_vision: FAIL (span %zu halves, expected %zu)\n", span.size(), expect ); return 1; }
		double wc = 1.0, wr = 0.0;
		for ( int ty = 0; ty < llm_h; ++ty ) {
			// span row of token (ty, tx): 1 + ty * (llm_w + 1) + tx
			double c, r;
			compare_rows( std::vector<float>( ref_emb.begin() + static_cast<size_t>( ty ) * llm_w * hidden, ref_emb.begin() + static_cast<size_t>( ty + 1 ) * llm_w * hidden ), span,
				1 + static_cast<size_t>( ty ) * ( llm_w + 1 ), llm_w, hidden, &c, &r );
			wc = std::fmin( wc, c );
			wr = std::fmax( wr, r );
		}
		std::printf( "tower on the %s patches: worst row cosine %.5f, worst error %.4f row rms\n", pass == 0 ? "reference's" : "engine's", wc, wr );
		{   // where the error is: per-row cosine summary and the worst rows
			std::vector<std::pair<double, int>> cs;
			double sumc = 0;
			for ( int ty = 0; ty < llm_h; ++ty ) {
				for ( int tx = 0; tx < llm_w; ++tx ) {
					double dot = 0, na = 0, nb = 0;
					for ( int c = 0; c < hidden; ++c ) {
						const double a = ref_emb[( static_cast<size_t>( ty ) * llm_w + tx ) * hidden + c], b = backend::to_float( span[( 1 + static_cast<size_t>( ty ) * ( llm_w + 1 ) + tx ) * hidden + c] );
						dot += a * b; na += a * a; nb += b * b;
					}
					const double cs1 = dot / ( std::sqrt( na ) * std::sqrt( nb ) + 1e-30 );
					cs.push_back( { cs1, ty * llm_w + tx } );
					sumc += cs1;
				}
			}
			std::sort( cs.begin(), cs.end() );
			std::printf( "  mean row cosine %.5f; worst rows (ty,tx):", sumc / cs.size() );
			for ( int i = 0; i < 6; ++i ) std::printf( " (%d,%d)=%.3f", cs[i].second / llm_w, cs[i].second % llm_w, cs[i].first );
			std::printf( "; median %.5f\n", cs[cs.size() / 2].first );
		}
		if ( wc < 0.99 ) { std::printf( "test_dsv41_vision: FAIL (embeddings)\n" ); return 1; }
		// delimiters: the first row, the last row and the newline after every token row are not image rows; the newline rows are all equal
		bool same = true;
		for ( int ty = 1; ty < llm_h && same; ++ty )
			for ( int c = 0; c < hidden && same; ++c )
				same = span[( 1 + static_cast<size_t>( ty ) * ( llm_w + 1 ) + llm_w ) * hidden + c] == span[( 1 + static_cast<size_t>( llm_w ) ) * hidden + c];
		if ( !same ) { std::printf( "test_dsv41_vision: FAIL (newline rows differ)\n" ); return 1; }
	}
	std::printf( "test_dsv41_vision: OK\n" );
	return 0;
}
