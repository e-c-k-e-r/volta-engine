// Gemma 4 audio tower (the small models') against llama.cpp's gemma4a encoder: tests/tools/gemma4_audio_golden.py runs llama-mtmd-cli with
// MTMD_DEBUG_EMBEDDINGS on a clip and keeps what it reports of the embeddings (their shape, mean / std / min / max, and the first and last 16
// values of the first token):
//   test_gemma4_audio <model dir> <clip> <golden json>
// The engine's features of the clip give llama.cpp's token count, and the tower's embeddings its statistics and first token.
#include "engine/models/gemma4_audio.h"
#include "engine/loader/audio.h"
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#if ENGINE_BACKEND_CUDA
#include "trellis_context.h"
#endif
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace engine;

int main( int argc, char** argv ) {
	if ( argc < 4 ) {
		std::fprintf( stderr, "usage: test_gemma4_audio <model dir> <clip> <golden json>\n" );
		return 2;
	}
	const std::string model = argv[1], clip = argv[2];
	std::ifstream gf( argv[3] );
	const nlohmann::json g = nlohmann::json::parse( std::string( ( std::istreambuf_iterator<char>( gf ) ), std::istreambuf_iterator<char>() ), nullptr, false );
	if ( g.is_discarded() ) {
		std::printf( "test_gemma4_audio: FAIL (golden)\n" );
		return 1;
	}

	std::ifstream cf( clip, std::ios::binary );
	const std::vector<uint8_t> bytes( ( std::istreambuf_iterator<char>( cf ) ), std::istreambuf_iterator<char>() );
	std::vector<float> pcm;
	int rate = 0;
	std::string err;
	std::vector<loader::PreparedAudio> chunks;
	loader::AudioParams params;
	if ( !loader::decode_audio( bytes.data(), bytes.size(), &pcm, &rate, &err ) || !loader::prepare_audio( pcm, rate, params, &chunks, &err ) || chunks.size() != 1 ) {
		std::printf( "test_gemma4_audio: FAIL (features: %s)\n", err.c_str() );
		return 1;
	}
	const int tokens = g["tokens"].get<int>(), hidden = g["hidden"].get<int>();
	bool ok = chunks[0].tokens == tokens;
	std::printf( "  %d samples at %d Hz -> %d frames -> %d tokens (llama.cpp %d)  %s\n", static_cast<int>( pcm.size() ), rate, chunks[0].frames, chunks[0].tokens, tokens,
		ok ? "PASS" : "FAIL" );

	backend::set_device( 0 );
#if ENGINE_BACKEND_CUDA
	engine::modules::trellis_context_init_all();
#endif
	pod::MemoryArena arena;
	engine::core::arena_create( arena, 0, 2ULL << 30 );
	auto stream = backend::stream_create();
	models::Gemma4AudioTower tower;
	if ( !tower.load( model.c_str(), arena, 750, stream ) ) {
		std::printf( "test_gemma4_audio: FAIL (load)\n" );
		return 1;
	}
	std::vector<backend::half_t> out;
	if ( !tower.encode( chunks[0], &out, stream ) || out.size() != static_cast<size_t>( chunks[0].tokens ) * hidden ) {
		std::printf( "test_gemma4_audio: FAIL (encode)\n" );
		return 1;
	}
	double sum = 0.0, sq = 0.0, mn = 1e30, mx = -1e30;
	for ( const auto& h : out ) {
		const double v = backend::to_float( h );
		sum += v;
		sq += v * v;
		mn = std::min( mn, v );
		mx = std::max( mx, v );
	}
	const double n = static_cast<double>( out.size() ), mean = sum / n, sd = std::sqrt( std::max( 0.0, sq / n - mean * mean ) );
	const double rmean = g["mean"].get<double>(), rsd = g["std"].get<double>();
	const bool stats = std::fabs( mean - rmean ) < 0.02 * rsd && std::fabs( sd - rsd ) < 0.02 * rsd;
	std::printf( "  mean %.5f std %.5f min %.3f max %.3f (llama.cpp %.5f %.5f %.3f %.3f)  %s\n", mean, sd, mn, mx, rmean, rsd, g["min"].get<double>(),
		g["max"].get<double>(), stats ? "PASS" : "FAIL" );
	// the first token's first and last 16 values: cosine and the largest difference against the spread
	double dot = 0.0, na = 0.0, nb = 0.0, worst = 0.0;
	for ( const char* part : { "first", "last" } ) {
		const auto& ref = g[part];
		for ( size_t i = 0; i < ref.size(); ++i ) {
			const size_t at = part[0] == 'f' ? i : static_cast<size_t>( hidden ) - ref.size() + i;
			const double a = backend::to_float( out[at] ), b = ref[i].get<double>();
			dot += a * b;
			na += a * a;
			nb += b * b;
			worst = std::max( worst, std::fabs( a - b ) );
		}
	}
	const double cosine = dot / std::sqrt( na * nb );
	const bool first = cosine > 0.995 && worst < 0.1 * rsd;
	std::printf( "  token 0, 32 values: cosine %.5f, largest difference %.4f  %s\n", cosine, worst, first ? "PASS" : "FAIL" );
	ok = ok && stats && first;
	std::printf( ok ? "test_gemma4_audio: PASS\n" : "test_gemma4_audio: FAIL\n" );
	return ok ? 0 : 1;
}
