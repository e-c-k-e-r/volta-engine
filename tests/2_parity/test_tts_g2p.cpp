// The TTS text front-end against vall_e's: a JSONL of {"text", "language", "phonemes"} (vall_e/emb/g2p.py encode's
// output, e.g. from store texts) phonemized here, the lines whose phonemes differ counted per language and the first few
// printed. Exit status 0 when every line matches.
//   test_tts_g2p <cases.jsonl> [--show n]
#include "engine/tts/g2p.h"
#include <nlohmann/json.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>

int main( int argc, char** argv ) {
	if ( argc < 2 ) {
		std::fprintf( stderr, "usage: test_tts_g2p <cases.jsonl> [--show n]\n" );
		return 2;
	}
	int show = 5;
	for ( int i = 2; i < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--show" ) && i + 1 < argc ) { show = std::atoi( argv[++i] ); }
	}
	engine::tts::G2P g2p;
	if ( !g2p.init( {} ) ) { return 1; }
	std::ifstream f( argv[1] );
	std::string line;
	std::map<std::string, std::pair<int, int>> per;   // language -> (lines, mismatches)
	int shown = 0, bad = 0;
	while ( std::getline( f, line ) ) {
		if ( line.empty() ) { continue; }
		const auto j = nlohmann::json::parse( line );
		const std::string text = j.at( "text" ), lang = j.at( "language" ), want = j.at( "phonemes" );
		const std::string got = g2p.phonemize( text, lang );
		auto& p = per[lang];
		++p.first;
		if ( got != want ) {
			++p.second;
			++bad;
			if ( shown++ < show ) {
				std::printf( "[%s] %s\n  vall_e: %s\n  engine: %s\n", lang.c_str(), text.c_str(), want.c_str(), got.c_str() );
			}
		}
	}
	for ( const auto& [lang, p] : per ) {
		std::printf( "%-4s %6d lines, %5d differ (%.3f%% match)\n", lang.c_str(), p.first, p.second, 100.0 * ( p.first - p.second ) / std::max( 1, p.first ) );
	}
	return bad ? 1 : 0;
}
