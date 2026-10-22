// Compares the engine tokenizer with reference token ids (HF tokenizers).
//   test_tokenizer_file <model_dir> <text file> <reference ids.json>   one file against a JSON array of ids
//   test_tokenizer_file <model_dir> --cases <cases.json>               [{"text", "ids"}] fixtures (tests/data/tokenizer_cases_*.json)
//   test_tokenizer_file <model_dir> --time <text file> [ids out.bin]   encode time of one file, optionally dumping the ids (int32) to compare runs
#include "engine/loader/tokenizer.h"

#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

int main( int argc, char** argv ) {
	if ( argc < 4 ) {
		std::printf( "usage: test_tokenizer_file <model_dir> <text file> <reference ids.json> | <model_dir> --cases <cases.json>\n" );
		return 2;
	}
	engine::loader::Tokenizer tok;
	if ( !tok.load( argv[1] ) ) {
		std::printf( "[-] cannot load tokenizer from %s\n", argv[1] );
		return 2;
	}
	if ( std::string( argv[2] ) == "--time" ) {
		std::ifstream tf( argv[3], std::ios::binary );
		std::stringstream ts;
		ts << tf.rdbuf();
		const std::string text = ts.str();
		const auto t0 = std::chrono::steady_clock::now();
		const std::vector<int> ids = tok.encode( text, false );
		const double ms = std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count();
		std::printf( "%zu bytes -> %zu tokens in %.1f ms\n", text.size(), ids.size(), ms );
		if ( argc > 4 ) {
			std::ofstream of( argv[4], std::ios::binary );
			of.write( reinterpret_cast<const char*>( ids.data() ), static_cast<std::streamsize>( ids.size() * sizeof( int ) ) );
		}
		return 0;
	}
	if ( std::string( argv[2] ) == "--cases" ) {
		std::ifstream cf( argv[3] );
		auto cases = nlohmann::json::parse( cf, nullptr, false );
		if ( cases.is_discarded() || !cases.is_array() ) {
			std::printf( "[-] bad cases file\n" );
			return 2;
		}
		int bad = 0;
		for ( size_t c = 0; c < cases.size(); ++c ) {
			const std::vector<int> ids = tok.encode( cases[c]["text"].get<std::string>(), false );
			const auto& want = cases[c]["ids"];
			size_t k = 0;
			while ( k < ids.size() && k < want.size() && ids[k] == want[k].get<int>() ) ++k;
			if ( k != ids.size() || k != want.size() ) {
				if ( ++bad <= 5 ) {
					std::printf( "[FAIL] case %zu differs at token %zu (engine %zu tokens, reference %zu): %.60s\n", c, k, ids.size(), want.size(),
						cases[c]["text"].get<std::string>().c_str() );
				}
			}
		}
		std::printf( "%zu cases, %d differ\n", cases.size(), bad );
		return bad ? 1 : 0;
	}
	std::ifstream tf( argv[2], std::ios::binary );
	std::stringstream ts;
	ts << tf.rdbuf();
	const std::string text = ts.str();
	std::ifstream rf( argv[3] );
	auto ref = nlohmann::json::parse( rf, nullptr, false );
	if ( ref.is_discarded() || !ref.is_array() ) {
		std::printf( "[-] bad reference ids\n" );
		return 2;
	}
	const std::vector<int> ids = tok.encode( text, false );
	std::printf( "engine %zu tokens, reference %zu tokens\n", ids.size(), ref.size() );
	size_t k = 0;
	while ( k < ids.size() && k < ref.size() && ids[k] == ref[k].get<int>() ) ++k;
	if ( k == ids.size() && k == ref.size() ) {
		std::printf( "identical\n" );
		return 0;
	}
	std::printf( "first difference at token %zu\n", k );
	auto show = [&]( const char* what, auto get, size_t n ) {
		std::printf( "  %s:", what );
		for ( size_t i = k > 3 ? k - 3 : 0; i < std::min( n, k + 8 ); ++i ) std::printf( " %d[%s]", get( i ), tok.decode_token( get( i ) ).c_str() );
		std::printf( "\n" );
	};
	show( "engine   ", [&]( size_t i ) { return ids[i]; }, ids.size() );
	show( "reference", [&]( size_t i ) { return ref[i].get<int>(); }, ref.size() );
	return 1;
}
