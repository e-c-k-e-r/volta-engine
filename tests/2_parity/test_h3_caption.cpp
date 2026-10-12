// The captioner (H3TextGroup::generate): greedy decoding must agree with itself across the prefill and the decode path. A text answer T of n tokens is generated
// once; the same prompt extended by T[0..n/2) (prefilled, not decoded) must continue with T[n/2..n).
//   test_h3_caption <exl3 dir> <tokenizer dir> <ngpus>
#include "engine/config.h"
#include "engine/loader/tokenizer.h"
#include "engine/video/h3_text_group.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main( int argc, char** argv ) {
	if ( argc < 4 ) { std::printf( "usage: test_h3_caption <exl3 dir> <tokenizer dir> <ngpus>\n" ); return 1; }
	std::vector<int> devs;
	for ( int i = 0; i < std::atoi( argv[3] ); ++i ) { devs.push_back( i ); }
	engine::loader::Tokenizer tok;
	ENGINE_ASSERT( tok.load( argv[2] ), "tokenizer" );
	engine::video::H3TextGroup grp;
	ENGINE_ASSERT( grp.load( argv[1], devs, 2048, 50, true ), "load" );
	std::vector<int> ids;
	auto push = [&]( const std::vector<int>& v ) { ids.insert( ids.end(), v.begin(), v.end() ); };
	push( { tok.get_special_token( "<|im_start|>" ) } );
	push( tok.encode( "user\nName the planets of the solar system in order from the sun, one short sentence each.", false ) );
	push( { tok.get_special_token( "<|im_end|>" ) } );
	push( tok.encode( "\n", false ) );
	push( { tok.get_special_token( "<|im_start|>" ) } );
	push( tok.encode( "assistant\n", false ) );
	engine::video::H3CaptionOptions opt;
	opt.temperature = 0.0f;
	opt.max_new_tokens = 40;
	std::vector<int> full, rest;
	const auto t0 = std::chrono::steady_clock::now();
	ENGINE_ASSERT( grp.generate( ids, -1, {}, opt, &full ), "generate" );
	const double secs = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
	std::printf( "%zu tokens in %.2f s: %s\n", full.size(), secs, tok.decode( full ).c_str() );
	int bad = 0;
	if ( full.size() < 8 ) { std::printf( "too short\n" ); return 1; }
	const size_t half = full.size() / 2;
	std::vector<int> ids2 = ids;
	ids2.insert( ids2.end(), full.begin(), full.begin() + half );
	opt.max_new_tokens = static_cast<int>( full.size() - half );
	ENGINE_ASSERT( grp.generate( ids2, -1, {}, opt, &rest ), "generate 2" );
	size_t same = 0;
	while ( same < rest.size() && same < full.size() - half && rest[same] == full[half + same] ) { ++same; }
	std::printf( "prefill-vs-decode continuation: %zu of %zu tokens agree\n", same, full.size() - half );
	if ( same + 2 < full.size() - half ) { ++bad; } // a near-tie argmax may flip the last tokens; anything earlier is a bug
	std::printf( bad ? "FAIL\n" : "ok\n" );
	return bad ? 1 : 0;
}
