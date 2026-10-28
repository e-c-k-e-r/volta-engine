// Tier 4: runtime steering on a real model (Qwen3.5-family, no speculator).
//  1. a slot with scope 0 decodes exactly what it decodes with steering never configured is checked by running this binary with ENGINE_STEER=off
//     (--tokens prints the greedy tokens) and diffing against ENGINE_STEER=on; here: greedy tokens of slot 0 alone.
//  2. an add vector on slot 1 changes slot 1's greedy text, and slot 0, decoding in the same batches, keeps the tokens it gives alone.
//  3. the logit penalty: the marker tokens never appear in a reasoning block that would otherwise open with them.
//   test_steer_model [--tokens]        (--tokens: print the plain greedy tokens and stop, for the steer=off comparison)
#include "engine/runtime/test_driver.h"
#include "engine/runtime/steer_file.h"
#include "engine/loader/tokenizer.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <climits>
#include <cstring>
#include <string>

using namespace engine::runtime;

int main( int argc, char** argv ) {
	const bool print_only = argc > 1 && std::string( argv[1] ) == "--tokens";
	const char* dir = test::QWEN_DIR;
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	if ( !std::getenv( "ENGINE_STEER" ) ) setenv( "ENGINE_STEER", "on", 1 );
	const bool steer_on = std::string( std::getenv( "ENGINE_STEER" ) ) != "off";
	engine::loader::Tokenizer tok;
	ENGINE_ASSERT( tok.load( dir ), "tokenizer load failed" );
	auto prompt = [&]( const std::string& user, bool think ) {
		return tok.encode( "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n" + ( think ? "<think>\n" : "<think>\n\n</think>\n\n" ), false );
	};
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = "qwen3_5";
	cfg.model_dir = dir;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	mc.warm_graphs( 4 );
	const std::vector<int> p0 = prompt( "Write a short poem about the sea.", false );
	const int kNew = 48;

	const test::GenStats alone = test::generate( mc, p0, kNew, false, 0 );
	std::printf( "slot 0 alone:" );
	for ( int t : alone.tokens ) std::printf( " %d", t );
	std::printf( "\n%s\n", tok.decode( alone.tokens ).c_str() );
	if ( print_only || !steer_on ) return 0;
	ENGINE_ASSERT( mc.steer_enabled(), "steering is not enabled in the model" );

	int failures = 0;
	int first = 0, count = 0;
	mc.steer_range( &first, &count );
	const int H = mc.metadata().hidden_size;

	// 2. an add vector (a pseudo-random direction, strong enough to move the text) on slot 1; slot 0 shares every decode batch with it
	std::vector<engine::modules::SteerEntry> entries;
	uint32_t seed = 11;
	for ( int l = first; l < first + count; ++l ) {
		engine::modules::SteerEntry e;
		e.layer = l;
		e.kind = engine::modules::STEER_ADD;
		e.vec.resize( H );
		double n2 = 0.0;
		for ( float& v : e.vec ) { seed = seed * 1664525u + 1013904223u; v = static_cast<float>( seed >> 8 ) / 16777216.0f - 0.5f; n2 += v * v; }
		for ( float& v : e.vec ) v = static_cast<float>( v / std::sqrt( n2 ) );
		e.par = 6.0f;
		entries.push_back( std::move( e ) );
	}
	mc.steer_set( 1, entries );
	mc.steer_control( 1, engine::modules::STEER_ALL, static_cast<int>( p0.size() ), 0 );
	mc.steer_control( 0, 0, static_cast<int>( p0.size() ), 0 );
	mc.reset_slot( 0 );
	mc.reset_slot( 1 );
	int t0 = mc.prefill( p0.data(), static_cast<int>( p0.size() ), 0 );
	int t1 = mc.prefill( p0.data(), static_cast<int>( p0.size() ), 1 );
	std::vector<int> s0 = { t0 }, s1 = { t1 };
	for ( int i = 1; i < kNew; ++i ) {
		const int ids[2] = { s0.back(), s1.back() }, slots[2] = { 0, 1 }, pos[2] = { static_cast<int>( p0.size() ) + i - 1, static_cast<int>( p0.size() ) + i - 1 };
		mc.forward( ids, pod::BatchRows::batched( 2, slots, pos ) );
		int out[2];
		mc.sample( out, 2 );
		s0.push_back( out[0] );
		s1.push_back( out[1] );
	}
	size_t same0 = 0, same1 = 0;
	for ( int i = 0; i < kNew; ++i ) { same0 += s0[i] == alone.tokens[i]; same1 += s1[i] == alone.tokens[i]; }
	std::printf( "unsteered slot 0 in a batch with a steered slot: %zu of %d tokens as alone\nsteered slot 1: %zu of %d tokens as the unsteered text\n%s\n", same0, kNew, same1, kNew, tok.decode( s1 ).c_str() );
	if ( same0 < static_cast<size_t>( kNew ) - 2 ) { std::printf( "[-] the neighbouring slot's text changed\n" ); ++failures; }
	if ( same1 > static_cast<size_t>( kNew ) / 2 ) { std::printf( "[-] the steered slot's text barely changed\n" ); ++failures; }

	// 3. the logit penalty: a prompt that makes the model open its reasoning with a marker word
	{
		const std::vector<int> pt = prompt( "Is 91 a prime number? Think it through.", true );
		mc.steer_set( 2, {} );
		mc.steer_control( 2, 0, static_cast<int>( pt.size() ), INT_MAX );
		mc.steer_penalty( 2, {}, 0.0f );
		const test::GenStats base = test::generate( mc, pt, 300, false, 2 );
		// the words the unpenalised reasoning starts with stand in for the markers (a given prompt need not produce "Wait")
		std::vector<int> markers;
		for ( int i = 0; i < 6 && i < static_cast<int>( base.tokens.size() ); ++i ) if ( std::find( markers.begin(), markers.end(), base.tokens[i] ) == markers.end() ) markers.push_back( base.tokens[i] );
		mc.steer_penalty( 2, markers, 30.0f );
		const test::GenStats pen = test::generate( mc, pt, 300, false, 2 );
		auto count_markers = [&]( const std::vector<int>& toks ) { int n = 0; for ( int t : toks ) for ( int m : markers ) n += t == m; return n; };
		const int nb = count_markers( base.tokens ), np = count_markers( pen.tokens );
		std::printf( "reasoning markers in 300 tokens: %d plain, %d with the penalty\n", nb, np );
		if ( nb == 0 ) { std::printf( "[-] the stand-in markers never appear without the penalty\n" ); ++failures; }
		else if ( np != 0 ) { std::printf( "[-] markers survived a penalty of 30\n" ); ++failures; }
	}
	std::printf( failures == 0 ? "[+] steering PASS\n" : "[-] steering FAIL\n" );
	mc.shutdown();
	return failures == 0 ? 0 : 1;
}
