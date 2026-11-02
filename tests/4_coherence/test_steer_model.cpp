// Tier 4: runtime steering on a real model (Qwen3.5 family, Flash-Next with STEER_FLASH=1, GLM-5.3-Flash with STEER_GLM=1 or DeepSeek-V4.1-Flash with STEER_DSV41=1; no speculator).
//  1. a slot with scope 0 decodes exactly what it decodes with steering never configured is checked by running this binary with ENGINE_STEER=off
//     (--tokens prints the greedy tokens) and diffing against ENGINE_STEER=on; here: greedy tokens of slot 0 alone.
//  2. an add vector on slot 1 changes slot 1's greedy text, and slot 0, decoding in the same batches, keeps the tokens it gives alone.
//  3. the logit penalty: the marker tokens never appear in a reasoning block that would otherwise open with them.
//   test_steer_model [--tokens]        (--tokens: print the plain greedy tokens and stop, for the steer=off comparison)
#include "engine/runtime/test_driver.h"
#include "engine/runtime/steer_file.h"
#include "engine/loader/tokenizer.h"
#include "engine/loader/deepseek_encoding.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <climits>
#include <cstring>
#include <string>

using namespace engine::runtime;

int main( int argc, char** argv ) {
	const bool print_only = argc > 1 && std::string( argv[1] ) == "--tokens";
	const bool flash = std::getenv( "STEER_FLASH" ) != nullptr;   // Flash-Next (qwen4_exp, ENGINE_TEST_FLASH_DIR) instead of the 27B
	const bool glm = std::getenv( "STEER_GLM" ) != nullptr;       // GLM-5.3-Flash (glm5_next, ENGINE_TEST_GLM_DIR)
	const bool dsv41 = std::getenv( "STEER_DSV41" ) != nullptr;   // DeepSeek-V4.1-Flash (deepseek_v41, ENGINE_TEST_DSV41_DIR)
	static const std::string dsv41_dir = test::model_path( "ENGINE_TEST_DSV41_DIR" );
	const char* dir = dsv41 ? dsv41_dir.c_str() : glm ? test::GLM_DIR : flash ? test::FLASH_DIR : test::QWEN_DIR;
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	if ( !std::getenv( "ENGINE_STEER" ) ) setenv( "ENGINE_STEER", "on", 1 );
	if ( !std::getenv( "ENGINE_MAX_SLOTS" ) ) setenv( "ENGINE_MAX_SLOTS", "3", 1 );   // slot 2 takes the penalty check
	const bool steer_on = std::string( std::getenv( "ENGINE_STEER" ) ) != "off";
	engine::loader::Tokenizer tok;
	ENGINE_ASSERT( tok.load( dir ), "tokenizer load failed" );
	auto prompt = [&]( const std::string& user, bool think ) {
		if ( dsv41 ) {
			engine::loader::DeepseekEncodeOptions o;
			o.thinking = think;
			std::string text, err;
			const nlohmann::json msgs = nlohmann::json::array( { { { "role", "user" }, { "content", user } } } );
			ENGINE_ASSERT( engine::loader::deepseek_v41_encode( msgs.dump(), std::string(), o, &text, &err ), "test_steer_model: DeepSeek prompt encoding" );
			return tok.encode( text, false );
		}
		if ( glm ) return tok.encode( "[gMASK]<sop><|user|>" + user + ( think ? "<|assistant|><think>" : "<|assistant|></think>" ), false );
		return tok.encode( "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n" + ( think ? "<think>\n" : "<think>\n\n</think>\n\n" ), false );
	};
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = dsv41 ? "deepseek_v41" : glm ? "glm5_next" : flash ? "qwen4_exp" : "qwen3_5";
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

	// 2. an add vector (a pseudo-random direction) on slot 1; slot 0 shares every decode batch with it. At strength 0 both slots must decode exactly
	//    what they decode alone (the edit path runs, the tables pick the right rows). At a strength that moves the text, slot 0 must keep its first
	//    16 tokens: an edit leaking into its rows would change them at once. Later tokens may differ on GLM, whose batched rounding depends on what
	//    the other rows route to, so a near-tie can flip once the neighbour's text is very different.
	auto run_pair = [&]( float strength, std::vector<int>& s0, std::vector<int>& s1 ) {
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
			e.par = strength;
			entries.push_back( std::move( e ) );
		}
		mc.steer_set( 1, entries );
		mc.steer_control( 1, engine::modules::STEER_ALL, static_cast<int>( p0.size() ), 0 );
		mc.steer_control( 0, 0, static_cast<int>( p0.size() ), 0 );
		mc.reset_slot( 0 );
		mc.reset_slot( 1 );
		s0 = { mc.prefill( p0.data(), static_cast<int>( p0.size() ), 0 ) };
		s1 = { mc.prefill( p0.data(), static_cast<int>( p0.size() ), 1 ) };
		for ( int i = 1; i < kNew; ++i ) {
			const int ids[2] = { s0.back(), s1.back() }, slots[2] = { 0, 1 }, pos[2] = { static_cast<int>( p0.size() ) + i - 1, static_cast<int>( p0.size() ) + i - 1 };
			mc.forward( ids, pod::BatchRows::batched( 2, slots, pos ) );
			int out[2];
			mc.sample( out, 2 );
			s0.push_back( out[0] );
			s1.push_back( out[1] );
		}
	};
	auto same = [&]( const std::vector<int>& a ) { int n = 0; while ( n < kNew && a[n] == alone.tokens[n] ) ++n; return n; };   // leading tokens as alone
	{
		std::vector<int> s0, s1;
		run_pair( 0.0f, s0, s1 );
		std::printf( "strength 0 on slot 1: slot 0 keeps %d, slot 1 %d of %d leading tokens\n", same( s0 ), same( s1 ), kNew );
		if ( same( s0 ) < kNew || same( s1 ) < kNew ) { std::printf( "[-] a zero-strength edit changed the text\n" ); ++failures; }
		run_pair( 6.0f, s0, s1 );
		size_t moved = 0;
		for ( int i = 0; i < kNew; ++i ) moved += s1[i] != alone.tokens[i];
		std::printf( "strength 6 on slot 1: slot 0 keeps %d of %d leading tokens, slot 1 differs in %zu\n%s\n", same( s0 ), kNew, moved, tok.decode( s1 ).c_str() );
		if ( same( s0 ) < 16 ) { std::printf( "[-] the neighbouring slot's text changed\n" ); ++failures; }
		if ( moved < static_cast<size_t>( kNew ) / 2 ) { std::printf( "[-] the steered slot's text barely changed\n" ); ++failures; }
	}

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
