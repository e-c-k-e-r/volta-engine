// DeepSeek-V4.1-Flash end to end: prefill a token-id prompt, then greedy decode, printing each token id. With trace_layers=<file> every
// eager forward appends each layer's hyper-connection streams (tag 100 + layer after attention, tag layer after the FFN; header tag, rows,
// first position; then rows x 4 x hidden FP32) for tests/tools/dsv41/trace_cmp.py against dsv41_ref.py's dumps.
//   test_dsv41 <model dir> <ids file: whitespace-separated token ids> [new tokens] [option=value ...]
// turns=<file> [turns_first=i turns_count=n]: instead, prefill recorded agent turns (tests/tools/agent_replay_build.py output: messages + target), each
// rendered with the built-in DeepSeek encoding plus its target text, for route statistics (expert_heat_file) and prefill timing
#include "engine/runtime/test_driver.h"
#include "engine/runtime/options.h"
#include "engine/loader/deepseek_encoding.h"
#include "engine/loader/tokenizer.h"
#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>

using namespace engine::runtime;

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	if ( argc < 3 ) {
		std::fprintf( stderr, "usage: test_dsv41 <model dir> <ids file> [new tokens] [option=value ...]\n" );
		return 2;
	}
	const std::string dir = argv[1];
	std::vector<int> prompt;
	{
		std::ifstream f( argv[2] );
		for ( int t; f >> t; ) prompt.push_back( t );
	}
	ENGINE_ASSERT( !prompt.empty(), "test_dsv41: empty prompt" );
	int gen = 8, a = 3;
	if ( argc > 3 && std::string( argv[3] ).find( '=' ) == std::string::npos ) { gen = std::atoi( argv[3] ); a = 4; }
	std::string turns;
	int snap_at = 0;   // snap=<P>: prefix-cache round trip with a snapshot chain ending near position P
	int turns_first = 0, turns_count = 1 << 30;
	std::string verify;   // verify=<L,L,..>: the greedy continuation fed back in blocks of L rows, the time of a block (the cost of a speculative verify)
	int adapt = 0;      // adapt=<N>: decode with an expert residency rebalance every N tokens, printing the time of each 16 tokens
	std::string dump;   // dump=<file>: teacher-forced prefill in chunks of 512 rows, every row's FP16 logits appended (needs glm_logit_rows=512)
	for ( ; a < argc; ++a ) {
		const std::string kv = argv[a];
		const size_t eq = kv.find( '=' );
		ENGINE_ASSERT( eq != std::string::npos, "test_dsv41: options are key=value" );
		if ( kv.substr( 0, eq ) == "verify" ) { verify = kv.substr( eq + 1 ); continue; }
		if ( kv.substr( 0, eq ) == "adapt" ) { adapt = std::atoi( kv.c_str() + eq + 1 ); continue; }
		if ( kv.substr( 0, eq ) == "dump" ) { dump = kv.substr( eq + 1 ); continue; }
		if ( kv.substr( 0, eq ) == "turns" ) { turns = kv.substr( eq + 1 ); continue; }
		if ( kv.substr( 0, eq ) == "snap" ) { snap_at = std::atoi( kv.c_str() + eq + 1 ); continue; }
		if ( kv.substr( 0, eq ) == "turns_first" ) { turns_first = std::atoi( kv.c_str() + eq + 1 ); continue; }
		if ( kv.substr( 0, eq ) == "turns_count" ) { turns_count = std::atoi( kv.c_str() + eq + 1 ); continue; }
		ENGINE_ASSERT( set_option( kv.substr( 0, eq ), kv.substr( eq + 1 ) ), "test_dsv41: bad option" );
	}
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = "deepseek_v41";
	cfg.model_dir = dir;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	if ( !turns.empty() ) {
		engine::loader::Tokenizer tok;
		ENGINE_ASSERT( tok.load( dir.c_str() ), "test_dsv41: tokenizer" );
		std::ifstream tf( turns );
		const nlohmann::json all = nlohmann::json::parse( tf, nullptr, false );
		ENGINE_ASSERT( all.is_array(), "test_dsv41: turns file" );
		const int max_seq = mc.metadata().max_seq_len;
		size_t total = 0;
		const auto t1 = std::chrono::steady_clock::now();
		for ( int i = turns_first; i < static_cast<int>( all.size() ) && i < turns_first + turns_count; ++i ) {
			std::string prompt, err;
			engine::loader::DeepseekEncodeOptions o;
			ENGINE_ASSERT( engine::loader::deepseek_v41_encode( all[i]["messages"].dump(), std::string(), o, &prompt, &err ), "test_dsv41: encode" );
			prompt += all[i].value( "target", std::string() ) + "<｜end▁of▁sentence｜>";
			std::vector<int> ids = tok.encode( prompt, false );
			if ( static_cast<int>( ids.size() ) > max_seq - 1 ) ids.erase( ids.begin(), ids.end() - ( max_seq - 1 ) );
			mc.reset_slot( 0 );
			const auto t2 = std::chrono::steady_clock::now();
			mc.prefill( ids.data(), static_cast<int>( ids.size() ), 0 );
			std::printf( "[*] turn %d: %zu tokens, %.1f s\n", i, ids.size(), test::ms_since( t2 ) / 1000.0 );
			total += ids.size();
		}
		std::printf( "[*] %zu tokens in %.1f s (%.0f tok/s)\n", total, test::ms_since( t1 ) / 1000.0, total * 1000.0 / test::ms_since( t1 ) );
		mc.shutdown();
		return 0;
	}
	if ( !dump.empty() ) {
		const int V = mc.metadata().vocab_size, n = static_cast<int>( prompt.size() );
		std::vector<engine::backend::half_t> buf;
		FILE* f = std::fopen( dump.c_str(), "wb" );
		ENGINE_ASSERT( f != nullptr, "test_dsv41: cannot write the dump" );
		mc.reset_slot( 0 );
		for ( int p = 0; p < n; p += 512 ) {
			const int c = std::min( 512, n - p );
			mc.forward( &prompt[p], c, p, 0 );
			buf.resize( static_cast<size_t>( c ) * V );
			mc.read_logits( c, buf.data() );
			std::fwrite( buf.data(), sizeof( engine::backend::half_t ), buf.size(), f );
			if ( adapt > 0 ) mc.rebalance_experts();   // residency changes between chunks: the logits must not
		}
		std::fclose( f );
		std::printf( "[*] logits of %d positions written to %s\n", n, dump.c_str() );
		mc.shutdown();
		return 0;
	}
	if ( adapt > 0 ) {
		mc.reset_slot( 0 );
		int tok = mc.prefill( prompt.data(), static_cast<int>( prompt.size() ), 0 );
		std::vector<int> toks{ tok };
		auto tw = std::chrono::steady_clock::now();
		for ( int i = 1, pos = static_cast<int>( prompt.size() ); i < gen && !mc.is_eos( tok ); ++i ) {
			if ( i % adapt == 0 ) mc.rebalance_experts();
			mc.forward( &tok, 1, pos++, 0 );
			tok = mc.sample_one();
			toks.push_back( tok );
			if ( i % 16 == 0 ) { std::printf( "[*] tokens %d..%d: %.1f ms/token\n", i - 15, i, test::ms_since( tw ) / 16.0 ); tw = std::chrono::steady_clock::now(); }
		}
		std::printf( "[*] tokens:" );
		for ( int t : toks ) std::printf( " %d", t );
		std::printf( "\n" );
		mc.shutdown();
		return 0;
	}
	const auto t0 = std::chrono::steady_clock::now();
	const test::GenStats st = test::generate( mc, prompt, gen, false );
	if ( snap_at > 0 ) {
		// prefix-cache round trip: snapshots [0, P1) and [P1, P2), the caches overwritten by another prompt, the chain restored (the parent
		// kv-only), the rest of the prompt prefilled from P2: the continuation must equal the plain run's
		const int P2 = ( std::min<int>( snap_at, static_cast<int>( prompt.size() ) - 2 ) / 2 ) * 2, P1 = ( P2 / 4 ) * 2;
		ModelCluster::SeqSnapshot s1, s2;
		mc.reset_slot( 0 );
		mc.prefill( prompt.data(), P1, 0 );
		ENGINE_ASSERT( mc.save_snapshot( 0, 0, P1, &s1, false ), "test_dsv41: save [0, P1)" );
		mc.prefill( prompt.data() + P1, P2 - P1, 0, 0, nullptr, P1 );
		ENGINE_ASSERT( mc.save_snapshot( 0, P1, P2, &s2, false ), "test_dsv41: save [P1, P2)" );
		std::vector<int> other( prompt.rbegin(), prompt.rend() );
		mc.reset_slot( 0 );
		mc.prefill( other.data(), static_cast<int>( other.size() ), 0 );
		mc.reset_slot( 0 );
		ENGINE_ASSERT( mc.load_snapshot( 0, s1, true ) && mc.load_snapshot( 0, s2, false ), "test_dsv41: load" );
		std::vector<int> toks;
		int tok = mc.prefill( prompt.data() + P2, static_cast<int>( prompt.size() ) - P2, 0, 0, nullptr, P2 );
		toks.push_back( tok );
		for ( int pos = static_cast<int>( prompt.size() ); static_cast<int>( toks.size() ) < static_cast<int>( st.tokens.size() ) && !mc.is_eos( tok ); ) {
			mc.forward( &tok, 1, pos++, 0 );
			tok = mc.sample_one();
			toks.push_back( tok );
		}
		const bool same = toks == st.tokens;
		std::printf( "[%c] snapshot round trip at %d / %d of %zu tokens: continuation %s (%zu tokens)\n", same ? '+' : '-', P1, P2, prompt.size(), same ? "identical" : "DIFFERS", toks.size() );
		mc.free_snapshot( &s1 );
		mc.free_snapshot( &s2 );
		if ( !same ) { mc.shutdown(); return 1; }
	}
	if ( !verify.empty() ) {
		for ( size_t at = 0; at < verify.size(); ) {
			const int L = std::atoi( verify.c_str() + at );
			at = verify.find( ',', at ) == std::string::npos ? verify.size() : verify.find( ',', at ) + 1;
			mc.reset_slot( 0 );
			mc.prefill( prompt.data(), static_cast<int>( prompt.size() ), 0 );
			int blocks = 0, pos = static_cast<int>( prompt.size() );
			const auto tv = std::chrono::steady_clock::now();
			for ( size_t i = 0; i + L <= st.tokens.size(); i += L, ++blocks ) {
				mc.forward( &st.tokens[i], L, pos, 0 );
				pos += L;
				mc.sample_one();
			}
			const double ms = test::ms_since( tv );
			std::printf( "[*] verify blocks of %d rows: %d blocks, %.1f ms per block, %.1f ms per row\n", L, blocks, ms / blocks, ms / ( blocks * L ) );
		}
	}
	const int dec = static_cast<int>( st.tokens.size() ) - 1;
	std::printf( "[*] prompt %zu tokens, %.1f s: prefill %.0f ms (%.0f tok/s), decode %d tokens in %.0f ms (%.1f ms/token, %.1f tok/s)\n", prompt.size(), test::ms_since( t0 ) / 1000.0,
		st.prefill_ms, prompt.size() * 1000.0 / st.prefill_ms, dec, st.decode_ms, dec > 0 ? st.decode_ms / dec : 0.0, dec > 0 ? dec * 1000.0 / st.decode_ms : 0.0 );
	std::printf( "[*] tokens:" );
	for ( int t : st.tokens ) std::printf( " %d", t );
	std::printf( "\n" );
	mc.shutdown();
	return 0;
}
