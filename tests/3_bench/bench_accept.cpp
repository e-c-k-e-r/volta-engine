// Draft acceptance characterisation (O5): speculative decode on a spread of prompts with Qwen's thinking
// enabled (the regime DFlash2 was trained on) or suppressed (--nothink), reporting per-position candidate
// quality: how often the drafter's best candidate / its top-k set / the selector walk hit the target's token.
//   bench_accept [--glm] [--nothink] [--tokens N] [--prompts K] [--temp T]   (T > 0 samples, top_p 0.95)
#include "engine/runtime/test_driver.h"
#include "engine/loader/tokenizer.h"
#include "engine/loader/chat_template.h"
#include <cstdlib>
#include <cstring>

using namespace engine::runtime;

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	bool think = true;
	int n_tokens = 256;
	int n_prompts = 100;
	float temp = 0.0f;
	bool glm = false;
	for ( int i = 1; i < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--nothink" ) ) think = false;
		else if ( !std::strcmp( argv[i], "--glm" ) ) glm = true;
		else if ( !std::strcmp( argv[i], "--tokens" ) && i + 1 < argc ) n_tokens = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--prompts" ) && i + 1 < argc ) n_prompts = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--temp" ) && i + 1 < argc ) temp = static_cast<float>( std::atof( argv[++i] ) );
	}
	const char* prompts[] = {
		"Solve step by step: what is 17 * 23 + 144 / 12? Show your reasoning.",
		"Write a Python function that returns the n-th Fibonacci number using memoization, and explain the complexity.",
		"Explain how a hash table handles collisions, and compare chaining with open addressing.",
		"A train leaves at 3pm going 60 mph, another leaves at 4pm going 80 mph on the same track. When does the second catch up?",
		"Write a C function to reverse a singly linked list in place, then walk through it on a 4-node example.",
		"What are the main differences between TCP and UDP? Give use cases for each.",
		"Prove that the square root of 2 is irrational.",
		"Write a SQL query to find the second highest salary in an employees table and explain how it works.",
	};
	const int total = static_cast<int>( sizeof( prompts ) / sizeof( prompts[0] ) );

	engine::loader::Tokenizer tok;
	ENGINE_ASSERT( tok.load( glm ? test::GLM_DIR : test::QWEN_DIR ), "tokenizer load failed" );
	engine::loader::ChatTemplate chat;
	ENGINE_ASSERT( chat.load( glm ? test::GLM_DIR : test::QWEN_DIR ), "chat template load failed" );
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = glm ? "glm53" : "qwen38";
	cfg.model_dir = glm ? test::GLM_DIR : test::QWEN_DIR;
	cfg.draft_model_dir = glm ? test::GLM_DRAFT_DIR : test::QWEN_DRAFT_DIR;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	mc.warm_graphs( 8 );
	mc.spec().set_diag( true );

	int rounds = 0, proposed = 0, accepted = 0, produced = 0;
	double ms = 0.0;
	for ( int pi = 0; pi < total && pi < n_prompts; ++pi ) {
		std::vector<int> ids;
		if ( glm ) {
			engine::loader::ChatTemplateOptions opts;
			opts.enable_thinking = think;
			ids = tok.encode( chat.render( { { "user", prompts[pi] } }, opts ), false );
		}
		else {
			ids = { 248045, 846, 198 };
			const std::vector<int> body = tok.encode( prompts[pi], false );
			ids.insert( ids.end(), body.begin(), body.end() );
			const std::vector<int> tail = think
				? std::vector<int>{ 248046, 198, 248045, 74455, 198, 248068, 198 }
				: std::vector<int>{ 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 };
			ids.insert( ids.end(), tail.begin(), tail.end() );
		}
		test::GenStats st;
		if ( temp <= 0.0f ) {
			st = test::generate( mc, ids, n_tokens, true );
		}
		else {
			SamplingParams sp;
			sp.temperature = temp;
			sp.top_p = 0.95f;
			sp.seed = 1000 + pi;
			Rng rng( sp.seed );
			mc.reset_slot( 0 );
			Candidates c;
			auto t0 = std::chrono::steady_clock::now();
			int tok = sample_token( ( mc.prefill( ids.data(), static_cast<int>( ids.size() ), 0, 0, &c ), c ), sp, rng );
			st.tokens.push_back( tok );
			int pos = static_cast<int>( ids.size() );
			t0 = std::chrono::steady_clock::now();
			while ( static_cast<int>( st.tokens.size() ) < n_tokens && !mc.is_eos( tok ) ) {
				const SpecResult r = mc.speculate( 0, tok, pos, -1.0f, &sp, &rng );
				st.rounds++;
				st.proposed += r.proposed;
				st.accepted += r.accepted;
				pos = r.next_pos;
				for ( int i = 0; i < r.emitted_count && static_cast<int>( st.tokens.size() ) < n_tokens; ++i ) {
					tok = r.emitted[i];
					st.tokens.push_back( tok );
					if ( mc.is_eos( tok ) ) break;
				}
			}
			st.decode_ms = test::ms_since( t0 );
		}
		std::printf( "  prompt %d: %3zu tokens, %2d rounds, acceptance %3d/%3d (%.0f%%), %.1f tok/s | \"%.50s\"\n", pi,
			st.tokens.size(), st.rounds, st.accepted, st.proposed, st.proposed ? 100.0 * st.accepted / st.proposed : 0.0,
			st.decode_tok_s(), tok.decode( std::vector<int>( st.tokens.begin(), st.tokens.begin() + MIN( 24, static_cast<int>( st.tokens.size() ) ) ) ).c_str() );
		rounds += st.rounds;
		proposed += st.proposed;
		accepted += st.accepted;
		produced += static_cast<int>( st.tokens.size() ) - 1;
		ms += st.decode_ms;
	}
	std::printf( "\n=== %s regime: %d rounds, acceptance %d/%d (%.1f%%), %.2f tokens/round, %.1f tok/s ===\n",
		think ? "thinking" : "no-think", rounds, accepted, proposed, proposed ? 100.0 * accepted / proposed : 0.0,
		rounds ? static_cast<double>( produced ) / rounds : 0.0, ms > 0.0 ? produced * 1000.0 / ms : 0.0 );
	const auto& d = mc.spec().diag();
	std::printf( "  pos |  seen | best==target | target in top-16 | walk==target\n" );
	for ( int k = 0; k < 7; ++k ) {
		if ( d.seen[k] == 0 ) continue;
		std::printf( "   %d  | %5d | %5.1f%%       | %5.1f%%           | %5.1f%%\n", k, d.seen[k],
			100.0 * d.top1[k] / d.seen[k], 100.0 * d.in_topk[k] / d.seen[k], 100.0 * d.walk[k] / d.seen[k] );
	}
	mc.shutdown();
	return 0;
}
