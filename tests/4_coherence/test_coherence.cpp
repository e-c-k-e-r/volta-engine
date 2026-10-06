// Tier-4 coherence gate: greedy (T=0) determinism against golden token sequences on TP=4.
//   default        : autoregressive decode per prompt (Mode A chunked prefill + M=1 decode)
//   --speculative  : DFlash2 rounds; output must be token-identical to greedy decode
//   Mode B check   : all prompts decoded concurrently in independent slots (when slots allow)
#include "engine/runtime/test_driver.h"
#include "engine/loader/tokenizer.h"
#include <cstring>

using namespace engine::runtime;

namespace impl {

struct Case {
	const char* prompt;
	std::vector<int> ids;
	std::vector<int> golden;
	const char* expect;
};

// Golden sequences: greedy outputs recorded on the pre-refactor M=1 token-by-token path.
std::vector<Case> qwen_cases() {
	return {
		{ "What is the capital of France?",
			{ 248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 },
			{ 760, 6511, 314, 9338, 369, 2972, 57590, 159034, 248046 }, "Paris" },
		{ "List the first 5 prime numbers:",
			{ 248045, 846, 198, 826, 279, 1118, 220, 20, 9944, 4947, 25, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 },
			{ 760, 1118, 220, 20, 9944, 4947, 513, 25, 271, 16, 13, 2972, 17, 332, 198 }, "2" },
		{ "Passage: The secret access code is 84920. Question: What is the secret access code?",
			{ 248045, 846, 198, 11839, 416, 25, 561, 6038, 2528, 1970, 369, 220, 23, 19, 24, 17, 15, 13, 15380,
			  25, 3437, 369, 279, 6038, 2528, 1970, 30, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 },
			{ 760, 6038, 2528, 1970, 369, 2972, 23, 19, 24, 17, 15, 159034, 248046 }, "84920" },
		{ "What is the capital of Japan?",
			{ 248045, 846, 198, 3710, 369, 279, 6511, 314, 6124, 30, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 },
			{ 760, 6511, 314, 6124, 369, 2972, 51076, 15560, 159034, 248046 }, "Tokyo" },
	};
}

std::vector<Case> glm_cases() {
	return {
		{ "What is the capital of France?",
			{ 154822, 154824, 154826, 25062, 287, 29905, 371, 25, 12035, 154827, 3838, 374, 279, 6722, 315, 9621, 30, 21754, 304, 825, 11646, 13, 154828, 154841 },
			{ 154842, 785, 6722, 315, 9621, 374, 12089, 13, 154827 }, "Paris" },
		{ "List the first 5 prime numbers:",
			{ 154822, 154824, 154826, 25062, 287, 29905, 371, 25, 12035, 154827, 852, 279, 1156, 220, 20, 10245, 5109, 25, 154828, 154841 },
			{ 154842, 785, 1156, 220, 20, 10245, 5109, 525, 25, 3070, 17, 11, 220, 18, 11, 220, 20, 11, 220, 22, 11, 220, 98965, 334, 154827 }, "2, 3, 5, 7, 11" },
		{ "Passage: The secret access code is 84920. Question: What is the secret access code?",
			{ 154822, 154824, 154826, 25062, 287, 29905, 371, 25, 12035, 154827, 12181, 424, 25, 576, 6234, 2615, 2038, 374, 220, 23, 101474, 98360, 13, 15834, 25, 3555, 374, 279, 6234, 2615, 2038, 30, 154828, 154841 },
			{ 16360, 13, 154842, 785, 6234, 2615, 2038, 374, 3070, 23, 101474, 98360, 334, 13, 154827 }, "84920" },
		{ "What is the capital of Japan?",
			{ 154822, 154824, 154826, 25062, 287, 29905, 371, 25, 12035, 154827, 3838, 374, 279, 6722, 315, 6323, 30, 154828, 154841 },
			{ 154842, 785, 6722, 315, 6323, 374, 3070, 52504, 16019, 334, 13, 1084, 594, 7407, 389, 279, 23086, 3108, 315, 16533, 927, 84, 11, 6323, 594, 7772, 12915, 11, 323, 374, 825, 315, 279, 1429, 93655, 57019, 5671, 304, 279, 1879, 13, 154827 }, "Tokyo" },
	};
}

bool check( const char* tag, const Case& c, const std::vector<int>& got, const engine::loader::Tokenizer& tok ) {
	const bool ok = ( got == c.golden );
	const std::string text = tok.decode( got );
	std::printf( "    [%s] %s \"%s\"\n", tag, ok ? "PASS" : "FAIL", text.c_str() );
	if ( !ok ) {
		std::printf( "        got:    %s\n        golden: %s\n", test::format_tokens( got ).c_str(), test::format_tokens( c.golden ).c_str() );
	}
	if ( text.find( c.expect ) == std::string::npos ) {
		std::printf( "        note: expected substring \"%s\" not found in decoded text\n", c.expect );
	}
	return ok;
}

// Mode B: every prompt in its own slot, decoded together one token per slot per forward.
int run_concurrent( ModelCluster& mc, const std::vector<Case>& cases, const engine::loader::Tokenizer& tok ) {
	const int n = static_cast<int>( cases.size() );
	std::vector<std::vector<int>> out( n );
	int slots[pod::MAX_BATCH], pos[pod::MAX_BATCH], last[pod::MAX_BATCH];
	for ( int i = 0; i < n; ++i ) {
		mc.reset_slot( i );
		last[i] = mc.prefill( cases[i].ids.data(), static_cast<int>( cases[i].ids.size() ), i );
		out[i].push_back( last[i] );
		pos[i] = static_cast<int>( cases[i].ids.size() );
	}
	auto t0 = std::chrono::steady_clock::now();
	int steps = 0;
	while ( true ) {
		int m = 0, map[pod::MAX_BATCH], toks[pod::MAX_BATCH];
		for ( int i = 0; i < n; ++i ) {
			if ( out[i].size() >= cases[i].golden.size() ) continue;
			map[m] = i;
			slots[m] = i;
			toks[m] = last[i];
			++m;
		}
		if ( m == 0 ) break;
		int rows_pos[pod::MAX_BATCH];
		for ( int r = 0; r < m; ++r ) rows_pos[r] = pos[map[r]];
		mc.forward( toks, pod::BatchRows::batched( m, slots, rows_pos ) );
		int preds[pod::MAX_BATCH];
		mc.sample( preds, m );
		for ( int r = 0; r < m; ++r ) {
			const int i = map[r];
			pos[i]++;
			last[i] = preds[r];
			out[i].push_back( preds[r] );
		}
		++steps;
	}
	std::printf( "    [Mode B] %d slots, %d batched steps in %.1f ms\n", n, steps, test::ms_since( t0 ) );
	int fail = 0;
	for ( int i = 0; i < n; ++i ) {
		fail += check( "Mode B", cases[i], out[i], tok ) ? 0 : 1;
	}
	return fail;
}

int run_model( const char* type, const char* dir, const char* draft, const std::vector<Case>& cases, bool spec ) {
	std::printf( "\n[*] %s (TP=4)\n", type );
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = type;
	cfg.model_dir = dir;
	if ( spec ) cfg.draft_model_dir = draft;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	engine::loader::Tokenizer tok;
	tok.load( dir );

	int fail = 0;
	for ( size_t i = 0; i < cases.size(); ++i ) {
		const auto& c = cases[i];
		std::printf( "[*] Prompt %zu: \"%s\"\n", i + 1, c.prompt );
		const auto st = test::generate( mc, c.ids, static_cast<int>( c.golden.size() ), spec );
		fail += check( spec ? "spec" : "greedy", c, st.tokens, tok ) ? 0 : 1;
		std::printf( "        prefill %.1f ms | decode %.1f tok/s", st.prefill_ms, st.decode_tok_s() );
		if ( spec ) {
			std::printf( " | acceptance %d/%d over %d rounds", st.accepted, st.proposed, st.rounds );
		}
		std::printf( "\n" );
	}
	if ( !spec && mc.metadata().max_slots >= static_cast<int>( cases.size() ) ) {
		fail += run_concurrent( mc, cases, tok );
	}
	mc.shutdown();
	std::printf( "[%c] %s: %d failure(s)\n", fail ? '-' : '+', type, fail );
	return fail;
}

} // namespace impl

int main( int argc, char** argv ) {
	bool run_qwen = true;
	bool run_glm = true;
	bool spec = false;
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	for ( int a = 1; a < argc; ++a ) {
		if ( !std::strcmp( argv[a], "--qwen-only" ) ) run_glm = false;
		else if ( !std::strcmp( argv[a], "--glm-only" ) ) run_qwen = false;
		else if ( !std::strcmp( argv[a], "--speculative" ) ) spec = true;
	}
	std::printf( "=== Tier-4 coherence gate (%s) ===\n", spec ? "DFlash2 speculative" : "greedy autoregressive" );

	int failures = 0;
	if ( run_qwen ) {
		failures += impl::run_model( "qwen38", test::QWEN_DIR, test::QWEN_DRAFT_DIR, impl::qwen_cases(), spec );
	}
	if ( run_glm ) {
		failures += impl::run_model( "glm53", test::GLM_DIR, test::GLM_DRAFT_DIR, impl::glm_cases(), spec );
	}
	ENGINE_ASSERT( failures == 0, "coherence gate failed" );
	std::printf( "=== COHERENCE GATE COMPLETE: ALL PASS ===\n" );
	return 0;
}
