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
	// Acceptance-only case. Qwen3.8-27B-DFlash2 was trained on Qwen's native reasoning traces, so a
	// prompt that suppresses thinking pushes the target off its trained distribution and draft
	// acceptance collapses (exllamav3 SESSION_OPTIMIZATIONS_20260920.md measured 9-24% suppressed
	// against 65-75% thinking). These measure that regime and assert only that decoding runs, so
	// ids are tokenized at run time and decode length comes from want_tokens instead of golden.
	bool probe;
	int want_tokens;
};

// Golden sequences: greedy outputs recorded on the pre-refactor M=1 token-by-token path.
std::vector<Case> qwen_cases() {
	std::vector<Case> v = {
		{ "What is the capital of France?",
			{ 248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 },
			{ 760, 6511, 314, 9338, 369, 2972, 57590, 159034, 248046 }, "Paris", false, 0 },
		{ "List the first 5 prime numbers:",
			{ 248045, 846, 198, 826, 279, 1118, 220, 20, 9944, 4947, 25, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 },
			{ 760, 1118, 220, 20, 9944, 4947, 513, 25, 271, 16, 13, 2972, 17, 332, 198 }, "2", false, 0 },
		{ "Passage: The secret access code is 84920. Question: What is the secret access code?",
			{ 248045, 846, 198, 11839, 416, 25, 561, 6038, 2528, 1970, 369, 220, 23, 19, 24, 17, 15, 13, 15380,
			  25, 3437, 369, 279, 6038, 2528, 1970, 30, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 },
			{ 760, 6038, 2528, 1970, 369, 2972, 23, 19, 24, 17, 15, 159034, 248046 }, "84920", false, 0 },
		{ "What is the capital of Japan?",
			{ 248045, 846, 198, 3710, 369, 279, 6511, 314, 6124, 30, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 },
			{ 760, 6511, 314, 6124, 369, 2972, 51076, 15560, 159034, 248046 }, "Tokyo", false, 0 },
	};
	if ( getenv( "ENGINE_TEST_REASONING_PROBE" ) ) {
		// Natural-reasoning prompt: no thinking-suppression instruction, so the target stays inside the
		// trace distribution DFlash2 was trained on. Assert only that decoding runs; we are measuring
		// acceptance against the target, not matching a recorded golden.
		// ids are filled in at run time from the tokenizer; decode length is set by want_tokens.
		v.push_back( { "Solve this step by step: what is 17 * 23? Show your reasoning.", {}, {}, "", true, 48 } );
	}
	return v;
}

std::vector<Case> glm_cases() {
	return {
		{ "What is the capital of France?",
			{ 154822, 154824, 154826, 25062, 287, 29905, 371, 25, 12035, 154827, 3838, 374, 279, 6722, 315, 9621, 30, 21754, 304, 825, 11646, 13, 154828, 154841 },
			{ 154842, 785, 6722, 315, 9621, 374, 12089, 13, 154827 }, "Paris", false, 0 },
		{ "List the first 5 prime numbers:",
			{ 154822, 154824, 154826, 25062, 287, 29905, 371, 25, 12035, 154827, 852, 279, 1156, 220, 20, 10245, 5109, 25, 154828, 154841 },
			{ 154842, 785, 1156, 220, 20, 10245, 5109, 525, 25, 3070, 17, 11, 220, 18, 11, 220, 20, 11, 220, 22, 11, 220, 98965, 334, 154827 }, "2, 3, 5, 7, 11", false, 0 },
		{ "Passage: The secret access code is 84920. Question: What is the secret access code?",
			{ 154822, 154824, 154826, 25062, 287, 29905, 371, 25, 12035, 154827, 12181, 424, 25, 576, 6234, 2615, 2038, 374, 220, 23, 101474, 98360, 13, 15834, 25, 3555, 374, 279, 6234, 2615, 2038, 30, 154828, 154841 },
			{ 16360, 13, 154842, 785, 6234, 2615, 2038, 374, 3070, 23, 101474, 98360, 334, 13, 154827 }, "84920", false, 0 },
		{ "What is the capital of Japan?",
			{ 154822, 154824, 154826, 25062, 287, 29905, 371, 25, 12035, 154827, 3838, 374, 279, 6722, 315, 6323, 30, 154828, 154841 },
			{ 154842, 785, 6722, 315, 6323, 374, 3070, 52504, 16019, 334, 13, 1084, 594, 7407, 389, 279, 23086, 3108, 315, 16533, 927, 84, 11, 6323, 594, 7772, 12915, 11, 323, 374, 825, 315, 279, 1429, 93655, 57019, 5671, 304, 279, 1879, 13, 154827 }, "Tokyo", false, 0 },
	};
}

// ENGINE_KV_FORMAT selects a quantized KV cache. The recorded goldens are FP16-KV greedy outputs, so a quantized cache is judged
// by the expected answer in plain decoding and Mode B, and speculation must match greedy decoding in the same format.
bool kv_quantized() {
	const char* e = getenv( "ENGINE_KV_FORMAT" );
	return e != nullptr && std::strcmp( e, "fp16" ) != 0;
}

bool check( const char* tag, const Case& c, const std::vector<int>& got, const engine::loader::Tokenizer& tok, bool by_answer = false ) {
	const std::string text = tok.decode( got );
	const bool ok = c.probe ? true : ( by_answer ? text.find( c.expect ) != std::string::npos : got == c.golden );
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
	std::vector<std::vector<int>> ids( n );
	int slots[pod::MAX_BATCH], pos[pod::MAX_BATCH], last[pod::MAX_BATCH];
	for ( int i = 0; i < n; ++i ) {
		ids[i] = cases[i].ids.empty() ? tok.encode( cases[i].prompt, true ) : cases[i].ids;
		mc.reset_slot( i );
		last[i] = mc.prefill( ids[i].data(), static_cast<int>( ids[i].size() ), i );
		out[i].push_back( last[i] );
		pos[i] = static_cast<int>( ids[i].size() );
	}
	auto t0 = std::chrono::steady_clock::now();
	int steps = 0;
	while ( true ) {
		int m = 0, map[pod::MAX_BATCH], toks[pod::MAX_BATCH];
		for ( int i = 0; i < n; ++i ) {
			const int want = cases[i].want_tokens ? cases[i].want_tokens : static_cast<int>( cases[i].golden.size() );
			if ( static_cast<int>( out[i].size() ) >= want ) continue;
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
		fail += check( "Mode B", cases[i], out[i], tok, kv_quantized() ) ? 0 : 1;
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
	if ( spec && getenv( "ENGINE_TEST_DFLASH2_PROFILE" ) ) mc.spec().set_profile( true );
	engine::loader::Tokenizer tok;
	tok.load( dir );

	int fail = 0;
	const bool kvq = kv_quantized();
	for ( size_t i = 0; i < cases.size(); ++i ) {
		Case c = cases[i];
		std::printf( "[*] Prompt %zu: \"%s\"\n", i + 1, c.prompt );
		const std::vector<int> ids = c.ids.empty() ? tok.encode( c.prompt, true ) : c.ids;
		const int want = c.want_tokens ? c.want_tokens : static_cast<int>( c.golden.size() );
		if ( kvq && spec && !c.probe ) c.golden = test::generate( mc, ids, want, false ).tokens;
		const auto st = test::generate( mc, ids, want, spec );
		fail += check( spec ? "spec" : "greedy", c, st.tokens, tok, kvq && !spec ) ? 0 : 1;
		std::printf( "        prefill %.1f ms | decode %.1f tok/s", st.prefill_ms, st.decode_tok_s() );
		if ( spec ) {
			std::printf( " | acceptance %d/%d over %d rounds", st.accepted, st.proposed, st.rounds );
			if ( !c.probe ) {
				// Where the draft chain dies. Step k is only sampled when 0..k-1 were all accepted, so
				// this is each position's conditional survival, not the marginal hit rate.
				int att[8], hit[8];
				mc.spec().step_stats( att, hit );
				std::printf( "\n    [steps] " );
				for ( int k = 0; k < 7; ++k ) {
					if ( att[k] == 0 ) { std::printf( " k%d:-", k ); continue; }
					std::printf( " k%d:%d/%d(%.0f%%)", k, hit[k], att[k], 100.0 * hit[k] / att[k] );
				}
				std::printf( "\n" );
			}
		}
		std::printf( "\n" );
	}
	if ( spec && getenv( "ENGINE_TEST_DFLASH2_PROFILE" ) ) mc.spec().profile_dump();
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
		failures += impl::run_model( "qwen3_5", test::QWEN_DIR, test::QWEN_DRAFT_DIR, impl::qwen_cases(), spec );
	}
	if ( run_glm ) {
		failures += impl::run_model( "glm5_next", test::GLM_DIR, test::GLM_DRAFT_DIR, impl::glm_cases(), spec );
	}
	ENGINE_ASSERT( failures == 0, "coherence gate failed" );
	std::printf( "=== COHERENCE GATE COMPLETE: ALL PASS ===\n" );
	return 0;
}
