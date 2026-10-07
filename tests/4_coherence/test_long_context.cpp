// Tier-4 long-context gate: a needle buried in a filler passage of several thousand tokens must be
// retrieved by greedy decode and by DFlash2 speculation. The speculative run exercises the drafter's
// context window and the target tap ring wrapping (ring cap 2048 < prompt length).
//   test_long_context [target_prompt_tokens] [needle_depth_percent]
#include "engine/runtime/test_driver.h"
#include "engine/loader/tokenizer.h"
#include <cstdlib>

using namespace engine::runtime;

namespace {

bool contains( const std::string& hay, const char* needle ) {
	return hay.find( needle ) != std::string::npos;
}

} // namespace

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	const int target = argc > 1 ? std::atoi( argv[1] ) : 4400;
	const int depth = argc > 2 ? std::atoi( argv[2] ) : 50;

	engine::loader::Tokenizer tok;
	ENGINE_ASSERT( tok.load( test::QWEN_DIR ), "tokenizer load failed" );
	const std::vector<int> filler = tok.encode( " The quick brown fox jumps over the lazy dog near the old river bank.", false );
	const std::vector<int> needle = tok.encode( " The secret access code is 84920.", false );
	const std::vector<int> head = { 248045, 846, 198 };
	const std::vector<int> question = tok.encode( " Question: What is the secret access code?", false );
	const std::vector<int> tail = { 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 };
	const std::vector<int> intro = tok.encode( "Passage:", false );

	const int fixed = static_cast<int>( head.size() + intro.size() + needle.size() + question.size() + tail.size() );
	const int n_filler = MAX( 0, ( target - fixed ) / static_cast<int>( filler.size() ) );
	const int before = n_filler * depth / 100;
	std::vector<int> ids = head;
	ids.insert( ids.end(), intro.begin(), intro.end() );
	for ( int i = 0; i < before; ++i ) ids.insert( ids.end(), filler.begin(), filler.end() );
	ids.insert( ids.end(), needle.begin(), needle.end() );
	for ( int i = before; i < n_filler; ++i ) ids.insert( ids.end(), filler.begin(), filler.end() );
	ids.insert( ids.end(), question.begin(), question.end() );
	ids.insert( ids.end(), tail.begin(), tail.end() );
	std::printf( "[*] prompt %zu tokens, needle at %d%%\n", ids.size(), depth );

	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = "qwen38";
	cfg.model_dir = test::QWEN_DIR;
	cfg.draft_model_dir = test::QWEN_DRAFT_DIR;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	ENGINE_ASSERT( static_cast<int>( ids.size() ) + 64 < mc.metadata().max_seq_len, "prompt does not fit max_seq" );

	int fail = 0;
	std::vector<int> out[2];
	for ( int spec = 0; spec < 2; ++spec ) {
		const auto st = test::generate( mc, ids, 40, spec == 1 );
		out[spec] = st.tokens;
		const std::string text = tok.decode( st.tokens );
		const bool ok = contains( text, "84920" );
		std::printf( "    [%s] prefill %.0f ms (%.0f tok/s) | decode %.1f tok/s | acceptance %d/%d over %d rounds | %s: \"%.60s\"\n",
			spec ? "spec" : "greedy", st.prefill_ms, ids.size() * 1000.0 / st.prefill_ms, st.decode_tok_s(),
			st.accepted, st.proposed, st.rounds, ok ? "PASS" : "FAIL", text.c_str() );
		fail += ok ? 0 : 1;
	}
	size_t i = 0;
	while ( i < out[0].size() && i < out[1].size() && out[0][i] == out[1][i] ) ++i;
	if ( i < out[0].size() && i < out[1].size() ) {
		std::printf( "    note: speculative output first differs from greedy at token %zu (%d vs %d)\n", i, out[0][i], out[1][i] );
	}
	else {
		std::printf( "    speculative output token-identical to greedy\n" );
	}
	mc.shutdown();
	std::printf( "=== LONG-CONTEXT GATE: %s ===\n", fail ? "FAIL" : "PASS" );
	return fail ? 1 : 0;
}
