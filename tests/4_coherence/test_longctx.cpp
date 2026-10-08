// Long-context needle retrieval: a filler prompt of ~N tokens with a secret code buried at a given depth, greedy
// (or speculative) decode, and a check that the answer carries the code. Context window set by EXL3_MAX_SEQ (Qwen).
//   test_longctx [--glm] [--tokens N] [--depth PCT] [--spec]
#include "engine/runtime/test_driver.h"
#include "engine/loader/tokenizer.h"
#include "engine/loader/chat_template.h"
#include <cstdlib>
#include <cstring>
#include <string>

using namespace engine::runtime;

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	bool glm = false, spec = false;
	int n_tokens = 8000;
	int depth = 50;
	for ( int i = 1; i < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--glm" ) ) glm = true;
		else if ( !std::strcmp( argv[i], "--spec" ) ) spec = true;
		else if ( !std::strcmp( argv[i], "--tokens" ) && i + 1 < argc ) n_tokens = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--depth" ) && i + 1 < argc ) depth = std::atoi( argv[++i] );
	}
	engine::loader::Tokenizer tok;
	ENGINE_ASSERT( tok.load( glm ? test::GLM_DIR : test::QWEN_DIR ), "tokenizer load failed" );
	engine::loader::ChatTemplate chat;
	ENGINE_ASSERT( chat.load( glm ? test::GLM_DIR : test::QWEN_DIR ), "chat template load failed" );

	// Filler: numbered sentences (no repetition a model could shortcut), then the needle at `depth` percent.
	std::vector<int> filler;
	for ( int i = 0; static_cast<int>( filler.size() ) < n_tokens; ++i ) {
		const std::string s = "Note " + std::to_string( i ) + ": the weather in district " + std::to_string( i * 7 % 101 ) + " was mild on day " + std::to_string( i % 365 ) + ". ";
		const std::vector<int> t = tok.encode( s, false );
		filler.insert( filler.end(), t.begin(), t.end() );
	}
	filler.resize( n_tokens );
	const std::vector<int> needle = tok.encode( " The secret access code is 84920. ", false );
	filler.insert( filler.begin() + static_cast<size_t>( n_tokens ) * depth / 100, needle.begin(), needle.end() );
	const std::string head = "Read the passage and answer the question.\n\nPassage: ";
	const std::string tail = "\n\nQuestion: What is the secret access code? Answer with the number only.";
	std::vector<int> ids;
	if ( glm ) {
		engine::loader::ChatTemplateOptions opts;
		opts.enable_thinking = false;
		const std::string marker = "@@BODY@@";
		const std::string r = chat.render( { { "user", marker } }, opts );
		const size_t at = r.find( marker );
		ENGINE_ASSERT( at != std::string::npos, "marker not in rendered template" );
		ids = tok.encode( r.substr( 0, at ) + head, false );
		ids.insert( ids.end(), filler.begin(), filler.end() );
		const std::vector<int> t = tok.encode( tail + r.substr( at + marker.size() ), false );
		ids.insert( ids.end(), t.begin(), t.end() );
	}
	else {
		ids = { 248045, 846, 198 };
		const std::vector<int> h = tok.encode( head, false );
		ids.insert( ids.end(), h.begin(), h.end() );
		ids.insert( ids.end(), filler.begin(), filler.end() );
		const std::vector<int> t = tok.encode( tail, false );
		ids.insert( ids.end(), t.begin(), t.end() );
		const std::vector<int> end = { 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 };
		ids.insert( ids.end(), end.begin(), end.end() );
	}

	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = glm ? "glm5_next" : "qwen3_5";
	cfg.model_dir = glm ? test::GLM_DIR : test::QWEN_DIR;
	if ( spec ) cfg.draft_model_dir = glm ? test::GLM_DRAFT_DIR : test::QWEN_DRAFT_DIR;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	mc.warm_graphs( 8 );
	std::printf( "[*] %s prompt %zu tokens, needle at %d%%, max_seq %d, %s\n", glm ? "glm5_next" : "qwen3_5", ids.size(), depth,
		mc.metadata().max_seq_len, spec ? "speculative" : "greedy" );
	ENGINE_ASSERT( static_cast<int>( ids.size() ) + 24 < mc.metadata().max_seq_len, "prompt does not fit the context window" );
	const test::GenStats st = test::generate( mc, ids, 24, spec );
	const std::string out = tok.decode( st.tokens );
	const bool ok = out.find( "84920" ) != std::string::npos;
	std::printf( "    prefill %.1f s (%.0f tok/s), decode %.1f tok/s\n    answer: \"%.60s\"\n", st.prefill_ms / 1000.0,
		ids.size() * 1000.0 / st.prefill_ms, st.decode_tok_s(), out.c_str() );
	std::printf( "%s needle retrieval at %zu tokens\n", ok ? "[+] PASS" : "[-] FAIL", ids.size() );
	mc.shutdown();
	return ok ? 0 : 1;
}
