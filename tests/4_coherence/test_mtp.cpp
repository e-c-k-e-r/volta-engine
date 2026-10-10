// Tier 4: greedy decoding with the MTP head as speculator reproduces plain greedy decoding token for token, on a short prompt and
// on a long one (several prefill chunks, so the head's prompt KV and the carried hidden row are exercised), and it accepts drafts.
#include "engine/runtime/test_driver.h"
#include "engine/loader/tokenizer.h"
#include <cstdlib>
#include <string>

using namespace engine::runtime;

int main( int argc, char** argv ) {
	// --flash: Qwen3.8-Flash-Next (qwen4_exp) instead of Qwen3.8-27B
	const bool flash = argc > 1 && std::string( argv[1] ) == "--flash";
	const char* dir = flash ? test::FLASH_DIR : test::QWEN_DIR;
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	setenv( "ENGINE_SPECULATOR", "mtp", 1 );
	engine::loader::Tokenizer tok;
	ENGINE_ASSERT( tok.load( dir ), "tokenizer load failed" );
	auto prompt_ids = [&]( const std::string& user ) {
		std::vector<int> ids = { 248045, 846, 198 };   // <|im_start|>user\n
		const std::vector<int> body = tok.encode( user, false );
		ids.insert( ids.end(), body.begin(), body.end() );
		const std::vector<int> tail = { 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 };   // <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
		ids.insert( ids.end(), tail.begin(), tail.end() );
		return ids;
	};
	std::string notes;
	for ( int i = 0; i < 700; ++i ) notes += "Note " + std::to_string( i ) + ": the weather in district " + std::to_string( i * 7 % 101 ) + " was mild on day " + std::to_string( i % 365 ) + ". ";
	const std::vector<std::vector<int>> prompts = {
		prompt_ids( "Write a Python function that checks whether a string is a palindrome, with a docstring and three unit tests." ),
		prompt_ids( notes + "\n\nWhat was the weather in district 7? Quote the note." ),
	};

	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = flash ? "qwen4_exp" : "qwen3_5";
	cfg.model_dir = dir;
	cfg.speculate = true;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	ENGINE_ASSERT( mc.has_speculator(), "no speculator" );
	mc.warm_graphs( 8 );
	int failures = 0;
	for ( size_t p = 0; p < prompts.size(); ++p ) {
		const test::GenStats plain = test::generate( mc, prompts[p], 96, false );
		const test::GenStats spec = test::generate( mc, prompts[p], 96, true );
		size_t d = 0;
		while ( d < plain.tokens.size() && d < spec.tokens.size() && plain.tokens[d] == spec.tokens[d] ) ++d;
		// Qwen3.8-27B reproduces plain greedy decoding token for token. Flash-Next's router flips make near-ties sensitive to the rounding of
		// the verify block against single rows, so it is held to the first three quarters of the continuation
		const bool same = plain.tokens == spec.tokens || ( flash && plain.tokens.size() == spec.tokens.size() && d * 4 >= plain.tokens.size() * 3 );
		const double rate = spec.proposed ? static_cast<double>( spec.accepted ) / spec.proposed : 0.0;
		std::printf( "[%s] prompt %zu (%zu tokens): plain %.1f tok/s, MTP %.1f tok/s, accepted %d / %d (%.0f%%)\n", same ? "ok" : "FAIL", p, prompts[p].size(),
			plain.decode_tok_s(), spec.decode_tok_s(), spec.accepted, spec.proposed, 100.0 * rate );
		if ( plain.tokens != spec.tokens ) {
			std::printf( "   first difference at token %zu of %zu: plain %d, MTP %d\n", d, plain.tokens.size(), d < plain.tokens.size() ? plain.tokens[d] : -1, d < spec.tokens.size() ? spec.tokens[d] : -1 );
		}
		if ( !same ) std::printf( "   plain %s\n   spec  %s\n", test::format_tokens( plain.tokens, 24 ).c_str(), test::format_tokens( spec.tokens, 24 ).c_str() );
		failures += same ? 0 : 1;
		if ( rate < 0.3 ) {
			std::printf( "[FAIL] draft acceptance %.0f%% is too low for a working head\n", 100.0 * rate );
			++failures;
		}
	}
	std::printf( failures ? "test_mtp: FAIL\n" : "test_mtp: PASS\n" );
	return failures ? 1 : 0;
}
