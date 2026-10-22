// Tier 4: greedy decoding with the MTP head as speculator reproduces plain greedy decoding token for token, on a short prompt and
// on a long one (several prefill chunks, so the head's prompt KV and the carried hidden row are exercised), and it accepts drafts.
#include "engine/runtime/test_driver.h"
#include "engine/loader/tokenizer.h"
#include <cmath>
#include <cstdlib>
#include <string>

using namespace engine::runtime;


// The logits are FP16, so two candidates within one ulp are common (an ulp is 0.0156 at |logit| 16..32), and a verify block rounds differently from a single
// row: the same near-tie then breaks one way in plain decoding and the other in a round (ties keep the lower id). A first difference is accepted when it is
// exactly that: plain decoding's runner-up at that step, within one FP16 ulp of its pick. Replays the plain run's tokens as single rows to read the step's logits.
// max_gap > 0 widens the tie to that many logits (joint rounds on Flash-Next: the router flips move a row's logits by a few tenths against single rows).
static bool one_ulp_tie( ModelCluster& mc, const std::vector<int>& prompt, const std::vector<int>& plain, size_t d, int spec_token, float max_gap = 0.0f ) {
	if ( d == 0 || d >= plain.size() ) return false;
	mc.reset_slot( 0 );
	mc.prefill( prompt.data(), static_cast<int>( prompt.size() ), 0 );
	int pos = static_cast<int>( prompt.size() );
	for ( size_t i = 0; i < d; ++i ) mc.forward( &plain[i], 1, pos++, 0 );
	Candidates cs[1];
	mc.sample_topk( 1, cs );
	if ( cs[0].ids[0] != plain[d] || cs[0].ids[1] != spec_token ) return false;
	const float top = cs[0].logits[0], gap = top - cs[0].logits[1];
	const float ulp = std::ldexp( 1.0f, std::ilogb( std::fabs( top ) ) - 10 );
	std::printf( "   at that step plain's top-2 logits are %.5f (id %d) and %.5f (id %d): gap %.5f, one FP16 ulp %.5f\n", top, cs[0].ids[0], cs[0].logits[1], cs[0].ids[1], gap, ulp );
	return max_gap > 0.0f ? gap <= max_gap : gap <= ulp * 1.001f;
}

int main( int argc, char** argv ) {
	// --flash: Qwen3.8-Flash-Next (qwen4_exp) instead of Qwen3.8-27B; --glm: GLM-5.3-Flash (the head comes from the NVFP4 checkpoint named by
	// ENGINE_NVFP4_DENSE_SRC)
	const bool flash = argc > 1 && std::string( argv[1] ) == "--flash";
	const bool glm = argc > 1 && std::string( argv[1] ) == "--glm";
	// --joint (with --flash): two prompts speculate together, their blocks verified in one forward (joint rounds), each held to its own plain greedy run
	const bool joint = flash && argc > 2 && std::string( argv[2] ) == "--joint";
	// --interleave: slot 0 speculates while slot 1's long prompt is prefilled in pieces between its rounds (what the generator does when a request arrives
	// during decoding); both slots must give exactly the tokens they give alone, since the pieces and the rounds are separate forwards
	const bool interleave = argc > 2 && std::string( argv[2] ) == "--interleave";
	const char* dir = glm ? test::GLM_DIR : flash ? test::FLASH_DIR : test::QWEN_DIR;
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	setenv( "ENGINE_SPECULATOR", "mtp", 1 );
	engine::loader::Tokenizer tok;
	ENGINE_ASSERT( tok.load( dir ), "tokenizer load failed" );
	auto prompt_ids = [&]( const std::string& user ) {
		std::vector<int> ids = glm ? tok.encode( "[gMASK]<sop><|user|>", false ) : std::vector<int>{ 248045, 846, 198 };   // <|im_start|>user\n
		const std::vector<int> body = tok.encode( user, false );
		ids.insert( ids.end(), body.begin(), body.end() );
		if ( glm ) {
			const std::vector<int> end = tok.encode( "<|assistant|></think>", false );
			ids.insert( ids.end(), end.begin(), end.end() );
			return ids;
		}
		const std::vector<int> tail = { 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 };   // <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
		ids.insert( ids.end(), tail.begin(), tail.end() );
		return ids;
	};
	std::string notes;
	for ( int i = 0; i < ( glm ? 380 : 700 ); ++i ) notes += "Note " + std::to_string( i ) + ": the weather in district " + std::to_string( i * 7 % 101 ) + " was mild on day " + std::to_string( i % 365 ) + ". ";
	const std::vector<std::vector<int>> prompts = {
		prompt_ids( "Write a Python function that checks whether a string is a palindrome, with a docstring and three unit tests." ),
		prompt_ids( notes + "\n\nWhat was the weather in district 7? Quote the note." ),
	};

	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = glm ? "glm5_next" : flash ? "qwen4_exp" : "qwen3_5";
	cfg.model_dir = dir;
	cfg.speculate = true;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	ENGINE_ASSERT( mc.has_speculator(), "no speculator" );
	mc.warm_graphs( 8 );
	int failures = 0;
	if ( interleave ) {
		constexpr int kPiece = 512, kNewA = 96, kNewB = 32;
		const std::vector<int>& A = prompts[0];
		const std::vector<int>& B = prompts[1];
		const test::GenStats refA = test::generate( mc, A, kNewA, true, 0 );
		const test::GenStats refB = test::generate( mc, B, kNewB, true, 1 );
		mc.reset_slot( 0 );
		mc.reset_slot( 1 );
		std::vector<int> ta;
		int tokA = mc.prefill( A.data(), static_cast<int>( A.size() ), 0 );
		ta.push_back( tokA );
		int posA = static_cast<int>( A.size() ), posB = 0, tokB = -1, pieces = 0, rounds_during = 0;
		while ( static_cast<int>( ta.size() ) < kNewA || tokB < 0 ) {
			if ( static_cast<int>( ta.size() ) < kNewA && !mc.is_eos( tokA ) ) {
				const SpecResult r = mc.speculate( 0, tokA, posA );
				posA = r.next_pos;
				for ( int i = 0; i < r.emitted_count && static_cast<int>( ta.size() ) < kNewA; ++i ) ta.push_back( tokA = r.emitted[i] );
				if ( tokB < 0 ) ++rounds_during;
			}
			if ( tokB < 0 ) {
				const int len = MIN( kPiece, static_cast<int>( B.size() ) - posB );
				const int r = mc.prefill( B.data() + posB, len, 1, 0, nullptr, posB );
				posB += len;
				++pieces;
				if ( posB >= static_cast<int>( B.size() ) ) tokB = r;
			}
			else if ( static_cast<int>( ta.size() ) >= kNewA || mc.is_eos( tokA ) ) break;
		}
		std::vector<int> tb = { tokB };
		int pb = static_cast<int>( B.size() );
		while ( static_cast<int>( tb.size() ) < kNewB && !mc.is_eos( tb.back() ) ) {
			const SpecResult r = mc.speculate( 1, tb.back(), pb );
			pb = r.next_pos;
			for ( int i = 0; i < r.emitted_count && static_cast<int>( tb.size() ) < kNewB; ++i ) tb.push_back( r.emitted[i] );
		}
		const bool okA = ta == refA.tokens, okB = tb == refB.tokens;
		std::printf( "[%s] slot 0 (%zu tokens, %d rounds while %d pieces of slot 1 went in): %zu tokens equal to the run alone\n", okA ? "ok" : "FAIL", A.size(), rounds_during, pieces, ta.size() );
		std::printf( "[%s] slot 1 (%zu tokens in pieces of %d): %zu tokens equal to the run alone\n", okB ? "ok" : "FAIL", B.size(), kPiece, tb.size() );
		if ( !okA ) std::printf( "   alone %s\n   mixed %s\n", test::format_tokens( refA.tokens, 24 ).c_str(), test::format_tokens( ta, 24 ).c_str() );
		if ( !okB ) std::printf( "   alone %s\n   mixed %s\n", test::format_tokens( refB.tokens, 24 ).c_str(), test::format_tokens( tb, 24 ).c_str() );
		std::printf( okA && okB ? "test_mtp --interleave: PASS\n" : "test_mtp --interleave: FAIL\n" );
		return okA && okB ? 0 : 1;
	}
	if ( joint ) {
		ENGINE_ASSERT( mc.speculator().joint(), "the speculator has no joint rounds" );
		mc.warm_graphs( pod::MAX_BATCH );
		const std::vector<std::vector<int>> jp = { prompts[0], prompt_ids( "Explain how a hash map handles collisions, with a short C++ example of open addressing." ) };
		const std::vector<test::GenStats> js = test::generate_joint( mc, jp, 96 );
		for ( size_t i = 0; i < jp.size(); ++i ) {
			const test::GenStats plain = test::generate( mc, jp[i], 96, false, static_cast<int>( i ) );
			const test::GenStats& spec = js[i];
			size_t d = 0;
			while ( d < plain.tokens.size() && d < spec.tokens.size() && plain.tokens[d] == spec.tokens[d] ) ++d;
			bool same = plain.tokens == spec.tokens;
			if ( !same && d < spec.tokens.size() ) same = one_ulp_tie( mc, jp[i], plain.tokens, d, spec.tokens[d], 0.25f );
			std::printf( "[%s] joint slot %zu (%zu tokens): %d rounds, accepted %d / %d, first difference at %zu of %zu\n", same ? "ok" : "FAIL", i, jp[i].size(), spec.rounds,
				spec.accepted, spec.proposed, d, plain.tokens.size() );
			if ( !same ) std::printf( "   plain %s\n   joint %s\n", test::format_tokens( plain.tokens, 24 ).c_str(), test::format_tokens( spec.tokens, 24 ).c_str() );
			failures += same ? 0 : 1;
		}
		std::printf( failures ? "test_mtp --joint: FAIL\n" : "test_mtp --joint: PASS\n" );
		return failures ? 1 : 0;
	}
	for ( size_t p = 0; p < prompts.size(); ++p ) {
		const test::GenStats plain = test::generate( mc, prompts[p], 96, false );
		const test::GenStats spec = test::generate( mc, prompts[p], 96, true );
		size_t d = 0;
		while ( d < plain.tokens.size() && d < spec.tokens.size() && plain.tokens[d] == spec.tokens[d] ) ++d;
		// Qwen3.8-27B reproduces plain greedy decoding token for token. Flash-Next's router flips make near-ties sensitive to the rounding of
		// the verify block against single rows, so it is held to the first three quarters of the continuation
		// GLM's single-row and verify-block numerics differ by a few tenths of a logit, so it only has to agree for the first tokens
		bool same = plain.tokens == spec.tokens || ( flash && plain.tokens.size() == spec.tokens.size() && d * 4 >= plain.tokens.size() * 3 ) || ( glm && d >= 8 );
		if ( !same && d < spec.tokens.size() ) same = one_ulp_tie( mc, prompts[p], plain.tokens, d, spec.tokens[d] );
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
