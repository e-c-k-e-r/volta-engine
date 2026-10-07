// End-to-end plumbing check for temperature sampling: with a vanishing temperature the sampled paths (plain decode
// and speculative rounds, including the shard top-k merge, the first token from prefill candidates and the
// sampled acceptance) must reproduce greedy decode; with a real temperature the same seed must reproduce the
// same stream and different seeds must differ.
#include "engine/runtime/test_driver.h"
#include "engine/loader/tokenizer.h"
#include <cstring>

using namespace engine::runtime;

namespace {

std::vector<int> run_sampled( ModelCluster& mc, const std::vector<int>& prompt, int n, bool spec, const SamplingParams& sp ) {
	Rng rng( sp.seed );
	mc.reset_slot( 0 );
	Candidates c;
	int tok = mc.prefill( prompt.data(), static_cast<int>( prompt.size() ), 0, 0, &c );
	tok = sample_token( c, sp, rng );
	std::vector<int> out = { tok };
	int pos = static_cast<int>( prompt.size() );
	while ( static_cast<int>( out.size() ) < n && !mc.is_eos( tok ) ) {
		if ( spec ) {
			const SpecResult r = mc.speculate( 0, tok, pos, -1.0f, &sp, &rng );
			pos = r.next_pos;
			for ( int i = 0; i < r.emitted_count && static_cast<int>( out.size() ) < n; ++i ) {
				tok = r.emitted[i];
				out.push_back( tok );
				if ( mc.is_eos( tok ) ) break;
			}
		}
		else {
			mc.forward( &tok, 1, pos++, 0 );
			Candidates cs[1];
			mc.sample_topk( 1, cs );
			tok = sample_token( cs[0], sp, rng );
			out.push_back( tok );
		}
	}
	return out;
}

} // namespace

int main( int argc, char** argv ) {
	const bool glm = argc > 1 && std::strcmp( argv[1], "--glm" ) == 0;
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = glm ? "glm53" : "qwen38";
	cfg.model_dir = glm ? test::GLM_DIR : test::QWEN_DIR;
	cfg.draft_model_dir = glm ? test::GLM_DRAFT_DIR : test::QWEN_DRAFT_DIR;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	mc.warm_graphs( 8 );

	std::vector<std::vector<int>> prompts = {
		{ 248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 },
		{ 248045, 846, 198, 826, 279, 1118, 220, 20, 9944, 4947, 25, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 },
	};
	// An open-ended prompt for the seed-sensitivity check: peaked answers legitimately repeat across seeds.
	engine::loader::Tokenizer tok;
	ENGINE_ASSERT( tok.load( glm ? test::GLM_DIR : test::QWEN_DIR ), "tokenizer load failed" );
	if ( glm ) {
		// Plain-text prompts: the T->0 == greedy and seed checks do not depend on chat formatting
		prompts.clear();
		prompts.push_back( tok.encode( "The capital of France is", false ) );
		prompts.push_back( tok.encode( "The first five prime numbers are 2, 3, 5,", false ) );
	}
	std::vector<std::vector<int>> all = prompts;
	if ( glm ) {
		all.push_back( tok.encode( "Once upon a time, a robot who wanted to learn to paint", false ) );
	}
	else {
		std::vector<int> ids = { 248045, 846, 198 };
		const auto body = tok.encode( "Write a short story about a robot who learns to paint.", false );
		ids.insert( ids.end(), body.begin(), body.end() );
		for ( int t : { 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 } ) ids.push_back( t );
		all.push_back( ids );
	}
	int fail = 0;
	const int n = 40;
	for ( size_t pi = 0; pi < all.size(); ++pi ) {
		const auto& prompt = all[pi];
		const auto g = test::generate( mc, prompt, n, false ).tokens;
		SamplingParams cold;
		cold.temperature = 1e-4f;
		cold.seed = 5;
		const auto a = run_sampled( mc, prompt, n, false, cold );
		const auto b = run_sampled( mc, prompt, n, true, cold );
		// The open-ended prompt (last) only checks seeds: its FP16 logits can tie exactly (GLM: two tokens at 17.73438),
		// greedy takes the lower id and the sampler legitimately picks either.
		const bool tie_prone = ( pi + 1 == all.size() );
		const bool ok_a = tie_prone || ( a == g ), ok_b = tie_prone || ( b == g );
		std::printf( "  prompt %zu: T->0 plain %s | T->0 speculative %s\n", pi, ok_a ? "== greedy" : "DIFFERS", ok_b ? "== greedy" : "DIFFERS" );
		fail += ( ok_a ? 0 : 1 ) + ( ok_b ? 0 : 1 );
		for ( const auto* v : { &a, &b } ) {
			for ( size_t i = 0; i < g.size() && i < v->size(); ++i ) {
				if ( g[i] != ( *v )[i] ) {
					std::printf( "    first divergence at %zu: greedy %d, sampled %d\n", i, g[i], ( *v )[i] );
					break;
				}
			}
		}

		SamplingParams hot;
		hot.temperature = 1.0f;
		hot.top_p = 0.95f;
		hot.seed = 11;
		const auto h1 = run_sampled( mc, prompt, n, true, hot );
		const auto h2 = run_sampled( mc, prompt, n, true, hot );
		hot.seed = 12;
		const auto h3 = run_sampled( mc, prompt, n, true, hot );
		const bool rep = ( h1 == h2 ), diff = ( h1 != h3 );
		std::printf( "            T=1.0 same seed %s | different seed %s\n", rep ? "reproduces" : "DIFFERS", diff ? "differs" : "IDENTICAL" );
		fail += ( rep ? 0 : 1 ) + ( ( pi + 1 == all.size() && !diff ) ? 1 : 0 );
	}
	mc.shutdown();
	std::printf( "=== SAMPLING GPU GATE: %s ===\n", fail ? "FAIL" : "PASS" );
	return fail ? 1 : 0;
}
