// Tier-3 speculative benchmark: DFlash2 rounds vs plain greedy decode on the same prompts.
// Speculative output must be token-identical to greedy (lossless at T=0); on divergence the first
// differing token is diagnosed as a near-tie flip or a speculative-round logic error. Exit 1 only for the latter.
#include "engine/runtime/test_driver.h"
#include <cstring>

using namespace engine::runtime;

namespace impl {

// Replays greedy decode up to the step that produced g[i] (i >= 1) and reports the target's top-2
// there and where the speculative token ranks. A tiny margin with spec == runner-up is a near-tie
// flip (M>1 vs M=1 numerics); a large margin is a logic error in the speculative round.
bool diagnose( ModelCluster& mc, const std::vector<int>& prompt, const std::vector<int>& g, const std::vector<int>& s, size_t i ) {
	std::printf( "    first divergence at token %zu: greedy %d, spec %d\n", i, g[i], s[i] );
	if ( i == 0 ) {
		std::printf( "    (prefill prediction differs: both paths share prefill, so this is a harness bug)\n" );
		return false;
	}
	mc.reset_slot( 0 );
	mc.prefill( prompt.data(), static_cast<int>( prompt.size() ), 0 );
	int pos = static_cast<int>( prompt.size() );
	for ( size_t t = 0; t < i; ++t ) {
		mc.forward( &g[t], 1, pos++, 0 );
	}
	const int V = mc.metadata().vocab_size;
	std::vector<half> h( V );
	mc.read_logits( 1, h.data() );
	int i1 = 0, i2 = -1;
	for ( int j = 1; j < V; ++j ) {
		if ( __half2float( h[j] ) > __half2float( h[i1] ) ) { i2 = i1; i1 = j; }
		else if ( i2 < 0 || __half2float( h[j] ) > __half2float( h[i2] ) ) { i2 = j; }
	}
	int rank = 0;
	const float vs = __half2float( h[s[i]] );
	for ( int j = 0; j < V; ++j ) rank += ( __half2float( h[j] ) > vs ) ? 1 : 0;
	std::printf( "    greedy step logits: top1 %d (%.4f), top2 %d (%.4f), margin %.4f; spec token rank %d (%.4f)\n",
		i1, __half2float( h[i1] ), i2, __half2float( h[i2] ), __half2float( h[i1] ) - __half2float( h[i2] ), rank, vs );
	const bool near_tie = ( rank <= 1 && __half2float( h[i1] ) - vs <= 0.0625f );
	std::printf( "    -> %s\n", near_tie ? "near-tie flip (batch numerics)" : "NOT a near tie: speculative-round logic error" );
	return near_tie;
}

} // namespace impl

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	const bool glm = ( argc > 1 && !std::strcmp( argv[1], "--glm" ) );
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = glm ? "glm53" : "qwen38";
	cfg.model_dir = glm ? test::GLM_DIR : test::QWEN_DIR;
	cfg.draft_model_dir = glm ? test::GLM_DRAFT_DIR : test::QWEN_DRAFT_DIR;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	mc.warm_graphs( 8 );

	// Qwen chat prompts ("List the first 5 prime numbers:" / "What is the capital of France?")
	const std::vector<std::vector<int>> prompts = {
		{ 248045, 846, 198, 826, 279, 1118, 220, 20, 9944, 4947, 25, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 },
		{ 248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 },
	};
	const int max_new = 96;
	double g_ms = 0.0, s_ms = 0.0;
	int g_tok = 0, s_tok = 0, proposed = 0, accepted = 0, rounds = 0, diverged = 0, hard = 0;
	for ( size_t pi = 0; pi < prompts.size(); ++pi ) {
		const auto& p = prompts[pi];
		const auto g = test::generate( mc, p, max_new, false );
		const auto s = test::generate( mc, p, max_new, true );
		if ( g.tokens != s.tokens ) {
			++diverged;
			std::printf( "[-] prompt %zu: speculative output diverged from greedy\n", pi );
			size_t i = 0;
			while ( i < g.tokens.size() && i < s.tokens.size() && g.tokens[i] == s.tokens[i] ) ++i;
			if ( i < g.tokens.size() && i < s.tokens.size() ) {
				if ( !impl::diagnose( mc, p, g.tokens, s.tokens, i ) ) ++hard;
			}
			else {
				std::printf( "    length differs only: greedy %zu, spec %zu tokens\n", g.tokens.size(), s.tokens.size() );
				++hard;
			}
		}
		g_ms += g.decode_ms;
		s_ms += s.decode_ms;
		g_tok += static_cast<int>( g.tokens.size() ) - 1;
		s_tok += static_cast<int>( s.tokens.size() ) - 1;
		proposed += s.proposed;
		accepted += s.accepted;
		rounds += s.rounds;
	}
	const double g_tps = g_tok * 1000.0 / g_ms;
	const double s_tps = s_tok * 1000.0 / s_ms;
	std::printf( "\n=== %s DFlash2 speculative benchmark (TP=4) ===\n", cfg.model_type.c_str() );
	std::printf( "   Greedy M=1        : %.2f tok/s\n", g_tps );
	std::printf( "   Speculative       : %.2f tok/s (%.2fx)\n", s_tps, s_tps / g_tps );
	std::printf( "   Acceptance        : %.1f%% (%d/%d), %.2f tokens/round\n",
		proposed ? 100.0 * accepted / proposed : 0.0, accepted, proposed, rounds ? static_cast<double>( s_tok ) / rounds : 0.0 );
	std::printf( "   Output            : %s\n", diverged == 0 ? "token-identical to greedy"
		: ( hard == 0 ? "diverged from greedy only at near-ties (accepted)" : "DIVERGED from greedy (not a near-tie)" ) );
	mc.shutdown();
	return hard ? 1 : 0;
}
