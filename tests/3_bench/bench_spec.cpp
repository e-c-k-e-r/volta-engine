// Tier-3 speculative benchmark: DFlash2 rounds vs plain greedy decode on the same prompts.
// Speculative output is asserted token-identical to greedy (lossless at T=0).
#include "engine/runtime/test_driver.h"
#include <cstring>

using namespace engine::runtime;

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	const bool glm = ( argc > 1 && !std::strcmp( argv[1], "--glm" ) );
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = glm ? "glm53" : "qwen38";
	cfg.model_dir = glm ? test::GLM_DIR : test::QWEN_DIR;
	cfg.draft_model_dir = glm ? test::GLM_DRAFT_DIR : test::QWEN_DRAFT_DIR;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );

	// Qwen chat prompts ("List the first 5 prime numbers:" / "What is the capital of France?")
	const std::vector<std::vector<int>> prompts = {
		{ 248045, 846, 198, 826, 279, 1118, 220, 20, 9944, 4947, 25, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 },
		{ 248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 },
	};
	const int max_new = 96;
	double g_ms = 0.0, s_ms = 0.0;
	int g_tok = 0, s_tok = 0, proposed = 0, accepted = 0, rounds = 0;
	for ( const auto& p : prompts ) {
		const auto g = test::generate( mc, p, max_new, false );
		const auto s = test::generate( mc, p, max_new, true );
		ENGINE_ASSERT( g.tokens == s.tokens, "speculative output diverged from greedy" );
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
	std::printf( "   Output            : token-identical to greedy\n" );
	mc.shutdown();
	return 0;
}
