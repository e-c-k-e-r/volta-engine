// R2 gate: after a partial accept, the next forward must see the same recurrent state whether it came from the
// per-row snapshot commit (current path) or from checkpoint + rollback + replay (the path R2 removed).
// For a = 0..7 accepted tokens, runs both on identical prompts and compares the logits of the next anchor row.
//   pass: max |dlogit| <= 2 FP16 ULP of the row's largest logit on every case and the argmax agrees.
#include "engine/runtime/test_driver.h"
#include <cmath>
#include <cstdio>

using namespace engine::runtime;

namespace {

float ulp_of( float v ) {
	const float a = std::fabs( v );
	if ( a < 6.1e-5f ) return 5.96e-8f;
	int e = 0;
	std::frexp( a, &e );
	return std::ldexp( 1.0f, e - 11 );
}

} // namespace

int main() {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = "qwen38";
	cfg.model_dir = test::QWEN_DIR;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	mc.warm_graphs( 8 );

	const std::vector<int> prompt = { 248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 };
	const int V = mc.metadata().vocab_size;
	const int slot = 0;

	// Ground-truth continuation so the first `a` block rows are the target's own tokens.
	std::vector<int> truth;
	{
		mc.reset_slot( slot );
		int tok = mc.prefill( prompt.data(), static_cast<int>( prompt.size() ), slot );
		int pos = static_cast<int>( prompt.size() );
		for ( int i = 0; i < 9; ++i ) {
			truth.push_back( tok );
			mc.forward( &tok, 1, pos++, slot );
			tok = mc.sample_one();
		}
	}

	int fails = 0;
	double worst = 0.0;
	for ( int a = 0; a <= 7; ++a ) {
		int block[8];
		for ( int i = 0; i < 8; ++i ) block[i] = truth[i];
		for ( int i = a + 1; i < 8; ++i ) block[i] = 1000 + 37 * i; // rejected tail
		const int pos = static_cast<int>( prompt.size() );
		const int next_anchor = truth[a + 1];
		std::vector<half> la( V ), lb( V );

		// Path A: snapshot commit.
		mc.reset_slot( slot );
		mc.prefill( prompt.data(), pos, slot );
		mc.set_spec_snapshot( true );
		mc.forward( block, 8, pos, slot );
		if ( a < 7 ) mc.commit_spec_row( slot, a );
		mc.set_spec_snapshot( false );
		mc.forward( &next_anchor, 1, pos + 1 + a, slot );
		mc.read_logits( 1, la.data() );

		// Path B: checkpoint + rollback + replay of [anchor .. a].
		mc.reset_slot( slot );
		mc.prefill( prompt.data(), pos, slot );
		mc.checkpoint( slot );
		mc.forward( block, 8, pos, slot );
		if ( a < 7 ) {
			mc.rollback( slot );
			mc.forward( block, a + 1, pos, slot );
		}
		mc.forward( &next_anchor, 1, pos + 1 + a, slot );
		mc.read_logits( 1, lb.data() );

		// Difference in FP16 ULPs of the row's largest-magnitude logit: a per-element ULP is meaningless for
		// logits near zero, where the same absolute noise spans thousands of tiny ULPs.
		double max_diff = 0.0;
		float scale = 0.0f;
		int ia = 0, ib = 0;
		for ( int j = 0; j < V; ++j ) {
			const float x = __half2float( la[j] ), y = __half2float( lb[j] );
			max_diff = std::fmax( max_diff, std::fabs( x - y ) );
			scale = std::fmax( scale, std::fmax( std::fabs( x ), std::fabs( y ) ) );
			if ( x > __half2float( la[ia] ) ) ia = j;
			if ( y > __half2float( lb[ib] ) ) ib = j;
		}
		const double max_ulps = max_diff / ulp_of( scale );
		worst = std::fmax( worst, max_ulps );
		const bool ok = max_ulps <= 2.0 && ia == ib;
		fails += ok ? 0 : 1;
		std::printf( "  accepted %d: max logit diff %.2f ULP, argmax %d vs %d  %s\n", a, max_ulps, ia, ib, ok ? "PASS" : "FAIL" );
	}
	mc.shutdown();
	std::printf( "=== SPEC STATE GATE: %s (worst %.2f ULP) ===\n", fails ? "FAIL" : "PASS", worst );
	return fails ? 1 : 0;
}
