// Diagnostic for test_mtp's first disagreement on the Qwen3.8-27B (prompt 0): the logits that decide continuation token IDX in the single-row decode
// path and in causal verify blocks of L rows ending at the same position (the way an MTP round verifies), for the two tokens that disagree.
//   diag_logits [IDX=22] [TOKEN_A=2574] [TOKEN_B=886]
#include "engine/runtime/test_driver.h"
#include "engine/loader/tokenizer.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using namespace engine::runtime;

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	const int idx = argc > 1 ? std::atoi( argv[1] ) : 22;
	const int ta = argc > 2 ? std::atoi( argv[2] ) : 2574, tb = argc > 3 ? std::atoi( argv[3] ) : 886;
	engine::loader::Tokenizer tok;
	ENGINE_ASSERT( tok.load( test::QWEN_DIR ), "tokenizer load failed" );
	std::vector<int> prompt = { 248045, 846, 198 };
	const std::vector<int> body = tok.encode( "Write a Python function that checks whether a string is a palindrome, with a docstring and three unit tests.", false );
	prompt.insert( prompt.end(), body.begin(), body.end() );
	const std::vector<int> tail = { 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 };
	prompt.insert( prompt.end(), tail.begin(), tail.end() );

	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = "qwen3_5";
	cfg.model_dir = test::QWEN_DIR;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	mc.warm_graphs( 8 );
	const int V = mc.metadata().vocab_size;
	const test::GenStats plain = test::generate( mc, prompt, 96, false );
	const std::vector<int>& T = plain.tokens;
	std::printf( "[*] prompt %zu tokens, plain continuation[%d] = %d (A=%d, B=%d)\n", prompt.size(), idx, T[idx], ta, tb );

	auto trial = [&]( int L, int ctx_block ) {
		mc.reset_slot( 0 );
		const int first = mc.prefill( prompt.data(), static_cast<int>( prompt.size() ), 0 );
		ENGINE_ASSERT( first == T[0], "prefill token differs from the plain run" );
		int pos = static_cast<int>( prompt.size() );
		// the block's context: single rows, or causal blocks of ctx_block rows (the way MTP rounds advance the state)
		for ( int i = 0; i <= idx - 1 - L; ) {
			const int n = std::min( ctx_block, idx - L - i );
			mc.forward( &T[i], n, pos, 0 );
			pos += n;
			i += n;
		}
		if ( L == 1 ) mc.forward( &T[idx - 1], 1, pos, 0 );
		else mc.forward( &T[idx - L], L, pos, 0 );
		std::vector<half> h( static_cast<size_t>( L ) * V );
		mc.read_logits( L, h.data() );
		std::vector<float> f( V );
		for ( int i = 0; i < V; ++i ) f[i] = __half2float( h[static_cast<size_t>( L - 1 ) * V + i] );
		return f;
	};

	const std::vector<float> ref = trial( 1, 1 );
	int order[3] = { 0, 1, 2 };
	(void) order;
	for ( int cb : { 1, 8, 4 } ) for ( int L : { 1, 4, 8 } ) {
		const std::vector<float> f = ( L == 1 && cb == 1 ) ? ref : trial( L, cb );
		int a1 = 0, a2 = -1;
		for ( int i = 1; i < V; ++i ) {
			if ( f[i] > f[a1] ) { a2 = a1; a1 = i; }
			else if ( a2 < 0 || f[i] > f[a2] ) { if ( i != a1 ) a2 = i; }
		}
		double maxd = 0.0;
		int ndiff = 0;
		for ( int i = 0; i < V; ++i ) {
			const double d = std::fabs( static_cast<double>( f[i] ) - ref[i] );
			maxd = std::max( maxd, d );
			if ( f[i] != ref[i] ) ++ndiff;
		}
		std::printf( "  ctx_block=%d L=%d: top1 %d (%.5f) top2 %d (%.5f) | A %.5f  B %.5f  A-B %+.5f | vs L=1: max|d| %.5f, %d of %d logits differ\n", cb, L, a1, f[a1], a2, f[a2],
			f[ta], f[tb], f[ta] - f[tb], maxd, ndiff, V );
	}
	mc.shutdown();
	return 0;
}
