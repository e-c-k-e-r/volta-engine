// Diagnostic for a greedy disagreement between two decoding modes on GLM: the target's logits for two candidate continuation words after a
// given assistant prefix, in the single-row decode path and in causal verify blocks of L rows ending at the same position (the shapes a
// speculative round verifies). A first difference that is a near-tie in the logits (a few FP16 ulps) is rounding, not a defect.
//   diag_glm_tie PROMPT_FILE PREFIX_FILE WORD_A WORD_B     (the user message, the assistant text before the word, two continuations)
#include "engine/runtime/test_driver.h"
#include "engine/loader/tokenizer.h"
#include "engine/loader/chat_template.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace engine::runtime;

static std::string slurp( const char* path ) {
	std::ifstream f( path );
	std::stringstream ss;
	ss << f.rdbuf();
	return ss.str();
}

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	ENGINE_ASSERT( argc >= 5, "usage: diag_glm_tie PROMPT_FILE PREFIX_FILE WORD_A WORD_B" );
	const std::string user = slurp( argv[1] ), prefix = slurp( argv[2] );
	engine::loader::Tokenizer tok;
	ENGINE_ASSERT( tok.load( test::GLM_DIR ), "tokenizer load failed" );
	engine::loader::ChatTemplate chat;
	ENGINE_ASSERT( chat.load( test::GLM_DIR ), "chat template load failed" );
	engine::loader::ChatTemplateOptions opts;
	opts.enable_thinking = false;
	std::vector<int> ids = tok.encode( chat.render( { { "user", user } }, opts ), false );
	const std::vector<int> pre = tok.encode( prefix, false );
	ids.insert( ids.end(), pre.begin(), pre.end() );
	const std::vector<int> wa = tok.encode( std::string( argv[3] ), false ), wb = tok.encode( std::string( argv[4] ), false );
	ENGINE_ASSERT( !wa.empty() && !wb.empty() && wa[0] != wb[0], "the two words must start with different tokens" );
	const int ta = wa[0], tb = wb[0];
	std::printf( "[*] context %zu tokens, A = token %d, B = token %d\n", ids.size(), ta, tb );

	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = "glm5_next";
	cfg.model_dir = test::GLM_DIR;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	mc.warm_graphs( 8 );
	const int V = mc.metadata().vocab_size;
	const int n = static_cast<int>( ids.size() );

	// Prefill all but the last L tokens in one go, feed the rest as one causal block of L rows, read the last row's logits
	auto trial = [&]( int L ) {
		mc.reset_slot( 0 );
		if ( n - L > 0 ) mc.prefill( ids.data(), n - L, 0 );
		mc.forward( &ids[n - L], L, n - L, 0 );
		std::vector<half> h( static_cast<size_t>( L ) * V );
		mc.read_logits( L, h.data() );
		std::vector<float> f( V );
		for ( int i = 0; i < V; ++i ) f[i] = __half2float( h[static_cast<size_t>( L - 1 ) * V + i] );
		return f;
	};
	const std::vector<float> ref = trial( 1 );
	for ( int L : { 1, 2, 4, 8 } ) {
		const std::vector<float> f = L == 1 ? ref : trial( L );
		int a1 = 0, a2 = -1;
		for ( int i = 1; i < V; ++i ) {
			if ( f[i] > f[a1] ) { a2 = a1; a1 = i; }
			else if ( a2 < 0 || f[i] > f[a2] ) { if ( i != a1 ) a2 = i; }
		}
		std::printf( "  L=%d: top1 %d (%.5f) top2 %d (%.5f) | A %.5f  B %.5f  A-B %+.5f\n", L, a1, f[a1], a2, f[a2], f[ta], f[tb], f[ta] - f[tb] );
	}
	mc.shutdown();
	return 0;
}
