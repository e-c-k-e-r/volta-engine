// GLM multi-row equivalence: the same prompt prefilled in chunks of 1 (the M=1 graph, the reference) and in wider
// chunks (batched MLA / KDA / mHC / MoE kernels). Reports the final-row top-16 spread and greedy continuation.
//   test_glm_rows [tokens] [chunk ...]
//   prints the pairwise max|d logit| matrix; the first chunk must rerun bit-identically.
#include "engine/runtime/test_driver.h"
#include "engine/loader/tokenizer.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace engine::runtime;

namespace {

struct Run {
	Candidates c;
	std::vector<int> cont;
};

Run prefill_run( ModelCluster& mc, const std::vector<int>& ids, int chunk, int n_cont ) {
	Run r;
	mc.reset_slot( 0 );
	int tok = mc.prefill( ids.data(), static_cast<int>( ids.size() ), 0, chunk, &r.c );
	int pos = static_cast<int>( ids.size() );
	r.cont.push_back( tok );
	for ( int i = 0; i < n_cont; ++i ) {
		mc.forward( &tok, 1, pos++, 0 );
		tok = mc.sample_one();
		r.cont.push_back( tok );
	}
	return r;
}

} // namespace

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	const int n = argc > 1 ? std::atoi( argv[1] ) : 100;
	std::vector<int> chunks;
	for ( int i = 2; i < argc; ++i ) chunks.push_back( std::atoi( argv[i] ) );
	if ( chunks.empty() ) chunks = { 1, 2, 4, 8, 9, 12, 16, 24 };
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = "glm53";
	cfg.model_dir = test::GLM_DIR;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );

	engine::loader::Tokenizer tok;
	ENGINE_ASSERT( tok.load( test::GLM_DIR ), "tokenizer load failed" );
	const std::vector<int> para = tok.encode( " The history of computing is a story of increasingly abstract machines: from mechanical looms and difference engines, through vacuum tubes and transistors, to the integrated circuits that now sit in every pocket. Each generation traded raw physical effort for a layer of symbolic description, and each layer made the one below it easier to forget.", false );
	std::vector<int> ids;
	for ( int i = 0; static_cast<int>( ids.size() ) < n; ++i ) ids.push_back( para[i % para.size()] );

	// Chunk 1 is the reference only by convention: M=1/2 take the int8-activation GEMV, M>=3 the FP16 GEMV or fused
	// GEMM, so the families differ by kernel noise. Rows of one family must not depend on the batching.
	std::vector<Run> runs;
	for ( int chunk : chunks ) {
		std::printf( "  chunk %d ...\n", chunk );
		runs.push_back( prefill_run( mc, ids, chunk, 16 ) );
	}
	const Run rerun = prefill_run( mc, ids, chunks[0], 16 );
	auto spread = [&]( const Run& x, const Run& y, int* same_ids ) {
		double d = 0.0;
		*same_ids = 0;
		for ( int i = 0; i < x.c.n; ++i ) {
			d = std::fmax( d, std::fabs( x.c.logits[i] - y.c.logits[i] ) );
			*same_ids += ( x.c.ids[i] == y.c.ids[i] ) ? 1 : 0;
		}
		return d;
	};
	int idn;
	bool ok = spread( runs[0], rerun, &idn ) == 0.0 && runs[0].cont == rerun.cont;
	std::printf( "  %d-token prompt, top logit %.2f; chunk %d rerun %s\n", n, runs[0].c.logits[0], chunks[0], ok ? "bit-identical" : "DIFFERS" );
	std::printf( "  max|d logit| between chunk sizes (top-16 ids equal):\n        " );
	for ( int c : chunks ) std::printf( " %9d", c );
	std::printf( "\n" );
	for ( size_t i = 0; i < chunks.size(); ++i ) {
		std::printf( "  %5d ", chunks[i] );
		for ( size_t j = 0; j < chunks.size(); ++j ) {
			const double d = spread( runs[i], runs[j], &idn );
			std::printf( " %5.3f/%2d ", d, idn );
		}
		std::printf( "\n" );
	}
	mc.shutdown();
	std::printf( "=== GLM ROWS: %s ===\n", ok ? "PASS" : "FAIL" );
	return ok ? 0 : 1;
}
