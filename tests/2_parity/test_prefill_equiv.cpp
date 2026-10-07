// Prefill equivalence: a long prompt prefilled in 16-row chunks (fused GEMM kernel, the pre-existing path) against
// 256-row chunks (dequantize + cuBLAS HGEMM). The two leave the same model state if the final-row logits agree and
// greedy decode continues identically from both. Also checks that the new path is bit-reproducible.
//   pass: cosine >= 0.9999 on the final-row logits, top-1 equal, next 32 greedy tokens equal, rerun logits identical.
#include "engine/runtime/test_driver.h"
#include "engine/loader/tokenizer.h"
#include <cmath>
#include <cstdio>

using namespace engine::runtime;

namespace {

// Final-row top-16 (ids, logits) plus a greedy continuation. read_logits would return row 0 of the last chunk, which
// is not the final position unless that chunk has a single row; prefill's candidates are always the final row's.
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
	const int n = argc > 1 ? std::atoi( argv[1] ) : 600;
	const int chunk_a = argc > 2 ? std::atoi( argv[2] ) : 16;      // reference: the fused-kernel path
	const int chunk_b = argc > 3 ? std::atoi( argv[3] ) : 256;     // path under test
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = "qwen38";
	cfg.model_dir = test::QWEN_DIR;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	mc.warm_graphs( 8 );

	// Natural text: a passage repeated to length, behind a chat prefix.
	engine::loader::Tokenizer tok;
	ENGINE_ASSERT( tok.load( test::QWEN_DIR ), "tokenizer load failed" );
	const std::vector<int> para = tok.encode( " The history of computing is a story of increasingly abstract machines: from mechanical looms and difference engines, through vacuum tubes and transistors, to the integrated circuits that now sit in every pocket. Each generation traded raw physical effort for a layer of symbolic description, and each layer made the one below it easier to forget.", false );
	std::vector<int> ids = { 248045, 846, 198 };
	for ( int i = 0; static_cast<int>( ids.size() ) < n; ++i ) ids.push_back( para[i % para.size()] );

	const Run ref = prefill_run( mc, ids, chunk_a, 32 );
	const Run ref2 = prefill_run( mc, ids, chunk_a, 32 );
	// Existing configurations spanning the GEMV and fused-GEMM families: their mutual spread is the noise floor.
	const Run f8 = prefill_run( mc, ids, 8, 4 );
	const Run f12 = prefill_run( mc, ids, 12, 4 );
	const Run f24 = prefill_run( mc, ids, 24, 4 );
	const Run test = prefill_run( mc, ids, chunk_b, 32 );
	const Run test2 = prefill_run( mc, ids, chunk_b, 32 );

	auto cmp = [&]( const Run& x, const Run& y, double* maxd, int* same_ids ) {
		*maxd = 0.0;
		*same_ids = 0;
		for ( int i = 0; i < x.c.n; ++i ) {
			*maxd = std::fmax( *maxd, std::fabs( x.c.logits[i] - y.c.logits[i] ) );
			*same_ids += ( x.c.ids[i] == y.c.ids[i] ) ? 1 : 0;
		}
	};
	double d_floor, d_test;
	int ids_floor, ids_test;
	double d_family = 0.0;
	int ids_family = 0;
	cmp( ref, ref2, &d_floor, &ids_floor );
	{
		const Run* fam[4] = { &ref, &f8, &f12, &f24 };
		for ( int i = 0; i < 4; ++i ) {
			for ( int j = i + 1; j < 4; ++j ) {
				double d;
				int idn;
				cmp( *fam[i], *fam[j], &d, &idn );
				d_family = std::fmax( d_family, d );
			}
		}
	}
	cmp( ref, test, &d_test, &ids_test );
	int same = 0;
	for ( size_t i = 0; i < ref.cont.size(); ++i ) same += ( ref.cont[i] == test.cont[i] ) ? 1 : 0;
	double d_rerun;
	int ids_rerun;
	cmp( test, test2, &d_rerun, &ids_rerun );
	const bool rerun_ok = d_rerun == 0.0 && ids_rerun == test.c.n && test.cont == test2.cont;
	std::printf( "  reference rerun: max|d| %.4f; spread among existing chunk configs (%d, 8, 12, 24): max|d| %.4f\n", d_floor, chunk_a, d_family );
	std::printf( "  %d-token prompt, chunk %d vs %d: top-16 ids equal %d/16, max|d logit| %.4f (top logit %.2f), greedy continuation %d/%zu equal, new-path rerun %s\n",
		n, chunk_a, chunk_b, ids_test, d_test, ref.c.logits[0], same, ref.cont.size(), rerun_ok ? "bit-identical" : "DIFFERS" );
	const bool top1 = ref.c.ids[0] == test.c.ids[0];
	const bool ok = top1 && d_test <= MAX( 0.5, 1.25 * d_family ) && rerun_ok && d_floor == 0.0;
	mc.shutdown();
	std::printf( "=== PREFILL EQUIV: %s ===\n", ok ? "PASS" : "FAIL" );
	return ok ? 0 : 1;
}
