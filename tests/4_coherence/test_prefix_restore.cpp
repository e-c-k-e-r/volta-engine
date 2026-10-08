// Tier 2: a slot restored from a sequence snapshot continues exactly like a cold prefill. Prefills the first `cut`
// tokens, snapshots the slot (recurrent state + KV rows), restores into other slots and prefills only the tail; the
// next-token candidates and 16 greedy continuation tokens must match the cold run. An aligned cut (a multiple of the
// prefill chunk) reproduces the cold chunk boundaries and must match bit for bit.
//   test_prefix_restore [--tokens N]
#include "engine/runtime/test_driver.h"
#include "engine/loader/tokenizer.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace engine::runtime;

static std::vector<int> decode( ModelCluster& mc, int first, int pos, int slot, int count ) {
	std::vector<int> out = { first };
	int tok = first;
	for ( int i = 1; i < count; ++i ) {
		mc.forward( &tok, 1, pos++, slot );
		tok = mc.sample_one();
		out.push_back( tok );
	}
	return out;
}

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	int n_tokens = 6000;
	for ( int i = 1; i < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--tokens" ) && i + 1 < argc ) n_tokens = std::atoi( argv[++i] );
	}
	engine::loader::Tokenizer tok;
	ENGINE_ASSERT( tok.load( test::QWEN_DIR ), "tokenizer load failed" );
	std::vector<int> ids = { 248045, 846, 198 };
	for ( int i = 0; static_cast<int>( ids.size() ) < n_tokens; ++i ) {
		const std::string s = "Note " + std::to_string( i ) + ": the weather in district " + std::to_string( i * 7 % 101 ) + " was mild on day " + std::to_string( i % 365 ) + ". ";
		const std::vector<int> t = tok.encode( s, false );
		ids.insert( ids.end(), t.begin(), t.end() );
	}
	ids.resize( n_tokens );
	const std::vector<int> end = { 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 };
	ids.insert( ids.end(), end.begin(), end.end() );
	const int n = static_cast<int>( ids.size() );

	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = "qwen3_5";
	cfg.model_dir = test::QWEN_DIR;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	ENGINE_ASSERT( mc.snapshot_bytes( 1024 ) > 0, "the model cannot snapshot" );
	mc.warm_graphs( 8 );
	const int chunk = mc.metadata().max_rows;
	std::printf( "[*] prompt %d tokens, prefill chunk %d, snapshot of 4096 tokens = %.1f MiB per rank\n", n, chunk, mc.snapshot_bytes( 4096 ) / 1048576.0 );

	// Cold reference
	mc.reset_slot( 0 );
	Candidates cold_c;
	const int cold_first = mc.prefill( ids.data(), n, 0, 0, &cold_c );
	const std::vector<int> cold = decode( mc, cold_first, n, 0, 16 );

	int failures = 0;
	for ( int cut : { chunk * ( ( n / 2 ) / chunk ), n / 2 + 37 } ) {
		const bool aligned = cut % chunk == 0;
		mc.reset_slot( 0 );
		mc.prefill( ids.data(), cut, 0 );
		ModelCluster::SeqSnapshot snap;
		auto t0 = std::chrono::steady_clock::now();
		ENGINE_ASSERT( mc.save_snapshot( 0, cut, &snap ), "save_snapshot failed" );
		const double save_ms = test::ms_since( t0 );
		for ( int slot : { 1, 2, 1 } ) {
			mc.reset_slot( slot );
			t0 = std::chrono::steady_clock::now();
			ENGINE_ASSERT( mc.load_snapshot( slot, snap ), "load_snapshot failed" );
			const double load_ms = test::ms_since( t0 );
			Candidates warm_c;
			const int first = mc.prefill( ids.data() + cut, n - cut, slot, 0, &warm_c, cut );
			const std::vector<int> warm = decode( mc, first, n, slot, 16 );
			float max_diff = 0.0f;
			for ( int j = 0; j < warm_c.n; ++j ) max_diff = std::max( max_diff, std::fabs( warm_c.logits[j] - cold_c.logits[j] ) );
			const bool same = warm == cold;
			const bool ok = same && ( aligned ? max_diff == 0.0f : max_diff < 0.25f );
			std::printf( "%s cut %d (%s) -> slot %d: save %.1f ms, load %.1f ms, max |dlogit| %.4g, tokens %s\n", ok ? "[+]" : "[-]", cut,
				aligned ? "aligned" : "unaligned", slot, save_ms, load_ms, max_diff, same ? "identical" : "DIFFER" );
			failures += ok ? 0 : 1;
		}
		mc.free_snapshot( &snap );
	}
	std::printf( "%s\n", failures ? "[-] FAIL" : "[+] PASS prefix restore" );
	mc.shutdown();
	return failures ? 1 : 0;
}
