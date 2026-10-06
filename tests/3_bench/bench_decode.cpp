// Tier-3 decode benchmark on TP=4 (real weights, eager launches, wall-clock).
//   M=1 decode   : single-slot autoregressive step (forward + argmax)
//   Mode A M=8   : causal verify block latency
//   Mode B       : one token per slot across `slots` independent requests
#include "engine/runtime/test_driver.h"
#include <cstring>

using namespace engine::runtime;

namespace impl {

void bench_model( const char* type, const char* dir, int steps ) {
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = type;
	cfg.model_dir = dir;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	const auto& meta = mc.metadata();
	const std::vector<int> prompt = { 15, 16, 17, 18, 19, 20, 21, 22 };

	// M=1 decode
	mc.reset_slot( 0 );
	int tok = mc.prefill( prompt.data(), static_cast<int>( prompt.size() ), 0 );
	int pos = static_cast<int>( prompt.size() );
	for ( int i = 0; i < 4; ++i ) { mc.forward( &tok, 1, pos++, 0 ); tok = mc.sample_one(); } // warmup
	auto t0 = std::chrono::steady_clock::now();
	for ( int i = 0; i < steps; ++i ) {
		mc.forward( &tok, 1, pos++, 0 );
		tok = mc.sample_one();
	}
	const double m1_ms = test::ms_since( t0 ) / steps;

	// Mode A: M=8 causal block (state rolled back each iteration)
	const int iters = 16;
	int block[8];
	for ( int i = 0; i < 8; ++i ) block[i] = prompt[i];
	int preds[pod::MAX_BATCH];
	t0 = std::chrono::steady_clock::now();
	for ( int i = 0; i < iters; ++i ) {
		mc.checkpoint( 0 );
		mc.forward( block, 8, pos, 0 );
		mc.sample( preds, 8 );
		mc.rollback( 0 );
	}
	const double m8_ms = test::ms_since( t0 ) / iters;

	std::printf( "\n=== %s decode benchmark (TP=4) ===\n", type );
	std::printf( "   M=1 decode        : %.2f ms/token (%.2f tok/s)\n", m1_ms, 1000.0 / m1_ms );
	std::printf( "   Mode A M=8 verify : %.2f ms/block (%.2fx M=1)\n", m8_ms, m8_ms / m1_ms );

	// Mode B: one row per slot
	const int slots = MIN( meta.max_slots, 8 );
	if ( slots > 1 ) {
		int ids[8], positions[8], toks[8];
		for ( int s = 0; s < slots; ++s ) {
			mc.reset_slot( s );
			toks[s] = mc.prefill( prompt.data(), static_cast<int>( prompt.size() ), s );
			ids[s] = s;
			positions[s] = static_cast<int>( prompt.size() );
		}
		t0 = std::chrono::steady_clock::now();
		const int b_steps = steps / 2;
		for ( int i = 0; i < b_steps; ++i ) {
			mc.forward( toks, pod::BatchRows::batched( slots, ids, positions ) );
			mc.sample( toks, slots );
			for ( int s = 0; s < slots; ++s ) positions[s]++;
		}
		const double b_ms = test::ms_since( t0 ) / b_steps;
		std::printf( "   Mode B x%d slots   : %.2f ms/step (%.1f aggregate tok/s)\n", slots, b_ms, slots * 1000.0 / b_ms );
	}
	mc.shutdown();
}

} // namespace impl

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	bool qwen = true;
	bool glm = false;
	for ( int i = 1; i < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--glm" ) || !std::strcmp( argv[i], "--glm-only" ) ) { qwen = false; glm = true; }
		else if ( !std::strcmp( argv[i], "--all" ) ) { qwen = true; glm = true; }
	}
	if ( qwen ) impl::bench_model( "qwen38", test::QWEN_DIR, 50 );
	if ( glm ) impl::bench_model( "glm53", test::GLM_DIR, 30 );
	std::printf( "\n[+] Benchmark complete.\n" );
	return 0;
}
