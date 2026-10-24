// Prefill throughput: tokens/s for a synthetic prompt at several chunk sizes. BENCH_PROFILE=1 brackets the final run
// with cudaProfilerStart/Stop for nvprof --profile-from-start off. BENCH_GLM=1 / BENCH_FLASH=1 pick GLM / Flash-Next (default Qwen); BENCH_TP sets the rank count.
//   bench_prefill [tokens] [chunk ...]
#include "engine/runtime/test_driver.h"
#include <cuda_profiler_api.h>
#include <cstdlib>

using namespace engine::runtime;

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	const int n = argc > 1 ? std::atoi( argv[1] ) : 2048;
	std::vector<int> chunks;
	for ( int i = 2; i < argc; ++i ) chunks.push_back( std::atoi( argv[i] ) );
	if ( chunks.empty() ) chunks = { 16, 64, 128 };
	ModelCluster mc;
	ClusterConfig cfg;
	const bool glm = std::getenv( "BENCH_GLM" ) != nullptr;
	const bool flash = std::getenv( "BENCH_FLASH" ) != nullptr;   // Qwen3.8-Flash-Next (qwen4_exp)
	cfg.model_type = glm ? "glm5_next" : flash ? "qwen4_exp" : "qwen3_5";
	cfg.model_dir = glm ? test::GLM_DIR : flash ? test::FLASH_DIR : test::QWEN_DIR;
	if ( const char* tp = std::getenv( "BENCH_TP" ) ) cfg.tp = std::atoi( tp );   // BENCH_TP=1: one GPU, no collectives (CUDA_VISIBLE_DEVICES picks which)
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	if ( !glm || std::getenv( "BENCH_WARM" ) ) mc.warm_graphs( 8 );  // BENCH_WARM=1: warm the decode graphs for GLM too
	std::vector<int> ids( n );
	uint32_t st = 12345u;
	for ( auto& t : ids ) { st = st * 1664525u + 1013904223u; t = 1000 + static_cast<int>( ( st >> 16 ) % 100000 ); }
	const bool prof = std::getenv( "BENCH_PROFILE" ) != nullptr;
	for ( size_t ci = 0; ci < chunks.size(); ++ci ) {
		for ( int rep = 0; rep < 2; ++rep ) {
			mc.reset_slot( 0 );
			const bool last = ( ci + 1 == chunks.size() ) && rep == 1;
			if ( prof && last ) CUDA_CHECK( cudaProfilerStart() );
			const auto t0 = std::chrono::steady_clock::now();
			mc.prefill( ids.data(), n, 0, chunks[ci] );
			const double ms = test::ms_since( t0 );
			if ( prof && last ) CUDA_CHECK( cudaProfilerStop() );
			if ( std::getenv( "BENCH_DECODE_AFTER" ) ) {  // one decode step (graph replay when warmed) after the prefill
				int tok = 1000;
				for ( int d = 0; d < std::atoi( std::getenv( "BENCH_DECODE_AFTER" ) ); ++d ) {
					mc.forward( &tok, 1, n + d, 0 );
					std::printf( "  decode step %d after prefill ok\n", d );
				}
			}
			if ( rep == 1 ) std::printf( "  chunk %3d: %d tokens in %8.1f ms = %7.0f tok/s  (%.1f ms per chunk)\n", chunks[ci], n, ms, n * 1000.0 / ms,
				ms / ( ( n + MAX( 1, chunks[ci] ) - 1 ) / MAX( 1, chunks[ci] ) ) );
		}
	}
	mc.shutdown();
	return 0;
}
