// Writes the full-vocabulary logits after a fixed prompt to a file (FP16), for comparing two builds or two checkpoints of one model
// (cosine and top-k overlap, tests/tools/compare_logits.py). GLM's test checkpoint by default.
//   diag_logits_dump <out.bin> [--model qwen3_5 | qwen4_exp | glm5_next | DIR]
#include "engine/runtime/test_driver.h"
#include <cstdio>
#include <cstring>
#include <vector>

using namespace engine::runtime;

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	if ( argc < 2 ) { std::printf( "usage: diag_logits_dump <out.bin> [--model M]\n" ); return 2; }
	const test::TestModel tm = test::model_arg( argc, argv, "glm5_next" );
	ENGINE_ASSERT( tm.is( "qwen3_5" ) || tm.is( "qwen4_exp" ) || tm.is( "glm5_next" ), "diag_logits_dump: --model qwen3_5, qwen4_exp or glm5_next" );
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = tm.type;
	cfg.model_dir = tm.dir;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	std::vector<int> ids( 48 );
	uint32_t st = 4242u;
	for ( auto& t : ids ) { st = st * 1664525u + 1013904223u; t = 1000 + static_cast<int>( ( st >> 16 ) % 50000 ); }
	mc.reset_slot( 0 );
	const int first = mc.prefill( ids.data(), static_cast<int>( ids.size() ), 0 );
	std::vector<engine::backend::half_t> logits( static_cast<size_t>( mc.metadata().vocab_size ) );
	mc.read_logits( 1, logits.data() );
	FILE* f = std::fopen( argv[1], "wb" );
	ENGINE_ASSERT( f, "cannot open the output file" );
	std::fwrite( logits.data(), sizeof( engine::backend::half_t ), logits.size(), f );
	std::fclose( f );
	std::printf( "[+] first token %d, %zu logits written to %s\n", first, logits.size(), argv[1] );
	mc.shutdown();
	return 0;
}
