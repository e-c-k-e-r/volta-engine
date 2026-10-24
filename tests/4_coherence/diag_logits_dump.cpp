// Writes the full-vocabulary logits after a fixed prompt to a file (FP16), for comparing two builds or two checkpoints of one model
// (cosine and top-k overlap, tests/tools/compare_logits.py). GLM by default; --flash and --qwen pick the others.
//   diag_logits_dump <out.bin> [--flash | --qwen]    (ENGINE_TEST_*_DIR picks the checkpoint)
#include "engine/runtime/test_driver.h"
#include <cstdio>
#include <cstring>
#include <vector>

using namespace engine::runtime;

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	if ( argc < 2 ) { std::printf( "usage: diag_logits_dump <out.bin> [--flash | --qwen]\n" ); return 2; }
	bool flash = false, qwen = false;
	for ( int i = 2; i < argc; ++i ) { flash = flash || !std::strcmp( argv[i], "--flash" ); qwen = qwen || !std::strcmp( argv[i], "--qwen" ); }
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = qwen ? "qwen3_5" : flash ? "qwen4_exp" : "glm5_next";
	cfg.model_dir = qwen ? test::QWEN_DIR : flash ? test::FLASH_DIR : test::GLM_DIR;
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
