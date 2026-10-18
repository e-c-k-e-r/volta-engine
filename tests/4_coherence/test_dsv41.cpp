// DeepSeek-V4.1-Flash end to end: prefill a token-id prompt, then greedy decode, printing each token id. With trace_layers=<file> every
// eager forward appends each layer's hyper-connection streams (tag 100 + layer after attention, tag layer after the FFN; header tag, rows,
// first position; then rows x 4 x hidden FP32) for tests/tools/dsv41/trace_cmp.py against dsv41_ref.py's dumps.
//   test_dsv41 <model dir> <ids file: whitespace-separated token ids> [new tokens] [option=value ...]
#include "engine/runtime/test_driver.h"
#include "engine/runtime/options.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace engine::runtime;

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	if ( argc < 3 ) {
		std::fprintf( stderr, "usage: test_dsv41 <model dir> <ids file> [new tokens] [option=value ...]\n" );
		return 2;
	}
	const std::string dir = argv[1];
	std::vector<int> prompt;
	{
		std::ifstream f( argv[2] );
		for ( int t; f >> t; ) prompt.push_back( t );
	}
	ENGINE_ASSERT( !prompt.empty(), "test_dsv41: empty prompt" );
	int gen = 8, a = 3;
	if ( argc > 3 && std::string( argv[3] ).find( '=' ) == std::string::npos ) { gen = std::atoi( argv[3] ); a = 4; }
	for ( ; a < argc; ++a ) {
		const std::string kv = argv[a];
		const size_t eq = kv.find( '=' );
		ENGINE_ASSERT( eq != std::string::npos && set_option( kv.substr( 0, eq ), kv.substr( eq + 1 ) ), "test_dsv41: bad option" );
	}
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = "deepseek_v41";
	cfg.model_dir = dir;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	const auto t0 = std::chrono::steady_clock::now();
	const test::GenStats st = test::generate( mc, prompt, gen, false );
	std::printf( "[*] prompt %zu tokens, %.1f s\n", prompt.size(), test::ms_since( t0 ) / 1000.0 );
	std::printf( "[*] tokens:" );
	for ( int t : st.tokens ) std::printf( " %d", t );
	std::printf( "\n" );
	mc.shutdown();
	return 0;
}
