// Flash-Next (qwen4_exp) bring-up driver: prefills the oracle prompt (results/flash_next_ref/ids.npy, the first N - 1 tokens, as the
// reference does) so the pipeline's debug trace (ENGINE_DEBUG_TRACE_LAYERS, ENGINE_DEBUG_QWEN4_HALT) can be compared with the dumps.
//   test_qwen4_exp [ids.npy] [--rows N] [--gen N]
// --gen N prefills the whole prompt with logits, prints the top candidates (compare with results/flash_next_ref/logits.npy) and decodes N tokens greedily.
#include "engine/runtime/test_driver.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace engine::runtime;

// int64 .npy, C order, 1-D
static std::vector<int> read_ids( const char* path ) {
	std::vector<int> ids;
	FILE* f = std::fopen( path, "rb" );
	if ( !f ) return ids;
	unsigned char magic[10];
	if ( std::fread( magic, 1, 10, f ) != 10 ) { std::fclose( f ); return ids; }
	const int header_len = magic[8] | ( magic[9] << 8 );
	std::fseek( f, 10 + header_len, SEEK_SET );
	long long v;
	while ( std::fread( &v, sizeof( v ), 1, f ) == 1 ) ids.push_back( static_cast<int>( v ) );
	std::fclose( f );
	return ids;
}

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	std::string path = "results/flash_next_ref/ids.npy";
	int rows = 0;
	int gen = 0;
	for ( int i = 1; i < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--rows" ) && i + 1 < argc ) rows = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--gen" ) && i + 1 < argc ) gen = std::atoi( argv[++i] );
		else path = argv[i];
	}
	std::vector<int> ids = read_ids( path.c_str() );
	ENGINE_ASSERT( !ids.empty(), "could not read the ids file" );
	const int n = rows > 0 ? rows : ( gen > 0 ? static_cast<int>( ids.size() ) : static_cast<int>( ids.size() ) - 1 );
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = "qwen4_exp";
	cfg.model_dir = test::FLASH_DIR;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	std::printf( "[*] forward of %d tokens\n", n );
	mc.reset_slot( 0 );
	if ( gen > 0 ) {
		Candidates c;
		int tok = mc.prefill( ids.data(), n, 0, 0, &c );
		std::printf( "[*] top candidates after the prompt:" );
		for ( int j = 0; j < 8 && j < c.n; ++j ) std::printf( " %d:%.3f", c.ids[j], c.logits[j] );
		std::printf( "\n[*] greedy:" );
		int pos = n;
		for ( int i = 0; i < gen; ++i ) {
			std::printf( " %d", tok );
			mc.forward( &tok, 1, pos++, 0 );
			tok = mc.sample_one();
		}
		std::printf( "\n" );
		return 0;
	}
	mc.forward( ids.data(), pod::BatchRows::causal( n, 0, 0, pod::BatchRows::LOGITS_NONE ) );
	std::printf( "[+] forward returned\n" );
	return 0;
}
