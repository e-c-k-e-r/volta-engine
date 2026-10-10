// Teacher-forced perplexity of a text file on a served model: the same tokens run through chunked prefill with every row's logits read back,
// so two builds (or two kernel paths) can be compared on a quality metric instead of greedy tokens. Optionally writes the per-token NLL.
//   test_ppl [--glm | --flash] [--file PATH] [--tokens N] [--chunk C] [--nll OUT.bin]
// The default text is the docs/ folder concatenated; ids never include special tokens.
#include "engine/runtime/test_driver.h"
#include "engine/loader/tokenizer.h"
#include "engine/backend/numeric.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>

using namespace engine::runtime;
namespace backend = engine::backend;

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	bool glm = false, flash = false;
	int n_tokens = 6000, chunk = 512;
	std::string file, nll_out;
	for ( int i = 1; i < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--glm" ) ) glm = true;
		else if ( !std::strcmp( argv[i], "--flash" ) ) flash = true;
		else if ( !std::strcmp( argv[i], "--file" ) && i + 1 < argc ) file = argv[++i];
		else if ( !std::strcmp( argv[i], "--tokens" ) && i + 1 < argc ) n_tokens = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--chunk" ) && i + 1 < argc ) chunk = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--nll" ) && i + 1 < argc ) nll_out = argv[++i];
	}
	const char* model_dir = glm ? test::GLM_DIR : ( flash ? test::FLASH_DIR : test::QWEN_DIR );
	engine::loader::Tokenizer tok;
	ENGINE_ASSERT( tok.load( model_dir ), "tokenizer load failed" );

	std::string text;
	if ( !file.empty() ) {
		std::ifstream f( file );
		std::stringstream ss;
		ss << f.rdbuf();
		text = ss.str();
	}
	else {
		std::vector<std::string> paths;
		for ( const auto& e : std::filesystem::directory_iterator( "docs" ) ) if ( e.path().extension() == ".md" ) paths.push_back( e.path().string() );
		std::sort( paths.begin(), paths.end() );
		for ( const auto& p : paths ) {
			std::ifstream f( p );
			std::stringstream ss;
			ss << f.rdbuf();
			text += ss.str() + "\n\n";
		}
	}
	std::vector<int> ids = tok.encode( text, false );
	ENGINE_ASSERT( static_cast<int>( ids.size() ) > n_tokens, "text too short for --tokens" );
	ids.resize( n_tokens );

	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = glm ? "glm5_next" : ( flash ? "qwen4_exp" : "qwen3_5" );
	cfg.model_dir = model_dir;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	const int V = mc.metadata().vocab_size;
	mc.reset_slot( 0 );
	std::vector<backend::half_t> logits( static_cast<size_t>( chunk ) * V );
	std::vector<float> nll;
	nll.reserve( n_tokens );
	double total = 0.0;
	for ( int off = 0; off + 1 < n_tokens; off += chunk ) {
		const int count = std::min( chunk, n_tokens - off );
		mc.forward( ids.data() + off, pod::BatchRows::causal( count, off, 0, pod::BatchRows::LOGITS_ALL ) );
		mc.read_logits( count, logits.data() );
		for ( int r = 0; r < count && off + r + 1 < n_tokens; ++r ) {
			const backend::half_t* l = logits.data() + static_cast<size_t>( r ) * V;
			float mx = -1e30f;
			for ( int v = 0; v < V; ++v ) mx = std::fmax( mx, backend::to_float( l[v] ) );
			double sum = 0.0;
			for ( int v = 0; v < V; ++v ) sum += std::exp( static_cast<double>( backend::to_float( l[v] ) - mx ) );
			const float nl = static_cast<float>( std::log( sum ) + mx - backend::to_float( l[ids[off + r + 1]] ) );
			nll.push_back( nl );
			total += nl;
		}
	}
	const double mean = total / nll.size();
	std::printf( "[*] %zu predicted tokens: mean NLL %.4f, perplexity %.3f\n", nll.size(), mean, std::exp( mean ) );
	if ( !nll_out.empty() ) {
		if ( FILE* f = std::fopen( nll_out.c_str(), "wb" ) ) {
			std::fwrite( nll.data(), sizeof( float ), nll.size(), f );
			std::fclose( f );
		}
	}
	mc.shutdown();
	return 0;
}
