// How much the next-token logits move when a prompt is prefilled in two parts at a different cut than the cold run's chunk
// boundaries (GLM). No snapshot needed, so it also runs with ENGINE_DSA=0 to separate the sparse-attention selection from
// the rest. Prints max |dlogit| of the candidate logits per cut; a cut that is a multiple of the prefill chunk must be 0.
//   test_chunk_boundary [--tokens N] [--cuts a,b,c] [--save file] [--ref file]
// --save writes the cold run's candidates; --ref compares them with a saved run by token id (noise-floor probes: ENGINE_DEBUG_PERTURB_EMBED=eps).
#include "engine/runtime/test_driver.h"
#include "engine/loader/tokenizer.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cstdio>

using namespace engine::runtime;

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	int n_tokens = 6000;
	std::vector<int> cuts = { 2560, 3008, 2572, 100 };
	const char* save = nullptr;
	const char* ref = nullptr;
	for ( int i = 1; i < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--tokens" ) && i + 1 < argc ) n_tokens = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--save" ) && i + 1 < argc ) save = argv[++i];
		else if ( !std::strcmp( argv[i], "--ref" ) && i + 1 < argc ) ref = argv[++i];
		else if ( !std::strcmp( argv[i], "--cuts" ) && i + 1 < argc ) {
			cuts.clear();
			for ( char* p = argv[++i]; *p; ) {
				cuts.push_back( static_cast<int>( std::strtol( p, &p, 10 ) ) );
				if ( *p == ',' ) ++p;
			}
		}
	}
	engine::loader::Tokenizer tok;
	ENGINE_ASSERT( tok.load( test::GLM_DIR ), "tokenizer load failed" );
	std::vector<int> ids = tok.encode( "[gMASK]<sop><|user|>", false );
	for ( int i = 0; static_cast<int>( ids.size() ) < n_tokens; ++i ) {
		const std::string s = "Note " + std::to_string( i ) + ": the weather in district " + std::to_string( i * 7 % 101 ) + " was mild on day " + std::to_string( i % 365 ) + ". ";
		const std::vector<int> t = tok.encode( s, false );
		ids.insert( ids.end(), t.begin(), t.end() );
	}
	ids.resize( n_tokens );
	const std::vector<int> end = tok.encode( "\n\nWhat is the weather in district 7?<|assistant|></think>", false );
	ids.insert( ids.end(), end.begin(), end.end() );
	const int n = static_cast<int>( ids.size() );

	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = "glm5_next";
	cfg.model_dir = test::GLM_DIR;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	mc.warm_graphs( 8 );
	std::printf( "[*] prompt %d tokens, prefill chunk %d\n", n, mc.metadata().max_rows );

	mc.reset_slot( 0 );
	Candidates cold;
	const int cold_first = mc.prefill( ids.data(), n, 0, 0, &cold );
	if ( save ) {
		if ( FILE* f = std::fopen( save, "wb" ) ) {
			std::fwrite( &cold, sizeof( cold ), 1, f );
			std::fclose( f );
		}
	}
	if ( ref ) {
		Candidates r;
		FILE* f = std::fopen( ref, "rb" );
		if ( f && std::fread( &r, sizeof( r ), 1, f ) == 1 ) {
			float d = 0.0f;
			int common = 0;
			for ( int i = 0; i < cold.n; ++i ) {
				for ( int j = 0; j < r.n; ++j ) {
					if ( r.ids[j] == cold.ids[i] ) { d = std::max( d, std::fabs( r.logits[j] - cold.logits[i] ) ); ++common; }
				}
			}
			std::printf( "[*] vs %s: %d common candidates, max |dlogit| %.4f, top-1 %s\n", ref, common, d, r.ids[0] == cold.ids[0] ? "same" : "DIFFERENT" );
		}
		if ( f ) std::fclose( f );
	}
	if ( std::getenv( "COLD_ONLY" ) ) return 0;
	for ( int cut : cuts ) {
		mc.reset_slot( 0 );
		mc.prefill( ids.data(), cut, 0 );
		Candidates c;
		const int first = mc.prefill( ids.data() + cut, n - cut, 0, 0, &c, cut );
		float d = 0.0f;
		for ( int j = 0; j < c.n && j < cold.n; ++j ) d = std::max( d, std::fabs( c.logits[j] - cold.logits[j] ) );
		std::printf( "[*] cut %5d: max |dlogit| %.4f, top-1 %s (cold logit0 %.3f)\n", cut, d, first == cold_first ? "same" : "DIFFERENT", cold.logits[0] );
	}
	return 0;
}
