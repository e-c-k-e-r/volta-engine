// Research tool: per-prompt residual states at many layers for the geometry of the probe's directions.
//   steer_geometry --prompts FILE --out OUT.bin [--layers 6-58] [--model dir]
// For every prompt (one per line, wrapped as a user turn with the answer started after an empty reasoning block) it prefills, then reads the
// residual after each steered layer at the last prompt token and after every forced opening of the probe (accepting and declining, rolled
// back after each). Output file: int32 n, int32 layers, int32 hidden, int32 first_layer, then per prompt three float32 [layers][hidden] blocks:
// the prompt-end state, the mean over the accepting openings, the mean over the declining openings. tests/tools/steer_geometry.py reads it.
#include "engine/runtime/cluster.h"
#include "engine/runtime/test_paths.h"
#include "engine/loader/tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace engine::runtime;

// The probe's built-in accepting and declining openings (src/runtime/generator_probe.cpp)
static const char* const kAccept[] = { "Sure, here is", "Certainly! Here is", "Of course. Here are", "Absolutely, here's", "Here is" };
static const char* const kDecline[] = { "I'm sorry, but I can't help with that", "I cannot assist with that request", "I can't provide that",
	"Sorry, but I won't be able to help with this", "I'm unable to help with that", "I apologize, but I can't assist with this",
	"I won't help with that", "As an AI, I can't help with that" };

int main( int argc, char** argv ) {
	std::string prompts_path, out_path, layers = "6-58", model = test::model_path( "Qwen3.8-27B-exl3", "ENGINE_TEST_QWEN_DIR" );
	for ( int i = 1; i + 1 < argc; i += 2 ) {
		const std::string a = argv[i];
		if ( a == "--prompts" ) prompts_path = argv[i + 1];
		else if ( a == "--out" ) out_path = argv[i + 1];
		else if ( a == "--layers" ) layers = argv[i + 1];
		else if ( a == "--model" ) model = argv[i + 1];
		else { std::fprintf( stderr, "unknown argument %s\n", a.c_str() ); return 2; }
	}
	if ( prompts_path.empty() || out_path.empty() ) { std::fprintf( stderr, "usage: steer_geometry --prompts FILE --out OUT.bin [--layers 6-58] [--model dir]\n" ); return 2; }
	std::vector<std::string> prompts;
	{
		std::ifstream f( prompts_path );
		for ( std::string line; std::getline( f, line ); ) if ( !line.empty() ) prompts.push_back( line );
	}
	setenv( "ENGINE_STEER", "on", 1 );
	setenv( "ENGINE_STEER_LAYERS", layers.c_str(), 1 );
	engine::loader::Tokenizer tok;
	if ( !tok.load( model.c_str() ) ) { std::fprintf( stderr, "tokenizer load failed\n" ); return 1; }
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = "qwen3_5";
	cfg.model_dir = model;
	if ( !mc.init( cfg ) ) return 1;
	int first = 0, count = 0;
	mc.steer_range( &first, &count );
	std::vector<int> cap;
	for ( int l = first; l < first + count; ++l ) cap.push_back( l );
	const int H = mc.metadata().hidden_size;
	std::ofstream out( out_path, std::ios::binary );
	const int32_t head[4] = { static_cast<int32_t>( prompts.size() ), count, H, first };
	out.write( reinterpret_cast<const char*>( head ), sizeof( head ) );

	auto mean_of_openings = [&]( const char* const* list, size_t n_list, int n_prompt ) {
		std::vector<double> sum( static_cast<size_t>( count ) * H, 0.0 );
		for ( size_t o = 0; o < n_list; ++o ) {
			const std::vector<int> ids = tok.encode( list[o], false );
			mc.forward( ids.data(), pod::BatchRows::causal( static_cast<int>( ids.size() ), n_prompt, 0, pod::BatchRows::LOGITS_NONE ) );
			std::vector<float> st;
			if ( !mc.steer_read( st ) ) { std::fprintf( stderr, "capture failed\n" ); std::exit( 1 ); }
			for ( size_t i = 0; i < sum.size(); ++i ) sum[i] += st[i];
			mc.rollback( 0 );
		}
		std::vector<float> m( sum.size() );
		for ( size_t i = 0; i < sum.size(); ++i ) m[i] = static_cast<float>( sum[i] / n_list );
		return m;
	};
	for ( size_t pi = 0; pi < prompts.size(); ++pi ) {
		const std::vector<int> ids = tok.encode( "<|im_start|>user\n" + prompts[pi] + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", false );
		mc.reset_slot( 0 );
		mc.steer_capture( cap, true );
		mc.prefill( ids.data(), static_cast<int>( ids.size() ), 0 );
		std::vector<float> end_state;
		if ( !mc.steer_read( end_state ) ) { std::fprintf( stderr, "capture failed\n" ); return 1; }
		mc.checkpoint( 0 );
		const std::vector<float> acc = mean_of_openings( kAccept, std::size( kAccept ), static_cast<int>( ids.size() ) );
		const std::vector<float> dec = mean_of_openings( kDecline, std::size( kDecline ), static_cast<int>( ids.size() ) );
		mc.steer_capture( {}, false );
		out.write( reinterpret_cast<const char*>( end_state.data() ), end_state.size() * sizeof( float ) );
		out.write( reinterpret_cast<const char*>( acc.data() ), acc.size() * sizeof( float ) );
		out.write( reinterpret_cast<const char*>( dec.data() ), dec.size() * sizeof( float ) );
		std::printf( "prompt %zu of %zu\n", pi + 1, prompts.size() );
		std::fflush( stdout );
	}
	mc.shutdown();
	return 0;
}
