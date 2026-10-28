// Builds a control-vector file (.steer) from two sets of prompts: the mean difference of the model's last-token residual states
// (after each steered layer) between the positive and the negative set, one add entry per layer. The entry's direction is the unit
// difference and its strength the length of the difference divided by the number of layers, so the layers together move the state by
// the contrast the prompts show at a steer_scale of 1. Prompts are one per line (\n inside a prompt written as \\n); each is wrapped as a user turn with the answer started after an empty reasoning block.
//   steer_build --pos pos.txt --neg neg.txt --out v.steer [--layers 21-42] [--model dir] [--scope generated]
// Needs the steer option (set here) and a checkpoint of a Qwen3.5-family model.
#include "engine/runtime/cluster.h"
#include "engine/runtime/steer_file.h"
#include "engine/runtime/test_paths.h"
#include "engine/loader/tokenizer.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace engine::runtime;

static std::vector<std::string> read_lines( const std::string& path ) {
	std::vector<std::string> out;
	std::ifstream f( path );
	std::string line;
	while ( std::getline( f, line ) ) {
		for ( size_t p; ( p = line.find( "\\n" ) ) != std::string::npos; ) line.replace( p, 2, "\n" );
		if ( !line.empty() ) out.push_back( line );
	}
	return out;
}

int main( int argc, char** argv ) {
	std::string pos_path, neg_path, out_path, layers = "", model = test::model_path( "Qwen3.8-27B-exl3", "ENGINE_TEST_QWEN_DIR" ), scope = "all";
	for ( int i = 1; i + 1 < argc; i += 2 ) {
		const std::string a = argv[i];
		if ( a == "--pos" ) pos_path = argv[i + 1];
		else if ( a == "--neg" ) neg_path = argv[i + 1];
		else if ( a == "--out" ) out_path = argv[i + 1];
		else if ( a == "--layers" ) layers = argv[i + 1];
		else if ( a == "--model" ) model = argv[i + 1];
		else if ( a == "--scope" ) scope = argv[i + 1];
		else { std::fprintf( stderr, "unknown argument %s\n", a.c_str() ); return 2; }
	}
	if ( pos_path.empty() || neg_path.empty() || out_path.empty() ) {
		std::fprintf( stderr, "usage: steer_build --pos pos.txt --neg neg.txt --out v.steer [--layers 21-42] [--model dir] [--scope generated]\n" );
		return 2;
	}
	int scope_bits = 0;
	if ( !parse_steer_scope( scope, &scope_bits ) ) { std::fprintf( stderr, "bad --scope\n" ); return 2; }
	setenv( "ENGINE_STEER", "on", 1 );
	if ( !layers.empty() ) setenv( "ENGINE_STEER_LAYERS", layers.c_str(), 1 );
	const std::vector<std::string> pos = read_lines( pos_path ), neg = read_lines( neg_path );
	if ( pos.empty() || neg.empty() ) { std::fprintf( stderr, "empty prompt set\n" ); return 2; }

	engine::loader::Tokenizer tok;
	if ( !tok.load( model.c_str() ) ) { std::fprintf( stderr, "tokenizer load failed (%s)\n", model.c_str() ); return 1; }
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

	auto mean_states = [&]( const std::vector<std::string>& prompts ) {
		std::vector<double> sum( static_cast<size_t>( count ) * H, 0.0 );
		for ( const std::string& p : prompts ) {
			const std::vector<int> ids = tok.encode( "<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", false );
			if ( ids.empty() || ids[0] != tok.im_start_id() ) { std::fprintf( stderr, "template special tokens were not recognised\n" ); std::exit( 1 ); }
			mc.reset_slot( 0 );
			mc.steer_capture( cap, true );
			mc.prefill( ids.data(), static_cast<int>( ids.size() ), 0 );
			std::vector<float> st;
			if ( !mc.steer_read( st ) ) { std::fprintf( stderr, "capture failed\n" ); std::exit( 1 ); }
			mc.steer_capture( {}, false );
			for ( size_t i = 0; i < sum.size(); ++i ) sum[i] += st[i];
		}
		for ( double& v : sum ) v /= static_cast<double>( prompts.size() );
		return sum;
	};
	const std::vector<double> mp = mean_states( pos ), mn = mean_states( neg );

	SteerFile f;
	f.hidden = H;
	f.layers = mc.metadata().num_layers;
	f.scope = scope_bits;
	for ( int li = 0; li < count; ++li ) {
		engine::modules::SteerEntry e;
		e.layer = first + li;
		e.kind = engine::modules::STEER_ADD;
		e.vec.resize( H );
		double n2 = 0.0;
		for ( int i = 0; i < H; ++i ) { const double d = mp[static_cast<size_t>( li ) * H + i] - mn[static_cast<size_t>( li ) * H + i]; e.vec[i] = static_cast<float>( d ); n2 += d * d; }
		const double n = std::sqrt( n2 );
		for ( float& v : e.vec ) v = static_cast<float>( v / ( n + 1e-12 ) );
		e.par = static_cast<float>( n / count );
		f.entries.push_back( std::move( e ) );
		std::printf( "layer %2d: contrast %.3f\n", first + li, n );
	}
	std::string err;
	if ( !save_steer_file( out_path, f, &err ) ) { std::fprintf( stderr, "%s\n", err.c_str() ); return 1; }
	std::printf( "wrote %s (%zu entries, scope %s)\n", out_path.c_str(), f.entries.size(), scope.c_str() );
	mc.shutdown();
	return 0;
}
