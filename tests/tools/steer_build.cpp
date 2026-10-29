// Builds a control-vector file (.steer) from two sets of prompts: the mean difference of the model's last-token residual states
// (after each steered layer) between the positive and the negative set, one add entry per layer. The entry's direction is the unit
// difference and its strength the length of the difference divided by the number of layers, so the layers together move the state by
// the contrast the prompts show at a steer_scale of 1. Prompts are one per line (\n inside a prompt written as \\n); each is wrapped as a user turn with the answer started after an empty reasoning block.
//   steer_build --pos pos.txt --neg neg.txt --out v.steer [--layers 21-42] [--model dir] [--scope generated] [--scale 1.5]
//   steer_build ... --answers   lines are `question ||| answer` (the model's own answers); the state is the mean over the answer's first 48 tokens, fed one at a time
//   steer_build --batch dir [...]     every NAME.pos.txt with a NAME.neg.txt in dir gives dir/NAME.steer (one model load for all)
// Needs the steer option (set here) and a checkpoint of a model with the steering hooks (qwen3_5, qwen4_exp; read from its config.json).
#include "engine/runtime/cluster.h"
#include "engine/runtime/steer_file.h"
#include "engine/runtime/test_paths.h"
#include "engine/loader/tokenizer.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
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
	float file_scale = 1.0f;   // the recommended steer_scale stored in the file
	bool answers = false;
	std::string pos_path, neg_path, out_path, batch, layers = "", model = test::model_path( "Qwen3.8-27B-exl3", "ENGINE_TEST_QWEN_DIR" ), scope = "all";
	for ( int i = 1; i < argc; i += ( std::string( argv[i] ) == "--answers" ? 1 : 2 ) ) {
		const std::string a = argv[i];
		if ( a == "--answers" ) { answers = true; continue; }
		if ( i + 1 >= argc ) { std::fprintf( stderr, "missing value for %s\n", a.c_str() ); return 2; }
		if ( a == "--pos" ) pos_path = argv[i + 1];
		else if ( a == "--neg" ) neg_path = argv[i + 1];
		else if ( a == "--out" ) out_path = argv[i + 1];
		else if ( a == "--batch" ) batch = argv[i + 1];
		else if ( a == "--layers" ) layers = argv[i + 1];
		else if ( a == "--model" ) model = argv[i + 1];
		else if ( a == "--scope" ) scope = argv[i + 1];
		else if ( a == "--scale" ) file_scale = static_cast<float>( std::atof( argv[i + 1] ) );
		else { std::fprintf( stderr, "unknown argument %s\n", a.c_str() ); return 2; }
	}
	struct Job { std::string pos, neg, out; };
	std::vector<Job> jobs;
	if ( !batch.empty() ) {
		for ( const auto& e : std::filesystem::directory_iterator( batch ) ) {
			const std::string f = e.path().filename().string(), suffix = ".pos.txt";
			if ( f.size() <= suffix.size() || f.compare( f.size() - suffix.size(), suffix.size(), suffix ) != 0 ) continue;
			const std::string name = f.substr( 0, f.size() - suffix.size() );
			if ( std::filesystem::exists( batch + "/" + name + ".neg.txt" ) ) jobs.push_back( { batch + "/" + f, batch + "/" + name + ".neg.txt", batch + "/" + name + ".steer" } );
		}
	}
	else if ( !pos_path.empty() && !neg_path.empty() && !out_path.empty() ) jobs.push_back( { pos_path, neg_path, out_path } );
	if ( jobs.empty() ) {
		std::fprintf( stderr, "usage: steer_build --pos pos.txt --neg neg.txt --out v.steer | --batch dir  [--layers 21-42] [--model dir] [--scope generated]\n" );
		return 2;
	}
	int scope_bits = 0;
	if ( !parse_steer_scope( scope, &scope_bits ) ) { std::fprintf( stderr, "bad --scope\n" ); return 2; }
	setenv( "ENGINE_STEER", "on", 1 );
	if ( !layers.empty() ) setenv( "ENGINE_STEER_LAYERS", layers.c_str(), 1 );

	engine::loader::Tokenizer tok;
	if ( !tok.load( model.c_str() ) ) { std::fprintf( stderr, "tokenizer load failed (%s)\n", model.c_str() ); return 1; }
	ModelCluster mc;
	ClusterConfig cfg;
	{
		// model_type of the checkpoint (qwen3_5 or qwen4_exp): the same file the server reads
		std::ifstream cf( model + "/config.json" );
		const nlohmann::json j = nlohmann::json::parse( cf, nullptr, false );
		cfg.model_type = !j.is_discarded() && j.contains( "model_type" ) && j["model_type"].is_string() ? j["model_type"].get<std::string>() : "qwen3_5";
		if ( cfg.model_type != "qwen3_5" && cfg.model_type != "qwen4_exp" ) { std::fprintf( stderr, "model_type %s has no steering hooks\n", cfg.model_type.c_str() ); return 1; }
	}
	cfg.model_dir = model;
	if ( !mc.init( cfg ) ) return 1;
	int first = 0, count = 0;
	mc.steer_range( &first, &count );
	std::vector<int> cap;
	for ( int l = first; l < first + count; ++l ) cap.push_back( l );
	const int H = mc.metadata().hidden_size;

	auto mean_states = [&]( const std::vector<std::string>& prompts ) {
		std::vector<double> sum( static_cast<size_t>( count ) * H, 0.0 );
		int rows = 0;
		for ( const std::string& line : prompts ) {
			std::string p = line, answer;
			if ( answers ) {
				const size_t cut = line.find( " ||| " );
				if ( cut == std::string::npos ) { std::fprintf( stderr, "--answers: a line without ' ||| '\n" ); std::exit( 1 ); }
				p = line.substr( 0, cut );
				answer = line.substr( cut + 5 );
			}
			const std::vector<int> ids = tok.encode( "<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", false );
			if ( ids.empty() || ids[0] != tok.im_start_id() ) { std::fprintf( stderr, "template special tokens were not recognised\n" ); std::exit( 1 ); }
			mc.reset_slot( 0 );
			mc.steer_capture( cap, true );
			mc.prefill( ids.data(), static_cast<int>( ids.size() ), 0 );
			std::vector<float> st;
			if ( !answers ) {
				if ( !mc.steer_read( st ) ) { std::fprintf( stderr, "capture failed\n" ); std::exit( 1 ); }
				for ( size_t i = 0; i < sum.size(); ++i ) sum[i] += st[i];
				++rows;
			}
			else {
				// the answer's tokens one at a time; each state is the residual after the token was read, the mean over the first 48
				const std::vector<int> a = tok.encode( answer, false );
				std::vector<double> acc( sum.size(), 0.0 );
				const int na = std::min( static_cast<int>( a.size() ), 48 );
				for ( int t = 0; t < na; ++t ) {
					mc.forward( &a[t], 1, static_cast<int>( ids.size() ) + t, 0 );
					if ( !mc.steer_read( st ) ) { std::fprintf( stderr, "capture failed\n" ); std::exit( 1 ); }
					for ( size_t i = 0; i < acc.size(); ++i ) acc[i] += st[i];
				}
				for ( size_t i = 0; i < sum.size(); ++i ) sum[i] += acc[i] / std::max( na, 1 );
				++rows;
			}
			mc.steer_capture( {}, false );
		}
		for ( double& v : sum ) v /= static_cast<double>( std::max( rows, 1 ) );
		return sum;
	};
	for ( const Job& job : jobs ) {
		const std::vector<std::string> pos = read_lines( job.pos ), neg = read_lines( job.neg );
		if ( pos.empty() || neg.empty() ) { std::fprintf( stderr, "empty prompt set in %s\n", job.out.c_str() ); return 2; }
		const std::vector<double> mp = mean_states( pos ), mn = mean_states( neg );
		SteerFile f;
		f.hidden = H;
		f.layers = mc.metadata().num_layers;
		f.scope = scope_bits;
		f.scale = file_scale;
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
			std::printf( "%s layer %2d: contrast %.3f\n", job.out.c_str(), first + li, n );
		}
		std::string err;
		if ( !save_steer_file( job.out, f, &err ) ) { std::fprintf( stderr, "%s\n", err.c_str() ); return 1; }
		std::printf( "wrote %s (%zu entries, scope %s)\n", job.out.c_str(), f.entries.size(), scope.c_str() );
	}
	mc.shutdown();
	return 0;
}
