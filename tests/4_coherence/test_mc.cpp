// Multiple-choice scoring: per task, one prefill of the prompt (tokenized without special tokens) and the log-softmax of the last row's
// logits at each answer's single token; the prediction is the most likely answer. The output rows follow llama.cpp's mc-score (task,
// predicted answer, correct answer, then every answer's log-probability), so a run compares task by task with one of llama.cpp.
//   test_mc [--flash | --glm] TASKS OUT [--start S] [--n N]
// TASKS: JSON list of { "prompt": text, "answers": [text, ...], "correct": index }; OUT is appended to from --start on (a resumed run)
#include "engine/runtime/test_driver.h"
#include "engine/loader/tokenizer.h"
#include "engine/backend/numeric.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace engine::runtime;
namespace backend = engine::backend;

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	bool glm = false, flash = false;
	const char* tasks_path = nullptr;
	const char* out_path = nullptr;
	int start = 0, count = -1;
	for ( int i = 1; i < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--glm" ) ) glm = true;
		else if ( !std::strcmp( argv[i], "--flash" ) ) flash = true;
		else if ( !std::strcmp( argv[i], "--start" ) && i + 1 < argc ) start = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--n" ) && i + 1 < argc ) count = std::atoi( argv[++i] );
		else if ( !tasks_path ) tasks_path = argv[i];
		else if ( !out_path ) out_path = argv[i];
	}
	if ( !tasks_path || !out_path ) {
		std::printf( "usage: test_mc [--flash | --glm] TASKS OUT [--start S] [--n N]\n" );
		return 2;
	}
	std::ifstream tf( tasks_path );
	ENGINE_ASSERT( tf.good(), "test_mc: cannot read the tasks" );
	const nlohmann::json tasks = nlohmann::json::parse( tf );
	const int end = count < 0 ? static_cast<int>( tasks.size() ) : std::min( static_cast<int>( tasks.size() ), start + count );

	const char* model_dir = glm ? test::GLM_DIR : ( flash ? test::FLASH_DIR : test::QWEN_DIR );
	engine::loader::Tokenizer tok;
	ENGINE_ASSERT( tok.load( model_dir ), "tokenizer load failed" );
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = glm ? "glm5_next" : ( flash ? "qwen4_exp" : "qwen3_5" );
	cfg.model_dir = model_dir;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	const int V = mc.metadata().vocab_size;
	std::vector<backend::half_t> logits( V );

	FILE* out = std::fopen( out_path, start > 0 ? "a" : "w" );
	ENGINE_ASSERT( out != nullptr, "test_mc: cannot write the output" );
	int right = 0, done = 0;
	const auto t0 = std::chrono::steady_clock::now();
	for ( int i = start; i < end; ++i ) {
		const auto& t = tasks[static_cast<size_t>( i )];
		const std::vector<int> ids = tok.encode( t["prompt"].get<std::string>(), false );
		std::vector<int> ans;
		for ( const auto& a : t["answers"] ) {
			const std::vector<int> at = tok.encode( a.get<std::string>(), false );
			ENGINE_ASSERT( at.size() == 1, "test_mc: an answer is not one token" );
			ans.push_back( at[0] );
		}
		ENGINE_ASSERT( ids.size() >= 2 && static_cast<int>( ids.size() ) < mc.metadata().max_seq_len, "test_mc: a prompt does not fit" );
		mc.reset_slot( 0 );
		const int n = static_cast<int>( ids.size() );
		mc.prefill( ids.data(), n - 1, 0 );
		mc.forward( ids.data() + n - 1, pod::BatchRows::causal( 1, n - 1, 0, pod::BatchRows::LOGITS_ALL ) );
		mc.read_logits( 1, logits.data() );
		float mx = -1e30f;
		for ( int v = 0; v < V; ++v ) mx = std::fmax( mx, backend::to_float( logits[v] ) );
		double z = 0.0;
		for ( int v = 0; v < V; ++v ) z += std::exp( static_cast<double>( backend::to_float( logits[v] ) ) - mx );
		const double logz = mx + std::log( z );
		size_t best = 0;
		std::vector<double> lp( ans.size() );
		for ( size_t s = 0; s < ans.size(); ++s ) {
			lp[s] = backend::to_float( logits[ans[s]] ) - logz;
			if ( lp[s] > lp[best] ) best = s;
		}
		const int correct = t["correct"].get<int>();
		std::fprintf( out, "%d\t%zu\t%d", i, best, correct );
		for ( double v : lp ) std::fprintf( out, "\t%.6f", v );
		std::fprintf( out, "\n" );
		std::fflush( out );
		right += static_cast<int>( best ) == correct ? 1 : 0;
		++done;
		if ( done % 25 == 0 ) {
			const double s = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
			std::printf( "%d / %d: accuracy %.2f%%, %.1f s per task\n", done, end - start, 100.0 * right / done, s / done );
		}
	}
	std::fclose( out );
	std::printf( "test_mc: %d tasks, %d right (%.2f%%)\n", done, right, done ? 100.0 * right / done : 0.0 );
	return 0;
}
