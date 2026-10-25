// Teacher-forced perplexity of a text file on a served model: the same tokens run through chunked prefill with every row's logits read back,
// so two builds (or two kernel paths) can be compared on a quality metric instead of greedy tokens. Optionally writes the per-token NLL.
//   test_ppl [--glm | --flash] [--file PATH] [--tokens N] [--chunk C] [--nll OUT.bin]
//   test_ppl --calibrate --file TEXT --seq 2048 --tokens N   cuts the text into independent rows of 2048 tokens (the slot restarts at each), as calibration rows are
//   test_ppl --calibrate --prompts DIR [--tokens N] [--chunk C]   replays the rendered prompts a server wrote with dump_requests (req_N.prompt.txt, chat-template tokens
//   intact): only the last request of each session is used, and the rows it shares with an earlier session (the system prompt and tool schemas) run without
//   adding to the Hessians, so every distinct token counts once; --tokens stops after N counted rows
//   test_ppl --flash --qbench qbench_prompts.md [--chunk C]   scores only the response tokens of each traced row (input as context), the
//   quantity the checkpoint's model card reports as PPL
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
	bool glm = false, flash = false, calibrate = false;
	int n_tokens = 6000, chunk = 512;
	std::string file, nll_out, qbench, prompts_dir;
	int seq = 0;
	for ( int i = 1; i < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--glm" ) ) glm = true;
		else if ( !std::strcmp( argv[i], "--flash" ) ) flash = true;
		else if ( !std::strcmp( argv[i], "--file" ) && i + 1 < argc ) file = argv[++i];
		else if ( !std::strcmp( argv[i], "--tokens" ) && i + 1 < argc ) n_tokens = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--chunk" ) && i + 1 < argc ) chunk = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--qbench" ) && i + 1 < argc ) qbench = argv[++i];
		else if ( !std::strcmp( argv[i], "--calibrate" ) ) calibrate = true;
		else if ( !std::strcmp( argv[i], "--nll" ) && i + 1 < argc ) nll_out = argv[++i];
		else if ( !std::strcmp( argv[i], "--prompts" ) && i + 1 < argc ) prompts_dir = argv[++i];
		else if ( !std::strcmp( argv[i], "--seq" ) && i + 1 < argc ) seq = std::atoi( argv[++i] );
	}
	if ( !calibrate && chunk > pod::MAX_BATCH ) {
		std::printf( "[test_ppl] chunk %d clamped to %d: the gathered logits hold a decode batch (--calibrate reads none and takes any chunk)\n", chunk, pod::MAX_BATCH );
		chunk = pod::MAX_BATCH;
	}
	const char* model_dir = glm ? test::GLM_DIR : ( flash ? test::FLASH_DIR : test::QWEN_DIR );
	engine::loader::Tokenizer tok;
	ENGINE_ASSERT( tok.load( model_dir ), "tokenizer load failed" );

	if ( !prompts_dir.empty() ) {
		ENGINE_ASSERT( calibrate, "--prompts replays sessions for calibration: add --calibrate" );
		std::vector<std::pair<std::string, std::string>> all;   // (name, text)
		for ( const auto& e : std::filesystem::directory_iterator( prompts_dir ) ) {
			const std::string name = e.path().filename().string();
			if ( name.size() < 11 || name.compare( name.size() - 11, 11, ".prompt.txt" ) != 0 ) continue;
			std::ifstream f( e.path() );
			std::stringstream ss;
			ss << f.rdbuf();
			all.emplace_back( name, ss.str() );
		}
		std::sort( all.begin(), all.end(), []( const auto& a, const auto& b ) { return a.second.size() > b.second.size(); } );
		std::vector<const std::string*> leaves;   // session tips: prompts that are not the start of a longer one
		for ( const auto& a : all ) {
			bool covered = false;
			for ( const std::string* l : leaves ) covered = covered || ( l->size() >= a.second.size() && l->compare( 0, a.second.size(), a.second ) == 0 );
			if ( !covered ) leaves.push_back( &a.second );
		}
		std::sort( leaves.begin(), leaves.end(), []( const std::string* a, const std::string* b ) { return *a < *b; } );
		std::printf( "[*] %zu prompt dumps, %zu sessions\n", all.size(), leaves.size() );
		ModelCluster mc;
		ClusterConfig cfg;
		cfg.model_type = glm ? "glm5_next" : ( flash ? "qwen4_exp" : "qwen3_5" );
		cfg.model_dir = model_dir;
		ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
		std::vector<std::vector<int>> done;
		long counted = 0, replayed = 0;
		for ( const std::string* text : leaves ) {
			if ( counted >= n_tokens ) break;
			std::vector<int> ids;
			for ( size_t at = 0; at < text->size(); ) {
				size_t end = std::min( text->size(), at + 4096 );
				if ( end < text->size() ) {
					const size_t nl = text->find( '\n', end );
					end = nl == std::string::npos ? text->size() : nl + 1;
				}
				const std::vector<int> part = tok.encode( text->substr( at, end - at ), false );
				ids.insert( ids.end(), part.begin(), part.end() );
				at = end;
			}
			size_t shared = 0;   // longest common prefix with a session already replayed
			for ( const auto& d : done ) {
				size_t k = 0;
				while ( k < d.size() && k < ids.size() && d[k] == ids[k] ) ++k;
				shared = std::max( shared, k );
			}
			mc.reset_slot( 0 );
			for ( size_t off = 0; off < ids.size(); ) {
				const bool counting = off >= shared;
				size_t end = std::min( ids.size(), off + static_cast<size_t>( chunk ) );
				if ( !counting && end > shared ) end = shared;
				const int count = static_cast<int>( end - off );
				mc.set_hessian_capture( counting );
				mc.forward( ids.data() + off, pod::BatchRows::causal( count, static_cast<int>( off ), 0, calibrate && !flash ? pod::BatchRows::LOGITS_NONE : pod::BatchRows::LOGITS_ALL ) );
				if ( counting ) counted += count;
				replayed += count;
				off = end;
			}
			done.push_back( std::move( ids ) );
			std::printf( "[*] session %zu: %zu tokens, %zu shared; counted %ld, replayed %ld\n", done.size(), done.back().size(), shared, counted, replayed );
		}
		mc.shutdown();
		return 0;
	}

	if ( !qbench.empty() ) {
		// rows: "### Input (N tokens)" fenced block, then "### Response (N tokens)" up to the next "## Row"
		std::ifstream f( qbench );
		std::stringstream ss;
		ss << f.rdbuf();
		std::string md = ss.str();
		md.erase( std::remove( md.begin(), md.end(), '\r' ), md.end() );  // the trace file has CRLF line endings
		struct Row { std::string in, out; int out_tokens; };
		std::vector<Row> rows;
		// rows are separated by "## Row" headings; inside one row the first "### Input" / "### Response (" headings are the real ones
		// (the model's own text may contain such headings)
		std::vector<size_t> starts;
		for ( size_t at = md.find( "## Row " ); at != std::string::npos; at = md.find( "\n## Row ", at + 1 ) ) starts.push_back( at == 0 ? 0 : at + 1 );
		starts.push_back( md.size() );
		for ( size_t k = 0; k + 1 < starts.size(); ++k ) {
			const std::string row = md.substr( starts[k], starts[k + 1] - starts[k] );
			Row r;
			const size_t in_head = row.find( "### Input" );
			const size_t out_head = row.find( "### Response (" );
			ENGINE_ASSERT( in_head != std::string::npos && out_head != std::string::npos, "row headings not found" );
			size_t a = row.find( "```", in_head );
			size_t ifl = 0;
			while ( row[a + ifl] == '`' ) ++ifl;
			a = row.find( '\n', a ) + 1;
			const size_t b = row.find( "\n" + std::string( ifl, '`' ) + "\n", a );
			r.in = row.substr( a, b - a );
			r.out_tokens = std::atoi( row.c_str() + out_head + 14 );
			// the fence is longer than any backtick run inside the text (four backticks when the response has code blocks)
			a = row.find( "```", out_head );
			size_t fl = 0;
			while ( row[a + fl] == '`' ) ++fl;
			a = row.find( '\n', a ) + 1;
			std::string body = row.substr( a );
			while ( !body.empty() && body.back() == '\n' ) body.pop_back();
			ENGINE_ASSERT( body.size() >= fl && body.find_first_not_of( '`', body.size() - fl ) == std::string::npos, "response fence not found" );
			body.resize( body.size() - fl );
			if ( !body.empty() && body.back() == '\n' ) body.pop_back();
			r.out = body;
			rows.push_back( r );
		}
		std::printf( "[*] qbench: %zu rows parsed\n", rows.size() );
		ModelCluster mc;
		ClusterConfig cfg;
		cfg.model_type = glm ? "glm5_next" : ( flash ? "qwen4_exp" : "qwen3_5" );
		cfg.model_dir = model_dir;
		ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
		const int V = mc.metadata().vocab_size;
		std::vector<backend::half_t> logits( static_cast<size_t>( chunk ) * V );
		double total = 0.0;
		size_t count_all = 0;
		for ( size_t ri = 0; ri < rows.size(); ++ri ) {
			std::vector<int> ids = tok.encode( rows[ri].in, true );
			const int n_in = static_cast<int>( ids.size() );
			std::vector<int> o = tok.encode( rows[ri].out, false );
			ids.insert( ids.end(), o.begin(), o.end() );
			const int n = static_cast<int>( ids.size() );
			mc.reset_slot( 0 );
			double row_total = 0.0;
			int row_count = 0;
			for ( int off = 0; off + 1 < n; off += chunk ) {
				const int count = std::min( chunk, n - off );
				mc.forward( ids.data() + off, pod::BatchRows::causal( count, off, 0, calibrate && !flash ? pod::BatchRows::LOGITS_NONE : pod::BatchRows::LOGITS_ALL ) );   // the logits buffers hold a decode batch; a calibration pass reads none
				if ( calibrate ) continue;   // forward only (ENGINE_HESSIAN_DIR collects its statistics in the layers); the host scoring below is the slow part
		mc.read_logits( count, logits.data() );
				for ( int r = 0; r < count && off + r + 1 < n; ++r ) {
					if ( off + r + 1 < n_in ) continue;
					const backend::half_t* l = logits.data() + static_cast<size_t>( r ) * V;
					float mx = -1e30f;
					for ( int v = 0; v < V; ++v ) mx = std::fmax( mx, backend::to_float( l[v] ) );
					double sum = 0.0;
					for ( int v = 0; v < V; ++v ) sum += std::exp( static_cast<double>( backend::to_float( l[v] ) - mx ) );
					row_total += std::log( sum ) + mx - backend::to_float( l[ids[off + r + 1]] );
					++row_count;
				}
			}
			std::printf( "[*] row %zu: input %d tokens, response %d tokens (trace says %d), ppl %.4f\n", ri, n_in, static_cast<int>( o.size() ), rows[ri].out_tokens, std::exp( row_total / row_count ) );
			total += row_total;
			count_all += row_count;
		}
		std::printf( "[*] qbench: %zu response tokens, mean NLL %.5f, perplexity %.4f\n", count_all, total / count_all, std::exp( total / count_all ) );
		mc.shutdown();
		return 0;
	}

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
	// Encode in pieces of ~4 KB cut at line ends (the tokenizer is quadratic in the length of a run of text without a split point)
	std::vector<int> ids;
	for ( size_t at = 0; at < text.size(); ) {
		size_t end = std::min( text.size(), at + 4096 );
		if ( end < text.size() ) {
			const size_t nl = text.find( '\n', end );
			end = nl == std::string::npos ? text.size() : nl + 1;
		}
		const std::vector<int> part = tok.encode( text.substr( at, end - at ), false );
		ids.insert( ids.end(), part.begin(), part.end() );
		at = end;
	}
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
	for ( int off = 0; off + 1 < n_tokens; ) {
		const int pos = seq > 0 ? off % seq : off;   // --seq: independent rows, the slot restarts at each
		if ( seq > 0 && pos == 0 ) mc.reset_slot( 0 );
		const int count = std::min( { chunk, n_tokens - off, seq > 0 ? seq - pos : chunk } );
		mc.forward( ids.data() + off, pod::BatchRows::causal( count, pos, 0, calibrate && !flash ? pod::BatchRows::LOGITS_NONE : pod::BatchRows::LOGITS_ALL ) );
		if ( calibrate ) { off += count; continue; }   // forward only: ENGINE_HESSIAN_DIR collects its statistics in the layers
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
		off += count;
	}
	if ( calibrate ) { mc.shutdown(); return 0; }
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
