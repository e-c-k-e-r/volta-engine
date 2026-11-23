// Teacher-forced perplexity of a text file on a served model: the same tokens run through chunked prefill with every row's logits read back,
// so two builds (or two kernel paths) can be compared on a quality metric instead of greedy tokens. Optionally writes the per-token NLL.
//   test_ppl [--glm | --flash] [--file PATH] [--tokens N] [--chunk C] [--nll OUT.bin] [--topk-out OUT.bin [--topk K]]
//     --topk-out: the model's distribution at every position as a teacher (volta_e2e): int32 magic 0x4b504f54, tokens N, seq, K, the N token ids,
//     then per position K ids (int32, most likely first), their probabilities (FP32) and the mass outside them (FP32)
//   --kl-teacher FILE (a --topk-out file of the same text and settings): also KL( teacher || model ) per position, over the teacher's top K plus
//     its lumped tail, averaged (--kl-from P: only the positions from P on, e.g. the sequences volta_e2e holds out)
//   --bins B1,B2,..: the NLL (and the KL) also averaged by position within a row (--seq) or session (--prompts), [0, B1), .., [Bn, end)
//   test_ppl --calibrate --file TEXT --seq 2048 --tokens N   cuts the text into independent rows of 2048 tokens (the slot restarts at each), as calibration rows are
//   test_ppl --calibrate --prompts DIR [--tokens N] [--chunk C]   replays the rendered prompts a server wrote with dump_requests (req_N.prompt.txt, chat-template tokens
//   intact): only the last request of each session is used, and the rows it shares with an earlier session (the system prompt and tool schemas) run without
//   adding to the Hessians, so every distinct token counts once; --tokens stops after N counted rows; --stride K counts every K-th chunk past that prefix
//   (the others run without adding), spreading a token budget over long sessions instead of spending it on the first ones
//   test_ppl --prompts DIR [--select hold|train] [--sessions N]   without --calibrate: perplexity of the assistant turns of the sessions (prefix rows run without logits; chat models are
//   not trained on system and user text, whose loss is meaningless, so only the tokens between <|im_start|>assistant and <|im_end|> after the shared prefix are scored); --select hold takes every 10th session in name order, train the rest
//   test_ppl --flash --qbench qbench_prompts.md [--chunk C]   scores only the response tokens of each traced row (input as context), the
//   quantity the checkpoint's model card reports as PPL
// The default text is the docs/ folder concatenated; ids never include special tokens.
#include "engine/runtime/test_driver.h"
#include "engine/loader/tokenizer.h"
#include "engine/backend/numeric.h"
#include "engine/modules/logit_rows.h"
#include "engine/runtime/options.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <filesystem>
#include <sched.h>

using namespace engine::runtime;
namespace backend = engine::backend;

// --bins: a value's mean by position (within a row or session), positions [0, B1), [B1, B2), .., [Bn, end)
struct PosBins {
	std::vector<int> edges;
	std::vector<double> sum;
	std::vector<size_t> n;
	void init( const std::vector<int>& e ) {
		edges = e;
		sum.assign( e.size() + 1, 0.0 );
		n.assign( e.size() + 1, 0 );
	}
	void add( int pos, double v ) {
		if ( edges.empty() ) return;
		const size_t b = std::upper_bound( edges.begin(), edges.end(), pos ) - edges.begin();
		sum[b] += v;
		++n[b];
	}
	void print( const char* what ) const {
		for ( size_t b = 0; b < n.size() && !edges.empty(); ++b ) {
			if ( n[b] == 0 ) continue;
			const int lo = b == 0 ? 0 : edges[b - 1];
			if ( b < edges.size() ) std::printf( "[*]   %s at positions %d-%d: %.6f (%zu tokens)\n", what, lo, edges[b] - 1, sum[b] / n[b], n[b] );
			else std::printf( "[*]   %s at positions %d-: %.6f (%zu tokens)\n", what, lo, sum[b] / n[b], n[b] );
		}
	}
};

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	bool glm = false, flash = false, calibrate = false;
	int n_tokens = 6000, chunk = 512;
	std::string file, nll_out, qbench, prompts_dir, topk_out, kl_teacher;
	int seq = 0, max_sessions = 0, skip_override = 0, show_miss = 0, stride = 1, topk = 128, kl_from = 0;
	std::string select;
	std::vector<int> bin_edges;
	for ( int i = 1; i < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--glm" ) ) glm = true;
		else if ( !std::strcmp( argv[i], "--flash" ) ) flash = true;
		else if ( !std::strcmp( argv[i], "--file" ) && i + 1 < argc ) file = argv[++i];
		else if ( !std::strcmp( argv[i], "--tokens" ) && i + 1 < argc ) n_tokens = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--chunk" ) && i + 1 < argc ) chunk = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--qbench" ) && i + 1 < argc ) qbench = argv[++i];
		else if ( !std::strcmp( argv[i], "--calibrate" ) ) calibrate = true;
		else if ( !std::strcmp( argv[i], "--nll" ) && i + 1 < argc ) nll_out = argv[++i];
		else if ( !std::strcmp( argv[i], "--topk-out" ) && i + 1 < argc ) topk_out = argv[++i];
		else if ( !std::strcmp( argv[i], "--topk" ) && i + 1 < argc ) topk = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--kl-teacher" ) && i + 1 < argc ) kl_teacher = argv[++i];
		else if ( !std::strcmp( argv[i], "--kl-from" ) && i + 1 < argc ) kl_from = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--bins" ) && i + 1 < argc ) {
			std::stringstream ss( argv[++i] );
			for ( std::string b; std::getline( ss, b, ',' ); ) if ( !b.empty() ) bin_edges.push_back( std::atoi( b.c_str() ) );
		}
		else if ( !std::strcmp( argv[i], "--prompts" ) && i + 1 < argc ) prompts_dir = argv[++i];
		else if ( !std::strcmp( argv[i], "--seq" ) && i + 1 < argc ) seq = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--select" ) && i + 1 < argc ) select = argv[++i];
		else if ( !std::strcmp( argv[i], "--show-miss" ) && i + 1 < argc ) show_miss = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--skip" ) && i + 1 < argc ) skip_override = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--sessions" ) && i + 1 < argc ) max_sessions = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--stride" ) && i + 1 < argc ) stride = std::max( 1, std::atoi( argv[++i] ) );
	}
	// --calibrate with ENGINE_CALIB_MTP set (Flash-Next): the MTP head runs after every forward on the same rows (hidden stack of position p with the token
	// of p + 1), so ENGINE_HESSIAN_DIR also collects the head's sites (layers 48 and 49 of the file names); the head keeps its own KV like in serving
	const bool mtp_calib = calibrate && flash && std::getenv( "ENGINE_CALIB_MTP" ) != nullptr;
	if ( mtp_calib ) setenv( "ENGINE_SPECULATOR", "mtp", 1 );
	if ( !calibrate && prompts_dir.empty() && chunk > pod::MAX_BATCH && !flash && !( glm && engine::runtime::options().glm_logit_rows >= chunk ) ) {
		// (Flash-Next gathers the logits of any chunk, GLM those of glm_logit_rows rows)
		std::printf( "[test_ppl] chunk %d clamped to %d: the gathered logits hold a decode batch (--calibrate reads none and takes any chunk; GLM: set glm_logit_rows)\n", chunk, pod::MAX_BATCH );
		chunk = pod::MAX_BATCH;
	}
	const char* model_dir = glm ? test::GLM_DIR : ( flash ? test::FLASH_DIR : test::QWEN_DIR );
	engine::loader::Tokenizer tok;
	ENGINE_ASSERT( tok.load( model_dir ), "tokenizer load failed" );

	if ( !prompts_dir.empty() ) {
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
		if ( !select.empty() ) {
			std::vector<const std::string*> kept;
			for ( size_t i = 0; i < leaves.size(); ++i ) if ( ( i % 10 == 0 ) == ( select == "hold" ) ) kept.push_back( leaves[i] );
			leaves = kept;
		}
		if ( max_sessions > 0 && static_cast<int>( leaves.size() ) > max_sessions ) leaves.resize( max_sessions );
		std::printf( "[*] %zu prompt dumps, %zu sessions\n", all.size(), leaves.size() );
		ModelCluster mc;
		ClusterConfig cfg;
		cfg.model_type = glm ? "glm5_next" : ( flash ? "qwen4_exp" : "qwen3_5" );
		cfg.model_dir = model_dir;
		cfg.speculate = mtp_calib;
		ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
		std::vector<std::vector<int>> done;
		long counted = 0, replayed = 0;
		struct PrefixSnap { ModelCluster::SeqSnapshot snap; std::vector<int> ids; };   // the start that a family of sessions shares (their system prompt), restored instead of replayed
		std::vector<PrefixSnap> snaps;
		bool use_snap = !mtp_calib;
		if ( !calibrate ) {
			// perplexity of the rows after the prefix the sessions share
			std::vector<std::vector<int>> all_ids;
			for ( const std::string* text : leaves ) {
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
				all_ids.push_back( std::move( ids ) );
			}
			size_t skip = all_ids.empty() ? 0 : all_ids[0].size();
			for ( const auto& v : all_ids ) {
				size_t k = 0;
				while ( k < v.size() && k < all_ids[0].size() && v[k] == all_ids[0][k] ) ++k;
				skip = std::min( skip, k );
			}
			if ( skip_override > 0 ) skip = static_cast<size_t>( skip_override );
			const int V = mc.metadata().vocab_size;
			const int score_rows = flash ? chunk : pod::MAX_BATCH;   // (Flash-Next gathers the logits of any chunk)
			std::vector<backend::half_t> logits( static_cast<size_t>( score_rows ) * V );
			double total = 0.0;
			long n = 0;
			PosBins sess_bins;
			sess_bins.init( bin_edges );
			// GLM's turns are <|assistant|> ... up to the next role tag (<|user|>, <|observation|> or <|system|>, which the model emits to end its turn); Qwen's are <|im_start|>assistant\n ... <|im_end|>
			const std::vector<int> open_assistant = tok.encode( glm ? "<|assistant|>" : "<|im_start|>assistant\n", false );
			std::vector<int> turn_end;
			if ( glm ) { for ( const char* t : { "<|user|>", "<|observation|>", "<|system|>" } ) { const std::vector<int> e = tok.encode( t, false ); ENGINE_ASSERT( e.size() == 1, "test_ppl: a GLM role tag is not one token" ); turn_end.push_back( e[0] ); } }
			else turn_end.push_back( tok.im_end_id() );
			for ( size_t si = 0; si < all_ids.size(); ++si ) {
				const std::vector<int>& ids = all_ids[si];
				std::vector<char> in_assistant( ids.size(), 0 );   // target tokens that the model itself produced: the assistant turns, their <|im_end|> included
				bool inside = false;
				for ( size_t j = 0; j < ids.size(); ++j ) {
					if ( inside ) {
						in_assistant[j] = 1;
						if ( std::find( turn_end.begin(), turn_end.end(), ids[j] ) != turn_end.end() ) inside = false;
					}
					else if ( j + open_assistant.size() <= ids.size() && std::equal( open_assistant.begin(), open_assistant.end(), ids.begin() + j ) ) {
						inside = true;
						j += open_assistant.size() - 1;
					}
				}
				mc.reset_slot( 0 );
				double st = 0.0;
				long sn = 0;
				for ( size_t off = 0; off + 1 < ids.size(); ) {
					const bool scoring = off + 1 >= skip;
					const size_t step = scoring ? static_cast<size_t>( score_rows ) : static_cast<size_t>( chunk );
					size_t end = std::min( ids.size(), off + step );
					if ( !scoring && end + 1 > skip ) end = skip - 1;
					const int count = static_cast<int>( end - off );
					mc.forward( ids.data() + off, pod::BatchRows::causal( count, static_cast<int>( off ), 0, scoring ? pod::BatchRows::LOGITS_ALL : pod::BatchRows::LOGITS_NONE ) );
					if ( scoring ) {
						std::vector<double> row_nl( count, 0.0 );
#if defined( ENGINE_BACKEND_CUDA ) && ENGINE_BACKEND_CUDA == 1
						// the rows' NLL on the device where the logits are (engine::modules::LogitRows)
						{
							static int* d_tgt = nullptr;
							static float* d_nll = nullptr;
							static int cap = 0;
							if ( count > cap ) {
								if ( d_tgt ) { backend::device_free( d_tgt ); backend::device_free( d_nll ); }
								d_tgt = static_cast<int*>( backend::device_malloc( count * sizeof( int ) ) );
								d_nll = static_cast<float*>( backend::device_malloc( count * sizeof( float ) ) );
								cap = count;
							}
							std::vector<int> tgt( count );
							for ( int r = 0; r < count; ++r ) tgt[r] = off + r + 1 < ids.size() && in_assistant[off + r + 1] ? ids[off + r + 1] : -1;
							backend::stream_t ls = nullptr;
							const backend::half_t* dl = mc.device_logits( count, &ls );
							backend::memcpy_async( d_tgt, tgt.data(), count * sizeof( int ), backend::Copy::H2D, ls );
							engine::modules::LogitRows::score( dl, count, V, d_tgt, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr, d_nll, nullptr, nullptr,
								ls );
							std::vector<float> h( count );
							backend::memcpy_async( h.data(), d_nll, count * sizeof( float ), backend::Copy::D2H, ls );
							backend::stream_sync( ls );
							for ( int r = 0; r < count; ++r ) row_nl[r] = h[r];
						}
#else
						mc.read_logits( count, logits.data() );
						// the rows' softmax over the vocabulary in parallel over the process's cores, summed in row order
						auto score_row = [&]( int r ) {
							if ( off + r + 1 >= ids.size() || !in_assistant[off + r + 1] ) return;
							const backend::half_t* l = logits.data() + static_cast<size_t>( r ) * V;
							float mx = -1e30f;
							for ( int v = 0; v < V; ++v ) mx = std::fmax( mx, backend::to_float( l[v] ) );
							double sum = 0.0;
							for ( int v = 0; v < V; ++v ) sum += std::exp( static_cast<double>( backend::to_float( l[v] ) - mx ) );
							row_nl[r] = std::log( sum ) + mx - backend::to_float( l[ids[off + r + 1]] );
						};
						const int nw = std::max( 1, std::min( count, 16 ) );
						std::vector<std::thread> th;
						for ( int w = 1; w < nw; ++w ) th.emplace_back( [&, w] { for ( int r = w; r < count; r += nw ) score_row( r ); } );
						for ( int r = 0; r < count; r += nw ) score_row( r );
						for ( auto& t : th ) t.join();
#endif
						for ( int r = 0; r < count; ++r ) {
							if ( off + r + 1 >= ids.size() || !in_assistant[off + r + 1] ) continue;
							st += row_nl[r];
							++sn;
							sess_bins.add( static_cast<int>( off ) + r, row_nl[r] );
						}
					}
					off = end;
				}
				total += st;
				n += sn;
				std::printf( "[*] session %zu: %zu tokens, %ld scored, NLL %.4f\n", si, ids.size(), sn, sn ? st / sn : 0.0 );
			}
			std::printf( "[*] %ld predicted tokens in %zu sessions (assistant turns only, after a prefix of %zu tokens): mean NLL %.4f, perplexity %.3f\n", n, all_ids.size(), skip, total / std::max( n, 1L ), std::exp( total / std::max( n, 1L ) ) );
			sess_bins.print( "NLL" );
			mc.shutdown();
			return 0;
		}
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
			size_t start = 0, snap_at = 0;
			if ( use_snap ) {
				int best = -1;   // the longest snapshot whose tokens are the start of this session
				for ( size_t i = 0; i < snaps.size(); ++i )
					if ( ids.size() > snaps[i].ids.size() && std::equal( snaps[i].ids.begin(), snaps[i].ids.end(), ids.begin() ) && ( best < 0 || snaps[i].ids.size() > snaps[best].ids.size() ) ) best = static_cast<int>( i );
				if ( best >= 0 ) {
					if ( mc.load_snapshot( 0, snaps[best].snap ) ) start = snaps[best].ids.size(); else { use_snap = false; mc.reset_slot( 0 ); }   // the prefix a family of sessions starts with (their system prompt) comes back from a snapshot instead of a replay
					if ( use_snap && snaps.size() < 24 && shared >= start + 2048 ) snap_at = shared / static_cast<size_t>( mc.snapshot_align() ) * static_cast<size_t>( mc.snapshot_align() );   // a longer prefix shared with an earlier session: a deeper snapshot
				}
				else if ( shared >= 1024 && snaps.size() < 24 ) {   // a prefix shared with an earlier session that no snapshot covers: save the slot there (aligned down) while replaying it
					const size_t al = static_cast<size_t>( mc.snapshot_align() );
					snap_at = shared / al * al;
					if ( snap_at < 1024 ) snap_at = 0;
				}
			}
			int chunk_i = 0;   // chunks past the shared prefix, for --stride
			for ( size_t off = start; off < ids.size(); ) {
				bool counting = off >= shared;
				if ( counting && stride > 1 ) counting = chunk_i++ % stride == 0;
				size_t end = std::min( ids.size(), off + static_cast<size_t>( chunk ) );
				if ( off < shared && end > shared ) end = shared;
				if ( snap_at && off < snap_at && end > snap_at ) end = snap_at;
				const int count = static_cast<int>( end - off );
				mc.set_hessian_capture( counting );
				mc.forward( ids.data() + off, pod::BatchRows::causal( count, static_cast<int>( off ), 0, calibrate && !flash ? pod::BatchRows::LOGITS_NONE : pod::BatchRows::LOGITS_ALL ) );
				if ( mtp_calib && off + 1 < ids.size() ) mc.mtp_step( 0, static_cast<int>( off ), static_cast<int>( std::min<size_t>( count, ids.size() - off - 1 ) ), ids.data() + off + 1, 0, 0, false );   // the prefix rows too: the head's KV stays continuous
				if ( counting ) counted += count;
				replayed += count;
				off = end;
				if ( snap_at && off == snap_at ) {
					snaps.emplace_back();
					if ( mc.save_snapshot( 0, 0, static_cast<int>( snap_at ), &snaps.back().snap, false ) ) { snaps.back().ids.assign( ids.begin(), ids.begin() + snap_at ); std::printf( "[*] snapshot %zu of the first %zu tokens: later sessions with this start restore it\n", snaps.size(), snap_at ); }
					else { snaps.pop_back(); use_snap = false; }
					snap_at = 0;
				}
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
	cfg.speculate = mtp_calib;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	const int V = mc.metadata().vocab_size;
	mc.reset_slot( 0 );
	// GLM with a prefill window and glm_logit_rows for it: several chunks per forward, layer by layer (the routed experts of a layer read once
	// per window: streamed experts are read from disk once per window instead of once per chunk), every row's logits kept
	const int win = glm && chunk > pod::MAX_BATCH ? std::max( 1, std::min( { mc.window_chunks(), engine::runtime::options().glm_logit_rows / chunk, 16 } ) ) : 1;
	const int span = win * chunk;   // rows of one forward at most
	if ( win > 1 ) std::printf( "[test_ppl] %d chunks of %d rows per forward (prefill window)\n", win, chunk );
	std::vector<backend::half_t> logits( static_cast<size_t>( span ) * V );
	FILE* tf = nullptr;
	if ( !topk_out.empty() ) {
		tf = std::fopen( topk_out.c_str(), "wb" );
		ENGINE_ASSERT( tf != nullptr, "test_ppl: cannot write --topk-out" );
		const int hdr[4] = { 0x4b504f54, n_tokens, seq, topk };
		std::fwrite( hdr, sizeof( int ), 4, tf );
		std::fwrite( ids.data(), sizeof( int ), n_tokens, tf );
	}
	// A stored teacher to measure against: its tokens must be these
	std::vector<int> kt_ids;
	std::vector<float> kt_p, kt_tail;
	int kt_k = 0;
	if ( !kl_teacher.empty() ) {
		FILE* kf = std::fopen( kl_teacher.c_str(), "rb" );
		int hdr[4] = {};
		ENGINE_ASSERT( kf && std::fread( hdr, sizeof( int ), 4, kf ) == 4 && hdr[0] == 0x4b504f54 && hdr[1] == n_tokens && hdr[2] == seq,
			"test_ppl: --kl-teacher is not a --topk-out file of this text and settings" );
		kt_k = hdr[3];
		std::vector<int> toks( n_tokens );
		ENGINE_ASSERT( std::fread( toks.data(), sizeof( int ), n_tokens, kf ) == static_cast<size_t>( n_tokens ) && toks == ids, "test_ppl: --kl-teacher holds other tokens" );
		kt_ids.resize( static_cast<size_t>( n_tokens ) * kt_k );
		kt_p.resize( kt_ids.size() );
		kt_tail.resize( n_tokens );
		for ( int r = 0; r < n_tokens; ++r ) {
			ENGINE_ASSERT( std::fread( kt_ids.data() + static_cast<size_t>( r ) * kt_k, sizeof( int ), kt_k, kf ) == static_cast<size_t>( kt_k ) &&
				std::fread( kt_p.data() + static_cast<size_t>( r ) * kt_k, sizeof( float ), kt_k, kf ) == static_cast<size_t>( kt_k ) &&
				std::fread( &kt_tail[r], sizeof( float ), 1, kf ) == 1, "test_ppl: --kl-teacher is short" );
		}
		std::fclose( kf );
	}
	double kl_sum = 0.0;
	size_t kl_n = 0;
	PosBins nll_bins, kl_bins;
	nll_bins.init( bin_edges );
	kl_bins.init( bin_edges );
	std::vector<float> nll;
	nll.reserve( n_tokens );
	double total = 0.0;
	// Per row of a chunk: the softmax's max and sum once, then the teacher's top K and the rest (--topk-out), the KL against a stored teacher
	// and the NLL; the rows run in parallel over the process's cores (the GPUs wait on this host work otherwise), the results are written and
	// summed in row order
	struct RowOut {
		std::vector<int> ids;
		std::vector<float> p;
		float tail = 0.0f, nl = 0.0f;
		double kl = 0.0;
		bool has_kl = false, has_nl = false;
		int top = 0;
	};
	std::vector<RowOut> outs( span );
	for ( RowOut& o : outs ) {
		o.ids.resize( topk );
		o.p.resize( topk );
	}
	int workers = 1;
	{
		cpu_set_t cs;
		if ( sched_getaffinity( 0, sizeof( cs ), &cs ) == 0 ) workers = CPU_COUNT( &cs );
		workers = std::max( 1, std::min( { workers, chunk, 16 } ) );
	}
#if defined( ENGINE_BACKEND_CUDA ) && ENGINE_BACKEND_CUDA == 1
	// The rows' statistics on the device where the logits are (engine::modules::LogitRows): only they come back to the host
	const size_t kmax = static_cast<size_t>( std::max( tf ? topk : 0, kt_k ) );
	auto dev = [&]( size_t bytes ) { return backend::device_malloc( std::max<size_t>( bytes, 16 ) ); };
	int *d_tgt = static_cast<int*>( dev( span * sizeof( int ) ) ), *d_ids = static_cast<int*>( dev( span * kmax * sizeof( int ) ) );
	int *d_top = static_cast<int*>( dev( span * sizeof( int ) ) ), *d_tids = static_cast<int*>( dev( span * kmax * sizeof( int ) ) );
	float *d_p = static_cast<float*>( dev( span * kmax * sizeof( float ) ) ), *d_tail = static_cast<float*>( dev( span * sizeof( float ) ) );
	float *d_tp = static_cast<float*>( dev( span * kmax * sizeof( float ) ) ), *d_ttail = static_cast<float*>( dev( span * sizeof( float ) ) );
	float *d_nll = static_cast<float*>( dev( span * sizeof( float ) ) ), *d_kl = static_cast<float*>( dev( span * sizeof( float ) ) );
	std::vector<int> h_tgt( span ), h_ids( span * kmax ), h_top( span );
	std::vector<float> h_p( span * kmax ), h_tail( span ), h_nll( span ), h_kl( span );
	auto device_rows = [&]( int count, int off ) {
		backend::stream_t ls = nullptr;
		const backend::half_t* dl = mc.device_logits( count, &ls );
		for ( int r = 0; r < count; ++r ) h_tgt[r] = off + r + 1 < n_tokens ? ids[off + r + 1] : -1;
		backend::memcpy_async( d_tgt, h_tgt.data(), count * sizeof( int ), backend::Copy::H2D, ls );
		if ( kt_k > 0 ) {
			const size_t o = static_cast<size_t>( off ) * kt_k;
			backend::memcpy_async( d_tids, kt_ids.data() + o, count * kt_k * sizeof( int ), backend::Copy::H2D, ls );
			backend::memcpy_async( d_tp, kt_p.data() + o, count * kt_k * sizeof( float ), backend::Copy::H2D, ls );
			backend::memcpy_async( d_ttail, kt_tail.data() + off, count * sizeof( float ), backend::Copy::H2D, ls );
		}
		const int k = tf ? topk : 0;
		engine::modules::LogitRows::score( dl, count, V, d_tgt, k, d_ids, d_p, d_tail, kt_k > 0 ? d_tids : nullptr, d_tp, d_ttail, kt_k, nullptr, d_nll, d_top,
			kt_k > 0 ? d_kl : nullptr, ls );
		backend::memcpy_async( h_nll.data(), d_nll, count * sizeof( float ), backend::Copy::D2H, ls );
		backend::memcpy_async( h_top.data(), d_top, count * sizeof( int ), backend::Copy::D2H, ls );
		if ( k > 0 ) {
			backend::memcpy_async( h_ids.data(), d_ids, count * k * sizeof( int ), backend::Copy::D2H, ls );
			backend::memcpy_async( h_p.data(), d_p, count * k * sizeof( float ), backend::Copy::D2H, ls );
			backend::memcpy_async( h_tail.data(), d_tail, count * sizeof( float ), backend::Copy::D2H, ls );
		}
		if ( kt_k > 0 ) backend::memcpy_async( h_kl.data(), d_kl, count * sizeof( float ), backend::Copy::D2H, ls );
		backend::stream_sync( ls );
		for ( int r = 0; r < count; ++r ) {
			RowOut& o = outs[r];
			o.top = h_top[r];
			if ( k > 0 ) {
				std::copy( h_ids.begin() + static_cast<size_t>( r ) * k, h_ids.begin() + static_cast<size_t>( r + 1 ) * k, o.ids.begin() );
				std::copy( h_p.begin() + static_cast<size_t>( r ) * k, h_p.begin() + static_cast<size_t>( r + 1 ) * k, o.p.begin() );
				o.tail = h_tail[r];
			}
			o.has_kl = kt_k > 0 && off + r >= kl_from;
			o.kl = h_kl[r];
			o.has_nl = off + r + 1 < n_tokens;
			o.nl = h_nll[r];
		}
	};
#endif
	std::vector<std::vector<float>> w_all( workers, std::vector<float>( V ) );
	std::vector<std::vector<int>> w_order( tf ? workers : 0, std::vector<int>( V ) );
	auto row_work = [&]( int w, int r, int off ) {
		RowOut& o = outs[r];
		const backend::half_t* l = logits.data() + static_cast<size_t>( r ) * V;
		std::vector<float>& t_all = w_all[w];
		float mx = -1e30f;
		int top = 0;
		for ( int v = 0; v < V; ++v ) {
			t_all[v] = backend::to_float( l[v] );
			if ( t_all[v] > mx ) {
				mx = t_all[v];
				top = v;
			}
		}
		double sum = 0.0;
		for ( int v = 0; v < V; ++v ) sum += std::exp( static_cast<double>( t_all[v] - mx ) );
		o.top = top;
		if ( tf ) {
			// the teacher at this position: softmax over the vocabulary, the top K and the rest
			std::vector<int>& order = w_order[w];
			for ( int v = 0; v < V; ++v ) order[v] = v;
			auto better = [&]( int a, int b ) { return t_all[a] > t_all[b] || ( t_all[a] == t_all[b] && a < b ); };
			std::nth_element( order.begin(), order.begin() + topk, order.end(), better );
			std::sort( order.begin(), order.begin() + topk, better );
			double in = 0.0;
			for ( int j = 0; j < topk; ++j ) {
				o.ids[j] = order[j];
				o.p[j] = static_cast<float>( std::exp( static_cast<double>( t_all[order[j]] - mx ) ) / sum );
				in += o.p[j];
			}
			o.tail = static_cast<float>( std::max( 0.0, 1.0 - in ) );
		}
		const size_t row = static_cast<size_t>( off + r );
		o.has_kl = kt_k > 0 && off + r >= kl_from;
		if ( o.has_kl ) {
			// KL( teacher || model ) at this position: the teacher's top entries and its tail against the model's mass outside them
			const double lse = std::log( sum ) + mx;
			double k = 0.0, qs = 0.0;
			for ( int j = 0; j < kt_k; ++j ) {
				const double lq = t_all[kt_ids[row * kt_k + j]] - lse, pp = kt_p[row * kt_k + j];
				qs += std::exp( lq );
				if ( pp > 0.0 ) k += pp * ( std::log( pp ) - lq );
			}
			const double pt = kt_tail[row], qt = std::max( 1.0 - qs, 1e-12 );
			if ( pt > 0.0 ) k += pt * ( std::log( pt ) - std::log( qt ) );
			o.kl = k;
		}
		o.has_nl = off + r + 1 < n_tokens;
		if ( o.has_nl ) o.nl = static_cast<float>( std::log( sum ) + mx - t_all[ids[off + r + 1]] );
	};
	for ( int off = 0; off + 1 < n_tokens; ) {
		const int pos = seq > 0 ? off % seq : off;   // --seq: independent rows, the slot restarts at each
		// --kl-from over independent rows (no teacher written): a whole row before the measured positions adds nothing, skip it
		if ( seq > 0 && pos == 0 && kt_k > 0 && !tf && off + seq <= kl_from ) {
			off += seq;
			continue;
		}
		if ( seq > 0 && pos == 0 ) mc.reset_slot( 0 );
		int count = std::min( { win > 1 && !calibrate ? span : chunk, n_tokens - off, seq > 0 ? seq - pos : span } );
		if ( count > chunk ) {
			// a window of whole chunks above the decode sizes (a short tail runs on its own next)
			pod::BatchRows w[16];
			int n = 0, rows = 0;
			while ( n < win && count - rows > pod::MAX_BATCH ) {
				const int c = std::min( chunk, count - rows );
				w[n++] = pod::BatchRows::causal( c, pos + rows, 0, pod::BatchRows::LOGITS_ALL );
				rows += c;
			}
			count = rows;
			if ( n >= 2 ) mc.forward_window( ids.data() + off, w, n );
			else mc.forward( ids.data() + off, w[0] );
		}
		else mc.forward( ids.data() + off, pod::BatchRows::causal( count, pos, 0, calibrate && !flash ? pod::BatchRows::LOGITS_NONE : pod::BatchRows::LOGITS_ALL ) );
		if ( mtp_calib ) {
			const int c = std::min( { count, n_tokens - off - 1, seq > 0 ? seq - pos - 1 : count } );   // a row's last position has no next token inside the row
			if ( c >= 1 ) mc.mtp_step( 0, pos, c, ids.data() + off + 1, 0, 0, false );
		}
		if ( calibrate ) { off += count; continue; }   // forward only: ENGINE_HESSIAN_DIR collects its statistics in the layers
#if defined( ENGINE_BACKEND_CUDA ) && ENGINE_BACKEND_CUDA == 1
		device_rows( count, off );
#else
		mc.read_logits( count, logits.data() );
		{
			const int nw = std::min( workers, count );
			std::vector<std::thread> th;
			for ( int w = 1; w < nw; ++w ) th.emplace_back( [&, w] { for ( int r = w; r < count; r += nw ) row_work( w, r, off ); } );
			for ( int r = 0; r < count; r += nw ) row_work( 0, r, off );
			for ( auto& t : th ) t.join();
		}
#endif
		for ( int r = 0; r < count; ++r ) {
			const RowOut& o = outs[r];
			if ( tf ) {
				std::fwrite( o.ids.data(), sizeof( int ), topk, tf );
				std::fwrite( o.p.data(), sizeof( float ), topk, tf );
				std::fwrite( &o.tail, sizeof( float ), 1, tf );
			}
			if ( o.has_kl ) {
				kl_sum += o.kl;
				++kl_n;
				kl_bins.add( pos + r, o.kl );
			}
			if ( !o.has_nl ) continue;
			if ( show_miss > 0 && o.nl > 10.0f && static_cast<int>( nll.size() ) % 25 == 0 ) {   // every 25th position of the badly predicted ones: the target and the model's top guess
				std::printf( "[miss] pos %d target %d (%s) top %d (%s) nll %.1f\n", off + r + 1, ids[off + r + 1], tok.decode( { ids[off + r + 1] } ).c_str(), o.top,
					tok.decode( { o.top } ).c_str(), o.nl );
				if ( --show_miss == 0 ) show_miss = -1;
			}
			nll.push_back( o.nl );
			total += o.nl;
			nll_bins.add( pos + r, o.nl );
		}
		off += count;
	}
	if ( tf ) std::fclose( tf );
	if ( calibrate ) { mc.shutdown(); return 0; }
	const double mean = total / nll.size();
	std::printf( "[*] %zu predicted tokens: mean NLL %.4f, perplexity %.3f\n", nll.size(), mean, std::exp( mean ) );
	if ( kl_n > 0 ) std::printf( "[*] KL( teacher || model ) over %zu positions: %.6f per token\n", kl_n, kl_sum / kl_n );
	nll_bins.print( "NLL" );
	if ( kl_n > 0 ) kl_bins.print( "KL" );
	if ( !nll_out.empty() ) {
		if ( FILE* f = std::fopen( nll_out.c_str(), "wb" ) ) {
			std::fwrite( nll.data(), sizeof( float ), nll.size(), f );
			std::fclose( f );
		}
	}
	mc.shutdown();
	return 0;
}
