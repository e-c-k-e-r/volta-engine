// Expected acceptance rate (EAR) of a quantized model against a reference model, on text the reference model generated itself:
// per scored token, sum over the vocabulary of min( p_ref, p_quant ), the acceptance rate the quantized model would get as the
// draft model of the reference in speculative sampling (one minus the total variation distance of the two distributions). Only
// generated tokens are scored, never the prompt. Three steps, each run with the model the step is about (ENGINE_TEST_QWEN_DIR,
// ENGINE_TEST_FLASH_DIR or ENGINE_TEST_GLM_DIR selects the checkpoint, as for test_ppl):
//   test_ear [--flash | --glm] --generate CORPUS [--prompts tests/data/ear_prompts.json] [--max-new 768] [--seed 1]
//       the reference model answers every prompt (its chat template, its generation_config.json sampling settings); writes the
//       token streams and which tokens it generated
//   test_ear [--flash | --glm] --reference CORPUS REF [--top 256] [--chunk 1]
//       the reference model teacher-forced over the corpus: the top entries of every scored token's distribution and the mass
//       left outside them
//   test_ear [--flash | --glm] --compare CORPUS REF [--chunk 1]
//       the quantized model over the same corpus: EAR per kind of text and overall (a lower bound: the reference's tail is
//       counted as disagreeing; the gap to the upper bound is printed), KL divergence over the top entries plus the lumped tail,
//       and how often both models' most likely token is the same
// --chunk sets the rows per teacher-forced forward (1: the decode path; up to 16). Reference and comparison should use the same.
#include "engine/runtime/test_driver.h"
#include "engine/runtime/sampling.h"
#include "engine/loader/tokenizer.h"
#include "engine/loader/chat_template.h"
#include "engine/backend/numeric.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace engine::runtime;
namespace backend = engine::backend;

namespace {

struct Sample {
	std::string kind;
	int prompt = 0;            // tokens before the generated ones
	std::vector<int> ids;      // prompt + generated
};

constexpr uint32_t kCorpusMagic = 0x43524145, kRefMagic = 0x52524145;   // "EARC", "EARR"

void write_corpus( const char* path, const std::vector<Sample>& samples ) {
	FILE* f = std::fopen( path, "wb" );
	ENGINE_ASSERT( f != nullptr, "test_ear: cannot write the corpus" );
	const uint32_t hdr[2] = { kCorpusMagic, static_cast<uint32_t>( samples.size() ) };
	std::fwrite( hdr, sizeof( hdr ), 1, f );
	for ( const Sample& s : samples ) {
		const uint32_t meta[3] = { static_cast<uint32_t>( s.kind.size() ), static_cast<uint32_t>( s.prompt ), static_cast<uint32_t>( s.ids.size() ) };
		std::fwrite( meta, sizeof( meta ), 1, f );
		std::fwrite( s.kind.data(), 1, s.kind.size(), f );
		std::fwrite( s.ids.data(), sizeof( int ), s.ids.size(), f );
	}
	std::fclose( f );
}

std::vector<Sample> read_corpus( const char* path ) {
	FILE* f = std::fopen( path, "rb" );
	ENGINE_ASSERT( f != nullptr, "test_ear: cannot read the corpus" );
	uint32_t hdr[2] = {};
	ENGINE_ASSERT( std::fread( hdr, sizeof( hdr ), 1, f ) == 1 && hdr[0] == kCorpusMagic, "test_ear: not a corpus file" );
	std::vector<Sample> samples( hdr[1] );
	for ( Sample& s : samples ) {
		uint32_t meta[3] = {};
		ENGINE_ASSERT( std::fread( meta, sizeof( meta ), 1, f ) == 1, "test_ear: truncated corpus" );
		s.kind.resize( meta[0] );
		s.prompt = static_cast<int>( meta[1] );
		s.ids.resize( meta[2] );
		ENGINE_ASSERT( std::fread( s.kind.data(), 1, meta[0], f ) == meta[0] && std::fread( s.ids.data(), sizeof( int ), meta[2], f ) == meta[2], "test_ear: truncated corpus" );
	}
	std::fclose( f );
	return samples;
}

// Probabilities of one row of logits (FP16 from the device), in double
void softmax_row( const backend::half_t* l, int V, std::vector<double>& p ) {
	float mx = -1e30f;
	for ( int v = 0; v < V; ++v ) mx = std::fmax( mx, backend::to_float( l[v] ) );
	double sum = 0.0;
	for ( int v = 0; v < V; ++v ) sum += ( p[v] = std::exp( static_cast<double>( backend::to_float( l[v] ) - mx ) ) );
	for ( int v = 0; v < V; ++v ) p[v] /= sum;
}

// Teacher-forced pass over every sample, `chunk` rows per forward; calls scored( sample, position, probabilities ) for each generated
// token (the distribution that predicts it)
template <typename F>
void teacher_forced( ModelCluster& mc, const std::vector<Sample>& samples, int chunk, F scored ) {
	const int V = mc.metadata().vocab_size;
	std::vector<backend::half_t> logits( static_cast<size_t>( chunk ) * V );
	std::vector<double> p( V );
	for ( size_t si = 0; si < samples.size(); ++si ) {
		const Sample& s = samples[si];
		const int n = static_cast<int>( s.ids.size() );
		mc.reset_slot( 0 );
		// the prompt (all but its last token) as plain prefill: no row of it is scored
		if ( s.prompt > 1 ) mc.prefill( s.ids.data(), s.prompt - 1, 0 );
		for ( int off = s.prompt - 1; off + 1 < n; off += chunk ) {
			const int count = std::min( chunk, n - 1 - off );
			mc.forward( s.ids.data() + off, pod::BatchRows::causal( count, off, 0, pod::BatchRows::LOGITS_ALL ) );
			mc.read_logits( count, logits.data() );
			for ( int r = 0; r < count; ++r ) {
				softmax_row( logits.data() + static_cast<size_t>( r ) * V, V, p );
				scored( si, off + r + 1, p );
			}
		}
	}
}

} // namespace

int main( int argc, char** argv ) {
	bool glm = false, flash = false;
	std::string mode, corpus_path, ref_path, prompts_path = std::string( TEST_DATA_DIR ) + "/ear_prompts.json";
	int max_new = 768, top = 256, chunk = 1;
	uint64_t seed = 1;
	for ( int i = 1; i < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--glm" ) ) glm = true;
		else if ( !std::strcmp( argv[i], "--flash" ) ) flash = true;
		else if ( !std::strcmp( argv[i], "--generate" ) && i + 1 < argc ) { mode = "generate"; corpus_path = argv[++i]; }
		else if ( ( !std::strcmp( argv[i], "--reference" ) || !std::strcmp( argv[i], "--compare" ) ) && i + 2 < argc ) {
			mode = argv[i] + 2;
			corpus_path = argv[++i];
			ref_path = argv[++i];
		}
		else if ( !std::strcmp( argv[i], "--prompts" ) && i + 1 < argc ) prompts_path = argv[++i];
		else if ( !std::strcmp( argv[i], "--max-new" ) && i + 1 < argc ) max_new = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--top" ) && i + 1 < argc ) top = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--chunk" ) && i + 1 < argc ) chunk = std::atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--seed" ) && i + 1 < argc ) seed = std::strtoull( argv[++i], nullptr, 10 );
		else {
			std::printf( "test_ear: unknown argument %s (see the header of tests/4_coherence/test_ear.cpp)\n", argv[i] );
			return 2;
		}
	}
	if ( mode.empty() ) {
		std::printf( "usage: test_ear [--flash | --glm] --generate CORPUS | --reference CORPUS REF | --compare CORPUS REF\n" );
		return 2;
	}
	chunk = std::max( 1, std::min( chunk, pod::MAX_BATCH ) );
	const char* model_dir = glm ? test::GLM_DIR : ( flash ? test::FLASH_DIR : test::QWEN_DIR );
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = glm ? "glm5_next" : ( flash ? "qwen4_exp" : "qwen3_5" );
	cfg.model_dir = model_dir;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	const int V = mc.metadata().vocab_size;

	if ( mode == "generate" ) {
		engine::loader::Tokenizer tok;
		ENGINE_ASSERT( tok.load( model_dir ), "tokenizer load failed" );
		engine::loader::ChatTemplate tmpl;
		ENGINE_ASSERT( tmpl.load( model_dir ), "chat template load failed" );
		// the checkpoint's own sampling settings (top_k is capped at the 16 candidates the device gathers, as in the server)
		SamplingParams sp;
		sp.temperature = 1.0f;
		std::ifstream gc( std::string( model_dir ) + "/generation_config.json" );
		if ( gc ) {
			const nlohmann::json g = nlohmann::json::parse( gc, nullptr, false );
			if ( g.is_object() ) {
				sp.temperature = g.value( "temperature", sp.temperature );
				sp.top_p = g.value( "top_p", sp.top_p );
				sp.top_k = g.value( "top_k", sp.top_k );
			}
		}
		std::ifstream pf( prompts_path );
		ENGINE_ASSERT( pf.good(), "test_ear: cannot read the prompts" );
		const nlohmann::json prompts = nlohmann::json::parse( pf );
		Rng rng( seed );
		std::vector<Sample> samples;
		for ( const auto& pr : prompts ) {
			Sample s;
			s.kind = pr.value( "kind", "text" );
			const std::string text = tmpl.render( { { "user", pr.at( "prompt" ).get<std::string>() } } );
			ENGINE_ASSERT( !text.empty(), "test_ear: the chat template rejected a prompt" );
			s.ids = tok.encode( text, false );
			s.prompt = static_cast<int>( s.ids.size() );
			mc.reset_slot( 0 );
			Candidates c;
			mc.prefill( s.ids.data(), s.prompt, 0, 0, &c );
			int pos = s.prompt;
			for ( int g = 0; g < max_new; ++g ) {
				const int t = sample_token( c, sp, rng );
				s.ids.push_back( t );
				if ( mc.is_eos( t ) ) break;
				mc.forward( &t, 1, pos++, 0 );
				mc.sample_topk( 1, &c );
			}
			std::printf( "[*] %-12s prompt %4d tokens, generated %4d\n", s.kind.c_str(), s.prompt, static_cast<int>( s.ids.size() ) - s.prompt );
			samples.push_back( std::move( s ) );
		}
		write_corpus( corpus_path.c_str(), samples );
		size_t gen = 0;
		for ( const Sample& s : samples ) gen += s.ids.size() - s.prompt;
		std::printf( "[*] corpus %s: %zu samples, %zu generated tokens (T %.2f, top-p %.2f, top-k %d, seed %llu, model %s)\n", corpus_path.c_str(), samples.size(), gen,
			sp.temperature, sp.top_p, sp.top_k, static_cast<unsigned long long>( seed ), model_dir );
		return 0;
	}

	const std::vector<Sample> samples = read_corpus( corpus_path.c_str() );
	if ( mode == "reference" ) {
		// per scored token: the top `top` ids and probabilities, then the probability mass outside them
		FILE* f = std::fopen( ref_path.c_str(), "wb" );
		ENGINE_ASSERT( f != nullptr, "test_ear: cannot write the reference" );
		const uint32_t hdr[3] = { kRefMagic, static_cast<uint32_t>( top ), static_cast<uint32_t>( chunk ) };
		std::fwrite( hdr, sizeof( hdr ), 1, f );
		std::vector<int> order( V );
		std::vector<int> ids( top );
		std::vector<float> probs( top );
		size_t n = 0;
		teacher_forced( mc, samples, chunk, [&]( size_t, int, const std::vector<double>& p ) {
			for ( int v = 0; v < V; ++v ) order[v] = v;
			std::partial_sort( order.begin(), order.begin() + top, order.end(), [&]( int a, int b ) { return p[a] > p[b]; } );
			double in = 0.0;
			for ( int i = 0; i < top; ++i ) {
				ids[i] = order[i];
				probs[i] = static_cast<float>( p[order[i]] );
				in += p[order[i]];
			}
			const float tail = static_cast<float>( std::max( 0.0, 1.0 - in ) );
			std::fwrite( ids.data(), sizeof( int ), top, f );
			std::fwrite( probs.data(), sizeof( float ), top, f );
			std::fwrite( &tail, sizeof( float ), 1, f );
			++n;
		} );
		std::fclose( f );
		std::printf( "[*] reference %s: %zu scored tokens, top %d, chunk %d, model %s\n", ref_path.c_str(), n, top, chunk, model_dir );
		return 0;
	}

	// compare
	FILE* f = std::fopen( ref_path.c_str(), "rb" );
	ENGINE_ASSERT( f != nullptr, "test_ear: cannot read the reference" );
	uint32_t hdr[3] = {};
	ENGINE_ASSERT( std::fread( hdr, sizeof( hdr ), 1, f ) == 1 && hdr[0] == kRefMagic, "test_ear: not a reference file" );
	const int rtop = static_cast<int>( hdr[1] );
	if ( static_cast<int>( hdr[2] ) != chunk ) std::printf( "[!] the reference was scored with chunk %u, this run uses %d\n", hdr[2], chunk );
	struct Acc { double ear = 0.0, gap = 0.0, kld = 0.0; long agree = 0, n = 0; };
	std::map<std::string, Acc> by_kind;
	Acc all;
	std::vector<int> ids( rtop );
	std::vector<float> probs( rtop );
	teacher_forced( mc, samples, chunk, [&]( size_t si, int, const std::vector<double>& q ) {
		float tail_p = 0.0f;
		ENGINE_ASSERT( std::fread( ids.data(), sizeof( int ), rtop, f ) == static_cast<size_t>( rtop ) && std::fread( probs.data(), sizeof( float ), rtop, f ) == static_cast<size_t>( rtop ) &&
			std::fread( &tail_p, sizeof( float ), 1, f ) == 1, "test_ear: the reference is shorter than the corpus (another corpus?)" );
		double ear = 0.0, in_q = 0.0, kld = 0.0;
		for ( int i = 0; i < rtop; ++i ) {
			const double pi = probs[i], qi = q[ids[i]];
			ear += std::min( pi, qi );
			in_q += qi;
			if ( pi > 0.0 ) kld += pi * std::log( pi / std::max( qi, 1e-30 ) );
		}
		const double tail_q = std::max( 0.0, 1.0 - in_q );
		if ( tail_p > 0.0f ) kld += tail_p * std::log( tail_p / std::max( tail_q, 1e-30 ) );
		const int qtop = static_cast<int>( std::max_element( q.begin(), q.end() ) - q.begin() );
		for ( Acc* a : { &by_kind[samples[si].kind], &all } ) {
			a->ear += ear;
			a->gap += std::min<double>( tail_p, tail_q );
			a->kld += kld;
			a->agree += qtop == ids[0];
			a->n += 1;
		}
	} );
	std::fclose( f );
	size_t max_ctx = 0;
	for ( const Sample& s : samples ) max_ctx = std::max( max_ctx, s.ids.size() );
	std::printf( "=== EAR against %s (corpus %s, %zu samples, %ld scored tokens, contexts up to %zu tokens, chunk %d, reference top %d) ===\n",
		ref_path.c_str(), corpus_path.c_str(), samples.size(), all.n, max_ctx, chunk, rtop );
	std::printf( "    model %s\n", model_dir );
	auto line = [&]( const char* name, const Acc& a ) {
		const double ear = a.ear / a.n;
		std::printf( "  %-12s %6ld tokens: EAR %.4f (a different token about 1 in %.0f, bound gap %.1e), KLD %.4f, top-1 agreement %.2f%%\n", name, a.n, ear,
			1.0 / std::max( 1.0 - ear, 1e-9 ), a.gap / a.n, a.kld / a.n, 100.0 * a.agree / a.n );
	};
	for ( const auto& [kind, a] : by_kind ) line( kind.c_str(), a );
	line( "all", all );
	return 0;
}
