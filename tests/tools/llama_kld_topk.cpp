// llama-perplexity --kl-divergence-base file -> test_ppl --kl-teacher file (top K + tail per position) for one chunk, and its token ids for
// test_ppl --ids, so our engine is scored against llama.cpp on the same tokens:
//   g++ -O2 -std=c++17 -o llama_kld_topk tests/tools/llama_kld_topk.cpp
//   llama-perplexity -m M.gguf -f TEXT -c N --chunks 1 --kl-divergence-base base.kld; llama_kld_topk base.kld base.topk [K=128] [chunk=0]
//   test_ppl --model DIR --ids base.topk.ids --tokens N --kl-teacher base.topk --kl-from N/2
// llama.cpp scores positions n_ctx/2 .. n_ctx-2 of each chunk (each predicts the next token); the other positions get an empty teacher
// (all mass in the tail), which --kl-from n_ctx/2 leaves out. llama-perplexity needs a text of at least 2 x n_ctx tokens, and exits 0 without
// writing the logits when it is shorter.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

int main( int argc, char** argv ) {
	if ( argc < 3 ) {
		std::fprintf( stderr, "usage: llama_kld_topk <in.kld> <out.topk> [K=128] [chunk=0]\n" );
		return 2;
	}
	const int K = argc > 3 ? std::atoi( argv[3] ) : 128, want = argc > 4 ? std::atoi( argv[4] ) : 0;
	std::FILE* in = std::fopen( argv[1], "rb" );
	if ( !in ) return 1;
	char magic[8];
	int n_ctx = 0, n_vocab = 0, n_chunk = 0;
	if ( std::fread( magic, 1, 8, in ) != 8 || std::memcmp( magic, "_logits_", 8 ) != 0 || std::fread( &n_ctx, 4, 1, in ) != 1 || std::fread( &n_vocab, 4, 1, in ) != 1 ||
		std::fread( &n_chunk, 4, 1, in ) != 1 || want >= n_chunk ) {
		std::fprintf( stderr, "llama_kld_topk: not a llama-perplexity logits file (or chunk %d of %d)\n", want, n_chunk );
		return 1;
	}
	std::vector<int32_t> tokens( static_cast<size_t>( n_chunk ) * n_ctx );
	if ( std::fread( tokens.data(), 4, tokens.size(), in ) != tokens.size() ) return 1;
	const int first = n_ctx / 2, scored = n_ctx - 1 - first;
	const size_t nv = 2 * ( ( n_vocab + 1 ) / 2 ) + 4;
	std::vector<uint16_t> rec( nv );
	if ( std::fseek( in, static_cast<long>( static_cast<size_t>( want ) * scored * nv * 2 ), SEEK_CUR ) != 0 ) return 1;
	std::printf( "n_ctx %d, vocab %d, chunks %d: chunk %d, positions %d..%d scored\n", n_ctx, n_vocab, n_chunk, want, first, n_ctx - 2 );

	std::FILE* out = std::fopen( argv[2], "wb" );
	if ( !out ) return 1;
	const int32_t* tk = tokens.data() + static_cast<size_t>( want ) * n_ctx;
	const int hdr[4] = { 0x4b504f54, n_ctx, 0, K };
	std::fwrite( hdr, 4, 4, out );
	std::fwrite( tk, 4, n_ctx, out );
	std::vector<int> ids( K );
	std::vector<float> p( K );
	std::vector<float> lp( n_vocab );
	std::vector<int> order( n_vocab );
	double tail_sum = 0.0;
	for ( int pos = 0; pos < n_ctx; ++pos ) {
		float tail = 1.0f;
		std::fill( ids.begin(), ids.end(), 0 );
		std::fill( p.begin(), p.end(), 0.0f );
		if ( pos >= first && pos < n_ctx - 1 ) {
			if ( std::fread( rec.data(), 2, nv, in ) != nv ) return 1;
			float scale, min_log_prob;
			std::memcpy( &scale, rec.data(), 4 );
			std::memcpy( &min_log_prob, rec.data() + 2, 4 );
			const uint16_t* code = rec.data() + 4;
			for ( int v = 0; v < n_vocab; ++v ) lp[v] = scale * code[v] + min_log_prob;   // as llama-perplexity decodes them
			std::iota( order.begin(), order.end(), 0 );
			auto better = [&]( int a, int b ) { return lp[a] > lp[b] || ( lp[a] == lp[b] && a < b ); };
			std::nth_element( order.begin(), order.begin() + K, order.end(), better );
			std::sort( order.begin(), order.begin() + K, better );
			double inside = 0.0;
			for ( int j = 0; j < K; ++j ) {
				ids[j] = order[j];
				p[j] = std::exp( lp[order[j]] );
				inside += p[j];
			}
			tail = static_cast<float>( std::max( 0.0, 1.0 - inside ) );
			tail_sum += tail;
		}
		std::fwrite( ids.data(), 4, K, out );
		std::fwrite( p.data(), 4, K, out );
		std::fwrite( &tail, 4, 1, out );
	}
	std::fclose( out );
	std::fclose( in );
	std::FILE* idf = std::fopen( ( std::string( argv[2] ) + ".ids" ).c_str(), "w" );
	for ( int pos = 0; idf && pos < n_ctx; ++pos ) std::fprintf( idf, "%d\n", tk[pos] );
	if ( idf ) std::fclose( idf );
	std::printf( "wrote %s: %d positions, top %d, mean tail of the scored ones %.5f\n", argv[2], n_ctx, K, tail_sum / scored );
	return 0;
}
