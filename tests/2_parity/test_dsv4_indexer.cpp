// DeepSeek-V4.1 indexer: the tensor-core scorer (two rows to a block, wmma) against the scalar kernel on random data. The same chain (score, then
// top-k select with the candidate-block bookkeeping) runs both ways for a candidate source (cand_mode 1: block maxima pick the candidates) and for a
// later layer (cand_mode 2: only the candidate blocks compete), at ratio 1 and 2 and across the n = TOPK boundary. Selections are compared as sets
// (a near tie at the cut may fall either way), the kept scores element by element, and the two are timed.
#include "engine/config.h"
#include "engine/core/batch.h"
#include "engine/modules/dsv4.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

using engine::modules::DSV4;

static uint32_t g_seed = 12345u;
static float rnd() { g_seed = g_seed * 1664525u + 1013904223u; return ( ( g_seed >> 8 ) / 16777216.0f ) * 2.0f - 1.0f; }

struct Case { int ratio, p0, R, mode; };

int main() {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	CUDA_CHECK( cudaSetDevice( 0 ) );
	constexpr int NH = DSV4::IDX_HEADS, D = DSV4::IDX_DIM;
	const int cap = 65536, cand_block = 8, cand_blocks = 2048, max_pos = 66000, RMAX = 300;
	std::vector<half> h_k( static_cast<size_t>( cap ) * D ), h_q( static_cast<size_t>( RMAX ) * NH * D ), h_w( static_cast<size_t>( RMAX ) * NH );
	for ( auto& v : h_k ) v = __float2half( rnd() * 2.0f );
	for ( auto& v : h_q ) v = __float2half( rnd() * 3.0f );
	for ( auto& v : h_w ) v = __float2half( rnd() );
	std::vector<float> h_rope( static_cast<size_t>( max_pos ) * DSV4::ROPE );
	for ( int p = 0; p < max_pos; ++p )
		for ( int i = 0; i < DSV4::ROPE / 2; ++i ) {
			const float a = p * std::pow( 10000.0f, -2.0f * i / DSV4::ROPE );
			h_rope[( static_cast<size_t>( p ) * ( DSV4::ROPE / 2 ) + i ) * 2] = std::cos( a );
			h_rope[( static_cast<size_t>( p ) * ( DSV4::ROPE / 2 ) + i ) * 2 + 1] = std::sin( a );
		}
	half *d_k, *d_q, *d_w;
	float *d_rope, *d_scores[2];
	uint8_t* d_cand[2];
	int *d_sel[2], *d_nsel[2];
	pod::BatchRows* d_rows;
	CUDA_CHECK( cudaMalloc( &d_k, h_k.size() * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_q, h_q.size() * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_w, h_w.size() * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_rope, h_rope.size() * 4 ) );
	CUDA_CHECK( cudaMalloc( &d_rows, sizeof( pod::BatchRows ) ) );
	for ( int v = 0; v < 2; ++v ) {
		CUDA_CHECK( cudaMalloc( &d_scores[v], static_cast<size_t>( DSV4::SCORE_ROWS ) * cap * 4 ) );
		CUDA_CHECK( cudaMalloc( &d_cand[v], static_cast<size_t>( RMAX ) * ( cap / cand_block ) ) );
		CUDA_CHECK( cudaMalloc( &d_sel[v], static_cast<size_t>( RMAX ) * DSV4::TOPK * 4 ) );
		CUDA_CHECK( cudaMalloc( &d_nsel[v], RMAX * 4 ) );
	}
	CUDA_CHECK( cudaMemcpy( d_k, h_k.data(), h_k.size() * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_q, h_q.data(), h_q.size() * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_w, h_w.data(), h_w.size() * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_rope, h_rope.data(), h_rope.size() * 4, cudaMemcpyHostToDevice ) );
	cudaEvent_t e0, e1;
	cudaEventCreate( &e0 );
	cudaEventCreate( &e1 );

	const Case cases[] = { { 1, 5000, 200, 1 }, { 1, 5000, 200, 2 }, { 2, 9000, 200, 1 }, { 1, 400, 250, 1 }, { 1, 7001, 129, 2 }, { 2, 1100, 300, 2 }, { 1, 60000, 64, 1 }, { 1, 60000, 64, 2 }, { 2, 60000, 128, 1 } };
	int failures = 0;
	for ( const Case& c : cases ) {
		pod::BatchRows rows = pod::BatchRows::causal( c.R, c.p0, 0 );
		CUDA_CHECK( cudaMemcpy( d_rows, &rows, rows.upload_bytes(), cudaMemcpyHostToDevice ) );
		const size_t cand_bytes = static_cast<size_t>( RMAX ) * ( cap / cand_block );
		// the candidate mask for mode 2: from a scalar mode-1 pass (both variants then use the same one)
		CUDA_CHECK( cudaMemset( d_cand[0], 0, cand_bytes ) );
		DSV4::indexer( d_q, d_w, d_k, d_scores[0], d_cand[0], d_sel[0], d_nsel[0], d_rope, c.ratio, cap, 1, cand_blocks, cand_block, c.R, d_rows, 0, false, nullptr );
		CUDA_CHECK( cudaDeviceSynchronize() );
		std::vector<uint8_t> h_cand( cand_bytes );
		CUDA_CHECK( cudaMemcpy( h_cand.data(), d_cand[0], cand_bytes, cudaMemcpyDeviceToHost ) );
		double ms[2] = { 0, 0 };
		for ( int v = 0; v < 2; ++v ) {   // v = 0 scalar, 1 tensor cores
			CUDA_CHECK( cudaMemset( d_sel[v], 0, static_cast<size_t>( RMAX ) * DSV4::TOPK * 4 ) );
			CUDA_CHECK( cudaMemset( d_nsel[v], 0, RMAX * 4 ) );
			CUDA_CHECK( cudaMemset( d_scores[v], 0, static_cast<size_t>( DSV4::SCORE_ROWS ) * cap * 4 ) );
			if ( c.mode == 2 ) CUDA_CHECK( cudaMemcpy( d_cand[v], h_cand.data(), cand_bytes, cudaMemcpyHostToDevice ) );
			else CUDA_CHECK( cudaMemset( d_cand[v], 0, cand_bytes ) );
			for ( int rep = 0; rep < 4; ++rep ) {   // rep 0 warms up (and leaves the results), the rest are timed
				if ( rep == 1 ) cudaEventRecord( e0 );
				DSV4::indexer( d_q, d_w, d_k, d_scores[v], d_cand[v], d_sel[v], d_nsel[v], d_rope, c.ratio, cap, c.mode, cand_blocks, cand_block, c.R, d_rows, 0, v == 1, nullptr );
			}
			cudaEventRecord( e1 );
			CUDA_CHECK( cudaEventSynchronize( e1 ) );
			float t = 0;
			cudaEventElapsedTime( &t, e0, e1 );
			ms[v] = t / 3.0;
		}
		std::vector<int> sel[2], nsel[2];
		std::vector<uint8_t> cand_out[2];
		for ( int v = 0; v < 2; ++v ) {
			sel[v].resize( static_cast<size_t>( RMAX ) * DSV4::TOPK );
			nsel[v].resize( RMAX );
			cand_out[v].resize( cand_bytes );
			CUDA_CHECK( cudaMemcpy( sel[v].data(), d_sel[v], sel[v].size() * 4, cudaMemcpyDeviceToHost ) );
			CUDA_CHECK( cudaMemcpy( nsel[v].data(), d_nsel[v], RMAX * 4, cudaMemcpyDeviceToHost ) );
			CUDA_CHECK( cudaMemcpy( cand_out[v].data(), d_cand[v], cand_bytes, cudaMemcpyDeviceToHost ) );
		}
		int worst = 0, count_bad = 0, cand_diff = 0;
		double mean_diff = 0.0;
		for ( int r = 0; r < c.R; ++r ) {
			if ( nsel[0][r] != nsel[1][r] ) { ++count_bad; continue; }
			std::set<int> a( sel[0].begin() + static_cast<size_t>( r ) * DSV4::TOPK, sel[0].begin() + static_cast<size_t>( r ) * DSV4::TOPK + nsel[0][r] );
			int miss = 0;
			for ( int i = 0; i < nsel[1][r]; ++i ) miss += a.count( sel[1][static_cast<size_t>( r ) * DSV4::TOPK + i] ) ? 0 : 1;
			worst = std::max( worst, miss );
			mean_diff += miss;
		}
		const int nb = ( c.p0 + c.R + 1 ) / c.ratio / cand_block;
		for ( int r = 0; r < c.R; ++r )
			for ( int b = 0; b < nb && b < cap / cand_block; ++b ) cand_diff += cand_out[0][static_cast<size_t>( r ) * ( cap / cand_block ) + b] != cand_out[1][static_cast<size_t>( r ) * ( cap / cand_block ) + b];
		// the scores of the last score block (rows beyond a multiple of SCORE_ROWS), where a row had something to score
		std::vector<float> s0( static_cast<size_t>( DSV4::SCORE_ROWS ) * cap ), s1( s0.size() );
		CUDA_CHECK( cudaMemcpy( s0.data(), d_scores[0], s0.size() * 4, cudaMemcpyDeviceToHost ) );
		CUDA_CHECK( cudaMemcpy( s1.data(), d_scores[1], s1.size() * 4, cudaMemcpyDeviceToHost ) );
		const int last0 = ( ( c.R - 1 ) / DSV4::SCORE_ROWS ) * DSV4::SCORE_ROWS;
		double max_rel = 0.0;
		long compared = 0;
		for ( int lr = 0; lr < c.R - last0; ++lr ) {
			const int n = ( c.p0 + last0 + lr + 1 ) / c.ratio;
			if ( n <= DSV4::TOPK ) continue;
			for ( int t = 0; t < n; ++t ) {
				if ( c.mode == 2 && !h_cand[static_cast<size_t>( last0 + lr ) * ( cap / cand_block ) + t / cand_block] ) continue;
				const float a = s0[static_cast<size_t>( lr ) * cap + t], b = s1[static_cast<size_t>( lr ) * cap + t];
				max_rel = std::max( max_rel, static_cast<double>( std::fabs( a - b ) / ( std::fabs( a ) + 1e-3f ) ) );
				++compared;
			}
		}
		const bool ok = count_bad == 0 && worst <= 3 && max_rel < 1e-3;
		failures += ok ? 0 : 1;
		std::printf( "[%c] ratio %d, rows %d at %d, mode %d: scalar %.2f ms, tensor cores %.2f ms (%.1fx); nsel differs on %d rows, worst set mismatch %d (mean %.2f), cand blocks differing %d, scores: %ld compared, max rel diff %.2e\n",
			ok ? '+' : '-', c.ratio, c.R, c.p0, c.mode, ms[0], ms[1], ms[0] / ms[1], count_bad, worst, mean_diff / c.R, cand_diff, compared, max_rel );
	}
	std::printf( "%s\n", failures ? "FAILED" : "all cases match" );
	return failures ? 1 : 0;
}
