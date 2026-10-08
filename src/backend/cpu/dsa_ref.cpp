// CPU backend: DSA indexer projection, scoring and selection as plain C++ (FP32 accumulate, FP16 storage), mirroring src/modules/dsa.cu.
//  score:  logits[h][p] = q[h] . key[p], relu, head-weighted sum (two 16-head halves added), scaled by 1/sqrt(DIM * HEADS), stored as FP16
//  select: the SEL_POOLS best pools by FP16 score; ties at the threshold are taken in pool order. Output order matches the CUDA radix
//          select: strictly better pools in pool order, then the threshold ties in pool order, each expanded to POOL tokens, then the raw tail.
#include "engine/modules/dsa.h"
#include "engine/backend/tiers.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <functional>
#include <vector>

namespace engine::modules {

namespace {

using h16 = backend::half_t;

constexpr int D = DSA::DIM;
constexpr int H = DSA::HEADS;
constexpr int P = DSA::POOL;

// Monotone map of an FP16 bit pattern to an unsigned key (the CUDA select's score_key)
inline unsigned score_key( h16 s ) {
	uint16_t b;
	std::memcpy( &b, &s, sizeof( b ) );
	return ( b & 0x8000u ) ? ( ~b & 0xFFFFu ) : ( b | 0x8000u );
}

void project( const h16* w, const h16* x, h16* out, int rows, int n, int k ) {
	for ( int r = 0; r < rows; ++r ) {
		for ( int j = 0; j < n; ++j ) {
			float acc = 0.0f;
			for ( int c = 0; c < k; ++c ) acc += static_cast<float>( w[static_cast<size_t>( j ) * k + c] ) * static_cast<float>( x[static_cast<size_t>( r ) * k + c] );
			out[static_cast<size_t>( r ) * n + j] = static_cast<h16>( acc );
		}
	}
}

}

void DSA::project_small( const backend::half_t* w, const backend::half_t* x, backend::half_t* proj, int rows, int n, int k, backend::stream_t ) {
	ENGINE_ASSERT( rows >= 1 && rows <= 16 && k % 256 == 0, "DSA::project_small: up to 16 rows, k a multiple of 256" );
	project( w, x, proj, rows, n, k );
}

void DSA::project_large( const backend::half_t* w, const backend::half_t* x, backend::half_t* proj, int rows, int n, int k, backend::stream_t ) {
	project( w, x, proj, rows, n, k );
}

void DSA::score( const backend::half_t* q_idx, const backend::half_t* w, const backend::half_t* pool_plane, backend::half_t* scores, int score_stride, int layer, int max_seq, int rows,
	const pod::BatchRows* d_rows, size_t pool_slot_stride, backend::stream_t ) {
	ENGINE_ASSERT( max_seq % 64 == 0, "DSA score: max_seq must be a multiple of 64" );
	const float norm = 1.0f / std::sqrt( static_cast<float>( D * H ) );
	for ( int r = 0; r < rows; ++r ) {
		const int bound = ( d_rows->pos[r] + 1 ) / P;
		if ( bound <= SEL_POOLS ) continue; // dense row
		const h16* keys = pool_plane + static_cast<size_t>( d_rows->slot[r] ) * pool_slot_stride + static_cast<size_t>( layer ) * ( max_seq / P ) * D;
		const h16* q = q_idx + static_cast<size_t>( r ) * H * D;
		float wf[H];
		for ( int h = 0; h < H; ++h ) wf[h] = static_cast<float>( w[static_cast<size_t>( r ) * H + h] );
		for ( int p = 0; p < bound; ++p ) {
			const h16* key = keys + static_cast<size_t>( p ) * D;
			float half_acc[2] = { 0.0f, 0.0f };
			for ( int h = 0; h < H; ++h ) {
				float dot = 0.0f;
				for ( int d = 0; d < D; ++d ) dot += static_cast<float>( q[h * D + d] ) * static_cast<float>( key[d] );
				half_acc[h / 16] = std::fma( std::fmax( dot, 0.0f ), wf[h], half_acc[h / 16] );
			}
			scores[static_cast<size_t>( r ) * score_stride + p] = static_cast<h16>( ( half_acc[0] + half_acc[1] ) * norm );
		}
	}
}

void DSA::select( const backend::half_t* scores, int score_stride, int* idx, int* nkeys, int rows, const pod::BatchRows* d_rows, backend::stream_t ) {
	const int K = SEL_POOLS;
	for ( int r = 0; r < rows; ++r ) {
		const int vis = d_rows->pos[r] + 1;
		const int bound = vis / P;
		if ( bound <= K ) {
			nkeys[r] = -1;
			continue;
		}
		const h16* sc = scores + static_cast<size_t>( r ) * score_stride;
		std::vector<unsigned> keys( bound );
		for ( int p = 0; p < bound; ++p ) keys[p] = score_key( sc[p] );
		std::vector<unsigned> sorted( keys );
		std::nth_element( sorted.begin(), sorted.begin() + ( K - 1 ), sorted.end(), std::greater<unsigned>() );
		const unsigned thr = sorted[K - 1]; // K-th largest key
		int total_gt = 0;
		for ( int p = 0; p < bound; ++p ) total_gt += keys[p] > thr;
		const int need_eq = K - total_gt;
		std::vector<int> sel( K );
		int ng = 0, ne = 0;
		for ( int p = 0; p < bound; ++p ) {
			if ( keys[p] > thr ) sel[ng++] = p;
			else if ( keys[p] == thr && ne < need_eq ) sel[total_gt + ne++] = p;
		}
		int* out = idx + static_cast<size_t>( r ) * KPAD;
		for ( int i = 0; i < K * P; ++i ) out[i] = sel[i / P] * P + ( i % P );
		const int tail = vis % P;
		for ( int t = 0; t < tail; ++t ) out[K * P + t] = vis - tail + t;
		nkeys[r] = K * P + tail;
	}
}

} // namespace engine::modules
