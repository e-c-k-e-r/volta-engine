#include "engine/modules/logit_rows.h"
#include <cuda_fp16.h>
#include <climits>
#include <cmath>

namespace engine::modules {

namespace {

constexpr int kThreads = 1024;

// Orders like the value: a larger key is a larger logit
__device__ __forceinline__ unsigned key16( half h ) {
	const unsigned b = __half_as_ushort( h );
	return ( b & 0x8000u ) ? ( ~b & 0xFFFFu ) : ( b | 0x8000u );
}

// Sum over the block (every thread gets it)
__device__ __forceinline__ double block_sum( double v, double* sh ) {
	for ( int o = 16; o > 0; o >>= 1 ) v += __shfl_xor_sync( 0xffffffffu, v, o );
	__syncthreads();
	if ( ( threadIdx.x & 31 ) == 0 ) sh[threadIdx.x >> 5] = v;
	__syncthreads();
	double s = 0.0;
	for ( int w = 0; w < kThreads / 32; ++w ) s += sh[w];
	return s;
}

// Block per row: the maximum (first of equals), the log-sum-exp, the target's NLL, the top k by radix selection over the FP16 keys (ties at
// the threshold to the lower ids) sorted larger first, the KL against a stored teacher
__global__ __launch_bounds__( kThreads )
void k_logit_rows( const half* __restrict__ logits, int vocab, const int* __restrict__ target, int k, int* __restrict__ ids, float* __restrict__ p,
	float* __restrict__ tail, const int* __restrict__ t_ids, const float* __restrict__ t_p, const float* __restrict__ t_tail, int tk, double* __restrict__ lse_out,
	float* __restrict__ nll, int* __restrict__ top, float* __restrict__ kl, float* __restrict__ ear ) {
	__shared__ double shd[32];
	__shared__ float shm[32];
	__shared__ int shi[32];
	__shared__ int hist[256];
	__shared__ int s_b1, s_above1, s_thr, s_need_eq, s_wsum[32], s_carry_gt, s_carry_eq;
	__shared__ unsigned s_key[LogitRows::MAX_K];
	__shared__ int s_id[LogitRows::MAX_K];
	const int r = blockIdx.x, t = threadIdx.x, lane = t & 31, w = t >> 5;
	const half* row = logits + static_cast<size_t>( r ) * vocab;
	// the maximum and its first index
	float m = -INFINITY;
	int mi = INT_MAX;
	for ( int v = t; v < vocab; v += kThreads ) {
		const float x = __half2float( row[v] );
		if ( x > m ) { m = x; mi = v; }
	}
	for ( int o = 16; o > 0; o >>= 1 ) {
		const float om = __shfl_xor_sync( 0xffffffffu, m, o );
		const int oi = __shfl_xor_sync( 0xffffffffu, mi, o );
		if ( om > m || ( om == m && oi < mi ) ) { m = om; mi = oi; }
	}
	if ( lane == 0 ) { shm[w] = m; shi[w] = mi; }
	__syncthreads();
	m = shm[0];
	mi = shi[0];
	for ( int x = 1; x < kThreads / 32; ++x ) if ( shm[x] > m || ( shm[x] == m && shi[x] < mi ) ) { m = shm[x]; mi = shi[x]; }
	double s = 0.0;
	for ( int v = t; v < vocab; v += kThreads ) s += static_cast<double>( expf( __half2float( row[v] ) - m ) );
	const double lse = m + log( block_sum( s, shd ) );
	if ( t == 0 ) {
		if ( lse_out ) lse_out[r] = lse;
		if ( top ) top[r] = mi;
		if ( nll && target ) nll[r] = target[r] >= 0 ? static_cast<float>( lse - __half2float( row[target[r]] ) ) : 0.0f;
	}
	if ( k > 0 ) {
		if ( t < 256 ) hist[t] = 0;
		__syncthreads();
		for ( int v = t; v < vocab; v += kThreads ) atomicAdd( &hist[key16( row[v] ) >> 8], 1 );
		__syncthreads();
		if ( t == 0 ) {
			int cum = 0, b = 255;
			for ( ; b > 0; --b ) {
				if ( cum + hist[b] >= k ) break;
				cum += hist[b];
			}
			s_b1 = b;
			s_above1 = cum;
		}
		__syncthreads();
		if ( t < 256 ) hist[t] = 0;
		__syncthreads();
		for ( int v = t; v < vocab; v += kThreads ) {
			const unsigned key = key16( row[v] );
			if ( ( key >> 8 ) == static_cast<unsigned>( s_b1 ) ) atomicAdd( &hist[key & 255u], 1 );
		}
		__syncthreads();
		if ( t == 0 ) {
			const int rem = k - s_above1;
			int cum = 0, b = 255;
			for ( ; b > 0; --b ) {
				if ( cum + hist[b] >= rem ) break;
				cum += hist[b];
			}
			s_thr = ( s_b1 << 8 ) | b;
			s_need_eq = rem - cum;
			s_carry_gt = 0;
			s_carry_eq = 0;
		}
		__syncthreads();
		const unsigned thr = static_cast<unsigned>( s_thr );
		const int need_eq = s_need_eq, total_gt = k - need_eq;
		// in id order: every larger key, then the first ties
		for ( int base = 0; base < vocab; base += kThreads ) {
			const int v = base + t;
			const unsigned key = v < vocab ? key16( row[v] ) : 0u;
			const bool gt = v < vocab && key > thr, eq = v < vocab && key == thr;
			const unsigned bg = __ballot_sync( 0xffffffffu, gt ), be = __ballot_sync( 0xffffffffu, eq );
			if ( lane == 0 ) s_wsum[w] = __popc( bg ) | ( __popc( be ) << 16 );
			__syncthreads();
			int wg = 0, we = 0;
			for ( int x = 0; x < w; ++x ) { wg += s_wsum[x] & 0xFFFF; we += s_wsum[x] >> 16; }
			const unsigned lm = ( 1u << lane ) - 1u;
			const int pg = s_carry_gt + wg + __popc( bg & lm ), pe = s_carry_eq + we + __popc( be & lm );
			if ( gt && pg < total_gt ) { s_key[pg] = key; s_id[pg] = v; }
			if ( eq && pe < need_eq ) { s_key[total_gt + pe] = key; s_id[total_gt + pe] = v; }
			__syncthreads();
			if ( t == 0 ) for ( int x = 0; x < kThreads / 32; ++x ) { s_carry_gt += s_wsum[x] & 0xFFFF; s_carry_eq += s_wsum[x] >> 16; }
			__syncthreads();
		}
		// sorted larger first, ties to the lower id (bitonic over a power of two, padding last)
		int P = 1;
		while ( P < k ) P <<= 1;
		if ( t >= k && t < P ) { s_key[t] = 0u; s_id[t] = INT_MAX; }
		__syncthreads();
		for ( int size = 2; size <= P; size <<= 1 ) {
			for ( int stride = size >> 1; stride > 0; stride >>= 1 ) {
				if ( t < P ) {
					const int o = t ^ stride;
					if ( o > t ) {
						const bool first_before = s_key[t] > s_key[o] || ( s_key[t] == s_key[o] && s_id[t] < s_id[o] );
						const bool up = ( t & size ) == 0;   // this pair's run sorts larger first
						if ( up != first_before ) {
							const unsigned tk2 = s_key[t]; s_key[t] = s_key[o]; s_key[o] = tk2;
							const int ti = s_id[t]; s_id[t] = s_id[o]; s_id[o] = ti;
						}
					}
				}
				__syncthreads();
			}
		}
		double in = 0.0;
		if ( t < k ) {
			const int id = s_id[t];
			const float pr = static_cast<float>( exp( static_cast<double>( __half2float( row[id] ) ) - lse ) );
			if ( ids ) ids[static_cast<size_t>( r ) * k + t] = id;
			if ( p ) p[static_cast<size_t>( r ) * k + t] = pr;
			in = pr;
		}
		in = block_sum( in, shd );
		if ( t == 0 && tail ) tail[r] = static_cast<float>( fmax( 0.0, 1.0 - in ) );
	}
	if ( tk > 0 && ( kl || ear ) ) {
		double qs = 0.0, part = 0.0, agree = 0.0;
		if ( t < tk ) {
			const double lq = __half2float( row[t_ids[static_cast<size_t>( r ) * tk + t]] ) - lse, pp = t_p[static_cast<size_t>( r ) * tk + t];
			qs = exp( lq );
			if ( pp > 0.0 ) part = pp * ( log( pp ) - lq );
			agree = fmin( pp, qs );
		}
		qs = block_sum( qs, shd );
		part = block_sum( part, shd );
		agree = block_sum( agree, shd );
		if ( t == 0 ) {
			const double pt = t_tail[r], qt = fmax( 1.0 - qs, 1e-12 );
			if ( kl ) kl[r] = static_cast<float>( part + ( pt > 0.0 ? pt * ( log( pt ) - log( qt ) ) : 0.0 ) );
			if ( ear ) {
				ear[2 * static_cast<size_t>( r )] = static_cast<float>( agree );
				ear[2 * static_cast<size_t>( r ) + 1] = static_cast<float>( agree + fmin( pt, fmax( 1.0 - qs, 0.0 ) ) );
			}
		}
	}
}

} // namespace

void LogitRows::score( const backend::half_t* logits, int rows, int vocab, const int* target, int k, int* ids, float* p, float* tail, const int* t_ids,
	const float* t_p, const float* t_tail, int tk, double* lse, float* nll, int* top, float* kl, backend::stream_t stream, float* ear ) {
	ENGINE_ASSERT( k <= MAX_K && tk <= kThreads, "LogitRows::score: at most MAX_K top entries and 1024 teacher entries" );
	if ( rows <= 0 ) return;
	k_logit_rows<<<rows, kThreads, 0, stream>>>( reinterpret_cast<const half*>( logits ), vocab, target, k, ids, p, tail, t_ids, t_p, t_tail, tk, lse, nll, top, kl,
		ear );
}

} // namespace engine::modules
