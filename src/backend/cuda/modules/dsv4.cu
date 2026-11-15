#include "engine/modules/dsv4.h"
#include "engine/modules/linear.h"
#include "engine/backend/cuda/reduce.cuh"
#include "../../../kernels/cuda/mma_sm80.cuh"
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>
#include <cmath>
#include <cfloat>

namespace wmma = nvcuda::wmma;

namespace engine::modules {

namespace {

constexpr int HD = DSV4::HD;
constexpr int ROPE = DSV4::ROPE;
constexpr int ROPE0 = HD - ROPE;   // first rotary dim of a head

// ---- quantization grids (round to nearest, ties to even, as the reference's casts) ----------------------------------------------------

// E4M3 (fn): 3 mantissa bits, normals from 2^-6, subnormal step 2^-9, max 448. |x| <= 448 already
__device__ __forceinline__ float round_e4m3( float x ) {
	const float a = fabsf( x );
	float step;
	if ( a < 0.015625f ) step = 0.001953125f;
	else {
		int e;
		frexpf( a, &e );           // a = m * 2^e, m in [0.5, 1)
		step = ldexpf( 1.0f, e - 4 );
	}
	return copysignf( rintf( a / step ) * step, x );
}

// E2M1: 0, 0.5, 1, 1.5, 2, 3, 4, 6. |x| <= 6 already
__device__ __forceinline__ float round_e2m1( float x ) {
	const float a = fabsf( x );
	float r;
	if ( a <= 0.25f ) r = 0.0f;
	else if ( a < 0.75f ) r = 0.5f;
	else if ( a <= 1.25f ) r = 1.0f;
	else if ( a < 1.75f ) r = 1.5f;
	else if ( a <= 2.5f ) r = 2.0f;
	else if ( a < 3.5f ) r = 3.0f;
	else if ( a <= 5.0f ) r = 4.0f;
	else r = 6.0f;
	return copysignf( r, x );
}

// 2^ceil(log2(v)), v > 0 normal
__device__ __forceinline__ float pow2_ceil( float v ) {
	int e;
	const float m = frexpf( v, &e );
	return ldexpf( 1.0f, m == 0.5f ? e - 1 : e );
}

__device__ __forceinline__ float e4m3_scale_pow2( float amax ) { return pow2_ceil( fmaxf( amax, 1e-4f ) / 448.0f ); }
__device__ __forceinline__ float q_e4m3( float x, float s ) { return round_e4m3( fminf( fmaxf( x / s, -448.0f ), 448.0f ) ) * s; }
__device__ __forceinline__ float q_e2m1( float x, float s ) { return round_e2m1( fminf( fmaxf( x / s, -6.0f ), 6.0f ) ) * s; }
// E2M1 scales: E8M0 (power of two) for the indexer, E4M3 for the compressed KV
__device__ __forceinline__ float e2m1_scale_pow2( float amax ) { return pow2_ceil( fmaxf( amax, 6.0f * 1.1754944e-38f ) / 6.0f ); }
__device__ __forceinline__ float e2m1_scale_e4m3( float amax ) { return round_e4m3( fmaxf( amax, 6.0f * 0.001953125f ) / 6.0f ); }

// max over lanes l ^ 1, 2, ..., width / 2 (a group of `width` consecutive lanes)
template <int WIDTH>
__device__ __forceinline__ float group_max( float v ) {
	#pragma unroll
	for ( int o = 1; o < WIDTH; o <<= 1 ) v = fmaxf( v, __shfl_xor_sync( 0xffffffffu, v, o ) );
	return v;
}

// interleaved pair rotation: (a, b) -> (a c - b s, a s + b c); inverse with -s
__device__ __forceinline__ void rot( float& a, float& b, float2 cs, bool inverse ) {
	const float s = inverse ? -cs.y : cs.y;
	const float x = a * cs.x - b * s, y = a * s + b * cs.x;
	a = x;
	b = y;
}

template <int THREADS>
__device__ __forceinline__ float block_sum( float v ) {
	__syncthreads();
	return engine::math::block_sum<THREADS / 32>( v );
}

// ---- window store: 128 threads, 4 dims each (a 32-block is 8 threads) -------------------------------------------------------------------

__global__ __launch_bounds__( 128 )
void window_store_kernel( const half* __restrict__ kv, half* __restrict__ ring, const float2* __restrict__ rope, int layer, const pod::BatchRows* __restrict__ rows,
	size_t ring_slot_stride ) {
	const int r = blockIdx.x, t = threadIdx.x, d0 = t * 4;
	const int pos = rows->pos[r];
	float v[4];
	#pragma unroll
	for ( int i = 0; i < 4; ++i ) v[i] = __half2float( kv[static_cast<size_t>( r ) * HD + d0 + i] );
	if ( d0 >= ROPE0 ) {
		const float2* cs = rope + static_cast<size_t>( pos ) * ( ROPE / 2 ) + ( d0 - ROPE0 ) / 2;
		rot( v[0], v[1], cs[0], false );
		rot( v[2], v[3], cs[1], false );
	}
	float amax = fmaxf( fmaxf( fabsf( v[0] ), fabsf( v[1] ) ), fmaxf( fabsf( v[2] ), fabsf( v[3] ) ) );
	amax = group_max<8>( amax );
	const float s = e4m3_scale_pow2( amax );
	half* dst = ring + static_cast<size_t>( rows->slot[r] ) * ring_slot_stride + ( static_cast<size_t>( layer ) * DSV4::RING + pos % DSV4::RING ) * HD + d0;
	#pragma unroll
	for ( int i = 0; i < 4; ++i ) dst[i] = __float2half_rn( q_e4m3( v[i], s ) );
}

// ---- compressor ---------------------------------------------------------------------------------------------------------------------------

__global__ __launch_bounds__( 128 )
void compress_stage_kernel( const half* __restrict__ ckv, const half* __restrict__ csc, float* __restrict__ cring_kv, float* __restrict__ cring_sc,
	const pod::BatchRows* __restrict__ rows, size_t cring_slot_stride ) {
	const int r = blockIdx.x;
	const int pos = rows->pos[r];
	const size_t base = static_cast<size_t>( rows->slot[r] ) * cring_slot_stride + static_cast<size_t>( pos % DSV4::RING ) * HD;
	for ( int d = threadIdx.x; d < HD; d += 128 ) {
		cring_kv[base + d] = __half2float( ckv[static_cast<size_t>( r ) * HD + d] );
		cring_sc[base + d] = __half2float( csc[static_cast<size_t>( r ) * HD + d] );
	}
}

// 128 threads, 4 dims each; the E2M1 scale covers 16 dims = 4 threads
__global__ __launch_bounds__( 128 )
void compress_pool_kernel( const half* __restrict__ ckv, const float* __restrict__ cring_kv, const float* __restrict__ cring_sc, const half* __restrict__ norm_w,
	half* __restrict__ latent_out, half* __restrict__ comp_kv, const float2* __restrict__ rope, int ratio, const pod::BatchRows* __restrict__ rows,
	size_t cring_slot_stride, size_t comp_slot_stride, float eps ) {
	const int r = blockIdx.x, t = threadIdx.x, d0 = t * 4;
	const int pos = rows->pos[r];
	if ( ( pos + 1 ) % ratio != 0 ) return;
	float v[4];
	if ( ratio == 1 ) {
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) v[i] = __half2float( ckv[static_cast<size_t>( r ) * HD + d0 + i] );
	}
	else {
		const size_t sbase = static_cast<size_t>( rows->slot[r] ) * cring_slot_stride;
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			float m = -FLT_MAX;
			for ( int j = 0; j < ratio; ++j ) m = fmaxf( m, cring_sc[sbase + static_cast<size_t>( ( pos - j ) % DSV4::RING ) * HD + d0 + i] );
			float num = 0.0f, den = 0.0f;
			for ( int j = ratio - 1; j >= 0; --j ) {   // oldest first, as the reference sums
				const size_t o = sbase + static_cast<size_t>( ( pos - j ) % DSV4::RING ) * HD + d0 + i;
				const float e = expf( cring_sc[o] - m );
				den += e;
				num += e * cring_kv[o];
			}
			v[i] = __half2float( __float2half_rn( num / den ) );   // the reference casts the pooled latent to the activation dtype before the norm
		}
	}
	const float ss = block_sum<128>( v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3] );
	const float inv = rsqrtf( ss / HD + eps );
	#pragma unroll
	for ( int i = 0; i < 4; ++i ) v[i] = __half2float( __float2half_rn( v[i] * inv * __half2float( norm_w[d0 + i] ) ) );
	#pragma unroll
	for ( int i = 0; i < 4; ++i ) latent_out[static_cast<size_t>( r ) * HD + d0 + i] = __float2half_rn( v[i] );
	const int gpos = pos + 1 - ratio;   // the group's first position
	if ( d0 >= ROPE0 ) {
		const float2* cs = rope + static_cast<size_t>( gpos ) * ( ROPE / 2 ) + ( d0 - ROPE0 ) / 2;
		rot( v[0], v[1], cs[0], false );
		rot( v[2], v[3], cs[1], false );
	}
	float amax = fmaxf( fmaxf( fabsf( v[0] ), fabsf( v[1] ) ), fmaxf( fabsf( v[2] ), fabsf( v[3] ) ) );
	amax = group_max<4>( amax );
	const float s = e2m1_scale_e4m3( amax );
	half* dst = comp_kv + static_cast<size_t>( rows->slot[r] ) * comp_slot_stride + static_cast<size_t>( pos / ratio ) * HD + d0;
	#pragma unroll
	for ( int i = 0; i < 4; ++i ) dst[i] = __float2half_rn( q_e2m1( v[i], s ) );
}

// 128 threads, one dim each; a warp is one 32-block
__global__ __launch_bounds__( 128 )
void index_key_kernel( const half* __restrict__ kraw, const half* __restrict__ knorm_w, half* __restrict__ idx_k, const float2* __restrict__ rope, int ratio,
	const pod::BatchRows* __restrict__ rows, size_t idx_slot_stride, float eps ) {
	constexpr int D = DSV4::IDX_DIM;
	const int r = blockIdx.x, d = threadIdx.x;
	const int pos = rows->pos[r];
	if ( ( pos + 1 ) % ratio != 0 ) return;
	__shared__ float xs[D];
	float x = __half2float( kraw[static_cast<size_t>( r ) * D + d] );
	const float ss = block_sum<128>( x * x );
	x = __half2float( __float2half_rn( x * rsqrtf( ss / D + eps ) * __half2float( knorm_w[d] ) ) );
	xs[d] = x;
	__syncthreads();
	if ( d >= D - ROPE ) {
		const float2 cs = rope[static_cast<size_t>( pos + 1 - ratio ) * ( ROPE / 2 ) + ( d - ( D - ROPE ) ) / 2];
		float a = xs[d & ~1], b = xs[d | 1];
		rot( a, b, cs, false );
		x = ( d & 1 ) ? b : a;
	}
	const float s = e2m1_scale_pow2( group_max<32>( fabsf( x ) ) );
	idx_k[static_cast<size_t>( rows->slot[r] ) * idx_slot_stride + static_cast<size_t>( pos / ratio ) * D + d] = __float2half_rn( q_e2m1( x, s ) );
}

// ---- indexer ----------------------------------------------------------------------------------------------------------------------------

constexpr int IDX_TILE = 64;   // keys per block

// grid (rows, tiles), 256 threads. q is roped and put on the E2M1 grid per block (cheap: 4096 values), transposed in shared memory so
// lane h reads q[d][h] without bank conflicts; each warp scores 8 keys, lane = head
__global__ __launch_bounds__( 256 )
void index_score_kernel( const half* __restrict__ q, const half* __restrict__ w, const half* __restrict__ idx_k, float* __restrict__ scores, const uint8_t* __restrict__ cand,
	int cand_block, const float2* __restrict__ rope, int ratio, int cap, int row0, const pod::BatchRows* __restrict__ rows, size_t idx_slot_stride ) {
	constexpr int NH = DSV4::IDX_HEADS, D = DSV4::IDX_DIM;
	const int r = row0 + blockIdx.x, t0 = blockIdx.y * IDX_TILE;
	const int pos = rows->pos[r];
	const int n = ( pos + 1 ) / ratio;
	if ( n <= DSV4::TOPK || t0 >= n ) return;   // every entry is kept, or the tile is past the visible ones
	if ( cand != nullptr ) {   // after the candidate source: a tile whose blocks were all dropped is never read
		const uint8_t* cm = cand + static_cast<size_t>( r ) * ( cap / cand_block );
		bool any = false;
		for ( int b = t0 / cand_block; b < ( t0 + IDX_TILE ) / cand_block && b * cand_block < n; ++b ) any = any || cm[b];
		if ( !any ) return;
	}
	__shared__ float qt[D][NH];
	__shared__ half kt[IDX_TILE][D];
	__shared__ float ws[NH];
	const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
	// q: 4096 values, 16 per thread; a 32-block (one head's 32 dims) is two threads' halves... use one warp per head pass
	for ( int h = warp; h < NH; h += 8 ) {
		float x[4];
		const half* src = q + static_cast<size_t>( r ) * NH * D + h * D + lane * 4;
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) x[i] = __half2float( src[i] );
		const int d0 = lane * 4;
		if ( d0 >= D - ROPE ) {
			const float2* cs = rope + static_cast<size_t>( pos ) * ( ROPE / 2 ) + ( d0 - ( D - ROPE ) ) / 2;
			rot( x[0], x[1], cs[0], false );
			rot( x[2], x[3], cs[1], false );
		}
		float amax = fmaxf( fmaxf( fabsf( x[0] ), fabsf( x[1] ) ), fmaxf( fabsf( x[2] ), fabsf( x[3] ) ) );
		amax = group_max<8>( amax );   // 32 dims = 8 lanes
		const float s = e2m1_scale_pow2( amax );
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) qt[d0 + i][h] = __half2float( __float2half_rn( q_e2m1( x[i], s ) ) );
	}
	if ( tid < NH ) ws[tid] = __half2float( w[static_cast<size_t>( r ) * NH + tid] ) * rsqrtf( static_cast<float>( D ) * NH );
	const half* kbase = idx_k + static_cast<size_t>( rows->slot[r] ) * idx_slot_stride;
	for ( int i = tid; i < IDX_TILE * D / 2; i += 256 ) {
		const int k = i / ( D / 2 ), c = i % ( D / 2 );
		const int t = t0 + k;
		reinterpret_cast<half2*>( &kt[k][0] )[c] = t < n ? reinterpret_cast<const half2*>( kbase + static_cast<size_t>( t ) * D )[c] : __float2half2_rn( 0.0f );
	}
	__syncthreads();
	for ( int k = warp; k < IDX_TILE; k += 8 ) {
		const int t = t0 + k;
		if ( t >= n ) break;
		float dot = 0.0f;
		#pragma unroll 8
		for ( int d = 0; d < D; ++d ) dot = fmaf( qt[d][lane], __half2float( kt[k][d] ), dot );
		float v = fmaxf( dot, 0.0f ) * ws[lane];
		v = engine::math::warp_sum( v );
		if ( lane == 0 ) scores[static_cast<size_t>( blockIdx.x ) * cap + t] = v;
	}
}

// ---- indexer scores on tensor cores --------------------------------------------------------------------------------------------------
// Block = 2 consecutive rows of one slot x a span of IDX_TC_SPAN keys, 8 warps. The rows' queries (rope, E2M1 grid) are prepared once, then the span is
// walked in tiles of 32 keys. Warp = (row, 16-key half, 16-head half): a 16 x 16 x 128 product on wmma m16n16k16 (fp16 operands, fp32 accumulate: the
// products are exact as in the scalar kernel, only the sum order differs). The two head halves of a (row, key) meet in shared memory for the ReLU, the
// head weight and the head sum. The key tile is read once for both rows.
constexpr int IDX_TC_KEYS = 32;     // keys per tile
constexpr int IDX_TC_SPAN = 1024;   // keys per block
constexpr int IDX_TC_LD = DSV4::IDX_DIM + 8;   // shared-memory row stride in halves: 272 bytes, so the fragment loads of 16 rows hit different banks

__global__ __launch_bounds__( 256 )
void index_score_tc_kernel( const half* __restrict__ q, const half* __restrict__ w, const half* __restrict__ idx_k, float* __restrict__ scores, const uint8_t* __restrict__ cand,
	int cand_block, const float2* __restrict__ rope, int ratio, int cap, int row0, int nr, const pod::BatchRows* __restrict__ rows, size_t idx_slot_stride ) {
	constexpr int NH = DSV4::IDX_HEADS, D = DSV4::IDX_DIM;
	const int lr0 = blockIdx.x * 2, span0 = blockIdx.y * IDX_TC_SPAN;
	int pos[2] = { 0, 0 }, n[2] = { 0, 0 };
	bool live[2];
	int nmax = 0;
	#pragma unroll
	for ( int i = 0; i < 2; ++i ) {
		const bool valid = lr0 + i < nr;
		if ( valid ) { pos[i] = rows->pos[row0 + lr0 + i]; n[i] = ( pos[i] + 1 ) / ratio; }
		live[i] = valid && n[i] > DSV4::TOPK && span0 < n[i];   // not a dense row, and the span starts within the visible entries
		if ( live[i] ) nmax = max( nmax, n[i] );
	}
	if ( !live[0] && !live[1] ) return;
	const int span_end = min( span0 + IDX_TC_SPAN, nmax );
	__shared__ __align__( 32 ) half sq[2][NH * IDX_TC_LD];
	__shared__ __align__( 32 ) half kt[IDX_TC_KEYS * IDX_TC_LD];
	__shared__ __align__( 32 ) float sl[8][16 * 16];
	__shared__ float ws[2][NH];
	const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
	// q of both rows: rope on the last ROPE dims, then the E2M1 grid (a power-of-two scale per 32 dims), as halves
	#pragma unroll
	for ( int i = 0; i < 2; ++i ) {
		for ( int h = warp; h < NH; h += 8 ) {
			float x[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			const int d0 = lane * 4;
			if ( live[i] ) {
				const half* src = q + ( static_cast<size_t>( row0 + lr0 + i ) * NH + h ) * D + d0;
				#pragma unroll
				for ( int k = 0; k < 4; ++k ) x[k] = __half2float( src[k] );
				if ( d0 >= D - ROPE ) {
					const float2* cs = rope + static_cast<size_t>( pos[i] ) * ( ROPE / 2 ) + ( d0 - ( D - ROPE ) ) / 2;
					rot( x[0], x[1], cs[0], false );
					rot( x[2], x[3], cs[1], false );
				}
			}
			float amax = fmaxf( fmaxf( fabsf( x[0] ), fabsf( x[1] ) ), fmaxf( fabsf( x[2] ), fabsf( x[3] ) ) );
			amax = group_max<8>( amax );   // 32 dims = 8 lanes
			const float s = e2m1_scale_pow2( amax );
			#pragma unroll
			for ( int k = 0; k < 4; ++k ) sq[i][h * IDX_TC_LD + d0 + k] = __float2half_rn( q_e2m1( x[k], s ) );
		}
	}
	if ( tid < 2 * NH ) {
		const int i = tid / NH, h = tid % NH;
		ws[i][h] = live[i] ? __half2float( w[static_cast<size_t>( row0 + lr0 + i ) * NH + h] ) * rsqrtf( static_cast<float>( D ) * NH ) : 0.0f;
	}
	const half* kbase = idx_k + static_cast<size_t>( rows->slot[row0 + lr0] ) * idx_slot_stride;   // both rows are of one slot
	const int ri = warp >> 2, nt = ( warp >> 1 ) & 1, slab = warp & 1;
	for ( int t0 = span0; t0 < span_end; t0 += IDX_TC_KEYS ) {
		if ( cand != nullptr ) {   // after the candidate source: a tile whose blocks were all dropped is never read
			bool any = false;
			#pragma unroll
			for ( int i = 0; i < 2; ++i ) {
				if ( !live[i] ) continue;
				const uint8_t* cm = cand + static_cast<size_t>( row0 + lr0 + i ) * ( cap / cand_block );
				for ( int b = t0 / cand_block; b < ( t0 + IDX_TC_KEYS ) / cand_block && b * cand_block < n[i]; ++b ) any = any || cm[b];
			}
			if ( !any ) continue;
		}
		__syncthreads();   // the previous tile's shared memory is free (and the first iteration waits for the query)
		for ( int i = tid; i < IDX_TC_KEYS * D / 2; i += 256 ) {
			const int k = i / ( D / 2 ), c = i % ( D / 2 );
			const int t = t0 + k;
			reinterpret_cast<half2*>( kt )[k * ( IDX_TC_LD / 2 ) + c] = t < nmax ? reinterpret_cast<const half2*>( kbase + static_cast<size_t>( t ) * D )[c] : __float2half2_rn( 0.0f );
		}
		__syncthreads();
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
		wmma::fill_fragment( acc, 0.0f );
		#pragma unroll
		for ( int k = 0; k < D; k += 16 ) {
			wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
			wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
			wmma::load_matrix_sync( fb, kt + nt * 16 * IDX_TC_LD + k, IDX_TC_LD );
			wmma::load_matrix_sync( fa, sq[ri] + slab * 16 * IDX_TC_LD + k, IDX_TC_LD );
			wmma::mma_sync( acc, fa, fb, acc );
		}
		wmma::store_matrix_sync( sl[warp], acc, 16, wmma::mem_row_major );
		__syncthreads();
		if ( tid < 64 ) {
			const int i = tid >> 5, tt = ( tid >> 4 ) & 1, kk = tid & 15;
			const int wa = i * 4 + tt * 2;   // the two head halves of this (row, key half)
			float v = 0.0f;
			#pragma unroll
			for ( int h = 0; h < 16; ++h ) {
				v = fmaf( fmaxf( sl[wa][h * 16 + kk], 0.0f ), ws[i][h], v );
				v = fmaf( fmaxf( sl[wa + 1][h * 16 + kk], 0.0f ), ws[i][16 + h], v );
			}
			const int t = t0 + tt * 16 + kk;
			if ( live[i] && t < n[i] ) scores[static_cast<size_t>( lr0 + i ) * cap + t] = v;
		}
	}
}

__device__ __forceinline__ uint32_t f2key( float f ) {
	const uint32_t u = __float_as_uint( f );
	return ( u & 0x80000000u ) ? ~u : ( u | 0x80000000u );
}

// The k-th largest key among keys(i), i < n, by an 8-bit MSB-first radix select (1024 threads). Returns the key and, in *above, how many keys exceed it
template <typename KeyFn>
__device__ uint32_t radix_kth( int n, int k, KeyFn key, int* above ) {
	__shared__ int hist[256];
	__shared__ uint32_t s_prefix;
	__shared__ int s_need;
	if ( threadIdx.x == 0 ) { s_prefix = 0; s_need = k; }
	uint32_t mask = 0;
	for ( int shift = 24; shift >= 0; shift -= 8 ) {
		for ( int i = threadIdx.x; i < 256; i += blockDim.x ) hist[i] = 0;
		__syncthreads();
		const uint32_t prefix = s_prefix;
		for ( int i = threadIdx.x; i < n; i += blockDim.x ) {
			const uint32_t kv = key( i );
			if ( ( kv & mask ) == prefix ) atomicAdd( &hist[( kv >> shift ) & 0xff], 1 );
		}
		__syncthreads();
		if ( threadIdx.x == 0 ) {
			int need = s_need;
			for ( int b = 255; b >= 0; --b ) {
				if ( hist[b] >= need ) { s_prefix = prefix | ( static_cast<uint32_t>( b ) << shift ); break; }
				need -= hist[b];
			}
			s_need = need;
		}
		__syncthreads();
		mask |= 0xffu << shift;
	}
	if ( above ) *above = k - s_need;
	return s_prefix;
}

// Exclusive prefix sum of v over the block's threads in thread order (blockDim.x a multiple of 32, at most 1024); *total gets the block's sum.
// Every thread of the block calls it
__device__ int block_excl_scan( int v, int* total ) {
	__shared__ int warp_sum[32];
	const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
	int x = v;
	for ( int o = 1; o < 32; o <<= 1 ) {
		const int y = __shfl_up_sync( 0xffffffff, x, o );
		if ( lane >= o ) x += y;
	}
	if ( lane == 31 ) warp_sum[warp] = x;
	__syncthreads();
	if ( warp == 0 ) {
		int w = lane < nw ? warp_sum[lane] : 0;
		for ( int o = 1; o < 32; o <<= 1 ) {
			const int y = __shfl_up_sync( 0xffffffff, w, o );
			if ( lane >= o ) w += y;
		}
		if ( lane < nw ) warp_sum[lane] = w;
	}
	__syncthreads();
	const int base = warp > 0 ? warp_sum[warp - 1] : 0;
	*total = warp_sum[nw - 1];
	__syncthreads();   // warp_sum is reused by the next call
	return base + x - v;
}

// One block (1024 threads) per row. cand_mode 1: mark the best cand_blocks blocks (by their best score; the newest block always) in cand[row];
// cand_mode 2: entries outside cand[row] do not compete. Then the TOPK best entries into sel[row]
__global__ __launch_bounds__( 1024 )
void index_select_kernel( float* __restrict__ scores, uint8_t* __restrict__ cand, int* __restrict__ sel, int* __restrict__ nsel, int ratio, int cap, int cand_mode,
	int cand_blocks, int cand_block, int row0, const pod::BatchRows* __restrict__ rows ) {
	const int r = row0 + blockIdx.x;
	const int pos = rows->pos[r];
	const int n = ( pos + 1 ) / ratio;
	int* out = sel + static_cast<size_t>( r ) * DSV4::TOPK;
	const int nblk_cap = cap / cand_block;
	uint8_t* cm = cand ? cand + static_cast<size_t>( r ) * nblk_cap : nullptr;
	const int nb = ( n + cand_block - 1 ) / cand_block;
	if ( n <= DSV4::TOPK ) {
		for ( int i = threadIdx.x; i < n; i += blockDim.x ) out[i] = i;
		if ( threadIdx.x == 0 ) nsel[r] = n;
		if ( cand_mode == 1 ) for ( int b = threadIdx.x; b < nb; b += blockDim.x ) cm[b] = 1;
		return;
	}
	float* sc = scores + static_cast<size_t>( blockIdx.x ) * cap;
	if ( cand_mode == 1 ) {
		if ( nb <= cand_blocks ) {
			for ( int b = threadIdx.x; b < nb; b += blockDim.x ) cm[b] = 1;
		}
		else {
			auto bkey = [&]( int b ) -> uint32_t {
				if ( b == nb - 1 ) return 0xffffffffu;   // the newest block is pinned
				float m = -FLT_MAX;
				const int e = min( n, ( b + 1 ) * cand_block );
				for ( int i = b * cand_block; i < e; ++i ) m = fmaxf( m, sc[i] );
				return f2key( m );
			};
			int above = 0;
			const uint32_t th = radix_kth( nb, cand_blocks, bkey, &above );
			// blocks tied at the threshold: the lowest indices fill the remaining places (in block order, so every run keeps the same ones)
			int ties_seen = 0;
			for ( int b0 = 0; b0 < nb; b0 += blockDim.x ) {
				const int b = b0 + threadIdx.x;
				const uint32_t k = b < nb ? bkey( b ) : 0u;
				const int tie = b < nb && k == th;
				int tie_total;
				const int rank = ties_seen + block_excl_scan( tie, &tie_total );
				if ( b < nb ) cm[b] = ( k > th || ( tie && rank < cand_blocks - above ) ) ? 1 : 0;
				ties_seen += tie_total;
			}
			__syncthreads();
		}
	}
	auto ekey = [&]( int i ) -> uint32_t {
		if ( cand_mode == 2 && !cm[i / cand_block] ) return 0u;
		return f2key( sc[i] );
	};
	int above = 0;
	const uint32_t th = radix_kth( n, DSV4::TOPK, ekey, &above );
	// the selection in ascending entry order, ties at the threshold to the lowest entries: the same list every run, so the attention sums
	// over it in the same order
	int cnt = 0, ties_seen = 0;
	for ( int i0 = 0; i0 < n; i0 += blockDim.x ) {
		const int i = i0 + threadIdx.x;
		const uint32_t k = i < n ? ekey( i ) : 0u;
		const int tie = k != 0u && k == th;
		int tie_total, keep_total;
		const int rank = ties_seen + block_excl_scan( tie, &tie_total );
		const int keep = k != 0u && ( k > th || ( tie && rank < DSV4::TOPK - above ) );
		const int at = cnt + block_excl_scan( keep, &keep_total );
		if ( keep ) out[at] = i;
		cnt += keep_total;
		ties_seen += tie_total;
	}
	if ( threadIdx.x == 0 ) nsel[r] = cnt;
}

// ---- attention ----------------------------------------------------------------------------------------------------------------------------

constexpr int ATT_TILE = 16;   // keys per shared-memory tile
constexpr int ATT_THREADS = 256;

// One block per row, 8 warps; a warp owns nh / 8 heads. Lane l holds dims {2 l + 64 i, 2 l + 1 + 64 i : i < 8} (half2 index l + 32 i), so the
// rotary pairs (half2 indices 224..255) sit in i = 7 and a 32-dim scale block is 16 lanes of one i
template <int HPW>
__global__ __launch_bounds__( ATT_THREADS )
void attend_kernel( const half* __restrict__ q, const half* __restrict__ ring, const half* __restrict__ comp_kv, const int* __restrict__ sel, const int* __restrict__ nsel,
	const float* __restrict__ sink, const float2* __restrict__ rope, half* __restrict__ o, int nh, int layer, int rows_n, const pod::BatchRows* __restrict__ rows,
	size_t ring_slot_stride, size_t comp_slot_stride, float* __restrict__ ws, int win_floor ) {
	extern __shared__ half2 att_smem[];
	half2* qs = att_smem;                       // [nh][HD / 2]
	half2* ks = att_smem + nh * ( HD / 2 );     // [ATT_TILE][HD / 2]
	const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
	const int pos = rows->pos[r];
	const float2* cs_row = rope + static_cast<size_t>( pos ) * ( ROPE / 2 );
	// q, roped
	for ( int i = tid; i < nh * ( HD / 2 ); i += ATT_THREADS ) {
		const int h = i / ( HD / 2 ), c = i % ( HD / 2 );
		float2 v = __half22float2( reinterpret_cast<const half2*>( q + ( static_cast<size_t>( r ) * nh + h ) * HD )[c] );
		if ( c * 2 >= ROPE0 ) rot( v.x, v.y, cs_row[c - ROPE0 / 2], false );
		qs[i] = __floats2half2_rn( v.x, v.y );
	}
	const half* wring = ring + static_cast<size_t>( rows->slot[r] ) * ring_slot_stride + static_cast<size_t>( layer ) * DSV4::RING * HD;
	const half* cbase = comp_kv ? comp_kv + static_cast<size_t>( rows->slot[r] ) * comp_slot_stride : nullptr;
	const int w0 = max( max( 0, win_floor ), pos - DSV4::WINDOW + 1 );
	const int n_win = pos - w0 + 1;
	const int n_cmp = comp_kv ? nsel[r] : 0;
	const int* srow = comp_kv ? sel + static_cast<size_t>( r ) * DSV4::TOPK : nullptr;
	const int total = n_win + n_cmp;
	const float scale = rsqrtf( static_cast<float>( HD ) );
	// split-KV (ws != null, decode): this block takes keys [kb, ke) and leaves a partial; att_combine_kernel merges the splits
	const int nsplit = gridDim.y;
	int kb = 0, ke = total;
	if ( ws != nullptr ) {
		const int chunk = ( ( total + nsplit - 1 ) / nsplit + ATT_TILE - 1 ) / ATT_TILE * ATT_TILE;
		kb = min( total, static_cast<int>( blockIdx.y ) * chunk);
		ke = min( total, kb + chunk );
	}

	float m[HPW], l[HPW];
	float2 acc[HPW][8];
	#pragma unroll
	for ( int j = 0; j < HPW; ++j ) {
		m[j] = -1e30f;
		l[j] = 0.0f;
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) acc[j][i] = make_float2( 0.0f, 0.0f );
	}
	half2 hq[HPW][8];
	bool q_loaded = false;
	for ( int k0 = kb; k0 < ke; k0 += ATT_TILE ) {
		__syncthreads();
		const int nk = min( ATT_TILE, ke - k0 );
		for ( int i = tid; i < nk * ( HD / 8 ); i += ATT_THREADS ) {   // 16-byte loads
			const int k = i / ( HD / 8 ), c = i % ( HD / 8 );
			const int e = k0 + k;
			const half* src = e < n_win ? wring + static_cast<size_t>( ( w0 + e ) % DSV4::RING ) * HD : cbase + static_cast<size_t>( srow[e - n_win] ) * HD;
			reinterpret_cast<int4*>( ks + k * ( HD / 2 ) )[c] = reinterpret_cast<const int4*>( src )[c];
		}
		__syncthreads();
		if ( !q_loaded ) {   // the warp's heads' query dims in registers once q is in shared memory
			#pragma unroll
			for ( int j = 0; j < HPW; ++j )
				#pragma unroll
				for ( int i = 0; i < 8; ++i ) hq[j][i] = qs[( warp * HPW + j ) * ( HD / 2 ) + lane + 32 * i];
			q_loaded = true;
		}
		// keys in pairs: both dot products and their warp sums in flight together, then the online-softmax steps of the two in key order
		for ( int k = 0; k < nk; k += 2 ) {
			const bool two = k + 1 < nk;
			float2 kv0[8], kv1[8];
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) {
				kv0[i] = __half22float2( ks[k * ( HD / 2 ) + lane + 32 * i] );
				kv1[i] = two ? __half22float2( ks[( k + 1 ) * ( HD / 2 ) + lane + 32 * i] ) : make_float2( 0.0f, 0.0f );
			}
			#pragma unroll
			for ( int j = 0; j < HPW; ++j ) {
				float d0 = 0.0f, d1 = 0.0f;
				#pragma unroll
				for ( int i = 0; i < 8; ++i ) {
					const float2 qv = __half22float2( hq[j][i] );
					d0 = fmaf( qv.x, kv0[i].x, fmaf( qv.y, kv0[i].y, d0 ) );
					d1 = fmaf( qv.x, kv1[i].x, fmaf( qv.y, kv1[i].y, d1 ) );
				}
				d0 = engine::math::warp_sum( d0 ) * scale;
				d1 = engine::math::warp_sum( d1 ) * scale;
				{
					const float mn = fmaxf( m[j], d0 );
					const float corr = expf( m[j] - mn ), p = expf( d0 - mn );
					l[j] = l[j] * corr + p;
					#pragma unroll
					for ( int i = 0; i < 8; ++i ) {
						acc[j][i].x = acc[j][i].x * corr + p * kv0[i].x;
						acc[j][i].y = acc[j][i].y * corr + p * kv0[i].y;
					}
					m[j] = mn;
				}
				if ( two ) {
					const float mn = fmaxf( m[j], d1 );
					const float corr = expf( m[j] - mn ), p = expf( d1 - mn );
					l[j] = l[j] * corr + p;
					#pragma unroll
					for ( int i = 0; i < 8; ++i ) {
						acc[j][i].x = acc[j][i].x * corr + p * kv1[i].x;
						acc[j][i].y = acc[j][i].y * corr + p * kv1[i].y;
					}
					m[j] = mn;
				}
			}
		}
	}
	if ( ws != nullptr ) {
		#pragma unroll
		for ( int j = 0; j < HPW; ++j ) {
			const int h = warp * HPW + j;
			float* p = ws + ( ( static_cast<size_t>( r ) * nsplit + blockIdx.y ) * nh + h ) * ( HD + 2 );
			float2* pa = reinterpret_cast<float2*>( p + 2 );
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) pa[lane + 32 * i] = acc[j][i];
			if ( lane == 0 ) { p[0] = m[j]; p[1] = l[j]; }
		}
		return;
	}
	const int G = DSV4::GROUP_HEADS;
	#pragma unroll
	for ( int j = 0; j < HPW; ++j ) {
		const int h = warp * HPW + j;
		const float mn = fmaxf( m[j], sink[h] );
		const float corr = expf( m[j] - mn );
		const float den = l[j] * corr + expf( sink[h] - mn );
		const float f = corr / den;
		float2 v[8];
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) v[i] = make_float2( acc[j][i].x * f, acc[j][i].y * f );
		// the output in the activation dtype, then the inverse rotation, then the E4M3 grid
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) v[i] = __half22float2( __floats2half2_rn( v[i].x, v[i].y ) );
		rot( v[7].x, v[7].y, cs_row[lane], true );
		half* dst = o + ( static_cast<size_t>( h / G ) * rows_n + r ) * ( G * HD ) + static_cast<size_t>( h % G ) * HD;
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) {
			const float amax = group_max<16>( fmaxf( fabsf( v[i].x ), fabsf( v[i].y ) ) );
			const float s = e4m3_scale_pow2( amax );
			reinterpret_cast<half2*>( dst )[lane + 32 * i] = __floats2half2_rn( q_e4m3( v[i].x, s ), q_e4m3( v[i].y, s ) );
		}
	}
}

// Merges the split-KV partials of one head of a row (grid rows x heads, 8 warps) and writes its output: every warp takes the row's maximum and
// denominator over the splits (lane = split), the warps sum the weighted partials of every eighth split (lane layout as attend_kernel), and
// warp i adds the eight sums of dims i in warp order and applies the rotary and FP8 epilogue to them
__global__ __launch_bounds__( ATT_THREADS )
void att_combine_kernel( const float* __restrict__ ws, int nsplit, const float* __restrict__ sink, const float2* __restrict__ rope, half* __restrict__ o, int nh,
	int rows_n, const pod::BatchRows* __restrict__ rows ) {
	const int r = blockIdx.x, h = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
	__shared__ float2 part[ATT_THREADS / 32][HD / 2];
	const size_t sstride = static_cast<size_t>( nh ) * ( HD + 2 );   // one split to the next
	const float* base = ws + ( static_cast<size_t>( r ) * nsplit * nh + h ) * ( HD + 2 );
	// every global read the epilogue needs up front: the splits' (m, l) and warp 7's rotary pairs
	const float2 ml = lane < nsplit ? *reinterpret_cast<const float2*>( base + lane * sstride ) : make_float2( -INFINITY, 0.0f );
	const float2 cs = warp == 7 ? rope[static_cast<size_t>( rows->pos[r] ) * ( ROPE / 2 ) + lane] : make_float2( 0.0f, 0.0f );
	const float m_l = ml.x;
	float m = m_l;
	#pragma unroll
	for ( int off = 16; off > 0; off >>= 1 ) m = fmaxf( m, __shfl_xor_sync( 0xffffffff, m, off ) );
	const float M = fmaxf( m, sink[h] );
	const float w_l = lane < nsplit ? expf( m_l - M ) : 0.0f;
	float2 v[8];
	#pragma unroll
	for ( int i = 0; i < 8; ++i ) v[i] = make_float2( 0.0f, 0.0f );
	for ( int sp = warp; sp < nsplit; sp += ATT_THREADS / 32 ) {
		const float w = __shfl_sync( 0xffffffff, w_l, sp );
		const float2* pa = reinterpret_cast<const float2*>( base + sp * sstride + 2 );
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) {
			const float2 a = pa[lane + 32 * i];
			v[i].x += a.x * w;
			v[i].y += a.y * w;
		}
	}
	#pragma unroll
	for ( int i = 0; i < 8; ++i ) part[warp][lane + 32 * i] = v[i];
	__syncthreads();
	// the epilogue, warp i the dims of i (the eight sums in warp order, the rotary pairs in i = 7, the E4M3 grid per 32 dims): every warp
	// takes the same denominator
	float den = lane < nsplit ? ml.y * w_l : 0.0f;
	#pragma unroll
	for ( int off = 16; off > 0; off >>= 1 ) den += __shfl_xor_sync( 0xffffffff, den, off );
	den += expf( sink[h] - M );
	const float f = 1.0f / den;
	const int i = warp;
	float2 t = part[0][lane + 32 * i];
	#pragma unroll
	for ( int w = 1; w < ATT_THREADS / 32; ++w ) { t.x += part[w][lane + 32 * i].x; t.y += part[w][lane + 32 * i].y; }
	float2 vi = __half22float2( __floats2half2_rn( t.x * f, t.y * f ) );
	if ( i == 7 ) rot( vi.x, vi.y, cs, true );
	const int G = DSV4::GROUP_HEADS;
	half* dst = o + ( static_cast<size_t>( h / G ) * rows_n + r ) * ( G * HD ) + static_cast<size_t>( h % G ) * HD;
	const float amax = group_max<16>( fmaxf( fabsf( vi.x ), fabsf( vi.y ) ) );
	const float sc = e4m3_scale_pow2( amax );
	reinterpret_cast<half2*>( dst )[lane + 32 * i] = __floats2half2_rn( q_e4m3( vi.x, sc ), q_e4m3( vi.y, sc ) );
}

// Split-KV attention on the tensor cores (mma.m8n8k4), the partials of attend_kernel for att_combine_kernel: a block takes one row, 16 of its
// heads as the queries and its split's share of the row's keys in tiles of 32 (the window, then the selected compressed entries, or for a
// drafter block its own keys). The tile layout follows latent_sparse_tc_kernel (mla_latent.cu): a tile's keys sit in registers, stored as
// K for the scores (each warp 16 x 16 over 128 of the dims), then as V^T for the output (each warp 64 dims); the softmax takes 16 lanes a
// head. The split's entry indices are staged in shared memory first and the next tile is fetched while this one is scored, so a tile waits
// on no global load
constexpr int TC_BK = 32, TC_KS = HD + 4, TC_VS = TC_BK + 4, TC_SS = TC_BK + 4;
constexpr int TC_OFF_Q = HD * TC_VS * 2;   // V^T is the larger use of the K / V buffer
constexpr int TC_OFF_S = TC_OFF_Q + 16 * TC_KS * 2;
constexpr int TC_OFF_P = TC_OFF_S + 4 * 16 * TC_SS * 4;
constexpr int TC_OFF_A = TC_OFF_P + 16 * TC_VS * 2;
constexpr int TC_OFF_I = TC_OFF_A + 16 * 12;
constexpr int TC_SMEM = TC_OFF_I + ( DSV4::WINDOW + DSV4::TOPK ) * 4;
static_assert( TC_BK * TC_KS * 2 <= TC_OFF_Q && 16 * TC_KS * 4 <= TC_OFF_Q, "K tile and output fit the K / V buffer" );

template <bool DSPARK>
__global__ __launch_bounds__( 256 )
void attend_tc_kernel( const half* __restrict__ q, const half* __restrict__ ring, const half* __restrict__ comp_kv, const int* __restrict__ sel,
	const int* __restrict__ nsel, const float2* __restrict__ rope, int nh, int layer, const pod::BatchRows* __restrict__ rows, size_t ring_slot_stride,
	size_t comp_slot_stride, float* __restrict__ ws, int win_floor, int dslot, int dw0, int dt, int rows_n ) {
	extern __shared__ __align__( 16 ) unsigned char tc_smem[];
	half* sB = reinterpret_cast<half*>( tc_smem );
	half* sQ = reinterpret_cast<half*>( tc_smem + TC_OFF_Q );
	float* sS = reinterpret_cast<float*>( tc_smem + TC_OFF_S );
	half* sP = reinterpret_cast<half*>( tc_smem + TC_OFF_P );
	float* sA = reinterpret_cast<float*>( tc_smem + TC_OFF_A );
	float* sM = sA + 16;
	float* sL = sM + 16;
	int* sI = reinterpret_cast<int*>( tc_smem + TC_OFF_I );
	float* sO = reinterpret_cast<float*>( tc_smem );

	const int r = blockIdx.x, h0 = blockIdx.z * 16, nsplit = gridDim.y;
	const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
	const int t = lane & 3, hi = lane >> 4, e = t + 4 * hi, qp = ( lane >> 2 ) & 3;
	const int pos = rows->pos[r];
	const float2* cs_row = rope + static_cast<size_t>( pos ) * ( ROPE / 2 );

	// the row's keys: the window [w0, pos] (a drafter block: [dw0, dt] of slot dslot), then the selected entries (the block's own keys)
	const half* wring = ring + static_cast<size_t>( DSPARK ? dslot : rows->slot[r] ) * ring_slot_stride + static_cast<size_t>( layer ) * DSV4::RING * HD;
	const int w0 = DSPARK ? dw0 : max( max( 0, win_floor ), pos - DSV4::WINDOW + 1 );
	const int n_win = ( DSPARK ? dt : pos ) - w0 + 1;
	const half* cbase = DSPARK || comp_kv == nullptr ? comp_kv : comp_kv + static_cast<size_t>( rows->slot[r] ) * comp_slot_stride;
	const int total = n_win + ( DSPARK ? rows_n : comp_kv ? nsel[r] : 0 );
	const int ntiles = ( total + TC_BK - 1 ) / TC_BK;
	const int kb0 = ( blockIdx.y * ntiles / nsplit ) * TC_BK;
	const int kb1 = min( total, ( ( blockIdx.y + 1 ) * ntiles / nsplit ) * TC_BK );
	for ( int j = max( kb0, n_win ) + tid; j < kb1; j += 256 ) {
		const int c = j - n_win;
		sI[j - kb0] = DSPARK ? rows->pos[c] % DSV4::RING : sel[static_cast<size_t>( r ) * DSV4::TOPK + c];
	}

	// q, roped (as attend_kernel: the rotary pairs through FP32, back to half)
	for ( int u = tid; u < 16 * ( HD / 8 ); u += 256 ) {
		const int qq = u >> 6, c = u & 63;
		uint4 raw = *reinterpret_cast<const uint4*>( q + ( static_cast<size_t>( r ) * nh + h0 + qq ) * HD + c * 8 );
		if ( c * 8 >= ROPE0 ) {
			half2* h2 = reinterpret_cast<half2*>( &raw );
			#pragma unroll
			for ( int i = 0; i < 4; ++i ) {
				float2 v = __half22float2( h2[i] );
				rot( v.x, v.y, cs_row[c * 4 + i - ROPE0 / 2], false );
				h2[i] = __floats2half2_rn( v.x, v.y );
			}
		}
		uint2* dst = reinterpret_cast<uint2*>( sQ + qq * TC_KS + c * 8 );
		dst[0] = make_uint2( raw.x, raw.y );
		dst[1] = make_uint2( raw.z, raw.w );
	}

	const int s_kg = warp & 1, dq = warp >> 1;
	const half* s_arow = sQ + ( 8 * ( qp >> 1 ) + e ) * TC_KS + dq * 128;
	const half* s_brow = sB + ( s_kg * 16 + 8 * ( qp & 1 ) + e ) * TC_KS + dq * 128;
	const half* v_row0 = sB + ( 64 * warp + qp + 4 * e ) * TC_VS;
	const half* p_row = sP + e * TC_VS;
	const float scale = rsqrtf( static_cast<float>( HD ) );

	// softmax role: 16 lanes a head, 2 keys each
	const int hrow = tid >> 4, sub = tid & 15;
	float m_run = -1e30f, l_run = 0.0f;
	float acc[2][2][8];
	#pragma unroll
	for ( int jj = 0; jj < 2; ++jj ) {
		#pragma unroll
		for ( int i = 0; i < 2; ++i ) {
			#pragma unroll
			for ( int c = 0; c < 8; ++c ) acc[jj][i][c] = 0.0f;
		}
	}

	const int st_p = lane & 15, st_cl = lane >> 4;
	auto fetch = [&]( int kb, uint4 ( &pk )[4][2] ) {
		const half* kp[2];
		#pragma unroll
		for ( int kk = 0; kk < 2; ++kk ) {
			const int j = kb + 2 * st_p + kk;
			kp[kk] = j >= kb1 ? nullptr : j < n_win ? wring + static_cast<size_t>( ( w0 + j ) % DSV4::RING ) * HD : cbase + static_cast<size_t>( sI[j - kb0] ) * HD;
		}
		#pragma unroll
		for ( int j = 0; j < 4; ++j ) {
			const int c = 2 * warp + st_cl + 16 * j;
			#pragma unroll
			for ( int kk = 0; kk < 2; ++kk ) pk[j][kk] = kp[kk] ? __ldg( reinterpret_cast<const uint4*>( kp[kk] + c * 8 ) ) : make_uint4( 0, 0, 0, 0 );
		}
	};
	// one tile from pk, the next fetched into pn
	auto tile = [&]( int kb, const uint4 ( &pk )[4][2], uint4 ( &pn )[4][2] ) {
		#pragma unroll
		for ( int j = 0; j < 4; ++j ) {
			const int c = 2 * warp + st_cl + 16 * j;
			#pragma unroll
			for ( int kk = 0; kk < 2; ++kk ) {
				uint2* kd = reinterpret_cast<uint2*>( sB + ( 2 * st_p + kk ) * TC_KS + c * 8 );
				kd[0] = make_uint2( pk[j][kk].x, pk[j][kk].y );
				kd[1] = make_uint2( pk[j][kk].z, pk[j][kk].w );
			}
		}
		__syncthreads();
		if ( kb + TC_BK < kb1 ) fetch( kb + TC_BK, pn );
		{
			// two accumulation chains over the dims
			float c0[8], c1[8];
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) c0[i] = c1[i] = 0.0f;
			#pragma unroll 8
			for ( int ks = 0; ks < 32; ks += 2 ) {
				const uint2 a0 = *reinterpret_cast<const uint2*>( s_arow + 4 * ks ), a1 = *reinterpret_cast<const uint2*>( s_arow + 4 * ks + 4 );
				const uint2 b0 = *reinterpret_cast<const uint2*>( s_brow + 4 * ks ), b1 = *reinterpret_cast<const uint2*>( s_brow + 4 * ks + 4 );
				mma884( c0, a0.x, a0.y, b0.x, b0.y );
				mma884( c1, a1.x, a1.y, b1.x, b1.y );
			}
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) {
				const int m = ( t & 1 ) + 2 * ( ( i >> 1 ) & 1 ) + 4 * hi;
				const int n = ( i & 1 ) + 2 * ( t >> 1 ) + 4 * ( i >> 2 );
				sS[dq * 16 * TC_SS + ( 8 * ( qp >> 1 ) + m ) * TC_SS + s_kg * 16 + 8 * ( qp & 1 ) + n] = c0[i] + c1[i];
			}
		}
		__syncthreads();
		// V^T over the scored K, beside the softmax
		#pragma unroll
		for ( int j = 0; j < 4; ++j ) {
			const int c = 2 * warp + st_cl + 16 * j;
			const half* v0 = reinterpret_cast<const half*>( &pk[j][0] );
			const half* v1 = reinterpret_cast<const half*>( &pk[j][1] );
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) *reinterpret_cast<half2*>( sB + ( c * 8 + i ) * TC_VS + 2 * st_p ) = __halves2half2( v0[i], v1[i] );
		}
		{
			float sv[2];
			float mx = -1e30f;
			#pragma unroll
			for ( int j = 0; j < 2; ++j ) {
				const int k = sub * 2 + j;
				float sum4 = 0.0f;
				#pragma unroll
				for ( int d4 = 0; d4 < 4; ++d4 ) sum4 += sS[d4 * 16 * TC_SS + hrow * TC_SS + k];
				sv[j] = ( kb + k < kb1 ) ? sum4 * scale : -1e30f;
				mx = fmaxf( mx, sv[j] );
			}
			#pragma unroll
			for ( int mask = 1; mask < 16; mask <<= 1 ) mx = fmaxf( mx, __shfl_xor_sync( 0xffffffff, mx, mask ) );
			const float m_new = fmaxf( m_run, mx );
			const float alpha = __expf( m_run - m_new );
			const float p0 = sv[0] > -1e29f ? __expf( sv[0] - m_new ) : 0.0f;
			const float p1 = sv[1] > -1e29f ? __expf( sv[1] - m_new ) : 0.0f;
			float sum = p0 + p1;
			#pragma unroll
			for ( int mask = 1; mask < 16; mask <<= 1 ) sum += __shfl_xor_sync( 0xffffffff, sum, mask );
			l_run = l_run * alpha + sum;
			m_run = m_new;
			*reinterpret_cast<half2*>( sP + hrow * TC_VS + sub * 2 ) = __floats2half2_rn( p0, p1 );
			if ( sub == 0 ) sA[hrow] = alpha;
		}
		__syncthreads();
		#pragma unroll
		for ( int jj = 0; jj < 2; ++jj ) {
			#pragma unroll
			for ( int i = 0; i < 2; ++i ) {
				#pragma unroll
				for ( int c = 0; c < 8; ++c ) acc[jj][i][c] *= sA[8 * i + ( t & 1 ) + 2 * ( ( c >> 1 ) & 1 ) + 4 * hi];
			}
		}
		#pragma unroll
		for ( int kk = 0; kk < TC_BK / 4; ++kk ) {
			uint2 b[2];
			#pragma unroll
			for ( int jj = 0; jj < 2; ++jj ) b[jj] = *reinterpret_cast<const uint2*>( v_row0 + 32 * jj * TC_VS + 4 * kk );
			#pragma unroll
			for ( int i = 0; i < 2; ++i ) {
				const uint2 a = *reinterpret_cast<const uint2*>( p_row + 8 * i * TC_VS + 4 * kk );
				#pragma unroll
				for ( int jj = 0; jj < 2; ++jj ) mma884( acc[jj][i], a.x, a.y, b[jj].x, b[jj].y );
			}
		}
		__syncthreads();
	};
	uint4 pa[4][2], pb[4][2];
	__syncthreads();   // the entry indices
	if ( kb0 < kb1 ) fetch( kb0, pa );
	for ( int kb = kb0; kb < kb1; kb += 2 * TC_BK ) {
		tile( kb, pa, pb );
		if ( kb + TC_BK < kb1 ) tile( kb + TC_BK, pb, pa );
	}

	// the partial: m, l and the unnormalized output in natural dim order (attend_kernel's layout)
	if ( sub == 0 ) {
		sM[hrow] = m_run;
		sL[hrow] = l_run;
	}
	#pragma unroll
	for ( int jj = 0; jj < 2; ++jj ) {
		#pragma unroll
		for ( int i = 0; i < 2; ++i ) {
			#pragma unroll
			for ( int c = 0; c < 8; ++c ) {
				const int m = ( t & 1 ) + 2 * ( ( c >> 1 ) & 1 ) + 4 * hi;
				const int n = ( c & 1 ) + 2 * ( t >> 1 ) + 4 * ( c >> 2 );
				sO[( 8 * i + m ) * TC_KS + 64 * warp + 32 * jj + qp + 4 * n] = acc[jj][i][c];
			}
		}
	}
	__syncthreads();
	for ( int u = tid; u < 16 * ( HD / 2 ); u += 256 ) {
		const int qq = u >> 8, d2 = u & 255;
		float* p = ws + ( ( static_cast<size_t>( r ) * nsplit + blockIdx.y ) * nh + h0 + qq ) * ( HD + 2 );
		if ( d2 == 0 ) {
			p[0] = sM[qq];
			p[1] = sL[qq];
		}
		reinterpret_cast<float2*>( p + 2 )[d2] = *reinterpret_cast<const float2*>( sO + qq * TC_KS + 2 * d2 );
	}
}


// ---- DSpark drafter -----------------------------------------------------------------------------------------------------------------

// ring[slot][pos % WINDOW][tap][hidden] = scale * mean of the row's four streams
__global__ __launch_bounds__( 256 )
void tap_store_kernel( const float* __restrict__ streams, half* __restrict__ ring, int tap, int taps, int hidden, float scale, int floor_pos,
	const pod::BatchRows* __restrict__ rows, size_t ring_slot_stride ) {
	const int r = blockIdx.x;
	const int pos = rows->pos[r];
	if ( pos < floor_pos ) return;
	half* dst = ring + static_cast<size_t>( rows->slot[r] ) * ring_slot_stride + ( static_cast<size_t>( pos % DSV4::WINDOW ) * taps + tap ) * hidden;
	const float* s = streams + static_cast<size_t>( r ) * 4 * hidden;
	for ( int d = threadIdx.x; d < hidden; d += blockDim.x ) dst[d] = __float2half_rn( 0.25f * scale * ( s[d] + s[hidden + d] + s[2 * hidden + d] + s[3 * hidden + d] ) );
}

__global__ __launch_bounds__( 256 )
void tap_gather_kernel( const half* __restrict__ ring, half* __restrict__ out, int slot, int first, int taps_hidden, size_t ring_slot_stride ) {
	const half* src = ring + static_cast<size_t>( slot ) * ring_slot_stride + static_cast<size_t>( ( first + blockIdx.x ) % DSV4::WINDOW ) * taps_hidden;
	half* dst = out + static_cast<size_t>( blockIdx.x ) * taps_hidden;
	for ( int d = threadIdx.x; d < taps_hidden; d += blockDim.x ) dst[d] = src[d];
}

// attend_kernel for a drafter block (see DSV4::dspark_attend): the keys are the ring window [w0, t] followed by the block's own rows_n keys
template <int HPW>
__global__ __launch_bounds__( ATT_THREADS )
void dspark_attend_kernel( const half* __restrict__ q, const half* __restrict__ ring, const half* __restrict__ blk, const float* __restrict__ sink, const float2* __restrict__ rope,
	half* __restrict__ o, int nh, int layer, int rows_n, const pod::BatchRows* __restrict__ rows, int slot, int w0, int t, size_t ring_slot_stride ) {
	extern __shared__ half2 att_smem[];
	half2* qs = att_smem;                       // [nh][HD / 2]
	half2* ks = att_smem + nh * ( HD / 2 );     // [ATT_TILE][HD / 2]
	const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
	const float2* cs_row = rope + static_cast<size_t>( rows->pos[r] ) * ( ROPE / 2 );
	for ( int i = tid; i < nh * ( HD / 2 ); i += ATT_THREADS ) {
		const int h = i / ( HD / 2 ), c = i % ( HD / 2 );
		float2 v = __half22float2( reinterpret_cast<const half2*>( q + ( static_cast<size_t>( r ) * nh + h ) * HD )[c] );
		if ( c * 2 >= ROPE0 ) rot( v.x, v.y, cs_row[c - ROPE0 / 2], false );
		qs[i] = __floats2half2_rn( v.x, v.y );
	}
	const half* wring = ring + static_cast<size_t>( slot ) * ring_slot_stride + static_cast<size_t>( layer ) * DSV4::RING * HD;
	const int n_win = t - w0 + 1;
	const int total = n_win + rows_n;
	const float scale = rsqrtf( static_cast<float>( HD ) );
	float m[HPW], l[HPW];
	float2 acc[HPW][8];
	#pragma unroll
	for ( int j = 0; j < HPW; ++j ) {
		m[j] = -1e30f;
		l[j] = 0.0f;
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) acc[j][i] = make_float2( 0.0f, 0.0f );
	}
	for ( int k0 = 0; k0 < total; k0 += ATT_TILE ) {
		__syncthreads();
		const int nk = min( ATT_TILE, total - k0 );
		for ( int i = tid; i < nk * ( HD / 8 ); i += ATT_THREADS ) {
			const int k = i / ( HD / 8 ), c = i % ( HD / 8 );
			const int e = k0 + k;
			const half* src = e < n_win ? wring + static_cast<size_t>( ( w0 + e ) % DSV4::RING ) * HD : blk + static_cast<size_t>( rows->pos[e - n_win] % DSV4::RING ) * HD;
			reinterpret_cast<int4*>( ks + k * ( HD / 2 ) )[c] = reinterpret_cast<const int4*>( src )[c];
		}
		__syncthreads();
		for ( int k = 0; k < nk; ++k ) {
			float2 kv[8];
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) kv[i] = __half22float2( ks[k * ( HD / 2 ) + lane + 32 * i] );
			#pragma unroll
			for ( int j = 0; j < HPW; ++j ) {
				const int h = warp * HPW + j;
				float dot = 0.0f;
				#pragma unroll
				for ( int i = 0; i < 8; ++i ) {
					const float2 qv = __half22float2( qs[h * ( HD / 2 ) + lane + 32 * i] );
					dot = fmaf( qv.x, kv[i].x, fmaf( qv.y, kv[i].y, dot ) );
				}
				dot = engine::math::warp_sum( dot ) * scale;
				const float mn = fmaxf( m[j], dot );
				const float corr = expf( m[j] - mn ), p = expf( dot - mn );
				l[j] = l[j] * corr + p;
				#pragma unroll
				for ( int i = 0; i < 8; ++i ) {
					acc[j][i].x = acc[j][i].x * corr + p * kv[i].x;
					acc[j][i].y = acc[j][i].y * corr + p * kv[i].y;
				}
				m[j] = mn;
			}
		}
	}
	const int G = DSV4::GROUP_HEADS;
	#pragma unroll
	for ( int j = 0; j < HPW; ++j ) {
		const int h = warp * HPW + j;
		const float mn = fmaxf( m[j], sink[h] );
		const float corr = expf( m[j] - mn );
		const float den = l[j] * corr + expf( sink[h] - mn );
		const float f = corr / den;
		float2 v[8];
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) v[i] = __half22float2( __floats2half2_rn( acc[j][i].x * f, acc[j][i].y * f ) );
		rot( v[7].x, v[7].y, cs_row[lane], true );
		half* dst = o + ( static_cast<size_t>( h / G ) * rows_n + r ) * ( G * HD ) + static_cast<size_t>( h % G ) * HD;
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) {
			const float amax = group_max<16>( fmaxf( fabsf( v[i].x ), fabsf( v[i].y ) ) );
			const float s = e4m3_scale_pow2( amax );
			reinterpret_cast<half2*>( dst )[lane + 32 * i] = __floats2half2_rn( q_e4m3( v[i].x, s ), q_e4m3( v[i].y, s ) );
		}
	}
}

__global__ void concat_groups_kernel( const half* __restrict__ in, half* __restrict__ out, int groups, int n, int rows ) {
	const size_t total = static_cast<size_t>( rows ) * groups * n;
	for ( size_t i = blockIdx.x * static_cast<size_t>( blockDim.x ) + threadIdx.x; i < total; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) {
		const int c = static_cast<int>( i % n );
		const size_t rg = i / n;
		const int g = static_cast<int>( rg % groups ), r = static_cast<int>( rg / groups );
		out[i] = in[( static_cast<size_t>( g ) * rows + r ) * n + c];
	}
}

// ---- routing: partial logits over hidden slices (lane = expert, so the [hidden, E] weight reads coalesce), then one block per row selects -----

constexpr int ROUTE_SPLITS = DSV4::ROUTE_SPLITS;   // 16: 192 blocks of short slices for one row (4 left a long dependent loop on 48)

// grid (rows, ROUTE_SPLITS, octet groups), 8 x kRouteOctets threads: thread (w, o) sums the eight experts of octet o (one 16-byte load of a w_t
// row) over the w-th eighth of the block's hidden slice, each expert in the order of one thread per expert, and the eighths are summed in order
constexpr int kRouteOctets = 4;
__global__ __launch_bounds__( 8 * kRouteOctets )
void route_partial_kernel( const half* __restrict__ x, const half* __restrict__ w_t, float* __restrict__ part, int hidden, int experts ) {
	const int r = blockIdx.x, z = blockIdx.y, ol = threadIdx.x % kRouteOctets, w = threadIdx.x / kRouteOctets, o = blockIdx.z * kRouteOctets + ol;
	const int slice = hidden / ROUTE_SPLITS, per_warp = slice / 8;
	const int d0 = z * slice + w * per_warp;
	const half* xr = x + static_cast<size_t>( r ) * hidden;
	float acc[8] = {};
	if ( o < experts / 8 ) {
		#pragma unroll 
		for ( int d = d0; d < d0 + per_warp; ++d ) {
			const float xv = __half2float( xr[d] );
			const uint4 wv = *reinterpret_cast<const uint4*>( w_t + static_cast<size_t>( d ) * experts + 8 * o );
			const half* wh = reinterpret_cast<const half*>( &wv );
			#pragma unroll
			for ( int j = 0; j < 8; ++j ) acc[j] = fmaf( xv, __half2float( wh[j] ), acc[j] );
		}
	}
	__shared__ float red[8][8 * kRouteOctets];   // [eighth][expert j of octet ol at j * kRouteOctets + ol]
	#pragma unroll
	for ( int j = 0; j < 8; ++j ) red[w][j * kRouteOctets + ol] = acc[j];
	__syncthreads();
	const int el = threadIdx.x, e = blockIdx.z * 8 * kRouteOctets + el;
	if ( e < experts ) {
		float v = 0.0f;
		#pragma unroll
		for ( int k = 0; k < 8; ++k ) v += red[k][( el % 8 ) * kRouteOctets + el / 8];
		part[( static_cast<size_t>( r ) * ROUTE_SPLITS + z ) * experts + e] = v;
	}
}

// One block per row, a thread per expert (whole warps): the scores, then the top_k by rank without sequential rounds (moe.cu's
// router_select_kernel): every warp ranks its 32 biased scores against each other (larger, or equal with the lower index, ranks first) and
// its top_k go to a packed candidate list (lanes past the experts as -inf), and the candidates are ranked against each other; the weights
// are the chosen experts' unbiased scores over their sum (summed in rank order)
__global__ __launch_bounds__( 512 )
void route_select_kernel( const float* __restrict__ part, const float* __restrict__ bias, const float* __restrict__ bias_vl, const uint8_t* __restrict__ img_flags, const int* __restrict__ perm, int* __restrict__ idx_out, half* __restrict__ w_out,
	int experts, int top_k, float scale ) {
	__shared__ float s_raw[512];
	__shared__ float c_val[16 * 16];
	__shared__ int c_idx[16 * 16];
	__shared__ int chosen[16];
	const int r = blockIdx.x, e = threadIdx.x, lane = e & 31, warp = e >> 5;
	float s = 0.0f, b = -FLT_MAX;
	if ( e < experts ) {
		float acc = 0.0f;
		#pragma unroll
		for ( int k = 0; k < ROUTE_SPLITS; ++k ) acc += part[( static_cast<size_t>( r ) * ROUTE_SPLITS + k ) * experts + e];
		const float sp = acc > 20.0f ? acc : log1pf( expf( acc ) );
		s = sqrtf( sp );
		b = s + ( ( img_flags != nullptr && bias_vl != nullptr && img_flags[r] ) ? bias_vl[e] : bias[e] );
		b = b != b ? -FLT_MAX : b;
	}
	s_raw[e] = s;
	{
		int rank = 0;
		#pragma unroll
		for ( int jj = 0; jj < 32; ++jj ) {
			const float u = __shfl_sync( 0xffffffff, b, jj );
			rank += ( u > b ) || ( u == b && jj < lane );
		}
		if ( rank < top_k ) {
			c_val[warp * top_k + rank] = e < experts ? b : -INFINITY;
			c_idx[warp * top_k + rank] = e < experts ? e : 0x7fffffff;
		}
	}
	__syncthreads();
	// candidate e / P against a P-th of the list on each of P consecutive lanes (P = 4, or 2 above top 8, so that P x candidates fit the block),
	// the counts summed over shuffles
	const int cands = static_cast<int>( blockDim.x / 32 ) * top_k, P = top_k > 8 ? 2 : 4, ci = e / P, pq = e % P, span = ( cands + P - 1 ) / P;
	if ( e < ( cands + 31 ) / 32 * 32 * P ) {   // whole warps, so the shuffles see every lane of a group
		const bool own = ci < cands;
		const float v = own ? c_val[ci] : -INFINITY;
		const int vi = own ? c_idx[ci] : 0x7fffffff;
		int rank = 0;
		const int c0 = pq * span, c1 = min( cands, c0 + span );
		#pragma unroll 6
		for ( int c = c0; c < c1; ++c ) rank += ( c_val[c] > v ) || ( c_val[c] == v && c_idx[c] < vi );
		rank += __shfl_xor_sync( 0xffffffff, rank, 1 );
		if ( P == 4 ) rank += __shfl_xor_sync( 0xffffffff, rank, 2 );
		if ( own && pq == 0 && rank < top_k && vi != 0x7fffffff ) chosen[rank] = vi;
	}
	__syncthreads();
	if ( warp != 0 ) return;
	const int ck = lane < top_k ? chosen[lane] : 0;
	const float sk = lane < top_k ? s_raw[ck] : 0.0f;
	float sum = 0.0f;
	for ( int k = 0; k < top_k; ++k ) sum += __shfl_sync( 0xffffffff, sk, k );
	if ( lane < top_k ) {
		idx_out[r * top_k + lane] = perm ? perm[ck] : ck;
		w_out[r * top_k + lane] = __float2half_rn( sk / ( sum + 1e-20f ) * scale );
	}
}

// ---- hyper-connections ------------------------------------------------------------------------------------------------------------------

// Element d is thread d % 256's, accumulated in ascending order; up to kHcnPer elements a thread all its loads are issued before any is used
// (the collapsed values stay in registers), wider rows go through shared memory: the same bits either way
constexpr int kHcnPer = 24;
__global__ __launch_bounds__( 256 )
void hc_collapse_norm_kernel( const float* __restrict__ streams, const float* __restrict__ pre, const half* __restrict__ norm_w, half* __restrict__ out,
	float* __restrict__ post, float post_scale, int hidden, float eps ) {
	extern __shared__ float hcn_smem[];
	const int r = blockIdx.x;
	const float p0 = pre[r * 4 + 0], p1 = pre[r * 4 + 1], p2 = pre[r * 4 + 2], p3 = pre[r * 4 + 3];
	const float* s = streams + static_cast<size_t>( r ) * 4 * hidden;
	float ss = 0.0f;
	if ( ( hidden + 255 ) / 256 <= kHcnPer ) {
		float c[kHcnPer], wv[kHcnPer];
		#pragma unroll
		for ( int k = 0; k < kHcnPer; ++k ) {
			const int d = threadIdx.x + 256 * k;
			c[k] = d < hidden ? p0 * s[d] + p1 * s[hidden + d] + p2 * s[2 * hidden + d] + p3 * s[3 * hidden + d] : 0.0f;
			wv[k] = d < hidden ? __half2float( norm_w[d] ) : 0.0f;
		}
		#pragma unroll
		for ( int k = 0; k < kHcnPer; ++k ) if ( threadIdx.x + 256 * k < hidden ) ss += c[k] * c[k];
		ss = block_sum<256>( ss );
		const float inv = rsqrtf( ss / hidden + eps );
		#pragma unroll
		for ( int k = 0; k < kHcnPer; ++k ) {
			const int d = threadIdx.x + 256 * k;
			if ( d < hidden ) out[static_cast<size_t>( r ) * hidden + d] = __float2half_rn( c[k] * inv * wv[k] );
		}
	}
	else {
		for ( int d = threadIdx.x; d < hidden; d += blockDim.x ) {
			const float c = p0 * s[d] + p1 * s[hidden + d] + p2 * s[2 * hidden + d] + p3 * s[3 * hidden + d];
			hcn_smem[d] = c;
			ss += c * c;
		}
		ss = block_sum<256>( ss );
		const float inv = rsqrtf( ss / hidden + eps );
		for ( int d = threadIdx.x; d < hidden; d += blockDim.x ) {
			out[static_cast<size_t>( r ) * hidden + d] = __float2half_rn( hcn_smem[d] * inv * __half2float( norm_w[d] ) );
		}
	}
	if ( post && threadIdx.x < 4 ) post[r * 4 + threadIdx.x] *= post_scale;
}

__global__ void hc_identity_pre_kernel( float* pre, int rows ) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < rows * 4 ) pre[i] = ( i % 4 ) == 0 ? 1.0f : 0.0f;
}

// grid (rows, 4 streams), 256 threads
__global__ __launch_bounds__( 256 )
void engram_gate_kernel( float* __restrict__ streams, const half* __restrict__ kv, const float* __restrict__ w, const uint8_t* __restrict__ skip, int hidden, float eps ) {
	const int r = blockIdx.x, h = blockIdx.y;
	if ( skip && skip[r] ) return;   // a row inside an image span: the reference shuts its gate
	float* s = streams + ( static_cast<size_t>( r ) * 4 + h ) * hidden;
	const half* key = kv + static_cast<size_t>( r ) * 5 * hidden + static_cast<size_t>( h ) * hidden;
	const half* val = kv + static_cast<size_t>( r ) * 5 * hidden + static_cast<size_t>( 4 ) * hidden;
	const float* wh = w + static_cast<size_t>( h ) * hidden;
	float s2 = 0.0f, k2 = 0.0f, dot = 0.0f;
	for ( int d = threadIdx.x; d < hidden; d += blockDim.x ) {
		const float x = s[d], k = __half2float( key[d] );
		s2 += x * x;
		k2 += k * k;
		dot += x * wh[d] * k;
	}
	s2 = block_sum<256>( s2 );
	k2 = block_sum<256>( k2 );
	dot = block_sum<256>( dot );
	const float g_in = dot * rsqrtf( s2 / hidden + eps ) * rsqrtf( k2 / hidden + eps ) * rsqrtf( static_cast<float>( hidden ) );
	const float gate = 1.0f / ( 1.0f + expf( -copysignf( sqrtf( fmaxf( fabsf( g_in ), 1e-6f ) ), g_in ) ) );
	for ( int d = threadIdx.x; d < hidden; d += blockDim.x ) s[d] += gate * __half2float( val[d] );
}

} // namespace

void DSV4::window_store( const half* kv, half* ring, const float* rope_f, int layer, int rows, const pod::BatchRows* d_rows, size_t ring_slot_stride, cudaStream_t stream ) {
	window_store_kernel<<<rows, 128, 0, stream>>>( kv, ring, reinterpret_cast<const float2*>( rope_f ), layer, d_rows, ring_slot_stride );
}

void DSV4::compress( const half* ckv, const half* cscore, const half* norm_w, float* cring_kv, float* cring_sc, half* latent_out, half* comp_kv, const float* rope_f,
	int ratio, int rows, const pod::BatchRows* d_rows, size_t cring_slot_stride, size_t comp_slot_stride, float eps, cudaStream_t stream ) {
	if ( ratio > 1 ) compress_stage_kernel<<<rows, 128, 0, stream>>>( ckv, cscore, cring_kv, cring_sc, d_rows, cring_slot_stride );
	compress_pool_kernel<<<rows, 128, 0, stream>>>( ckv, cring_kv, cring_sc, norm_w, latent_out, comp_kv, reinterpret_cast<const float2*>( rope_f ), ratio, d_rows, cring_slot_stride, comp_slot_stride, eps );
}

void DSV4::index_key_store( const half* kraw, const half* knorm_w, half* idx_k, const float* rope_f, int ratio, int rows, const pod::BatchRows* d_rows,
	size_t idx_slot_stride, float eps, cudaStream_t stream ) {
	index_key_kernel<<<rows, DSV4::IDX_DIM, 0, stream>>>( kraw, knorm_w, idx_k, reinterpret_cast<const float2*>( rope_f ), ratio, d_rows, idx_slot_stride, eps );
}

void DSV4::indexer( const half* q, const half* w, const half* idx_k, float* scores, uint8_t* cand, int* sel, int* nsel, const float* rope_f, int ratio, int cap,
	int cand_mode, int cand_blocks, int cand_block, int rows, const pod::BatchRows* d_rows, size_t idx_slot_stride, bool tensor_cores, cudaStream_t stream ) {
	// the tensor-core scorer pairs consecutive rows of one slot: prefill chunks (decode blocks may mix slots and are a few rows anyway)
	const bool tc = tensor_cores && rows > pod::MAX_BATCH;
	// blocks of SCORE_ROWS rows share the score buffer
	for ( int row0 = 0; row0 < rows; row0 += SCORE_ROWS ) {
		const int nr = rows - row0 < SCORE_ROWS ? rows - row0 : SCORE_ROWS;
		if ( tc ) {
			const dim3 grid( ( nr + 1 ) / 2, ( cap + IDX_TC_SPAN - 1 ) / IDX_TC_SPAN );
			index_score_tc_kernel<<<grid, 256, 0, stream>>>( q, w, idx_k, scores, cand_mode == 2 ? cand : nullptr, cand_block, reinterpret_cast<const float2*>( rope_f ), ratio, cap, row0, nr,
				d_rows, idx_slot_stride );
		}
		else {
			const dim3 grid( nr, ( cap + IDX_TILE - 1 ) / IDX_TILE );
			index_score_kernel<<<grid, 256, 0, stream>>>( q, w, idx_k, scores, cand_mode == 2 ? cand : nullptr, cand_block, reinterpret_cast<const float2*>( rope_f ), ratio, cap, row0,
				d_rows, idx_slot_stride );
		}
		index_select_kernel<<<nr, 1024, 0, stream>>>( scores, cand, sel, nsel, ratio, cap, cand_mode, cand_blocks, cand_block, row0, d_rows );
	}
}

namespace {

// attend_tc_kernel: its mma.m8n8k4 form is SM_70's (the SM_80+ expansion is ~80 FFMA an instruction), 16 heads a block
struct AttDevice {
	int major = 0, sms = 0;
};

const AttDevice& att_device() {
	static AttDevice d[ENGINE_NUM_GPUS];
	int dev = 0;
	cudaGetDevice( &dev );
	if ( d[dev].major == 0 ) {
		cudaDeviceGetAttribute( &d[dev].sms, cudaDevAttrMultiProcessorCount, dev );
		cudaFuncSetAttribute( attend_tc_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, TC_SMEM );
		cudaFuncSetAttribute( attend_tc_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, TC_SMEM );
		cudaDeviceGetAttribute( &d[dev].major, cudaDevAttrComputeCapabilityMajor, dev );
	}
	return d[dev];
}

bool att_tc_ok( int nh ) { return nh % 16 == 0 && att_device().major == 7; }

// A tile is a block's serial unit (its shared memory bound), so the splits spread a row's tiles evenly over at most a block an SM
int att_splits( int tiles, int rows ) {
	const int cap = std::max( 1, std::min( att_device().sms, DSV4::ATTEND_PARTIALS ) / rows );
	const int per = ( tiles + cap - 1 ) / cap;
	return std::min( 32, ( tiles + per - 1 ) / per );
}

} // namespace

void DSV4::attend( const half* q, const half* ring, const half* comp_kv, const int* sel, const int* nsel, const float* sink, const float* rope_f, half* o, int nh,
	int layer, int rows, const pod::BatchRows* d_rows, size_t ring_slot_stride, size_t comp_slot_stride, float* ws, cudaStream_t stream, int win_floor ) {
	const float2* rope = reinterpret_cast<const float2*>( rope_f );
	const size_t smem = static_cast<size_t>( nh + ATT_TILE ) * ( HD / 2 ) * sizeof( half2 );
	ENGINE_ASSERT( nh == 8 || nh == 16 || nh == 32, "DSV4::attend: heads per rank must be 8, 16 or 32" );
	if ( ws != nullptr && att_tc_ok( nh ) ) {
		const int nsplit = att_splits( ( DSV4::WINDOW + ( comp_kv ? DSV4::TOPK : 0 ) + TC_BK - 1 ) / TC_BK, rows );
		attend_tc_kernel<false><<<dim3( rows, nsplit, nh / 16 ), 256, TC_SMEM, stream>>>( q, ring, comp_kv, sel, nsel, rope, nh, layer, d_rows, ring_slot_stride,
			comp_slot_stride, ws, win_floor, 0, 0, 0, rows );
		att_combine_kernel<<<dim3( rows, nh ), ATT_THREADS, 0, stream>>>( ws, nsplit, sink, rope, o, nh, rows, d_rows );
		return;
	}
	// decode batches split the keys over blocks (a single block per row leaves almost every SM idle, and its dependent loads miss the cache
	// once the compressed cache outgrows it); a window-only layer's 128 keys take a tile a block; prefill chunks stay on one block per row
	int nsplit = 1;
	if ( ws != nullptr && comp_kv != nullptr ) nsplit = rows == 1 ? 32 : rows == 2 ? 16 : rows <= 4 ? 8 : rows <= 16 ? 4 : 1;
	else if ( ws != nullptr ) nsplit = rows <= 4 ? DSV4::WINDOW / ATT_TILE : rows <= 16 ? 2 : 1;
	float* wsp = nsplit > 1 ? ws : nullptr;
	const dim3 grid( rows, nsplit );
	if ( nh == 32 ) {
		static bool attr[ENGINE_NUM_GPUS] = {};
		int dev = 0;
		cudaGetDevice( &dev );
		if ( !attr[dev] ) { cudaFuncSetAttribute( attend_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024 ); attr[dev] = true; }
	}
	switch ( nh / 8 ) {
		case 1: attend_kernel<1><<<grid, ATT_THREADS, smem, stream>>>( q, ring, comp_kv, sel, nsel, sink, rope, o, nh, layer, rows, d_rows, ring_slot_stride, comp_slot_stride, wsp, win_floor ); break;
		case 2: attend_kernel<2><<<grid, ATT_THREADS, smem, stream>>>( q, ring, comp_kv, sel, nsel, sink, rope, o, nh, layer, rows, d_rows, ring_slot_stride, comp_slot_stride, wsp, win_floor ); break;
		default: attend_kernel<4><<<grid, ATT_THREADS, smem, stream>>>( q, ring, comp_kv, sel, nsel, sink, rope, o, nh, layer, rows, d_rows, ring_slot_stride, comp_slot_stride, wsp, win_floor ); break;
	}
	ENGINE_ASSERT( nsplit <= 32, "DSV4::attend: the combine reads one split per lane" );
	if ( nsplit > 1 ) att_combine_kernel<<<dim3( rows, nh ), ATT_THREADS, 0, stream>>>( wsp, nsplit, sink, rope, o, nh, rows, d_rows );
}

void DSV4::tap_store( const float* streams, half* ring, int tap, int taps, int hidden, float scale, int floor_pos, int rows, const pod::BatchRows* d_rows, size_t ring_slot_stride,
	cudaStream_t stream ) {
	tap_store_kernel<<<rows, 256, 0, stream>>>( streams, ring, tap, taps, hidden, scale, floor_pos, d_rows, ring_slot_stride );
}

void DSV4::tap_gather( const half* ring, half* out, int slot, int first, int n, int taps, int hidden, size_t ring_slot_stride, cudaStream_t stream ) {
	tap_gather_kernel<<<n, 256, 0, stream>>>( ring, out, slot, first, taps * hidden, ring_slot_stride );
}

void DSV4::dspark_attend( const half* q, const half* ring, const half* blk, const float* sink, const float* rope_f, half* o, int nh, int layer, int rows, const pod::BatchRows* d_rows,
	int slot, int w0, int t, size_t ring_slot_stride, float* ws, cudaStream_t stream ) {
	const float2* rope = reinterpret_cast<const float2*>( rope_f );
	const size_t smem = static_cast<size_t>( nh + ATT_TILE ) * ( HD / 2 ) * sizeof( half2 );
	ENGINE_ASSERT( nh == 8 || nh == 16 || nh == 32, "DSV4::dspark_attend: heads per rank must be 8, 16 or 32" );
	if ( ws != nullptr && rows <= pod::MAX_BATCH && att_tc_ok( nh ) ) {
		const int nsplit = att_splits( ( t - w0 + 1 + rows + TC_BK - 1 ) / TC_BK, rows );
		attend_tc_kernel<true><<<dim3( rows, nsplit, nh / 16 ), 256, TC_SMEM, stream>>>( q, ring, blk, nullptr, nullptr, rope, nh, layer, d_rows, ring_slot_stride, 0, ws, 0,
			slot, w0, t, rows );
		att_combine_kernel<<<dim3( rows, nh ), ATT_THREADS, 0, stream>>>( ws, nsplit, sink, rope, o, nh, rows, d_rows );
		return;
	}
	if ( nh == 32 ) {
		static bool attr[ENGINE_NUM_GPUS] = {};
		int dev = 0;
		cudaGetDevice( &dev );
		if ( !attr[dev] ) { cudaFuncSetAttribute( dspark_attend_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024 ); attr[dev] = true; }
	}
	switch ( nh / 8 ) {
		case 1: dspark_attend_kernel<1><<<rows, ATT_THREADS, smem, stream>>>( q, ring, blk, sink, rope, o, nh, layer, rows, d_rows, slot, w0, t, ring_slot_stride ); break;
		case 2: dspark_attend_kernel<2><<<rows, ATT_THREADS, smem, stream>>>( q, ring, blk, sink, rope, o, nh, layer, rows, d_rows, slot, w0, t, ring_slot_stride ); break;
		default: dspark_attend_kernel<4><<<rows, ATT_THREADS, smem, stream>>>( q, ring, blk, sink, rope, o, nh, layer, rows, d_rows, slot, w0, t, ring_slot_stride ); break;
	}
}

void DSV4::concat_groups( const half* in, half* out, int groups, int n, int rows, cudaStream_t stream ) {
	const size_t total = static_cast<size_t>( rows ) * groups * n;
	concat_groups_kernel<<<static_cast<int>( ( total + 255 ) / 256 > 4096 ? 4096 : ( total + 255 ) / 256 ), 256, 0, stream>>>( in, out, groups, n, rows );
}

void DSV4::route( const half* x, const half* w_t, const float* bias, const float* bias_vl, const uint8_t* img_flags, const int* perm, int* idx_out, half* w_out, float* scratch, int hidden, int experts, int top_k,
	float scale, int rows, cudaStream_t stream ) {
	ENGINE_ASSERT( experts <= 512 && experts % 8 == 0 && top_k <= 16 && hidden % ( ROUTE_SPLITS * 8 ) == 0,
		"DSV4::route: at most 512 experts (a multiple of 8), top 16, hidden a multiple of 128" );
	route_partial_kernel<<<dim3( rows, ROUTE_SPLITS, ( experts / 8 + kRouteOctets - 1 ) / kRouteOctets ), 8 * kRouteOctets, 0, stream>>>( x, w_t, scratch, hidden, experts );
	route_select_kernel<<<rows, ( experts + 31 ) / 32 * 32, 0, stream>>>( scratch, bias, bias_vl, img_flags, perm, idx_out, w_out, experts, top_k, scale );
}

void DSV4::hc_collapse_norm( const float* streams, const float* pre, const half* norm_w, half* out, float* post, float post_scale, int hidden, float eps, int rows,
	cudaStream_t stream ) {
	const size_t smem = ( hidden + 255 ) / 256 <= kHcnPer ? 0 : static_cast<size_t>( hidden ) * sizeof( float );
	hc_collapse_norm_kernel<<<rows, 256, smem, stream>>>( streams, pre, norm_w, out, post, post_scale, hidden, eps );
}

void DSV4::hc_identity_pre( float* pre, int rows, cudaStream_t stream ) {
	hc_identity_pre_kernel<<<( rows * 4 + 255 ) / 256, 256, 0, stream>>>( pre, rows );
}

void DSV4::engram_gate( float* streams, const half* kv, const float* w, const uint8_t* skip, int hidden, float eps, int rows, cudaStream_t stream ) {
	engram_gate_kernel<<<dim3( rows, 4 ), 256, 0, stream>>>( streams, kv, w, skip, hidden, eps );
}

// ---------------------------------------------------------------------------------------------------------------------
// FP8 E4M3 weights with E8M0 block scales (the original DeepSeek-V4.1 tensors, kept as they are in device memory)

namespace {

// Two E4M3 bytes of x (picked by the __byte_perm selector 0x4140 / 0x4342) as a half2 holding value * 2^-8: the sign, the 4 exponent bits and the
// 3 mantissa bits of an E4M3 byte placed at a half's sign / low exponent / top mantissa bits read as the same number with the exponent bias 15
// instead of 7 (subnormals included); fp8_block_scale puts the 2^8 back
__device__ __forceinline__ half2 fp8x2_to_half2( uint32_t x, uint32_t sel ) {
	const uint32_t t = __byte_perm( x, 0u, sel );
	const uint32_t h = ( ( t & 0x007f007fu ) << 7 ) | ( ( t & 0x00800080u ) << 8 );
	return *reinterpret_cast<const half2*>( &h );
}

// 2^(e - 127) * 2^8, undoing the half read's 2^-8
__device__ __forceinline__ float fp8_block_scale( uint32_t e ) {
	const uint32_t biased = min( max( e + 8u, 1u ), 254u );
	return __uint_as_float( biased << 23 );
}

template <int MAXM>
__global__ __launch_bounds__( 256 )
void fp8_gemv_kernel( const half* __restrict__ x, const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc, half* __restrict__ out, int m0, int m, int n, int k ) {
	const int row = blockIdx.x * 8 + threadIdx.x / 32;
	if ( row >= n ) return;
	const int lane = threadIdx.x % 32;
	const uint8_t* wr = w + static_cast<size_t>( row ) * k;
	const uint8_t* sr = sc + static_cast<size_t>( row / 32 ) * ( k / 32 );
	float acc[MAXM];
	#pragma unroll
	for ( int r = 0; r < MAXM; ++r ) acc[r] = 0.0f;
	for ( int kk = lane * 16; kk < k; kk += 512 ) {
		const uint4 wb = *reinterpret_cast<const uint4*>( wr + kk );
		const float s = fp8_block_scale( sr[kk / 32] );
		float2 wf[8];
		const uint32_t words[4] = { wb.x, wb.y, wb.z, wb.w };
		#pragma unroll
		for ( int q = 0; q < 4; ++q ) {
			wf[2 * q] = __half22float2( fp8x2_to_half2( words[q], 0x4140u ) );
			wf[2 * q + 1] = __half22float2( fp8x2_to_half2( words[q], 0x4342u ) );
		}
		#pragma unroll
		for ( int r = 0; r < MAXM; ++r ) {
			if ( m0 + r >= m ) break;
			const uint4 xa = *reinterpret_cast<const uint4*>( x + static_cast<size_t>( m0 + r ) * k + kk );
			const uint4 xb = *reinterpret_cast<const uint4*>( x + static_cast<size_t>( m0 + r ) * k + kk + 8 );
			const uint32_t xs[8] = { xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w };
			float sum = 0.0f;
			#pragma unroll
			for ( int q = 0; q < 8; ++q ) {
				const float2 xf = __half22float2( *reinterpret_cast<const half2*>( &xs[q] ) );
				sum = fmaf( wf[q].x, xf.x, sum );
				sum = fmaf( wf[q].y, xf.y, sum );
			}
			acc[r] += s * sum;
		}
	}
	#pragma unroll
	for ( int r = 0; r < MAXM; ++r ) {
		float v = acc[r];
		#pragma unroll
		for ( int o = 16; o > 0; o >>= 1 ) v += __shfl_xor_sync( 0xffffffffu, v, o );
		if ( lane == 0 && m0 + r < m ) out[static_cast<size_t>( m0 + r ) * n + row] = __float2half_rn( v );
	}
}

__global__ __launch_bounds__( 256 )
void fp8_dequant_kernel( const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc, half* __restrict__ out, int n, int k ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;   // 16 elements each
	if ( i * 16 >= static_cast<size_t>( n ) * k ) return;
	const size_t at = i * 16;
	const int row = static_cast<int>( at / k ), col = static_cast<int>( at % k );
	const uint4 wb = *reinterpret_cast<const uint4*>( w + at );
	const float s = fp8_block_scale( sc[static_cast<size_t>( row / 32 ) * ( k / 32 ) + col / 32] );
	const uint32_t words[4] = { wb.x, wb.y, wb.z, wb.w };
	uint32_t o[8];
	#pragma unroll
	for ( int q = 0; q < 4; ++q ) {
		const float2 a = __half22float2( fp8x2_to_half2( words[q], 0x4140u ) );
		const float2 b = __half22float2( fp8x2_to_half2( words[q], 0x4342u ) );
		const half2 ha = __floats2half2_rn( a.x * s, a.y * s ), hb = __floats2half2_rn( b.x * s, b.y * s );
		o[2 * q] = *reinterpret_cast<const uint32_t*>( &ha );
		o[2 * q + 1] = *reinterpret_cast<const uint32_t*>( &hb );
	}
	*reinterpret_cast<uint4*>( out + at ) = make_uint4( o[0], o[1], o[2], o[3] );
	*reinterpret_cast<uint4*>( out + at + 8 ) = make_uint4( o[4], o[5], o[6], o[7] );
}

} // namespace

void DSV4::linear_fp8( const half* x, const uint8_t* w, const uint8_t* scales, half* out, int m, int n, int k, half* scratch, cudaStream_t stream ) {
	ENGINE_ASSERT( k % 32 == 0 && n % 32 == 0, "DSV4::linear_fp8: block-aligned shapes" );
	if ( m <= 16 ) {
		for ( int m0 = 0; m0 < m; m0 += 8 ) fp8_gemv_kernel<8><<<( n + 7 ) / 8, 256, 0, stream>>>( x, w, scales, out, m0, m, n, k );
		return;
	}
	ENGINE_ASSERT( scratch != nullptr, "DSV4::linear_fp8: a scratch for the dequantized weight" );
	const size_t groups = static_cast<size_t>( n ) * k / 16;
	fp8_dequant_kernel<<<( groups + 255 ) / 256, 256, 0, stream>>>( w, scales, scratch, n, k );
	LinearPlainGEMV::forward( scratch, x, out, m, n, k, stream );
}

} // namespace engine::modules
