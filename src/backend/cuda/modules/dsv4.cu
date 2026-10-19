#include "engine/modules/dsv4.h"
#include "engine/backend/cuda/reduce.cuh"
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>

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
			__shared__ int ties;
			if ( threadIdx.x == 0 ) ties = 0;
			__syncthreads();
			for ( int b = threadIdx.x; b < nb; b += blockDim.x ) {
				const uint32_t k = bkey( b );
				uint8_t keep = k > th;
				if ( k == th ) keep = atomicAdd( &ties, 1 ) < cand_blocks - above;
				cm[b] = keep;
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
	__shared__ int cnt, ties;
	if ( threadIdx.x == 0 ) { cnt = 0; ties = 0; }
	__syncthreads();
	for ( int i = threadIdx.x; i < n; i += blockDim.x ) {
		const uint32_t k = ekey( i );
		if ( k == 0u ) continue;
		if ( k > th ) out[atomicAdd( &cnt, 1 )] = i;
		else if ( k == th && atomicAdd( &ties, 1 ) < DSV4::TOPK - above ) out[atomicAdd( &cnt, 1 )] = i;
	}
	__syncthreads();
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
	size_t ring_slot_stride, size_t comp_slot_stride ) {
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
	const int w0 = max( 0, pos - DSV4::WINDOW + 1 );
	const int n_win = pos - w0 + 1;
	const int n_cmp = comp_kv ? nsel[r] : 0;
	const int* srow = comp_kv ? sel + static_cast<size_t>( r ) * DSV4::TOPK : nullptr;
	const int total = n_win + n_cmp;
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
		for ( int i = tid; i < nk * ( HD / 8 ); i += ATT_THREADS ) {   // 16-byte loads
			const int k = i / ( HD / 8 ), c = i % ( HD / 8 );
			const int e = k0 + k;
			const half* src = e < n_win ? wring + static_cast<size_t>( ( w0 + e ) % DSV4::RING ) * HD : cbase + static_cast<size_t>( srow[e - n_win] ) * HD;
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

constexpr int ROUTE_SPLITS = 4;

// grid (rows, experts / 32, ROUTE_SPLITS), 256 threads: warp w sums its share of the block's hidden slice for 32 experts
__global__ __launch_bounds__( 256 )
void route_partial_kernel( const half* __restrict__ x, const half* __restrict__ w_t, float* __restrict__ part, int hidden, int experts ) {
	const int r = blockIdx.x, e = blockIdx.y * 32 + ( threadIdx.x & 31 ), warp = threadIdx.x >> 5;
	const int slice = hidden / ROUTE_SPLITS, per_warp = slice / 8;
	const int d0 = blockIdx.z * slice + warp * per_warp;
	const half* xr = x + static_cast<size_t>( r ) * hidden;
	float acc = 0.0f;
	if ( e < experts ) {
		for ( int d = d0; d < d0 + per_warp; ++d ) acc = fmaf( __half2float( xr[d] ), __half2float( w_t[static_cast<size_t>( d ) * experts + e] ), acc );
	}
	__shared__ float red[8][32];
	red[warp][threadIdx.x & 31] = acc;
	__syncthreads();
	if ( warp == 0 && e < experts ) {
		float v = 0.0f;
		#pragma unroll
		for ( int w = 0; w < 8; ++w ) v += red[w][threadIdx.x];
		part[( static_cast<size_t>( r ) * ROUTE_SPLITS + blockIdx.z ) * experts + e] = v;
	}
}

__global__ __launch_bounds__( 512 )
void route_select_kernel( const float* __restrict__ part, const float* __restrict__ bias, const int* __restrict__ perm, int* __restrict__ idx_out, half* __restrict__ w_out,
	int experts, int top_k, float scale ) {
	__shared__ float sel_v[512];
	__shared__ int sel_i[512];
	const int r = blockIdx.x, e = threadIdx.x;
	float s = 0.0f, b = -FLT_MAX;
	if ( e < experts ) {
		float acc = 0.0f;
		#pragma unroll
		for ( int k = 0; k < ROUTE_SPLITS; ++k ) acc += part[( static_cast<size_t>( r ) * ROUTE_SPLITS + k ) * experts + e];
		const float sp = acc > 20.0f ? acc : log1pf( expf( acc ) );
		s = sqrtf( sp );
		b = s + bias[e];
	}
	__shared__ int chosen[16];
	__shared__ float chosen_s[16];
	for ( int k = 0; k < top_k; ++k ) {
		sel_v[e] = b;
		sel_i[e] = e;
		__syncthreads();
		for ( int st = blockDim.x / 2; st > 0; st >>= 1 ) {
			if ( e < st ) {
				const float o = sel_v[e + st];
				const int oi = sel_i[e + st];
				if ( o > sel_v[e] || ( o == sel_v[e] && oi < sel_i[e] ) ) { sel_v[e] = o; sel_i[e] = oi; }
			}
			__syncthreads();
		}
		if ( e == sel_i[0] ) { chosen[k] = e; chosen_s[k] = s; b = -FLT_MAX; }
		__syncthreads();
	}
	if ( e == 0 ) {
		float sum = 0.0f;
		for ( int k = 0; k < top_k; ++k ) sum += chosen_s[k];
		for ( int k = 0; k < top_k; ++k ) {
			idx_out[r * top_k + k] = perm ? perm[chosen[k]] : chosen[k];
			w_out[r * top_k + k] = __float2half_rn( chosen_s[k] / ( sum + 1e-20f ) * scale );
		}
	}
}

// ---- hyper-connections ------------------------------------------------------------------------------------------------------------------

__global__ __launch_bounds__( 256 )
void hc_collapse_norm_kernel( const float* __restrict__ streams, const float* __restrict__ pre, const half* __restrict__ norm_w, half* __restrict__ out,
	float* __restrict__ post, float post_scale, int hidden, float eps ) {
	extern __shared__ float hcn_smem[];
	const int r = blockIdx.x;
	const float p0 = pre[r * 4 + 0], p1 = pre[r * 4 + 1], p2 = pre[r * 4 + 2], p3 = pre[r * 4 + 3];
	const float* s = streams + static_cast<size_t>( r ) * 4 * hidden;
	float ss = 0.0f;
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
	if ( post && threadIdx.x < 4 ) post[r * 4 + threadIdx.x] *= post_scale;
}

__global__ void hc_identity_pre_kernel( float* pre, int rows ) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < rows * 4 ) pre[i] = ( i % 4 ) == 0 ? 1.0f : 0.0f;
}

// grid (rows, 4 streams), 256 threads
__global__ __launch_bounds__( 256 )
void engram_gate_kernel( float* __restrict__ streams, const half* __restrict__ kv, const float* __restrict__ w, int hidden, float eps ) {
	const int r = blockIdx.x, h = blockIdx.y;
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
	int cand_mode, int cand_blocks, int cand_block, int rows, const pod::BatchRows* d_rows, size_t idx_slot_stride, cudaStream_t stream ) {
	// blocks of SCORE_ROWS rows share the score buffer
	for ( int row0 = 0; row0 < rows; row0 += SCORE_ROWS ) {
		const int nr = rows - row0 < SCORE_ROWS ? rows - row0 : SCORE_ROWS;
		const dim3 grid( nr, ( cap + IDX_TILE - 1 ) / IDX_TILE );
		index_score_kernel<<<grid, 256, 0, stream>>>( q, w, idx_k, scores, cand_mode == 2 ? cand : nullptr, cand_block, reinterpret_cast<const float2*>( rope_f ), ratio, cap, row0,
			d_rows, idx_slot_stride );
		index_select_kernel<<<nr, 1024, 0, stream>>>( scores, cand, sel, nsel, ratio, cap, cand_mode, cand_blocks, cand_block, row0, d_rows );
	}
}

void DSV4::attend( const half* q, const half* ring, const half* comp_kv, const int* sel, const int* nsel, const float* sink, const float* rope_f, half* o, int nh,
	int layer, int rows, const pod::BatchRows* d_rows, size_t ring_slot_stride, size_t comp_slot_stride, cudaStream_t stream ) {
	const float2* rope = reinterpret_cast<const float2*>( rope_f );
	const size_t smem = static_cast<size_t>( nh + ATT_TILE ) * ( HD / 2 ) * sizeof( half2 );
	switch ( nh / 8 ) {
		case 1: attend_kernel<1><<<rows, ATT_THREADS, smem, stream>>>( q, ring, comp_kv, sel, nsel, sink, rope, o, nh, layer, rows, d_rows, ring_slot_stride, comp_slot_stride ); break;
		case 2: attend_kernel<2><<<rows, ATT_THREADS, smem, stream>>>( q, ring, comp_kv, sel, nsel, sink, rope, o, nh, layer, rows, d_rows, ring_slot_stride, comp_slot_stride ); break;
		case 4: {
			static bool attr[ENGINE_NUM_GPUS] = {};
			int dev = 0;
			cudaGetDevice( &dev );
			if ( !attr[dev] ) { cudaFuncSetAttribute( attend_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024 ); attr[dev] = true; }
			attend_kernel<4><<<rows, ATT_THREADS, smem, stream>>>( q, ring, comp_kv, sel, nsel, sink, rope, o, nh, layer, rows, d_rows, ring_slot_stride, comp_slot_stride );
		} break;
		default: ENGINE_ASSERT( false, "DSV4::attend: heads per rank must be 8, 16 or 32" );
	}
}

void DSV4::concat_groups( const half* in, half* out, int groups, int n, int rows, cudaStream_t stream ) {
	const size_t total = static_cast<size_t>( rows ) * groups * n;
	concat_groups_kernel<<<static_cast<int>( ( total + 255 ) / 256 > 4096 ? 4096 : ( total + 255 ) / 256 ), 256, 0, stream>>>( in, out, groups, n, rows );
}

void DSV4::route( const half* x, const half* w_t, const float* bias, const int* perm, int* idx_out, half* w_out, float* scratch, int hidden, int experts, int top_k,
	float scale, int rows, cudaStream_t stream ) {
	ENGINE_ASSERT( experts <= 512 && top_k <= 16 && hidden % ( ROUTE_SPLITS * 8 ) == 0, "DSV4::route: at most 512 experts, top 16, hidden a multiple of 32" );
	route_partial_kernel<<<dim3( rows, ( experts + 31 ) / 32, ROUTE_SPLITS ), 256, 0, stream>>>( x, w_t, scratch, hidden, experts );
	route_select_kernel<<<rows, 512, 0, stream>>>( scratch, bias, perm, idx_out, w_out, experts, top_k, scale );
}

void DSV4::hc_collapse_norm( const float* streams, const float* pre, const half* norm_w, half* out, float* post, float post_scale, int hidden, float eps, int rows,
	cudaStream_t stream ) {
	hc_collapse_norm_kernel<<<rows, 256, static_cast<size_t>( hidden ) * sizeof( float ), stream>>>( streams, pre, norm_w, out, post, post_scale, hidden, eps );
}

void DSV4::hc_identity_pre( float* pre, int rows, cudaStream_t stream ) {
	hc_identity_pre_kernel<<<( rows * 4 + 255 ) / 256, 256, 0, stream>>>( pre, rows );
}

void DSV4::engram_gate( float* streams, const half* kv, const float* w, int hidden, float eps, int rows, cudaStream_t stream ) {
	engram_gate_kernel<<<dim3( rows, 4 ), 256, 0, stream>>>( streams, kv, w, hidden, eps );
}

} // namespace engine::modules
