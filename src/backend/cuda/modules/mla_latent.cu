#include "engine/modules/mla.h"
#include "kv_quant.cuh"
#include "../../../kernels/cuda/mma_sm80.cuh"
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace wmma = nvcuda::wmma;

namespace engine::modules {

namespace {

constexpr int LORA = MLALatent::LORA;
constexpr int HD = MLALatent::HEAD;
constexpr int ROWG = 16; // rows per pass of the projection kernels

// Both projections are per-head GEMMs on tensor cores (m16n16k16, FP32 accumulate). A block owns (head, 16 output columns, up to
// RCH rows); its four warps split the contraction four ways and the partial tiles are summed through shared memory, so the weight
// streams over 16 x heads x (columns / 16) blocks even for a single decode row.
constexpr int RCH = 64; // rows per block
constexpr int APAD = 8;

// q_lat[r][h][c] = sum_d q[r][h][d] * W_UK[h][d][c]  (W_UK rows d, row-major over c; K rows come first in kv_b)
__global__ __launch_bounds__( 128 )
void absorb_q_kernel( const half* __restrict__ q, const half* __restrict__ kv_b, half* __restrict__ q_lat, int heads, int rows ) {
	__shared__ __align__( 32 ) half sa[ROWG][HD + APAD];
	__shared__ __align__( 32 ) float sc[4][ROWG * 16];
	const int h = blockIdx.x;
	const int c0 = blockIdx.y * 16;
	const int warp = threadIdx.x >> 5;
	const int rbeg = blockIdx.z * RCH;
	const int rend = min( rows, rbeg + RCH );
	const half* w = kv_b + static_cast<size_t>( h ) * 2 * HD * LORA + c0;
	for ( int r0 = rbeg; r0 < rend; r0 += ROWG ) {
		const int nr = min( ROWG, rend - r0 );
		__syncthreads();
		for ( int i = threadIdx.x; i < ROWG * ( HD / 8 ); i += 128 ) {
			const int r = i / ( HD / 8 ), v = i % ( HD / 8 );
			uint4 val = make_uint4( 0, 0, 0, 0 );
			if ( r < nr ) val = *reinterpret_cast<const uint4*>( q + ( static_cast<size_t>( r0 + r ) * heads + h ) * HD + v * 8 );
			*reinterpret_cast<uint4*>( &sa[r][v * 8] ) = val;
		}
		__syncthreads();
		wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa[HD / 64];
		wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb[HD / 64];
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
		wmma::fill_fragment( acc, 0.0f );
		#pragma unroll
		for ( int ks = 0; ks < HD / 64; ++ks ) {
			const int kk = warp * ( HD / 4 ) + ks * 16;
			wmma::load_matrix_sync( fb[ks], w + static_cast<size_t>( kk ) * LORA, LORA );
			wmma::load_matrix_sync( fa[ks], &sa[0][kk], HD + APAD );
		}
		#pragma unroll
		for ( int ks = 0; ks < HD / 64; ++ks ) wmma::mma_sync( acc, fa[ks], fb[ks], acc );
		wmma::store_matrix_sync( sc[warp], acc, 16, wmma::mem_row_major );
		__syncthreads();
		{
			const int idx = threadIdx.x * 2;
			const int r = idx >> 4, c = idx & 15;
			if ( r < nr ) {
				const float x = sc[0][idx] + sc[1][idx] + sc[2][idx] + sc[3][idx];
				const float y = sc[0][idx + 1] + sc[1][idx + 1] + sc[2][idx + 1] + sc[3][idx + 1];
				*reinterpret_cast<half2*>( q_lat + ( static_cast<size_t>( r0 + r ) * heads + h ) * LORA + c0 + c ) = __floats2half2_rn( x, y );
			}
		}
	}
}

// out[r][h][v] = sum_c W_UV[h][v][c] * o_lat[r][h][c]  (V rows follow the K rows; B read column-major: element (c, v) at v * LORA + c)
__global__ __launch_bounds__( 128 )
void expand_out_kernel( const half* __restrict__ o_lat, const half* __restrict__ kv_b, half* __restrict__ out, int heads, int rows ) {
	__shared__ __align__( 32 ) half so[ROWG][LORA + APAD];
	__shared__ __align__( 32 ) float sc[4][ROWG * 16];
	const int h = blockIdx.x;
	const int v0 = blockIdx.y * 16;
	const int warp = threadIdx.x >> 5;
	const int rbeg = blockIdx.z * RCH;
	const int rend = min( rows, rbeg + RCH );
	const half* w = kv_b + ( static_cast<size_t>( h ) * 2 * HD + HD + v0 ) * LORA;
	for ( int r0 = rbeg; r0 < rend; r0 += ROWG ) {
		const int nr = min( ROWG, rend - r0 );
		__syncthreads();
		for ( int i = threadIdx.x; i < ROWG * ( LORA / 8 ); i += 128 ) {
			const int r = i / ( LORA / 8 ), v = i % ( LORA / 8 );
			uint4 val = make_uint4( 0, 0, 0, 0 );
			if ( r < nr ) val = *reinterpret_cast<const uint4*>( o_lat + ( static_cast<size_t>( r0 + r ) * heads + h ) * LORA + v * 8 );
			*reinterpret_cast<uint4*>( &so[r][v * 8] ) = val;
		}
		__syncthreads();
		wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa[LORA / 64];
		wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb[LORA / 64];
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
		wmma::fill_fragment( acc, 0.0f );
		#pragma unroll
		for ( int ks = 0; ks < LORA / 64; ++ks ) {
			const int kk = warp * ( LORA / 4 ) + ks * 16;
			wmma::load_matrix_sync( fb[ks], w + kk, LORA );
			wmma::load_matrix_sync( fa[ks], &so[0][kk], LORA + APAD );
		}
		#pragma unroll
		for ( int ks = 0; ks < LORA / 64; ++ks ) wmma::mma_sync( acc, fa[ks], fb[ks], acc );
		wmma::store_matrix_sync( sc[warp], acc, 16, wmma::mem_row_major );
		__syncthreads();
		{
			const int idx = threadIdx.x * 2;
			const int r = idx >> 4, c = idx & 15;
			if ( r < nr ) {
				const float x = sc[0][idx] + sc[1][idx] + sc[2][idx] + sc[3][idx];
				const float y = sc[0][idx + 1] + sc[1][idx + 1] + sc[2][idx + 1] + sc[3][idx + 1];
				*reinterpret_cast<half2*>( out + ( static_cast<size_t>( r0 + r ) * heads + h ) * HD + v0 + c ) = __floats2half2_rn( x, y );
			}
		}
	}
}

// Eight consecutive latent elements of any cache format as packed halves
template <KVQ Q>
__device__ __forceinline__ uint4 latent_ld8( const void* data, const void* scales, size_t idx ) {
	return kv_decode8_h<Q>( kv_fetch8<Q>( data, scales, idx ) );
}

// One block per row; a thread quantizes one unit (kv_unit<Q>() elements) at a time
template <KVQ Q>
__global__ void store_latent_kernel( void* __restrict__ data, void* __restrict__ scales, const half* __restrict__ c, int layer, int max_seq, int rows, pod::SlotRef slot ) {
	constexpr int U = kv_unit<Q>();
	const int r = blockIdx.x;
	const int pos = slot.rows->pos[slot.row + r];
	const size_t off = static_cast<size_t>( slot.rows->slot[slot.row + r] ) * slot.stride + ( static_cast<size_t>( layer ) * max_seq + pos ) * LORA;
	for ( int u = threadIdx.x; u < LORA / U; u += blockDim.x ) kv_store_unit<Q>( data, scales, off + static_cast<size_t>( u ) * U, c + static_cast<size_t>( r ) * LORA + u * U );
}

constexpr int CHUNK_KEYS = 1024;
constexpr int DSA_KPAD = 2080; // DSA::KPAD
constexpr int TILE = 16;
constexpr int CS = LORA + 8; // padded key row (halves) in shared memory

// Latent attention, one block per (row, chunk of 1024 keys), all heads of the row: a 16-key tile of c is staged once and
// serves the 16 heads' scores and their value sums. 512 threads: scores take two threads per (head, key) pair, the value
// sum one thread per latent dimension holding all 16 heads' accumulators.
template <int HEADS, KVQ Q>
__global__ __launch_bounds__( 512 )
void latent_attn_kernel(
	const half* __restrict__ q_lat,
	const void* __restrict__ cache,
	const void* __restrict__ cscale,
	half* __restrict__ o_lat,
	float* __restrict__ workspace,
	size_t slot_stride,
	int layer,
	int max_seq,
	const pod::BatchRows* __restrict__ rows,
	int row0,
	int n_chunks,
	float scale,
	const int* __restrict__ sel_idx,
	const int* __restrict__ sel_n
) {
	extern __shared__ __align__( 16 ) unsigned char smem_raw[];
	float* sq = reinterpret_cast<float*>( smem_raw );                       // [HEADS][LORA]
	half* sc = reinterpret_cast<half*>( sq + HEADS * LORA );                // [TILE][CS]
	float* ss = reinterpret_cast<float*>( sc + TILE * CS );                 // [HEADS][TILE] scores, then probabilities
	float* sal = ss + HEADS * TILE;                                         // [HEADS] rescale factors
	const int m = blockIdx.x, chunk = blockIdx.y, t = threadIdx.x;
	const int pos = rows->pos[row0 + m];
	// DSA rows (sel_n >= 0) attend to a list of token indices, others to every position up to their own
	const int nsel = sel_n ? sel_n[row0 + m] : -1;
	const int* lst = sel_idx ? sel_idx + static_cast<size_t>( row0 + m ) * DSA_KPAD : nullptr;
	const int total = nsel >= 0 ? nsel : min( pos + 1, max_seq );
	const int kv_start = chunk * CHUNK_KEYS;
	const int kv_end = min( kv_start + CHUNK_KEYS, total );
	const size_t base = static_cast<size_t>( rows->slot[row0 + m] ) * slot_stride + static_cast<size_t>( layer ) * max_seq * LORA;

	float o[HEADS];
	#pragma unroll
	for ( int h = 0; h < HEADS; ++h ) o[h] = 0.0f;
	float m_run = -1e30f, l_run = 0.0f; // meaningful on t < HEADS

	if ( kv_start < total ) {
		for ( int i = t; i < HEADS * LORA; i += 512 ) sq[i] = __half2float( q_lat[( static_cast<size_t>( m ) * HEADS ) * LORA + i] );
		for ( int kb = kv_start; kb < kv_end; kb += TILE ) {
			const int nk = min( TILE, kv_end - kb );
			__syncthreads();
			for ( int u = t; u < TILE * ( LORA / 8 ); u += 512 ) {
				const int k = u / ( LORA / 8 ), c8 = u % ( LORA / 8 );
				uint4 raw = make_uint4( 0, 0, 0, 0 );
				if ( k < nk ) {
					const int key = nsel >= 0 ? lst[kb + k] : kb + k;
					raw = latent_ld8<Q>( cache, cscale, base + static_cast<size_t>( key ) * LORA + c8 * 8 );
				}
				*reinterpret_cast<uint4*>( sc + k * CS + c8 * 8 ) = raw;
			}
			__syncthreads();
			// Scores: pair p = (head p / TILE, key p % TILE), lanes 2p and 2p + 1 take half of the dimensions each
			{
				const int p = t >> 1, half_sel = t & 1;
				const int h = p / TILE, k = p % TILE;
				const float* qh = sq + h * LORA + half_sel * 256;
				const half* ck = sc + k * CS + half_sel * 256;
				float dot = 0.0f;
				#pragma unroll 4
				for ( int d = 0; d < 256; d += 8 ) {
					const uint4 craw = *reinterpret_cast<const uint4*>( ck + d );
					const half2* c2 = reinterpret_cast<const half2*>( &craw );
					const float4 qa = *reinterpret_cast<const float4*>( qh + d );
					const float4 qb = *reinterpret_cast<const float4*>( qh + d + 4 );
					const float2 f0 = __half22float2( c2[0] ), f1 = __half22float2( c2[1] ), f2 = __half22float2( c2[2] ), f3 = __half22float2( c2[3] );
					dot = fmaf( qa.x, f0.x, dot ); dot = fmaf( qa.y, f0.y, dot );
					dot = fmaf( qa.z, f1.x, dot ); dot = fmaf( qa.w, f1.y, dot );
					dot = fmaf( qb.x, f2.x, dot ); dot = fmaf( qb.y, f2.y, dot );
					dot = fmaf( qb.z, f3.x, dot ); dot = fmaf( qb.w, f3.y, dot );
				}
				dot += __shfl_xor_sync( 0xffffffff, dot, 1 );
				if ( half_sel == 0 ) ss[h * TILE + k] = ( k < nk ) ? dot * scale : -1e30f;
			}
			__syncthreads();
			if ( t < HEADS ) {
				float tile_max = -1e30f;
				#pragma unroll
				for ( int k = 0; k < TILE; ++k ) tile_max = fmaxf( tile_max, ss[t * TILE + k] );
				const float m_new = fmaxf( m_run, tile_max );
				const float alpha = __expf( m_run - m_new );
				float lsum = 0.0f;
				#pragma unroll
				for ( int k = 0; k < TILE; ++k ) {
					const float pk = ( k < nk ) ? __expf( ss[t * TILE + k] - m_new ) : 0.0f;
					ss[t * TILE + k] = pk;
					lsum += pk;
				}
				l_run = l_run * alpha + lsum;
				m_run = m_new;
				sal[t] = alpha;
			}
			__syncthreads();
			#pragma unroll
			for ( int h = 0; h < HEADS; ++h ) o[h] *= sal[h];
			for ( int k = 0; k < nk; ++k ) {
				const float cv = __half2float( sc[k * CS + t] );
				#pragma unroll
				for ( int h = 0; h < HEADS; ++h ) o[h] = fmaf( ss[h * TILE + k], cv, o[h] );
			}
		}
	}

	if ( n_chunks == 1 ) {
		// Row's whole context in one chunk: write the normalized result
		__shared__ float s_inv[HEADS];
		if ( t < HEADS ) s_inv[t] = l_run > 0.0f ? 1.0f / l_run : 0.0f;
		__syncthreads();
		#pragma unroll
		for ( int h = 0; h < HEADS; ++h ) o_lat[( static_cast<size_t>( m ) * HEADS + h ) * LORA + t] = __float2half_rn( o[h] * s_inv[h] );
	}
	else {
		#pragma unroll
		for ( int h = 0; h < HEADS; ++h ) {
			float* ws = workspace + ( ( static_cast<size_t>( m ) * HEADS + h ) * n_chunks + chunk ) * ( LORA + 2 );
			ws[2 + t] = o[h];
			if ( t == h ) {
				ws[0] = m_run;
				ws[1] = l_run;
			}
		}
	}
}


// Tensor-core prefill attention in latent space (Volta mma.m8n8k4, FP32 accumulate). A block owns 32 queries = 2 token rows x 16
// heads, which share every key, so a 32-key tile of the latent cache serves all 32. S = Q c^T: 8 warps, each a 16 x 16 tile over
// half of the 512 dimensions (the two halves are summed through shared memory). The softmax runs 8 lanes per query row. O += P c
// gives each warp 64 of the 512 latent dimensions (interleaved d = 64w + 32jj + qp + 4n, so the transposed rows are bank
// conflict free). K and V are the same data, so one shared buffer holds the tile first as c[key][d] (for S) and then as
// c^T[d][key] (for PV), refilled from the raw words the thread keeps in registers; the next tile is fetched during PV.
// SM_80+ (MMA_SM80): the same tiles on mma.m16n8k16 read from the same shared buffers. S: warp (dh, qg, kg) is one 16-row A tile of Q
// against two 8-key B tiles of c. PV: each warp's 64 dimensions are 64w + 8j + n (j = 0..7), the 32 queries two 16-row tiles.
#define LT_BQ 32
#define LT_BK 32
#define LT_KS 516
#define LT_VS 36
#define LT_SS 36
#define LT_OFF_Q 36864
#define LT_OFF_S ( LT_OFF_Q + LT_BQ * LT_KS * 2 )
#define LT_OFF_P ( LT_OFF_S + 2 * LT_BQ * LT_SS * 4 )
#define LT_OFF_A ( LT_OFF_P + LT_BQ * LT_VS * 2 )
#define LT_SMEM ( LT_OFF_A + LT_BQ * 12 )

__device__ __forceinline__ void lt_mma( float* c, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1 ) {
#if defined(USE_ROCM)
	const uint32_t a[2] = { a0, a1 }, b[2] = { b0, b1 };
	mma_emu_m8n8k4_f32( a, b, c, threadIdx.x & 31, rocm_shfl() );
#else
	asm volatile (
		"mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
		"{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
		: "+f"( c[0] ), "+f"( c[1] ), "+f"( c[2] ), "+f"( c[3] ), "+f"( c[4] ), "+f"( c[5] ), "+f"( c[6] ), "+f"( c[7] )
		: "r"( a0 ), "r"( a1 ), "r"( b0 ), "r"( b1 )
	);
#endif
}

template <KVQ Q>
__global__ __launch_bounds__( 256 )
void latent_prefill_tc_kernel(
	const half* __restrict__ q_lat,
	const void* __restrict__ cache,
	const void* __restrict__ cscale,
	half* __restrict__ o_lat,
	float* __restrict__ workspace,
	size_t slot_stride,
	int layer,
	int max_seq,
	const pod::BatchRows* __restrict__ rows,
	int count,
	float scale,
	int nsplit
) {
	constexpr int HEADS = 16;
	extern __shared__ __align__( 16 ) unsigned char lt_smem[];
	half* sB = reinterpret_cast<half*>( lt_smem );
	half* sQ = reinterpret_cast<half*>( lt_smem + LT_OFF_Q );
	float* sS = reinterpret_cast<float*>( lt_smem + LT_OFF_S );
	half* sP = reinterpret_cast<half*>( lt_smem + LT_OFF_P );
	float* sA = reinterpret_cast<float*>( lt_smem + LT_OFF_A );
	float* sM = sA + LT_BQ;
	float* sL = sM + LT_BQ;
	float* sO = reinterpret_cast<float*>( lt_smem );

	const int q_base = blockIdx.x * LT_BQ;
	const int tid = threadIdx.x;
	const int warp = tid >> 5;
	const int lane = tid & 31;
	const int t = lane & 3;
	const int hi = lane >> 4;
	const int e = t + 4 * hi;
	const int qp = ( lane >> 2 ) & 3;

	const int last_row = min( ( q_base + LT_BQ - 1 ) / HEADS, count - 1 );
	int max_pos = 0;
	for ( int r = q_base / HEADS; r <= last_row; ++r ) max_pos = max( max_pos, rows->pos[r] );
	max_pos = min( max_pos, max_seq - 1 );
	const size_t kvb = static_cast<size_t>( rows->slot[q_base / HEADS] ) * slot_stride + static_cast<size_t>( layer ) * max_seq * LORA;

	// Softmax role: 8 lanes per query row, 4 keys each
	const int srow = tid >> 3;
	const int sub = tid & 7;
	const int sqi = q_base + srow;
	const bool svalid = sqi < count * HEADS;
	const int spos = svalid ? rows->pos[sqi / HEADS] : -1;
	float m_run = -1e30f, l_run = 0.0f;

	for ( int u = tid; u < LT_BQ * ( LORA / 8 ); u += 256 ) {
		const int qq = u >> 6, c = u & 63;
		const int qi = q_base + qq;
		uint4 raw = make_uint4( 0, 0, 0, 0 );
		if ( qi < count * HEADS ) raw = *reinterpret_cast<const uint4*>( q_lat + static_cast<size_t>( qi ) * LORA + c * 8 );
		uint2* dst = reinterpret_cast<uint2*>( sQ + qq * LT_KS + c * 8 );
		dst[0] = make_uint2( raw.x, raw.y );
		dst[1] = make_uint2( raw.z, raw.w );
	}

	const int dh = warp >> 2, s_qg = ( warp >> 1 ) & 1, s_kg = warp & 1;
	const half* s_arow = sQ + ( s_qg * 16 + 8 * ( qp >> 1 ) + e ) * LT_KS + dh * 256;
	const half* s_brow = sB + ( s_kg * 16 + 8 * ( qp & 1 ) + e ) * LT_KS + dh * 256;
	const half* v_row0 = sB + ( 64 * warp + qp + 4 * e ) * LT_VS;
	const half* p_row = sP + e * LT_VS;

#if MMA_SM80
	float acc[2][8][4];   // [query tile][dimension tile][frag_c]
	#pragma unroll
	for ( int i = 0; i < 2; ++i ) {
		#pragma unroll
		for ( int j = 0; j < 8; ++j ) {
			#pragma unroll
			for ( int r = 0; r < 4; ++r ) acc[i][j][r] = 0.0f;
		}
	}
	(void) s_arow; (void) s_brow; (void) v_row0; (void) p_row;
#else
	float acc[2][4][8];
	#pragma unroll
	for ( int jj = 0; jj < 2; ++jj ) {
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			#pragma unroll
			for ( int r = 0; r < 8; ++r ) acc[jj][i][r] = 0.0f;
		}
	}
#endif

	// Staging role: lane = (key pair p, chunk parity cl); 4 chunks of 8 dimensions per thread, each for two keys
	const int st_p = lane & 15, st_cl = lane >> 4;
	uint4 pk[4][2];
	auto fetch = [&]( int kb ) {
		#pragma unroll
		for ( int j = 0; j < 4; ++j ) {
			const int c = 2 * warp + st_cl + 16 * j;
			#pragma unroll
			for ( int kk = 0; kk < 2; ++kk ) {
				const int key = 2 * st_p + kk;
				pk[j][kk] = ( kb + key <= max_pos ) ? latent_ld8<Q>( cache, cscale, kvb + static_cast<size_t>( kb + key ) * LORA + c * 8 ) : make_uint4( 0, 0, 0, 0 );
			}
		}
	};
	const int ntiles = ( max_pos + LT_BK ) / LT_BK;
	const int kb0 = ( blockIdx.y * ntiles / nsplit ) * LT_BK;
	const int kb1 = ( ( blockIdx.y + 1 ) * ntiles / nsplit ) * LT_BK;
	if ( kb0 < kb1 ) fetch( kb0 );
	__syncthreads();

	for ( int kb = kb0; kb < kb1; kb += LT_BK ) {
		// c[key][d] for S
		#pragma unroll
		for ( int j = 0; j < 4; ++j ) {
			const int c = 2 * warp + st_cl + 16 * j;
			#pragma unroll
			for ( int kk = 0; kk < 2; ++kk ) {
				uint2* kd = reinterpret_cast<uint2*>( sB + ( 2 * st_p + kk ) * LT_KS + c * 8 );
				kd[0] = make_uint2( pk[j][kk].x, pk[j][kk].y );
				kd[1] = make_uint2( pk[j][kk].z, pk[j][kk].w );
			}
		}
		__syncthreads();

		// S partial: this warp's 16 x 16 tile over its 256 dimensions
#if MMA_SM80
		{
			float c[2][4] = {};
			const half* qa = sQ + s_qg * 16 * LT_KS + dh * 256;
			const half* kt = sB + s_kg * 16 * LT_KS + dh * 256;
			#pragma unroll 4
			for ( int ks = 0; ks < 16; ++ks ) {
				uint32_t a[4];
				lds_frag_a( qa + 16 * ks, LT_KS, a );
				#pragma unroll
				for ( int j = 0; j < 2; ++j ) {
					uint32_t b[2];
					lds_frag_b( kt + 8 * j * LT_KS + 16 * ks, LT_KS, b );
					mma16816( c[j], a, b );
				}
			}
			#pragma unroll
			for ( int j = 0; j < 2; ++j ) {
				#pragma unroll
				for ( int r = 0; r < 4; ++r ) {
					sS[dh * LT_BQ * LT_SS + ( s_qg * 16 + frag_c_row( r, lane ) ) * LT_SS + s_kg * 16 + 8 * j + frag_c_col( r, lane )] = c[j][r];
				}
			}
		}
#else
		{
			float c[8];
			#pragma unroll
			for ( int r = 0; r < 8; ++r ) c[r] = 0.0f;
			#pragma unroll 16
			for ( int ks = 0; ks < 64; ++ks ) {
				const uint2 a = *reinterpret_cast<const uint2*>( s_arow + 4 * ks );
				const uint2 b = *reinterpret_cast<const uint2*>( s_brow + 4 * ks );
				lt_mma( c, a.x, a.y, b.x, b.y );
			}
			#pragma unroll
			for ( int r = 0; r < 8; ++r ) {
				const int m = ( t & 1 ) + 2 * ( ( r >> 1 ) & 1 ) + 4 * hi;
				const int n = ( r & 1 ) + 2 * ( t >> 1 ) + 4 * ( r >> 2 );
				sS[dh * LT_BQ * LT_SS + ( s_qg * 16 + 8 * ( qp >> 1 ) + m ) * LT_SS + s_kg * 16 + 8 * ( qp & 1 ) + n] = c[r];
			}
		}
#endif
		__syncthreads();

		// Online softmax of the tile, one query row per 8 lanes
		{
			float sv[4];
			float mx = -1e30f;
			#pragma unroll
			for ( int j = 0; j < 4; ++j ) {
				const int key = sub * 4 + j;
				const bool ok = svalid && kb + key <= spos;
				sv[j] = ok ? ( sS[srow * LT_SS + key] + sS[LT_BQ * LT_SS + srow * LT_SS + key] ) * scale : -1e30f;
				mx = fmaxf( mx, sv[j] );
			}
			#pragma unroll
			for ( int mask = 1; mask < 8; mask <<= 1 ) mx = fmaxf( mx, __shfl_xor_sync( 0xffffffff, mx, mask ) );
			const float m_new = fmaxf( m_run, mx );
			const float alpha = __expf( m_run - m_new );
			float sum = 0.0f;
			half2 ph[2];
			#pragma unroll
			for ( int j = 0; j < 2; ++j ) {
				const float p0 = sv[2 * j] > -1e29f ? __expf( sv[2 * j] - m_new ) : 0.0f;
				const float p1 = sv[2 * j + 1] > -1e29f ? __expf( sv[2 * j + 1] - m_new ) : 0.0f;
				sum += p0 + p1;
				ph[j] = __floats2half2_rn( p0, p1 );
			}
			#pragma unroll
			for ( int mask = 1; mask < 8; mask <<= 1 ) sum += __shfl_xor_sync( 0xffffffff, sum, mask );
			l_run = l_run * alpha + sum;
			m_run = m_new;
			*reinterpret_cast<uint2*>( sP + srow * LT_VS + sub * 4 ) = make_uint2( *reinterpret_cast<uint32_t*>( &ph[0] ), *reinterpret_cast<uint32_t*>( &ph[1] ) );
			if ( sub == 0 ) sA[srow] = alpha;
		}
		__syncthreads();

		// c^T[d][key] for PV (the S reads of the buffer are done)
		#pragma unroll
		for ( int j = 0; j < 4; ++j ) {
			const int c = 2 * warp + st_cl + 16 * j;
			const half* v0 = reinterpret_cast<const half*>( &pk[j][0] );
			const half* v1 = reinterpret_cast<const half*>( &pk[j][1] );
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) {
				*reinterpret_cast<half2*>( sB + ( c * 8 + i ) * LT_VS + 2 * st_p ) = __halves2half2( v0[i], v1[i] );
			}
		}
		__syncthreads();
		if ( kb + LT_BK < kb1 ) fetch( kb + LT_BK );

		// O = O * alpha + P c for this warp's 64 dimensions
#if MMA_SM80
		#pragma unroll
		for ( int i = 0; i < 2; ++i ) {
			#pragma unroll
			for ( int j = 0; j < 8; ++j ) {
				#pragma unroll
				for ( int r = 0; r < 4; ++r ) acc[i][j][r] *= sA[16 * i + frag_c_row( r, lane )];
			}
		}
		#pragma unroll
		for ( int kk = 0; kk < LT_BK / 16; ++kk ) {
			uint32_t a[2][4];
			#pragma unroll
			for ( int i = 0; i < 2; ++i ) lds_frag_a( sP + 16 * i * LT_VS + 16 * kk, LT_VS, a[i] );
			#pragma unroll
			for ( int j = 0; j < 8; ++j ) {
				uint32_t b[2];
				lds_frag_b( sB + ( 64 * warp + 8 * j ) * LT_VS + 16 * kk, LT_VS, b );
				#pragma unroll
				for ( int i = 0; i < 2; ++i ) mma16816( acc[i][j], a[i], b );
			}
		}
#else
		#pragma unroll
		for ( int jj = 0; jj < 2; ++jj ) {
			#pragma unroll
			for ( int i = 0; i < 4; ++i ) {
				#pragma unroll
				for ( int r = 0; r < 8; ++r ) {
					const int m = ( t & 1 ) + 2 * ( ( r >> 1 ) & 1 ) + 4 * hi;
					acc[jj][i][r] *= sA[8 * i + m];
				}
			}
		}
		#pragma unroll
		for ( int kk = 0; kk < LT_BK / 4; ++kk ) {
			uint2 b[2];
			#pragma unroll
			for ( int jj = 0; jj < 2; ++jj ) b[jj] = *reinterpret_cast<const uint2*>( v_row0 + 32 * jj * LT_VS + 4 * kk );
			#pragma unroll
			for ( int i = 0; i < 4; ++i ) {
				const uint2 a = *reinterpret_cast<const uint2*>( p_row + 8 * i * LT_VS + 4 * kk );
				#pragma unroll
				for ( int jj = 0; jj < 2; ++jj ) lt_mma( acc[jj][i], a.x, a.y, b[jj].x, b[jj].y );
			}
		}
#endif
		__syncthreads();
	}

	if ( sub == 0 ) {
		sA[srow] = nsplit > 1 ? 1.0f : ( l_run > 0.0f ? 1.0f / l_run : 0.0f );
		sM[srow] = m_run;
		sL[srow] = l_run;
	}
	__syncthreads();
#if MMA_SM80
	#pragma unroll
	for ( int i = 0; i < 2; ++i ) {
		#pragma unroll
		for ( int j = 0; j < 8; ++j ) {
			#pragma unroll
			for ( int r = 0; r < 4; ++r ) {
				const int m = 16 * i + frag_c_row( r, lane );
				sO[m * LT_KS + 64 * warp + 8 * j + frag_c_col( r, lane )] = acc[i][j][r] * sA[m];
			}
		}
	}
#else
	#pragma unroll
	for ( int jj = 0; jj < 2; ++jj ) {
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			#pragma unroll
			for ( int r = 0; r < 8; ++r ) {
				const int m = ( t & 1 ) + 2 * ( ( r >> 1 ) & 1 ) + 4 * hi;
				const int n = ( r & 1 ) + 2 * ( t >> 1 ) + 4 * ( r >> 2 );
				sO[( 8 * i + m ) * LT_KS + 64 * warp + 32 * jj + qp + 4 * n] = acc[jj][i][r] * sA[8 * i + m];
			}
		}
	}
#endif
	__syncthreads();

	if ( nsplit > 1 ) {
		// Unnormalized partial (m, l, O) in the layout of latent_reduce_kernel
		for ( int u = tid; u < LT_BQ * LORA; u += 256 ) {
			const int qq = u >> 9, d = u & 511;
			const int qi = q_base + qq;
			if ( qi >= count * HEADS ) continue;
			float* ws = workspace + ( static_cast<size_t>( qi ) * nsplit + blockIdx.y ) * ( LORA + 2 );
			if ( d == 0 ) {
				ws[0] = sM[qq];
				ws[1] = sL[qq];
			}
			ws[2 + d] = sO[qq * LT_KS + d];
		}
		return;
	}
	for ( int u = tid; u < LT_BQ * ( LORA / 8 ); u += 256 ) {
		const int qq = u >> 6, c = u & 63;
		const int qi = q_base + qq;
		if ( qi >= count * HEADS ) continue;
		const float* o = sO + qq * LT_KS + c * 8;
		uint4 res;
		half2* r2 = reinterpret_cast<half2*>( &res );
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) r2[i] = __floats2half2_rn( o[2 * i], o[2 * i + 1] );
		*reinterpret_cast<uint4*>( o_lat + static_cast<size_t>( qi ) * LORA + c * 8 ) = res;
	}
}


// Sparse (DSA) variant of the tensor-core latent attention: one block per row, its 16 heads as the 16 queries, and the row's own list
// of key positions (nsel >= 0) or every position up to its own (nsel < 0) walked in tiles of 32 gathered keys. Tile handling, the
// time-multiplexed K / V^T buffer and the softmax follow latent_prefill_tc_kernel; S is split four ways over the dimensions
// (each warp 16 x 16 over 128 of them) because only 16 queries share the tile.
#define LS_BQ 16
#define LS_OFF_Q 36864
#define LS_OFF_S ( LS_OFF_Q + LS_BQ * LT_KS * 2 )
#define LS_OFF_P ( LS_OFF_S + 4 * LS_BQ * LT_SS * 4 )
#define LS_OFF_A ( LS_OFF_P + LS_BQ * LT_VS * 2 )
#define LS_SMEM ( LS_OFF_A + LS_BQ * 12 )

template <KVQ Q>
__global__ __launch_bounds__( 256 )
void latent_sparse_tc_kernel(
	const half* __restrict__ q_lat,
	const void* __restrict__ cache,
	const void* __restrict__ cscale,
	half* __restrict__ o_lat,
	float* __restrict__ workspace,
	size_t slot_stride,
	int layer,
	int max_seq,
	const pod::BatchRows* __restrict__ rows,
	float scale,
	int nsplit,
	const int* __restrict__ sel_idx,
	const int* __restrict__ sel_n
) {
	constexpr int HEADS = 16;
	extern __shared__ __align__( 16 ) unsigned char ls_smem[];
	half* sB = reinterpret_cast<half*>( ls_smem );
	half* sQ = reinterpret_cast<half*>( ls_smem + LS_OFF_Q );
	float* sS = reinterpret_cast<float*>( ls_smem + LS_OFF_S );
	half* sP = reinterpret_cast<half*>( ls_smem + LS_OFF_P );
	float* sA = reinterpret_cast<float*>( ls_smem + LS_OFF_A );
	float* sM = sA + LS_BQ;
	float* sL = sM + LS_BQ;
	float* sO = reinterpret_cast<float*>( ls_smem );

	const int r = blockIdx.x;
	const int tid = threadIdx.x;
	const int warp = tid >> 5;
	const int lane = tid & 31;
	const int t = lane & 3;
	const int hi = lane >> 4;
	const int e = t + 4 * hi;
	const int qp = ( lane >> 2 ) & 3;

	const int nsel = sel_n ? sel_n[r] : -1;
	// A row below the selection threshold (nsel < 0) attends densely even though the list pointer is set
	const int* lst = ( sel_idx && nsel >= 0 ) ? sel_idx + static_cast<size_t>( r ) * DSA_KPAD : nullptr;
	const int n_keys = nsel >= 0 ? nsel : min( rows->pos[r] + 1, max_seq );
	const size_t kvb = static_cast<size_t>( rows->slot[r] ) * slot_stride + static_cast<size_t>( layer ) * max_seq * LORA;

	for ( int u = tid; u < LS_BQ * ( LORA / 8 ); u += 256 ) {
		const int qq = u >> 6, c = u & 63;
		const uint4 raw = *reinterpret_cast<const uint4*>( q_lat + ( static_cast<size_t>( r ) * HEADS + qq ) * LORA + c * 8 );
		uint2* dst = reinterpret_cast<uint2*>( sQ + qq * LT_KS + c * 8 );
		dst[0] = make_uint2( raw.x, raw.y );
		dst[1] = make_uint2( raw.z, raw.w );
	}

	const int s_kg = warp & 1, dq = warp >> 1;
	const half* s_arow = sQ + ( 8 * ( qp >> 1 ) + e ) * LT_KS + dq * 128;
	const half* s_brow = sB + ( s_kg * 16 + 8 * ( qp & 1 ) + e ) * LT_KS + dq * 128;
	const half* v_row0 = sB + ( 64 * warp + qp + 4 * e ) * LT_VS;
	const half* p_row = sP + e * LT_VS;

	// Softmax role: 16 lanes per query row, 2 keys each
	const int srow = tid >> 4, sub = tid & 15;
	float m_run = -1e30f, l_run = 0.0f;

#if MMA_SM80
	float acc[8][4];   // [dimension tile][frag_c], the 16 queries one row tile
	#pragma unroll
	for ( int j = 0; j < 8; ++j ) {
		#pragma unroll
		for ( int q = 0; q < 4; ++q ) acc[j][q] = 0.0f;
	}
	(void) s_arow; (void) s_brow; (void) v_row0; (void) p_row;
#else
	float acc[2][2][8];
	#pragma unroll
	for ( int jj = 0; jj < 2; ++jj ) {
		#pragma unroll
		for ( int i = 0; i < 2; ++i ) {
			#pragma unroll
			for ( int q = 0; q < 8; ++q ) acc[jj][i][q] = 0.0f;
		}
	}
#endif

	const int st_p = lane & 15, st_cl = lane >> 4;
	uint4 pk[4][2];
	auto fetch = [&]( int kb ) {
		int kpos[2];
		#pragma unroll
		for ( int kk = 0; kk < 2; ++kk ) {
			const int j = kb + 2 * st_p + kk;
			kpos[kk] = j < n_keys ? ( lst ? lst[j] : j ) : -1;
		}
		#pragma unroll
		for ( int j = 0; j < 4; ++j ) {
			const int c = 2 * warp + st_cl + 16 * j;
			#pragma unroll
			for ( int kk = 0; kk < 2; ++kk ) {
				pk[j][kk] = kpos[kk] >= 0 ? latent_ld8<Q>( cache, cscale, kvb + static_cast<size_t>( kpos[kk] ) * LORA + c * 8 ) : make_uint4( 0, 0, 0, 0 );
			}
		}
	};
	const int ntiles = ( n_keys + LT_BK - 1 ) / LT_BK;
	const int kb0 = ( blockIdx.y * ntiles / nsplit ) * LT_BK;
	const int kb1 = ( ( blockIdx.y + 1 ) * ntiles / nsplit ) * LT_BK;
	if ( kb0 < kb1 ) fetch( kb0 );
	__syncthreads();

	for ( int kb = kb0; kb < kb1; kb += LT_BK ) {
		#pragma unroll
		for ( int j = 0; j < 4; ++j ) {
			const int c = 2 * warp + st_cl + 16 * j;
			#pragma unroll
			for ( int kk = 0; kk < 2; ++kk ) {
				uint2* kd = reinterpret_cast<uint2*>( sB + ( 2 * st_p + kk ) * LT_KS + c * 8 );
				kd[0] = make_uint2( pk[j][kk].x, pk[j][kk].y );
				kd[1] = make_uint2( pk[j][kk].z, pk[j][kk].w );
			}
		}
		__syncthreads();

#if MMA_SM80
		{
			float c[2][4] = {};
			const half* qa = sQ + dq * 128;
			const half* kt = sB + s_kg * 16 * LT_KS + dq * 128;
			#pragma unroll
			for ( int ks = 0; ks < 8; ++ks ) {
				uint32_t a[4];
				lds_frag_a( qa + 16 * ks, LT_KS, a );
				#pragma unroll
				for ( int j = 0; j < 2; ++j ) {
					uint32_t b[2];
					lds_frag_b( kt + 8 * j * LT_KS + 16 * ks, LT_KS, b );
					mma16816( c[j], a, b );
				}
			}
			#pragma unroll
			for ( int j = 0; j < 2; ++j ) {
				#pragma unroll
				for ( int q = 0; q < 4; ++q ) {
					sS[dq * LS_BQ * LT_SS + frag_c_row( q, lane ) * LT_SS + s_kg * 16 + 8 * j + frag_c_col( q, lane )] = c[j][q];
				}
			}
		}
#else
		{
			float c[8];
			#pragma unroll
			for ( int q = 0; q < 8; ++q ) c[q] = 0.0f;
			#pragma unroll 16
			for ( int ks = 0; ks < 32; ++ks ) {
				const uint2 a = *reinterpret_cast<const uint2*>( s_arow + 4 * ks );
				const uint2 b = *reinterpret_cast<const uint2*>( s_brow + 4 * ks );
				lt_mma( c, a.x, a.y, b.x, b.y );
			}
			#pragma unroll
			for ( int q = 0; q < 8; ++q ) {
				const int m = ( t & 1 ) + 2 * ( ( q >> 1 ) & 1 ) + 4 * hi;
				const int n = ( q & 1 ) + 2 * ( t >> 1 ) + 4 * ( q >> 2 );
				sS[dq * LS_BQ * LT_SS + ( 8 * ( qp >> 1 ) + m ) * LT_SS + s_kg * 16 + 8 * ( qp & 1 ) + n] = c[q];
			}
		}
#endif
		__syncthreads();

		{
			float sv[2];
			float mx = -1e30f;
			#pragma unroll
			for ( int j = 0; j < 2; ++j ) {
				const int key = sub * 2 + j;
				float sum4 = 0.0f;
				#pragma unroll
				for ( int d4 = 0; d4 < 4; ++d4 ) sum4 += sS[d4 * LS_BQ * LT_SS + srow * LT_SS + key];
				sv[j] = ( kb + key < n_keys ) ? sum4 * scale : -1e30f;
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
			*reinterpret_cast<half2*>( sP + srow * LT_VS + sub * 2 ) = __floats2half2_rn( p0, p1 );
			if ( sub == 0 ) sA[srow] = alpha;
		}
		__syncthreads();

		#pragma unroll
		for ( int j = 0; j < 4; ++j ) {
			const int c = 2 * warp + st_cl + 16 * j;
			const half* v0 = reinterpret_cast<const half*>( &pk[j][0] );
			const half* v1 = reinterpret_cast<const half*>( &pk[j][1] );
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) {
				*reinterpret_cast<half2*>( sB + ( c * 8 + i ) * LT_VS + 2 * st_p ) = __halves2half2( v0[i], v1[i] );
			}
		}
		__syncthreads();
		if ( kb + LT_BK < kb1 ) fetch( kb + LT_BK );

#if MMA_SM80
		#pragma unroll
		for ( int j = 0; j < 8; ++j ) {
			#pragma unroll
			for ( int q = 0; q < 4; ++q ) acc[j][q] *= sA[frag_c_row( q, lane )];
		}
		#pragma unroll
		for ( int kk = 0; kk < LT_BK / 16; ++kk ) {
			uint32_t a[4];
			lds_frag_a( sP + 16 * kk, LT_VS, a );
			#pragma unroll
			for ( int j = 0; j < 8; ++j ) {
				uint32_t b[2];
				lds_frag_b( sB + ( 64 * warp + 8 * j ) * LT_VS + 16 * kk, LT_VS, b );
				mma16816( acc[j], a, b );
			}
		}
#else
		#pragma unroll
		for ( int jj = 0; jj < 2; ++jj ) {
			#pragma unroll
			for ( int i = 0; i < 2; ++i ) {
				#pragma unroll
				for ( int q = 0; q < 8; ++q ) {
					const int m = ( t & 1 ) + 2 * ( ( q >> 1 ) & 1 ) + 4 * hi;
					acc[jj][i][q] *= sA[8 * i + m];
				}
			}
		}
		#pragma unroll
		for ( int kk = 0; kk < LT_BK / 4; ++kk ) {
			uint2 b[2];
			#pragma unroll
			for ( int jj = 0; jj < 2; ++jj ) b[jj] = *reinterpret_cast<const uint2*>( v_row0 + 32 * jj * LT_VS + 4 * kk );
			#pragma unroll
			for ( int i = 0; i < 2; ++i ) {
				const uint2 a = *reinterpret_cast<const uint2*>( p_row + 8 * i * LT_VS + 4 * kk );
				#pragma unroll
				for ( int jj = 0; jj < 2; ++jj ) lt_mma( acc[jj][i], a.x, a.y, b[jj].x, b[jj].y );
			}
		}
#endif
		__syncthreads();
	}

	if ( sub == 0 ) {
		sA[srow] = nsplit > 1 ? 1.0f : ( l_run > 0.0f ? 1.0f / l_run : 0.0f );
		sM[srow] = m_run;
		sL[srow] = l_run;
	}
	__syncthreads();
#if MMA_SM80
	#pragma unroll
	for ( int j = 0; j < 8; ++j ) {
		#pragma unroll
		for ( int q = 0; q < 4; ++q ) {
			const int m = frag_c_row( q, lane );
			sO[m * LT_KS + 64 * warp + 8 * j + frag_c_col( q, lane )] = acc[j][q] * sA[m];
		}
	}
#else
	#pragma unroll
	for ( int jj = 0; jj < 2; ++jj ) {
		#pragma unroll
		for ( int i = 0; i < 2; ++i ) {
			#pragma unroll
			for ( int q = 0; q < 8; ++q ) {
				const int m = ( t & 1 ) + 2 * ( ( q >> 1 ) & 1 ) + 4 * hi;
				const int n = ( q & 1 ) + 2 * ( t >> 1 ) + 4 * ( q >> 2 );
				sO[( 8 * i + m ) * LT_KS + 64 * warp + 32 * jj + qp + 4 * n] = acc[jj][i][q] * sA[8 * i + m];
			}
		}
	}
#endif
	__syncthreads();

	if ( nsplit > 1 ) {
		for ( int u = tid; u < LS_BQ * LORA; u += 256 ) {
			const int qq = u >> 9, d = u & 511;
			float* ws = workspace + ( ( static_cast<size_t>( r ) * HEADS + qq ) * nsplit + blockIdx.y ) * ( LORA + 2 );
			if ( d == 0 ) {
				ws[0] = sM[qq];
				ws[1] = sL[qq];
			}
			ws[2 + d] = sO[qq * LT_KS + d];
		}
		return;
	}
	for ( int u = tid; u < LS_BQ * ( LORA / 8 ); u += 256 ) {
		const int qq = u >> 6, c = u & 63;
		const float* o = sO + qq * LT_KS + c * 8;
		uint4 res;
		half2* r2 = reinterpret_cast<half2*>( &res );
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) r2[i] = __floats2half2_rn( o[2 * i], o[2 * i + 1] );
		*reinterpret_cast<uint4*>( o_lat + ( static_cast<size_t>( r ) * HEADS + qq ) * LORA + c * 8 ) = res;
	}
}


// Diagnostic: dense attention logits of one row, scores[h][t] = q_lat[h] . c_t * scale for t < n
__global__ __launch_bounds__( 256 )
void diag_scores_kernel( const half* __restrict__ q_lat, const half* __restrict__ kv, float* __restrict__ scores, int n, int heads, float scale ) {
	__shared__ float sq[LORA];
	const int h = blockIdx.y;
	for ( int i = threadIdx.x; i < LORA; i += 256 ) sq[i] = __half2float( q_lat[static_cast<size_t>( h ) * LORA + i] );
	__syncthreads();
	const int t = blockIdx.x * 256 + threadIdx.x;
	if ( t >= n ) return;
	float acc = 0.0f;
	for ( int c = 0; c < LORA; c += 8 ) {
		const uint4 raw = *reinterpret_cast<const uint4*>( kv + static_cast<size_t>( t ) * LORA + c );
		const half2* h2 = reinterpret_cast<const half2*>( &raw );
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			const float2 f = __half22float2( h2[i] );
			acc = fmaf( sq[c + 2 * i], f.x, acc );
			acc = fmaf( sq[c + 2 * i + 1], f.y, acc );
		}
	}
	scores[static_cast<size_t>( h ) * n + t] = acc * scale;
}

// Merge a row's chunk partials: out = sum_c exp(m_c - m) o_c / sum_c exp(m_c - m) l_c; empty chunks (l = 0) are skipped.
__global__ __launch_bounds__( 512 )
void latent_reduce_kernel( const float* __restrict__ workspace, half* __restrict__ o_lat, int heads, int n_chunks ) {
	const int m = blockIdx.y, h = blockIdx.x, t = threadIdx.x;
	const float* ws = workspace + ( static_cast<size_t>( m ) * heads + h ) * n_chunks * ( LORA + 2 );
	float mx = -1e30f;
	for ( int c = 0; c < n_chunks; ++c ) {
		if ( ws[c * ( LORA + 2 ) + 1] > 0.0f ) mx = fmaxf( mx, ws[c * ( LORA + 2 )] );
	}
	float l = 0.0f, acc = 0.0f;
	for ( int c = 0; c < n_chunks; ++c ) {
		const float lc = ws[c * ( LORA + 2 ) + 1];
		if ( lc <= 0.0f ) continue;
		const float w = __expf( ws[c * ( LORA + 2 )] - mx );
		l += w * lc;
		acc += w * ws[c * ( LORA + 2 ) + 2 + t];
	}
	o_lat[( static_cast<size_t>( m ) * heads + h ) * LORA + t] = __float2half_rn( l > 0.0f ? acc / l : 0.0f );
}

} // namespace

void MLALatent::absorb_q( const pod::TensorView& q, const pod::TensorView& kv_b, pod::TensorView& q_lat, int heads, int rows, cudaStream_t stream ) {
	absorb_q_kernel<<<dim3( heads, LORA / 16, ( rows + RCH - 1 ) / RCH ), dim3( 128 ), 0, stream>>>( q.ptr<const half>(), kv_b.ptr<const half>(), q_lat.ptr<half>(), heads, rows );
}

void MLALatent::expand_out( const pod::TensorView& o_lat, const pod::TensorView& kv_b, pod::TensorView& out, int heads, int rows, cudaStream_t stream ) {
	expand_out_kernel<<<dim3( heads, HEAD / 16, ( rows + RCH - 1 ) / RCH ), dim3( 128 ), 0, stream>>>( o_lat.ptr<const half>(), kv_b.ptr<const half>(), out.ptr<half>(), heads, rows );
}

void MLALatent::store( const pod::KVRef& kv, const pod::TensorView& c, int layer, int max_seq, int rows, const pod::SlotRef& slot, cudaStream_t stream ) {
	ENGINE_ASSERT( slot.rows != nullptr, "MLALatent::store: device-resolved positions only" );
	ENGINE_ASSERT( kv.kq == pod::KVQ::F16 || kv.kq == pod::KVQ::E5M2 || kv.kq == pod::KVQ::I8 || kv.kq == pod::KVQ::FP4, "MLALatent::store: cache format" );
	switch ( kv.kq ) {
		case pod::KVQ::E5M2: store_latent_kernel<KVQ::E5M2><<<dim3( rows ), dim3( 128 ), 0, stream>>>( kv.k, kv.ks, c.ptr<const half>(), layer, max_seq, rows, slot ); break;
		case pod::KVQ::I8: store_latent_kernel<KVQ::I8><<<dim3( rows ), dim3( 128 ), 0, stream>>>( kv.k, kv.ks, c.ptr<const half>(), layer, max_seq, rows, slot ); break;
		case pod::KVQ::FP4: store_latent_kernel<KVQ::FP4><<<dim3( rows ), dim3( 128 ), 0, stream>>>( kv.k, kv.ks, c.ptr<const half>(), layer, max_seq, rows, slot ); break;
		default: store_latent_kernel<KVQ::F16><<<dim3( rows ), dim3( 128 ), 0, stream>>>( kv.k, kv.ks, c.ptr<const half>(), layer, max_seq, rows, slot ); break;
	}
}

template <KVQ Q>
void attend_q( const pod::KVRef& kv, const pod::TensorView& q_lat, pod::TensorView& o_lat, int layer, int max_seq, int rows, int heads,
	const pod::SlotRef& slot, float* workspace, int key_hint, cudaStream_t stream, const int* sel_idx, const int* sel_n ) {
	const void* cache = kv.k;
	const void* cscale = kv.ks;
	ENGINE_ASSERT( heads == 16 && slot.rows != nullptr, "MLALatent::attend: 16 heads per rank, device-resolved positions" );
	const int n_chunks = MLALatent::workspace_chunks( max_seq );
	const float scale = 1.0f / sqrtf( static_cast<float>( MLALatent::HEAD ) );
	const size_t smem = static_cast<size_t>( 16 ) * LORA * 4 + static_cast<size_t>( TILE ) * CS * 2 + ( 16 * TILE + 16 ) * 4;
	static bool attr[ENGINE_NUM_GPUS] = {};
	int device = 0;
	(void) cudaGetDevice( &device );
	if ( !attr[device] ) {
		CUDA_CHECK( cudaFuncSetAttribute( latent_attn_kernel<16, Q>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>( smem ) ) );
		attr[device] = true;
	}
	// Decode rows (up to MAX_BATCH) take the same per-row tensor-core kernel: its blocks split the keys of a row, where
	// the chunked kernel below walks 1024 keys per block (8 blocks at 8K keys, 540 us per layer). It resolves each
	// row's own slot, so verify blocks and rows of distinct slots both qualify.
	const bool decode_rows = rows <= pod::MAX_BATCH && slot.row == 0;
#if defined(USE_ROCM)
	// RDNA grants a workgroup 64 KB: the two-row prefill tile (~80 KB) does not fit, the per-row kernel (~62 KB, dense when sel_idx is
	// null) does and computes the same attention
	const bool prefill_per_row = rows > pod::MAX_BATCH && LT_SMEM > rocm_smem_optin_max();
#else
	const bool prefill_per_row = false;
#endif
	if ( decode_rows || ( rows > pod::MAX_BATCH && ( sel_idx != nullptr || prefill_per_row ) ) ) {
		ENGINE_ASSERT( slot.row == 0, "MLALatent::attend: prefill chunks start at row 0" );
		static bool sp_attr[ENGINE_NUM_GPUS] = {};
		if ( !sp_attr[device] ) {
			CUDA_CHECK( cudaFuncSetAttribute( latent_sparse_tc_kernel<Q>, cudaFuncAttributeMaxDynamicSharedMemorySize, LS_SMEM ) );
			sp_attr[device] = true;
		}
		int nsplit = rows < 160 ? std::min( ( 320 + rows - 1 ) / rows, decode_rows ? 32 : 8 ) : 1;
		while ( nsplit > 1 && static_cast<size_t>( rows ) * heads * nsplit > MLALatent::workspace_slots( max_seq, heads ) ) --nsplit;
		latent_sparse_tc_kernel<Q><<<dim3( rows, nsplit ), dim3( 256 ), LS_SMEM, stream>>>(
			q_lat.ptr<const half>(), cache, cscale, o_lat.ptr<half>(), workspace, slot.stride, layer, max_seq, slot.rows, scale, nsplit, sel_idx, sel_n );
		if ( nsplit > 1 ) {
			latent_reduce_kernel<<<dim3( heads, rows ), dim3( 512 ), 0, stream>>>( workspace, o_lat.ptr<half>(), heads, nsplit );
		}
		return;
	}
	if ( rows > pod::MAX_BATCH && sel_idx == nullptr ) {
		ENGINE_ASSERT( slot.row == 0, "MLALatent::attend: prefill chunks start at row 0" );
		static bool tc_attr[ENGINE_NUM_GPUS] = {};
		if ( !tc_attr[device] ) {
			CUDA_CHECK( cudaFuncSetAttribute( latent_prefill_tc_kernel<Q>, cudaFuncAttributeMaxDynamicSharedMemorySize, LT_SMEM ) );
			tc_attr[device] = true;
		}
		// Spread the key tiles over extra blocks when the query tiles alone leave the GPU under two waves
		const int base_blocks = ( rows * heads + LT_BQ - 1 ) / LT_BQ;
		int nsplit = 1;
		if ( base_blocks < 160 ) nsplit = std::min( ( 320 + base_blocks - 1 ) / base_blocks, 8 );
		if ( key_hint > 0 ) nsplit = std::max( 1, std::min( nsplit, key_hint / ( 4 * LT_BK ) ) );
		while ( nsplit > 1 && static_cast<size_t>( rows ) * heads * nsplit > MLALatent::workspace_slots( max_seq, heads ) ) --nsplit;
		latent_prefill_tc_kernel<Q><<<dim3( base_blocks, nsplit ), dim3( 256 ), LT_SMEM, stream>>>(
			q_lat.ptr<const half>(), cache, cscale, o_lat.ptr<half>(), workspace, slot.stride, layer, max_seq, slot.rows, rows, scale, nsplit );
		if ( nsplit > 1 ) {
			latent_reduce_kernel<<<dim3( heads, rows ), dim3( 512 ), 0, stream>>>( workspace, o_lat.ptr<half>(), heads, nsplit );
		}
		return;
	}
	for ( int row0 = slot.row; row0 < slot.row + rows; row0 += pod::MAX_BATCH ) {
		const int M = min( pod::MAX_BATCH, slot.row + rows - row0 );
		const size_t off = static_cast<size_t>( row0 - slot.row ) * heads * LORA;
		latent_attn_kernel<16, Q><<<dim3( M, n_chunks ), dim3( 512 ), smem, stream>>>(
			q_lat.ptr<const half>() + off, cache, cscale, o_lat.ptr<half>() + off, workspace, slot.stride, layer, max_seq, slot.rows, row0, n_chunks, scale, sel_idx, sel_n );
		if ( n_chunks > 1 ) {
			latent_reduce_kernel<<<dim3( heads, M ), dim3( 512 ), 0, stream>>>( workspace, o_lat.ptr<half>() + off, heads, n_chunks );
		}
	}
}

void MLALatent::attend( const pod::KVRef& kv, const pod::TensorView& q_lat, pod::TensorView& o_lat, int layer, int max_seq, int rows, int heads,
	const pod::SlotRef& slot, float* workspace, int key_hint, cudaStream_t stream, const int* sel_idx, const int* sel_n ) {
	switch ( kv.kq ) {
		case pod::KVQ::E5M2: attend_q<KVQ::E5M2>( kv, q_lat, o_lat, layer, max_seq, rows, heads, slot, workspace, key_hint, stream, sel_idx, sel_n ); break;
		case pod::KVQ::I8: attend_q<KVQ::I8>( kv, q_lat, o_lat, layer, max_seq, rows, heads, slot, workspace, key_hint, stream, sel_idx, sel_n ); break;
		case pod::KVQ::FP4: attend_q<KVQ::FP4>( kv, q_lat, o_lat, layer, max_seq, rows, heads, slot, workspace, key_hint, stream, sel_idx, sel_n ); break;
		case pod::KVQ::F16: attend_q<KVQ::F16>( kv, q_lat, o_lat, layer, max_seq, rows, heads, slot, workspace, key_hint, stream, sel_idx, sel_n ); break;
		default: ENGINE_ASSERT( false, "MLALatent::attend: cache format" ); break;
	}
}

void MLALatent::diag_scores( const half* q_lat_row, const half* cache_slot_layer, int n, int heads, float* scores, cudaStream_t stream ) {
	diag_scores_kernel<<<dim3( ( n + 255 ) / 256, heads ), dim3( 256 ), 0, stream>>>( q_lat_row, cache_slot_layer, scores, n, heads, 1.0f / sqrtf( static_cast<float>( HEAD ) ) );
}

} // namespace engine::modules
