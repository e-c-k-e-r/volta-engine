#include "engine/modules/mla.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace engine::modules {

namespace {

constexpr int LORA = MLALatent::LORA;
constexpr int HD = MLALatent::HEAD;
constexpr int ROWG = 16; // rows per pass of the projection kernels

// q_lat[r][h][c] = sum_d q[r][h][d] * W_UK[h][d][c]. Block = (head, 64-column slice), 32 threads of 2 columns; the q rows of
// a group of 16 are staged in shared memory, the weight column slice streams once per group.
__global__ __launch_bounds__( 32 )
void absorb_q_kernel( const half* __restrict__ q, const half* __restrict__ kv_b, half* __restrict__ q_lat, int heads, int rows ) {
	__shared__ float sq[ROWG][HD];
	const int h = blockIdx.x;
	const int c0 = blockIdx.y * 64 + threadIdx.x * 2;
	const half* w = kv_b + static_cast<size_t>( h ) * 2 * HD * LORA; // K rows first
	for ( int r0 = 0; r0 < rows; r0 += ROWG ) {
		const int nr = min( ROWG, rows - r0 );
		__syncthreads();
		for ( int i = threadIdx.x; i < nr * HD; i += 32 ) {
			const int r = i / HD, d = i % HD;
			sq[r][d] = __half2float( q[( static_cast<size_t>( r0 + r ) * heads + h ) * HD + d] );
		}
		__syncthreads();
		float acc[ROWG][2];
		#pragma unroll
		for ( int r = 0; r < ROWG; ++r ) { acc[r][0] = 0.0f; acc[r][1] = 0.0f; }
		for ( int d = 0; d < HD; ++d ) {
			const float2 wf = __half22float2( *reinterpret_cast<const half2*>( w + static_cast<size_t>( d ) * LORA + c0 ) );
			#pragma unroll
			for ( int r = 0; r < ROWG; ++r ) {
				const float qv = sq[r][d];
				acc[r][0] = fmaf( qv, wf.x, acc[r][0] );
				acc[r][1] = fmaf( qv, wf.y, acc[r][1] );
			}
		}
		for ( int r = 0; r < nr; ++r ) {
			*reinterpret_cast<half2*>( q_lat + ( static_cast<size_t>( r0 + r ) * heads + h ) * LORA + c0 ) = __floats2half2_rn( acc[r][0], acc[r][1] );
		}
	}
}

// out[r][h][v] = sum_c W_UV[h][v][c] * o_lat[r][h][c]. Block = (head, 64 values), one thread per value streaming its 1 KiB weight
// row once per group of 16 rows (o_lat of the group staged in shared memory).
__global__ __launch_bounds__( 64 )
void expand_out_kernel( const half* __restrict__ o_lat, const half* __restrict__ kv_b, half* __restrict__ out, int heads, int rows ) {
	__shared__ float so[ROWG][LORA];
	const int h = blockIdx.x;
	const int v = blockIdx.y * 64 + threadIdx.x;
	const half* w = kv_b + ( static_cast<size_t>( h ) * 2 * HD + HD + v ) * LORA; // V rows follow the K rows
	for ( int r0 = 0; r0 < rows; r0 += ROWG ) {
		const int nr = min( ROWG, rows - r0 );
		__syncthreads();
		for ( int i = threadIdx.x; i < nr * LORA; i += 64 ) {
			const int r = i / LORA, c = i % LORA;
			so[r][c] = __half2float( o_lat[( static_cast<size_t>( r0 + r ) * heads + h ) * LORA + c] );
		}
		__syncthreads();
		float acc[ROWG];
		#pragma unroll
		for ( int r = 0; r < ROWG; ++r ) acc[r] = 0.0f;
		for ( int c = 0; c < LORA; c += 8 ) {
			const uint4 raw = *reinterpret_cast<const uint4*>( w + c );
			const half2* w2 = reinterpret_cast<const half2*>( &raw );
			float wf[8];
			#pragma unroll
			for ( int i = 0; i < 4; ++i ) {
				const float2 f = __half22float2( w2[i] );
				wf[2 * i] = f.x;
				wf[2 * i + 1] = f.y;
			}
			#pragma unroll
			for ( int r = 0; r < ROWG; ++r ) {
				#pragma unroll
				for ( int i = 0; i < 8; ++i ) acc[r] = fmaf( wf[i], so[r][c + i], acc[r] );
			}
		}
		for ( int r = 0; r < nr; ++r ) {
			out[( static_cast<size_t>( r0 + r ) * heads + h ) * HD + v] = __float2half_rn( acc[r] );
		}
	}
}

__global__ void store_latent_kernel( half* __restrict__ cache, const half* __restrict__ c, int layer, int max_seq, int rows, pod::SlotRef slot ) {
	const int r = blockIdx.x;
	const int pos = slot.rows->pos[slot.row + r];
	const size_t off = static_cast<size_t>( slot.rows->slot[slot.row + r] ) * slot.stride + ( static_cast<size_t>( layer ) * max_seq + pos ) * LORA;
	for ( int i = threadIdx.x; i < LORA; i += blockDim.x ) cache[off + i] = c[static_cast<size_t>( r ) * LORA + i];
}

constexpr int CHUNK_KEYS = 1024;
constexpr int TILE = 16;
constexpr int CS = LORA + 8; // padded key row (halves) in shared memory

// Latent attention, one block per (row, chunk of 1024 keys), all heads of the row: a 16-key tile of c is staged once and
// serves the 16 heads' scores and their value sums. 512 threads: scores take two threads per (head, key) pair, the value
// sum one thread per latent dimension holding all 16 heads' accumulators.
template <int HEADS>
__global__ __launch_bounds__( 512 )
void latent_attn_kernel(
	const half* __restrict__ q_lat,
	const half* __restrict__ cache,
	half* __restrict__ o_lat,
	float* __restrict__ workspace,
	size_t slot_stride,
	int layer,
	int max_seq,
	const pod::BatchRows* __restrict__ rows,
	int row0,
	int n_chunks,
	float scale
) {
	extern __shared__ __align__( 16 ) unsigned char smem_raw[];
	float* sq = reinterpret_cast<float*>( smem_raw );                       // [HEADS][LORA]
	half* sc = reinterpret_cast<half*>( sq + HEADS * LORA );                // [TILE][CS]
	float* ss = reinterpret_cast<float*>( sc + TILE * CS );                 // [HEADS][TILE] scores, then probabilities
	float* sal = ss + HEADS * TILE;                                         // [HEADS] rescale factors
	const int m = blockIdx.x, chunk = blockIdx.y, t = threadIdx.x;
	const int pos = rows->pos[row0 + m];
	const int total = min( pos + 1, max_seq );
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
				if ( k < nk ) raw = *reinterpret_cast<const uint4*>( cache + base + static_cast<size_t>( kb + k ) * LORA + c8 * 8 );
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
	asm volatile (
		"mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
		"{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
		: "+f"( c[0] ), "+f"( c[1] ), "+f"( c[2] ), "+f"( c[3] ), "+f"( c[4] ), "+f"( c[5] ), "+f"( c[6] ), "+f"( c[7] )
		: "r"( a0 ), "r"( a1 ), "r"( b0 ), "r"( b1 )
	);
}

__global__ __launch_bounds__( 256 )
void latent_prefill_tc_kernel(
	const half* __restrict__ q_lat,
	const half* __restrict__ cache,
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
	const half* kv = cache + static_cast<size_t>( rows->slot[q_base / HEADS] ) * slot_stride + static_cast<size_t>( layer ) * max_seq * LORA;

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

	float acc[2][4][8];
	#pragma unroll
	for ( int jj = 0; jj < 2; ++jj ) {
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			#pragma unroll
			for ( int r = 0; r < 8; ++r ) acc[jj][i][r] = 0.0f;
		}
	}

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
				pk[j][kk] = ( kb + key <= max_pos ) ? *reinterpret_cast<const uint4*>( kv + static_cast<size_t>( kb + key ) * LORA + c * 8 ) : make_uint4( 0, 0, 0, 0 );
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
		__syncthreads();
	}

	if ( sub == 0 ) {
		sA[srow] = nsplit > 1 ? 1.0f : ( l_run > 0.0f ? 1.0f / l_run : 0.0f );
		sM[srow] = m_run;
		sL[srow] = l_run;
	}
	__syncthreads();
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
	absorb_q_kernel<<<dim3( heads, LORA / 64 ), dim3( 32 ), 0, stream>>>( q.ptr<const half>(), kv_b.ptr<const half>(), q_lat.ptr<half>(), heads, rows );
}

void MLALatent::expand_out( const pod::TensorView& o_lat, const pod::TensorView& kv_b, pod::TensorView& out, int heads, int rows, cudaStream_t stream ) {
	expand_out_kernel<<<dim3( heads, HEAD / 64 ), dim3( 64 ), 0, stream>>>( o_lat.ptr<const half>(), kv_b.ptr<const half>(), out.ptr<half>(), heads, rows );
}

void MLALatent::store( half* cache, const pod::TensorView& c, int layer, int max_seq, int rows, const pod::SlotRef& slot, cudaStream_t stream ) {
	ENGINE_ASSERT( slot.rows != nullptr, "MLALatent::store: device-resolved positions only" );
	store_latent_kernel<<<dim3( rows ), dim3( 128 ), 0, stream>>>( cache, c.ptr<const half>(), layer, max_seq, rows, slot );
}

void MLALatent::attend( const half* cache, const pod::TensorView& q_lat, pod::TensorView& o_lat, int layer, int max_seq, int rows, int heads,
	const pod::SlotRef& slot, float* workspace, int key_hint, cudaStream_t stream ) {
		ENGINE_ASSERT( heads == 16 && slot.rows != nullptr, "MLALatent::attend: 16 heads per rank, device-resolved positions" );
	const int n_chunks = workspace_chunks( max_seq );
	const float scale = 1.0f / sqrtf( static_cast<float>( HEAD ) );
	const size_t smem = static_cast<size_t>( 16 ) * LORA * 4 + static_cast<size_t>( TILE ) * CS * 2 + ( 16 * TILE + 16 ) * 4;
	static bool attr[ENGINE_NUM_GPUS] = {};
	int device = 0;
	cudaGetDevice( &device );
	if ( !attr[device] ) {
		CUDA_CHECK( cudaFuncSetAttribute( latent_attn_kernel<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>( smem ) ) );
		attr[device] = true;
	}
	static const bool simt_only = std::getenv( "EXL3_MLA_LATENT_SIMT" ) != nullptr;
	if ( rows > pod::MAX_BATCH && !simt_only ) {
		ENGINE_ASSERT( slot.row == 0, "MLALatent::attend: prefill chunks start at row 0" );
		static bool tc_attr[ENGINE_NUM_GPUS] = {};
		if ( !tc_attr[device] ) {
			CUDA_CHECK( cudaFuncSetAttribute( latent_prefill_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, LT_SMEM ) );
			tc_attr[device] = true;
		}
		// Spread the key tiles over extra blocks when the query tiles alone leave the GPU under two waves
		const int base_blocks = ( rows * heads + LT_BQ - 1 ) / LT_BQ;
		int nsplit = 1;
		if ( base_blocks < 160 ) nsplit = std::min( ( 320 + base_blocks - 1 ) / base_blocks, 8 );
		if ( key_hint > 0 ) nsplit = std::max( 1, std::min( nsplit, key_hint / ( 4 * LT_BK ) ) );
		while ( nsplit > 1 && static_cast<size_t>( rows ) * heads * nsplit > workspace_slots( max_seq, heads ) ) --nsplit;
		latent_prefill_tc_kernel<<<dim3( base_blocks, nsplit ), dim3( 256 ), LT_SMEM, stream>>>(
			q_lat.ptr<const half>(), cache, o_lat.ptr<half>(), workspace, slot.stride, layer, max_seq, slot.rows, rows, scale, nsplit );
		if ( nsplit > 1 ) {
			latent_reduce_kernel<<<dim3( heads, rows ), dim3( 512 ), 0, stream>>>( workspace, o_lat.ptr<half>(), heads, nsplit );
		}
		return;
	}
	for ( int row0 = slot.row; row0 < slot.row + rows; row0 += pod::MAX_BATCH ) {
		const int M = min( pod::MAX_BATCH, slot.row + rows - row0 );
		const size_t off = static_cast<size_t>( row0 - slot.row ) * heads * LORA;
		latent_attn_kernel<16><<<dim3( M, n_chunks ), dim3( 512 ), smem, stream>>>(
			q_lat.ptr<const half>() + off, cache, o_lat.ptr<half>() + off, workspace, slot.stride, layer, max_seq, slot.rows, row0, n_chunks, scale );
		if ( n_chunks > 1 ) {
			latent_reduce_kernel<<<dim3( heads, M ), dim3( 512 ), 0, stream>>>( workspace, o_lat.ptr<half>() + off, heads, n_chunks );
		}
	}
}

} // namespace engine::modules
