// Tensor-core prefill attention (Volta mma.m8n8k4, FP32 accumulate) as the PTX of the Vulkan backend's launches through
// VK_NV_cuda_kernel_launch (src/backend/vulkan/mod_cuda.cpp), within the 48 KiB of shared memory such a launch has: the tiles of
// attention.cu's attn_prefill_tc_kernel on 32-key tiles. A block owns 32 query vectors (row, head pairs of one slot) of one kv head and walks
// the keys in tiles of 32. S = Q K^T: warp (half, qg, kg) computes a 16 x 16 tile over one 128-dimension half of a 256-dimension part, the two
// halves summed in the softmax; the softmax runs on 8 lanes per query row, P is written as FP16, and O += P V gives each warp 32 dimensions
// of each 256-dimension part (interleaved: d = 32w + dn + 4c) for all 32 queries. The K tile and the transposed V tile share one buffer (V is
// staged once S is done); head_dim 512 stages Q, K and V one 256-dimension part at a time. FP16 KV; causal, sliding windows, ring caches,
// bidirectional image spans, the output gate, and key splits whose partials (unnormalized output, max, sum) take fa.slang's layout for its
// fa_reduce.
#include <cuda_fp16.h>
#include <cstdint>
#include "../mma_sm80.cuh"

namespace {

constexpr int BQ = 32, BK = 32, KS = 260, VS = BK + 4, SS = BK + 4;
constexpr int OFF_Q = 256 * VS * 2;           // [0, OFF_Q): the K tile (BK x KS) or the transposed V tile (256 x VS)
constexpr int OFF_S = OFF_Q + BQ * KS * 2;    // Q, one 256-dimension part
constexpr int OFF_P = OFF_S + 2 * BQ * SS * 4;   // S, the two halves' partial sums
constexpr int OFF_A = OFF_P + BQ * VS * 2;    // P
constexpr int SMEM = OFF_A + BQ * 12;         // alpha, max, sum per query
static_assert( BK * KS * 2 <= OFF_Q, "the K tile fits the V tile's buffer" );
static_assert( BQ * KS * 4 <= OFF_S, "the output part (FP32) fits below S" );
static_assert( SMEM <= 48 * 1024, "a CUDA launch from Vulkan has 48 KiB of shared memory" );

__device__ __forceinline__ float sigmoid( float x ) { return 1.0f / ( 1.0f + __expf( -x ) ); }

template <int HD>
__device__ __forceinline__ void attn_prefill_body(
	const half* __restrict__ q,
	const half* __restrict__ kc,
	const half* __restrict__ vc,
	const half* __restrict__ gate,
	half* __restrict__ out,
	long long slot_stride,
	int layer_idx,
	const int* __restrict__ rows_slot,
	const int* __restrict__ rows_pos,
	const int* __restrict__ rows_key_end,
	const int* __restrict__ rows_bidir,
	int count,
	int num_q_heads,
	int num_kv_heads,
	int max_seq,
	float scale,
	float* __restrict__ part,
	int nsplit,
	int window,
	int ring
) {
	constexpr int NP = HD / 256;   // 256-dimension parts
	extern __shared__ __align__( 16 ) unsigned char smem[];
	half* sK = reinterpret_cast<half*>( smem );
	half* sVt = reinterpret_cast<half*>( smem );
	half* sQ = reinterpret_cast<half*>( smem + OFF_Q );
	float* sS = reinterpret_cast<float*>( smem + OFF_S );
	half* sP = reinterpret_cast<half*>( smem + OFF_P );
	float* sA = reinterpret_cast<float*>( smem + OFF_A );
	float* sM = sA + BQ;
	float* sL = sM + BQ;
	float* sO = reinterpret_cast<float*>( smem );

	const int G = num_q_heads / num_kv_heads;
	const int kvh = blockIdx.x;
	const int q_base = blockIdx.y * BQ;
	const int q_end = min( q_base + BQ, count * G );
	const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
	const int t = lane & 3, hi = lane >> 4, e = t + 4 * hi, qp = ( lane >> 2 ) & 3;
	const bool bidir = window > 0 && *rows_bidir != 0;

	const int last_row = ( q_end - 1 ) / G;
	int max_pos = 0, min_pos = 0x7fffffff;
	for ( int r = q_base / G; r <= last_row; ++r ) {
		max_pos = max( max_pos, bidir ? rows_key_end[r] - 1 : rows_pos[r] );
		min_pos = min( min_pos, rows_pos[r] );
	}
	if ( ring == 0 ) max_pos = min( max_pos, max_seq - 1 );
	// the block's keys start at the tile below its earliest row's window
	const int key_lo = window > 0 ? ( max( 0, min_pos - window + 1 ) / BK ) * BK : 0;
	const size_t kv_stride = static_cast<size_t>( num_kv_heads ) * HD;
	const size_t kv_base = static_cast<size_t>( rows_slot[q_base / G] ) * slot_stride + static_cast<size_t>( layer_idx ) * max_seq * kv_stride +
		static_cast<size_t>( kvh ) * HD;

	// softmax role: 8 lanes per query row, 4 keys a lane
	const int srow = tid >> 3, sub = tid & 7;
	const int sqi = q_base + srow;
	const bool svalid = sqi < q_end;
	const int spos = svalid ? ( bidir ? rows_key_end[sqi / G] - 1 : rows_pos[sqi / G] ) : -1;   // the row's last key
	const int sp0 = svalid ? rows_pos[sqi / G] : -1;   // its position (the window counts back from it)
	float m_run = -1e30f, l_run = 0.0f;

	// S: warp (dh, qg, kg); PV: warp w's 32 dimensions of each part
	const int s_dh = warp >> 2, s_qg = ( warp >> 1 ) & 1, s_kg = warp & 1;
	const half* s_arow = sQ + ( s_qg * 16 + 8 * ( qp >> 1 ) + e ) * KS + s_dh * 128;
	const half* s_brow = sK + ( s_kg * 16 + 8 * ( qp & 1 ) + e ) * KS + s_dh * 128;
	const half* v_row = sVt + ( 32 * warp + qp + 4 * e ) * VS;
	const half* p_row = sP + e * VS;

	float acc[NP][4][8];
	#pragma unroll
	for ( int p = 0; p < NP; ++p ) {
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			#pragma unroll
			for ( int r = 0; r < 8; ++r ) acc[p][i][r] = 0.0f;
		}
	}

	// staging: Q part (BQ x 32 chunks of 8), K part (16 keys x 32 chunks, twice: lanes 0-15 and 16-31 the two chunks of a 32-byte sector),
	// V part transposed (16 key pairs x 32 chunks, two keys' chunks into half2 pairs)
	auto stage_q = [&]( int part0 ) {
		#pragma unroll
		for ( int u = 0; u < 4; ++u ) {
			const int it = tid + 256 * u, qq = it >> 5, c = it & 31;
			const int qi = q_base + qq;
			uint4 raw = make_uint4( 0, 0, 0, 0 );
			if ( qi < q_end ) raw = *reinterpret_cast<const uint4*>( q + ( static_cast<size_t>( qi / G ) * num_q_heads + kvh * G + qi % G ) * HD + part0 + c * 8 );
			uint2* dst = reinterpret_cast<uint2*>( sQ + qq * KS + c * 8 );
			dst[0] = make_uint2( raw.x, raw.y );
			dst[1] = make_uint2( raw.z, raw.w );
		}
	};
	auto key_row = [&]( int kb, int key ) -> long long {   // the cache row of key kb + key, -1 past the block's last key
		if ( kb + key > max_pos ) return -1;
		int tok = kb + key;
		if ( ring > 0 ) tok %= ring;
		return static_cast<long long>( tok );
	};
	auto fetch_k = [&]( int kb, int part0, uint4 ( &rk )[4] ) {
		#pragma unroll
		for ( int u = 0; u < 4; ++u ) {
			const int it = tid + 256 * u, key = ( it & 15 ) + 16 * ( it >> 9 ), c = ( it >> 4 ) & 31;
			const long long row = key_row( kb, key );
			rk[u] = row < 0 ? make_uint4( 0, 0, 0, 0 ) : *reinterpret_cast<const uint4*>( kc + kv_base + row * kv_stride + part0 + c * 8 );
		}
	};
	auto store_k = [&]( const uint4 ( &rk )[4] ) {
		#pragma unroll
		for ( int u = 0; u < 4; ++u ) {
			const int it = tid + 256 * u, key = ( it & 15 ) + 16 * ( it >> 9 ), c = ( it >> 4 ) & 31;
			uint2* dst = reinterpret_cast<uint2*>( sK + key * KS + c * 8 );
			dst[0] = make_uint2( rk[u].x, rk[u].y );
			dst[1] = make_uint2( rk[u].z, rk[u].w );
		}
	};
	auto fetch_v = [&]( int kb, int part0, uint4 ( &rv )[2][2] ) {
		#pragma unroll
		for ( int u = 0; u < 2; ++u ) {
			const int it = tid + 256 * u, kp = it & 15, c = it >> 4;
			#pragma unroll
			for ( int kk = 0; kk < 2; ++kk ) {
				const long long row = key_row( kb, 2 * kp + kk );
				rv[u][kk] = row < 0 ? make_uint4( 0, 0, 0, 0 ) : *reinterpret_cast<const uint4*>( vc + kv_base + row * kv_stride + part0 + c * 8 );
			}
		}
	};
	auto store_v = [&]( const uint4 ( &rv )[2][2] ) {
		#pragma unroll
		for ( int u = 0; u < 2; ++u ) {
			const int it = tid + 256 * u, kp = it & 15, c = it >> 4;
			const half* v0 = reinterpret_cast<const half*>( &rv[u][0] );
			const half* v1 = reinterpret_cast<const half*>( &rv[u][1] );
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) *reinterpret_cast<half2*>( sVt + ( c * 8 + i ) * VS + 2 * kp ) = __halves2half2( v0[i], v1[i] );
		}
	};

	// the key tiles split over blockIdx.z; a split with no tiles leaves an empty partial for the merge
	const int ntiles = ( max_pos - key_lo + BK ) / BK;
	const int kb0 = key_lo + ( blockIdx.z * ntiles / nsplit ) * BK;
	const int kb1 = key_lo + ( ( blockIdx.z + 1 ) * ntiles / nsplit ) * BK;

	uint4 rk[4];
	uint4 rv[2][2];
	if ( NP == 1 ) {
		stage_q( 0 );
		if ( kb0 < kb1 ) fetch_k( kb0, 0, rk );
	}
	for ( int kb = kb0; kb < kb1; kb += BK ) {
		// S = Q K^T, the 256-dimension parts in turn
		float c[8];
		#pragma unroll
		for ( int r = 0; r < 8; ++r ) c[r] = 0.0f;
		#pragma unroll
		for ( int p = 0; p < NP; ++p ) {
			if ( NP > 1 ) {
				if ( p > 0 ) __syncthreads();   // the previous part's products have read sQ and sK
				stage_q( 256 * p );
				fetch_k( kb, 256 * p, rk );
			}
			store_k( rk );
			if ( NP == 1 ) fetch_v( kb, 0, rv );   // in flight during the products
			__syncthreads();
			#pragma unroll 8
			for ( int ks = 0; ks < 32; ++ks ) {
				const uint2 a = *reinterpret_cast<const uint2*>( s_arow + 4 * ks );
				const uint2 b = *reinterpret_cast<const uint2*>( s_brow + 4 * ks );
				mma884( c, a.x, a.y, b.x, b.y );
			}
		}
		#pragma unroll
		for ( int r = 0; r < 8; ++r ) {
			const int m = ( t & 1 ) + 2 * ( ( r >> 1 ) & 1 ) + 4 * hi;
			const int n = ( r & 1 ) + 2 * ( t >> 1 ) + 4 * ( r >> 2 );
			sS[s_dh * BQ * SS + ( s_qg * 16 + 8 * ( qp >> 1 ) + m ) * SS + s_kg * 16 + 8 * ( qp & 1 ) + n] = c[r];
		}
		__syncthreads();

		// online softmax of this tile, one query row per 8 lanes; the V tile (first part) goes into the K tile's buffer meanwhile
		if ( NP == 1 ) store_v( rv );
		{
			float sv[4];
			float mx = -1e30f;
			#pragma unroll
			for ( int j = 0; j < 4; ++j ) {
				const int key = kb + sub * 4 + j;
				const bool ok = svalid && key <= spos && ( window <= 0 || key > sp0 - window );
				const int at = srow * SS + sub * 4 + j;
				sv[j] = ok ? ( sS[at] + sS[BQ * SS + at] ) * scale : -1e30f;
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
			*reinterpret_cast<uint2*>( sP + srow * VS + sub * 4 ) = make_uint2( *reinterpret_cast<uint32_t*>( &ph[0] ), *reinterpret_cast<uint32_t*>( &ph[1] ) );
			if ( sub == 0 ) sA[srow] = alpha;
		}
		__syncthreads();

		// O = O * alpha + P V, the parts in turn
		#pragma unroll
		for ( int p = 0; p < NP; ++p ) {
			if ( NP > 1 ) {
				fetch_v( kb, 256 * p, rv );
				if ( p > 0 ) __syncthreads();   // the previous part's products have read sVt
				store_v( rv );
				__syncthreads();
			}
			else if ( kb + BK < kb1 ) fetch_k( kb + BK, 0, rk );   // the next tile's keys in flight during the products
			#pragma unroll
			for ( int i = 0; i < 4; ++i ) {
				#pragma unroll
				for ( int r = 0; r < 8; ++r ) {
					const int m = ( t & 1 ) + 2 * ( ( r >> 1 ) & 1 ) + 4 * hi;
					acc[p][i][r] *= sA[8 * i + m];
				}
			}
			#pragma unroll
			for ( int kk = 0; kk < BK / 4; ++kk ) {
				const uint2 b = *reinterpret_cast<const uint2*>( v_row + 4 * kk );
				#pragma unroll
				for ( int i = 0; i < 4; ++i ) {
					const uint2 a = *reinterpret_cast<const uint2*>( p_row + 8 * i * VS + 4 * kk );
					mma884( acc[p][i], a.x, a.y, b.x, b.y );
				}
			}
		}
		__syncthreads();
	}

	if ( sub == 0 ) {
		sA[srow] = nsplit > 1 ? 1.0f : ( l_run > 0.0f ? 1.0f / l_run : 0.0f );
		sM[srow] = m_run;
		sL[srow] = l_run;
	}
	const size_t q_all = static_cast<size_t>( count ) * num_q_heads;
	#pragma unroll
	for ( int p = 0; p < NP; ++p ) {
		__syncthreads();   // sA / sM / sL written; the previous part's output read
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			#pragma unroll
			for ( int r = 0; r < 8; ++r ) {
				const int m = ( t & 1 ) + 2 * ( ( r >> 1 ) & 1 ) + 4 * hi;
				const int n = ( r & 1 ) + 2 * ( t >> 1 ) + 4 * ( r >> 2 );
				sO[( 8 * i + m ) * KS + 32 * warp + qp + 4 * n] = acc[p][i][r] * sA[8 * i + m];
			}
		}
		__syncthreads();
		for ( int u = tid; u < BQ * 32; u += 256 ) {
			const int qq = u >> 5, cc = u & 31;
			const int qi = q_base + qq;
			if ( qi >= q_end ) continue;
			const size_t oo = static_cast<size_t>( qi / G ) * num_q_heads + kvh * G + qi % G;
			const float* o = sO + qq * KS + cc * 8;
			if ( nsplit > 1 ) {
				// fa.slang's partials: split z's output at (z q_all + oo) HD, its max and sum after every split's output
				float4* dst = reinterpret_cast<float4*>( part + ( blockIdx.z * q_all + oo ) * HD + 256 * p + cc * 8 );
				dst[0] = make_float4( o[0], o[1], o[2], o[3] );
				dst[1] = make_float4( o[4], o[5], o[6], o[7] );
				if ( p == 0 && cc == 0 ) {
					float* ml = part + nsplit * q_all * HD + ( blockIdx.z * q_all + oo ) * 2;
					ml[0] = sL[qq] > 0.0f ? sM[qq] : -1e30f;
					ml[1] = sL[qq];
				}
				continue;
			}
			const size_t off = oo * HD + 256 * p + cc * 8;
			float sig[8];
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) sig[i] = 1.0f;
			if ( gate != nullptr ) {
				const uint4 graw = *reinterpret_cast<const uint4*>( gate + off );
				const half2* g2 = reinterpret_cast<const half2*>( &graw );
				#pragma unroll
				for ( int i = 0; i < 4; ++i ) {
					const float2 f = __half22float2( g2[i] );
					sig[2 * i] = sigmoid( f.x );
					sig[2 * i + 1] = sigmoid( f.y );
				}
			}
			uint4 res;
			half2* r2 = reinterpret_cast<half2*>( &res );
			#pragma unroll
			for ( int i = 0; i < 4; ++i ) r2[i] = __floats2half2_rn( o[2 * i] * sig[2 * i], o[2 * i + 1] * sig[2 * i + 1] );
			*reinterpret_cast<uint4*>( out + off ) = res;
		}
	}
}

} // namespace

#define VK_ATTN_ARGS                                                                                                                              \
	const half *q, const half *kc, const half *vc, const half *gate, half *out, long long slot_stride, int layer_idx, const int *rows_slot,       \
		const int *rows_pos, const int *rows_key_end, const int *rows_bidir, int count, int num_q_heads, int num_kv_heads, int max_seq,           \
		float scale, float *part, int nsplit, int window, int ring
#define VK_ATTN_PASS                                                                                                                              \
	q, kc, vc, gate, out, slot_stride, layer_idx, rows_slot, rows_pos, rows_key_end, rows_bidir, count, num_q_heads, num_kv_heads, max_seq,       \
		scale, part, nsplit, window, ring

extern "C" __global__ __launch_bounds__( 256 ) void vk_attn_prefill_256( VK_ATTN_ARGS ) { attn_prefill_body<256>( VK_ATTN_PASS ); }
extern "C" __global__ __launch_bounds__( 256 ) void vk_attn_prefill_512( VK_ATTN_ARGS ) { attn_prefill_body<512>( VK_ATTN_PASS ); }
