#include "engine/modules/attention.h"
#include "kv_quant.cuh"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdlib>
#include <algorithm>

namespace engine::modules {

namespace {

#define ATTN_PAGE_SIZE 256
#define ATTN_MAX_CHUNKS 128
#define ATTN_MAX_HEAD_DIM 256
#define ATTN_MAX_ROW_HEADS ( pod::MAX_BATCH * 16 )

__device__ float g_attn_workspace[ATTN_MAX_ROW_HEADS * ATTN_MAX_CHUNKS * ( ATTN_MAX_HEAD_DIM + 2 )];

// Device address of g_attn_workspace on the current GPU. The bare symbol in host code is the
// host shadow, not a usable kernel argument; resolve once per device.
float* attn_workspace_ptr() {
	static float* ptrs[ENGINE_NUM_GPUS] = {};
	int dev = 0;
	CUDA_CHECK( cudaGetDevice( &dev ) );
	ENGINE_ASSERT( dev >= 0 && dev < ENGINE_NUM_GPUS, "attn_workspace_ptr: device out of range" );
	if ( !ptrs[dev] ) {
		void* p = nullptr;
		CUDA_CHECK( cudaGetSymbolAddress( &p, g_attn_workspace ) );
		ptrs[dev] = static_cast<float*>( p );
	}
	return ptrs[dev];
}

__global__ void unpack_q_gate_kernel(
	const half* __restrict__ q_gate,
	half* __restrict__ q,
	half* __restrict__ gate,
	int num_q_heads,
	int head_dim,
	int M
) {
	int head = blockIdx.x;
	int m = blockIdx.y;
	int tid = threadIdx.x;
	if ( head >= num_q_heads || m >= M || tid >= head_dim ) return;

	size_t src_offset = ( static_cast<size_t>( m ) * num_q_heads + head ) * 2 * head_dim;
	size_t dst_offset = ( static_cast<size_t>( m ) * num_q_heads + head ) * head_dim + tid;

	q[dst_offset] = q_gate[src_offset + tid];
	gate[dst_offset] = q_gate[src_offset + head_dim + tid];
}

// One thread quantizes one unit (element, 16- or 64-element group) of one row's K or V into its slot/layer/position.
template <KVQ Q>
__global__ void kv_store_kernel(
	void* __restrict__ data,
	void* __restrict__ scales,
	const half* __restrict__ src,
	size_t slot_stride,
	int layer_idx,
	const pod::BatchRows* __restrict__ rows,
	int count,
	int total_elems,
	int max_seq
) {
	constexpr int U = kv_unit<Q>();
	const int units_per_row = total_elems / U;
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ( tid >= units_per_row * count ) return;
	const int m = tid / units_per_row;
	const int u = tid % units_per_row;
	const size_t idx0 = static_cast<size_t>( rows->slot[m] ) * slot_stride
		+ ( static_cast<size_t>( layer_idx ) * max_seq + rows->pos[m] ) * total_elems + static_cast<size_t>( u ) * U;
	kv_store_unit<Q>( data, scales, idx0, src + static_cast<size_t>( m ) * total_elems + static_cast<size_t>( u ) * U );
}

// Block-wide orthonormal 64-point Walsh-Hadamard transform of one value per thread (blockDim a multiple of 64);
// s holds blockDim floats. Its own inverse, used to undo the rotation applied to Q4 values on store.
__device__ __forceinline__ float kv_fwht64_block( float v, int tid, float* s ) {
	s[tid] = v;
	__syncthreads();
	for ( int h = 1; h < 64; h <<= 1 ) {
		const float a = s[tid], b = s[tid ^ h];
		__syncthreads();
		s[tid] = ( tid & h ) ? b - a : a + b;
		__syncthreads();
	}
	const float r = s[tid] * 0.125f;
	__syncthreads();
	return r;
}

template <KVQ KQ, KVQ VQ>
__global__ void attn_chunked_paged_kernel(
	const half* __restrict__ q,
	const pod::KVRef kv,
	const half* __restrict__ gate,
	half* __restrict__ out,
	float* __restrict__ workspace,
	size_t slot_stride,
	int layer_idx,
	const pod::BatchRows* __restrict__ rows,
	int num_q_heads,
	int num_kv_heads,
	int head_dim,
	int max_seq,
	int n_chunks,
	int chunk_len,
	int row0,
	float scale
) {
	int head = blockIdx.x;
	int m = blockIdx.y;
	int chunk = blockIdx.z;
	int tid = threadIdx.x;
	if ( head >= num_q_heads || tid >= head_dim ) return;

	int heads_per_kv = num_q_heads / num_kv_heads;
	if ( heads_per_kv < 1 ) heads_per_kv = 1;
	int kv_head = head / heads_per_kv;

	// Chunks past this row's context write an empty (l = 0) partial that the reduce skips, so
	// the grid can be sized by max_seq and stay valid across positions (graph replay)
	int cur_pos = rows->pos[row0 + m];
	int total_tokens = min( cur_pos + 1, max_seq );
	const size_t slot_off = static_cast<size_t>( rows->slot[row0 + m] ) * slot_stride;

	int kv_start = chunk * chunk_len;
	int kv_end = min( kv_start + chunk_len, total_tokens );

	const half* q_head = q + ( static_cast<size_t>( m ) * num_q_heads + head ) * head_dim;
	half* out_head = out + ( static_cast<size_t>( m ) * num_q_heads + head ) * head_dim;

	float my_q = __half2float( q_head[tid] );

	__shared__ float s_warp_sums[8];
	__shared__ float s_score;

	int warp_id = tid >> 5;
	int lane_id = tid & 31;
	int num_warps = ( head_dim + 31 ) >> 5;

	float m_reg = -1e30f;
	float l_reg = 0.0f;
	float o_dim = 0.0f;

	if ( kv_start < total_tokens ) {
		for ( int t = kv_start; t < kv_end; ++t ) {
			size_t kv_offset = slot_off + ( static_cast<size_t>( layer_idx ) * max_seq + t ) * ( num_kv_heads * head_dim ) + kv_head * head_dim + tid;
			float my_k = kv_load1<KQ>( kv.k, kv.ks, kv_offset );
			float dot = my_q * my_k;

			#pragma unroll
			for ( int mask = 16; mask > 0; mask >>= 1 ) {
				dot += __shfl_xor_sync( 0xffffffff, dot, mask );
			}

			if ( lane_id == 0 ) {
				s_warp_sums[warp_id] = dot;
			}
			__syncthreads();

			if ( warp_id == 0 ) {
				float b_sum = ( lane_id < num_warps ) ? s_warp_sums[lane_id] : 0.0f;
				#pragma unroll
				for ( int mask = 4; mask > 0; mask >>= 1 ) {
					b_sum += __shfl_xor_sync( 0xffffffff, b_sum, mask );
				}
				if ( lane_id == 0 ) {
					s_score = b_sum * scale;
				}
			}
			__syncthreads();

			float my_v = kv_load1<VQ>( kv.v, kv.vs, kv_offset );
			float score = s_score;

			float m_new = fmaxf( m_reg, score );
			float alpha = expf( m_reg - m_new );
			float exp_s = expf( score - m_new );
			o_dim = alpha * o_dim + exp_s * my_v;
			l_reg = alpha * l_reg + exp_s;
			m_reg = m_new;
		}
	}

	if ( n_chunks == 1 ) {
		float inv_l = ( l_reg > 0.0f ) ? ( 1.0f / l_reg ) : 0.0f;
		float val = o_dim * inv_l;
		if constexpr ( VQ == KVQ::Q4 ) {
			__shared__ float s_rot[ATTN_MAX_HEAD_DIM];
			val = kv_fwht64_block( val, tid, s_rot );
		}
		float sig = 1.0f;
		if ( gate != nullptr ) {
			size_t g_off = ( static_cast<size_t>( m ) * num_q_heads + head ) * head_dim + tid;
			float g_val = __half2float( gate[g_off] );
			sig = 1.0f / ( 1.0f + __expf( -g_val ) );
		}
		out_head[tid] = __float2half_rn( val * sig );
	}
	else {
		float* ws = workspace + ( ( static_cast<size_t>( m ) * num_q_heads + head ) * n_chunks + chunk ) * ( ATTN_MAX_HEAD_DIM + 2 );
		if ( tid == 0 ) {
			ws[0] = m_reg;
			ws[1] = l_reg;
		}
		ws[2 + tid] = o_dim;
	}
}

// Prefill attention. The decode kernel above does one block-wide reduction (two barriers) per key and re-reads K/V for
// every query row; here a block shares each 32-key tile of K and V (staged once in shared memory) across 8 queries,
// one warp per query. Lanes own 8 head dimensions; scores are warp-reduced, the softmax is online in FP32 once per
// tile, and keys past a query's own position are masked. head_dim 256 only. Queries are (row, head) pairs ordered
// row-major within a kv head's group, so the 3 heads of a row share a block.
#define PF_TILE 32
#define PF_QPB 8
template <KVQ KQ, KVQ VQ>
__global__ __launch_bounds__( 256 )
void attn_prefill_kernel(
	const half* __restrict__ q,
	const pod::KVRef kv,
	const half* __restrict__ gate,
	half* __restrict__ out,
	size_t slot_stride,
	int layer_idx,
	const pod::BatchRows* __restrict__ rows,
	int count,
	int num_q_heads,
	int num_kv_heads,
	int max_seq,
	float scale
) {
	constexpr int HD = 256;
	__shared__ __align__( 16 ) half sK[PF_TILE * HD];
	__shared__ __align__( 16 ) half sV[PF_TILE * HD];

	const int G = num_q_heads / num_kv_heads;
	const int kvh = blockIdx.x;
	const int q_base = blockIdx.y * PF_QPB;
	const int warp = threadIdx.x >> 5;
	const int lane = threadIdx.x & 31;
	const int qi = q_base + warp;
	const bool valid = qi < count * G;
	const int row = valid ? qi / G : 0;
	const int head = kvh * G + ( valid ? qi % G : 0 );

	// Keys needed by any query of this block: up to the largest position among its rows (all one slot).
	const int last_row = min( ( q_base + PF_QPB - 1 ) / G, count - 1 );
	int max_pos = 0;
	for ( int r = q_base / G; r <= last_row; ++r ) max_pos = max( max_pos, rows->pos[r] );
	max_pos = min( max_pos, max_seq - 1 );
	const size_t slot_off = static_cast<size_t>( rows->slot[valid ? row : q_base / G] ) * slot_stride;
	const int my_pos = valid ? rows->pos[row] : -1;

	float qv[8];
	if ( valid ) {
		const half* qh = q + ( static_cast<size_t>( row ) * num_q_heads + head ) * HD + lane * 8;
		const uint4 raw = *reinterpret_cast<const uint4*>( qh );
		const half2* h2 = reinterpret_cast<const half2*>( &raw );
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			const float2 f = __half22float2( h2[i] );
			qv[2 * i] = f.x * scale;
			qv[2 * i + 1] = f.y * scale;
		}
	}
	else {
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) qv[i] = 0.0f;
	}

	float m_run = -1e30f, l_run = 0.0f;
	float o[8];
	#pragma unroll
	for ( int i = 0; i < 8; ++i ) o[i] = 0.0f;

	const size_t kv_stride = static_cast<size_t>( num_kv_heads ) * HD;
	const size_t kv_base = slot_off + static_cast<size_t>( layer_idx ) * max_seq * kv_stride + static_cast<size_t>( kvh ) * HD;

	for ( int kb = 0; kb <= max_pos; kb += PF_TILE ) {
		// Stage the tile: 32 keys x 512 B each for K and V (1024 uint4 apiece); keys past max_pos load as zero.
		for ( int u = threadIdx.x; u < PF_TILE * ( HD / 8 ); u += blockDim.x ) {
			const int j = u / ( HD / 8 );
			const int c = u % ( HD / 8 );
			const int key = kb + j;
			uint4 kk = make_uint4( 0, 0, 0, 0 ), vv = make_uint4( 0, 0, 0, 0 );
			if ( key <= max_pos ) {
				const size_t at = kv_base + static_cast<size_t>( key ) * kv_stride + c * 8;
				float f[8];
				half2* o2 = reinterpret_cast<half2*>( &kk );
				kv_load8<KQ>( kv.k, kv.ks, at, f );
				#pragma unroll
				for ( int i = 0; i < 4; ++i ) o2[i] = __floats2half2_rn( f[2 * i], f[2 * i + 1] );
				o2 = reinterpret_cast<half2*>( &vv );
				kv_load8<VQ>( kv.v, kv.vs, at, f );
				#pragma unroll
				for ( int i = 0; i < 4; ++i ) o2[i] = __floats2half2_rn( f[2 * i], f[2 * i + 1] );
			}
			*reinterpret_cast<uint4*>( sK + j * HD + c * 8 ) = kk;
			*reinterpret_cast<uint4*>( sV + j * HD + c * 8 ) = vv;
		}
		__syncthreads();

		if ( valid && kb <= my_pos ) {
			float my_s = -1e30f;
			#pragma unroll
			for ( int j = 0; j < PF_TILE; ++j ) {
				const uint4 raw = *reinterpret_cast<const uint4*>( sK + j * HD + lane * 8 );
				const half2* h2 = reinterpret_cast<const half2*>( &raw );
				float d = 0.0f;
				#pragma unroll
				for ( int i = 0; i < 4; ++i ) {
					const float2 f = __half22float2( h2[i] );
					d = fmaf( qv[2 * i], f.x, d );
					d = fmaf( qv[2 * i + 1], f.y, d );
				}
				#pragma unroll
				for ( int mask = 16; mask > 0; mask >>= 1 ) d += __shfl_xor_sync( 0xffffffff, d, mask );
				if ( lane == j ) my_s = ( kb + j <= my_pos ) ? d : -1e30f;
			}
			float tile_max = my_s;
			#pragma unroll
			for ( int mask = 16; mask > 0; mask >>= 1 ) tile_max = fmaxf( tile_max, __shfl_xor_sync( 0xffffffff, tile_max, mask ) );
			const float m_new = fmaxf( m_run, tile_max );
			const float alpha = __expf( m_run - m_new );
			const float p = ( my_s > -1e29f ) ? __expf( my_s - m_new ) : 0.0f;
			float p_sum = p;
			#pragma unroll
			for ( int mask = 16; mask > 0; mask >>= 1 ) p_sum += __shfl_xor_sync( 0xffffffff, p_sum, mask );
			l_run = l_run * alpha + p_sum;
			m_run = m_new;
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) o[i] *= alpha;
			#pragma unroll
			for ( int j = 0; j < PF_TILE; ++j ) {
				const float pj = __shfl_sync( 0xffffffff, p, j );
				const uint4 raw = *reinterpret_cast<const uint4*>( sV + j * HD + lane * 8 );
				const half2* h2 = reinterpret_cast<const half2*>( &raw );
				#pragma unroll
				for ( int i = 0; i < 4; ++i ) {
					const float2 f = __half22float2( h2[i] );
					o[2 * i] = fmaf( pj, f.x, o[2 * i] );
					o[2 * i + 1] = fmaf( pj, f.y, o[2 * i + 1] );
				}
			}
		}
		__syncthreads();
	}

	if constexpr ( VQ == KVQ::Q4 ) {
		// Undo the per-64 rotation of V: bits 0..2 of an element index are the register, bits 3..5 the lane within its group of 8
		#pragma unroll
		for ( int h = 1; h < 8; h <<= 1 ) {
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) {
				if ( ( i & h ) == 0 ) {
					const float a = o[i], b = o[i | h];
					o[i] = a + b;
					o[i | h] = a - b;
				}
			}
		}
		#pragma unroll
		for ( int h = 1; h < 8; h <<= 1 ) {
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) {
				const float other = __shfl_xor_sync( 0xffffffff, o[i], h );
				o[i] = ( lane & h ) ? other - o[i] : o[i] + other;
			}
		}
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) o[i] *= 0.125f;
	}

	if ( valid ) {
		const float inv_l = ( l_run > 0.0f ) ? ( 1.0f / l_run ) : 0.0f;
		const size_t off = ( static_cast<size_t>( row ) * num_q_heads + head ) * HD + lane * 8;
		float sig[8];
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) sig[i] = 1.0f;
		if ( gate != nullptr ) {
			const uint4 graw = *reinterpret_cast<const uint4*>( gate + off );
			const half2* g2 = reinterpret_cast<const half2*>( &graw );
			#pragma unroll
			for ( int i = 0; i < 4; ++i ) {
				const float2 f = __half22float2( g2[i] );
				sig[2 * i] = 1.0f / ( 1.0f + __expf( -f.x ) );
				sig[2 * i + 1] = 1.0f / ( 1.0f + __expf( -f.y ) );
			}
		}
		uint4 res;
		half2* r2 = reinterpret_cast<half2*>( &res );
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			r2[i] = __floats2half2_rn( o[2 * i] * inv_l * sig[2 * i], o[2 * i + 1] * inv_l * sig[2 * i + 1] );
		}
		*reinterpret_cast<uint4*>( out + off ) = res;
	}
}

// Tensor-core prefill attention (Volta mma.m8n8k4, FP32 accumulate). A block owns 32 queries (row, head pairs) of one
// kv head and walks the keys in tiles of 64. S = Q K^T: warp (qg, kg) computes a 16 x 16 tile over the full head_dim, the
// four quad-pairs of each mma being its four 8 x 8 tiles. The softmax runs on S staged through shared memory (8 lanes per
// query row), P is written as FP16, and O += P V gives each warp 32 head dimensions (interleaved: d = 32w + dn + 4c so the
// transposed V rows are bank conflict free) for all 32 queries. V is staged transposed so a B fragment (4 consecutive
// keys of one dimension) is one 8-byte load. head_dim 256 only; keys past a query's own position are masked.
#define PA_BQ 32
#define PA_BK 64
#define PA_KS 260
#define PA_VS 68
#define PA_SS 68
#define PA_OFF_VT ( PA_BK * PA_KS * 2 )
#define PA_OFF_Q ( PA_OFF_VT + 256 * PA_VS * 2 )
#define PA_OFF_S ( PA_OFF_Q + PA_BQ * PA_KS * 2 )
#define PA_OFF_P ( PA_OFF_S + PA_BQ * PA_SS * 4 )
#define PA_OFF_A ( PA_OFF_P + PA_BQ * PA_VS * 2 )
#define PA_SMEM ( PA_OFF_A + PA_BQ * 12 )

__device__ __forceinline__ void pa_mma( float* c, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1 ) {
	asm volatile (
		"mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
		"{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
		: "+f"( c[0] ), "+f"( c[1] ), "+f"( c[2] ), "+f"( c[3] ), "+f"( c[4] ), "+f"( c[5] ), "+f"( c[6] ), "+f"( c[7] )
		: "r"( a0 ), "r"( a1 ), "r"( b0 ), "r"( b1 )
	);
}

template <KVQ KQ, KVQ VQ>
__global__ __launch_bounds__( 256 )
void attn_prefill_tc_kernel(
	const half* __restrict__ q,
	const pod::KVRef kv,
	const half* __restrict__ gate,
	half* __restrict__ out,
	size_t slot_stride,
	int layer_idx,
	const pod::BatchRows* __restrict__ rows,
	int count,
	int num_q_heads,
	int num_kv_heads,
	int max_seq,
	float scale,
	float* __restrict__ workspace,
	int nsplit
) {
	constexpr int HD = 256;
	extern __shared__ __align__( 16 ) unsigned char pa_smem[];
	half* sK = reinterpret_cast<half*>( pa_smem );
	half* sVt = reinterpret_cast<half*>( pa_smem + PA_OFF_VT );
	half* sQ = reinterpret_cast<half*>( pa_smem + PA_OFF_Q );
	float* sS = reinterpret_cast<float*>( pa_smem + PA_OFF_S );
	half* sP = reinterpret_cast<half*>( pa_smem + PA_OFF_P );
	float* sA = reinterpret_cast<float*>( pa_smem + PA_OFF_A );
	float* sM = sA + PA_BQ;
	float* sL = sM + PA_BQ;
	float* sO = reinterpret_cast<float*>( pa_smem );

	const int G = num_q_heads / num_kv_heads;
	const int kvh = blockIdx.x;
	const int q_base = blockIdx.y * PA_BQ;
	const int tid = threadIdx.x;
	const int warp = tid >> 5;
	const int lane = tid & 31;
	const int t = lane & 3;
	const int hi = lane >> 4;
	const int e = t + 4 * hi;
	const int qp = ( lane >> 2 ) & 3;

	const int last_row = min( ( q_base + PA_BQ - 1 ) / G, count - 1 );
	int max_pos = 0;
	for ( int r = q_base / G; r <= last_row; ++r ) max_pos = max( max_pos, rows->pos[r] );
	max_pos = min( max_pos, max_seq - 1 );
	const size_t slot_off = static_cast<size_t>( rows->slot[q_base / G] ) * slot_stride;
	const size_t kv_stride = static_cast<size_t>( num_kv_heads ) * HD;
	const size_t kv_base = slot_off + static_cast<size_t>( layer_idx ) * max_seq * kv_stride + static_cast<size_t>( kvh ) * HD;

	// Softmax role: 8 lanes per query row
	const int srow = tid >> 3;
	const int sub = tid & 7;
	const int sqi = q_base + srow;
	const bool svalid = sqi < count * G;
	const int spos = svalid ? rows->pos[sqi / G] : -1;
	float m_run = -1e30f, l_run = 0.0f;

	for ( int u = tid; u < PA_BQ * ( HD / 8 ); u += 256 ) {
		const int qq = u >> 5, c = u & 31;
		const int qi = q_base + qq;
		uint4 raw = make_uint4( 0, 0, 0, 0 );
		if ( qi < count * G ) {
			raw = *reinterpret_cast<const uint4*>( q + ( static_cast<size_t>( qi / G ) * num_q_heads + kvh * G + qi % G ) * HD + c * 8 );
		}
		uint2* dst = reinterpret_cast<uint2*>( sQ + qq * PA_KS + c * 8 );
		dst[0] = make_uint2( raw.x, raw.y );
		dst[1] = make_uint2( raw.z, raw.w );
	}

	const int s_qg = warp >> 2, s_kg = warp & 3;
	const half* s_arow = sQ + ( s_qg * 16 + 8 * ( qp >> 1 ) + e ) * PA_KS;
	const half* s_brow = sK + ( s_kg * 16 + 8 * ( qp & 1 ) + e ) * PA_KS;
	const half* v_row = sVt + ( 32 * warp + qp + 4 * e ) * PA_VS;
	const half* p_row = sP + e * PA_VS;

	float acc[4][8];
	#pragma unroll
	for ( int i = 0; i < 4; ++i ) {
		#pragma unroll
		for ( int r = 0; r < 8; ++r ) acc[i][r] = 0.0f;
	}

	// Each thread stages two keys (2 kp) x a 32-byte pair of chunks, twice; the next tile is fetched into registers while
	// the current one is computed on, and V is written transposed two keys at a time
	const int st_kp = tid & 31;
	KVRaw8 pk[2][2][2], pv[2][2][2];
	auto fetch = [&]( int kb ) {
		#pragma unroll
		for ( int ui = 0; ui < 2; ++ui ) {
			#pragma unroll
			for ( int kk = 0; kk < 2; ++kk ) {
				const int key = 2 * st_kp + kk;
				#pragma unroll
				for ( int cc = 0; cc < 2; ++cc ) {
					KVRaw8 k4 = {}, v4 = {};
					if ( kb + key <= max_pos ) {
						const int c = ( ( tid >> 5 ) + 8 * ui ) * 2 + cc;
						const size_t at = kv_base + static_cast<size_t>( kb + key ) * kv_stride + c * 8;
						k4 = kv_fetch8<KQ>( kv.k, kv.ks, at );
						v4 = kv_fetch8<VQ>( kv.v, kv.vs, at );
					}
					pk[ui][kk][cc] = k4;
					pv[ui][kk][cc] = v4;
				}
			}
		}
	};
	// Split the key tiles over blockIdx.z; a split with no tiles leaves an empty partial for the merge
	const int ntiles = ( max_pos + PA_BK ) / PA_BK;
	const int kb0 = ( blockIdx.z * ntiles / nsplit ) * PA_BK;
	const int kb1 = ( ( blockIdx.z + 1 ) * ntiles / nsplit ) * PA_BK;
	if ( kb0 < kb1 ) fetch( kb0 );

	for ( int kb = kb0; kb < kb1; kb += PA_BK ) {
		#pragma unroll
		for ( int ui = 0; ui < 2; ++ui ) {
			#pragma unroll
			for ( int cc = 0; cc < 2; ++cc ) {
				const int c = ( ( tid >> 5 ) + 8 * ui ) * 2 + cc;
				#pragma unroll
				for ( int kk = 0; kk < 2; ++kk ) {
					const uint4 kh = kv_decode8_h<KQ>( pk[ui][kk][cc] );
					uint2* kd = reinterpret_cast<uint2*>( sK + ( 2 * st_kp + kk ) * PA_KS + c * 8 );
					kd[0] = make_uint2( kh.x, kh.y );
					kd[1] = make_uint2( kh.z, kh.w );
				}
				const uint4 va = kv_decode8_h<VQ>( pv[ui][0][cc] ), vb = kv_decode8_h<VQ>( pv[ui][1][cc] );
				const half* v0 = reinterpret_cast<const half*>( &va );
				const half* v1 = reinterpret_cast<const half*>( &vb );
				#pragma unroll
				for ( int i = 0; i < 8; ++i ) {
					*reinterpret_cast<half2*>( sVt + ( c * 8 + i ) * PA_VS + 2 * st_kp ) = __halves2half2( v0[i], v1[i] );
				}
			}
		}
		__syncthreads();
		if ( kb + PA_BK < kb1 ) fetch( kb + PA_BK );

		// S = Q K^T for this warp's 16 x 16 tile
		{
			float c[8];
			#pragma unroll
			for ( int r = 0; r < 8; ++r ) c[r] = 0.0f;
			#pragma unroll 16
			for ( int ks = 0; ks < HD / 4; ++ks ) {
				const uint2 a = *reinterpret_cast<const uint2*>( s_arow + 4 * ks );
				const uint2 b = *reinterpret_cast<const uint2*>( s_brow + 4 * ks );
				pa_mma( c, a.x, a.y, b.x, b.y );
			}
			#pragma unroll
			for ( int r = 0; r < 8; ++r ) {
				const int m = ( t & 1 ) + 2 * ( ( r >> 1 ) & 1 ) + 4 * hi;
				const int n = ( r & 1 ) + 2 * ( t >> 1 ) + 4 * ( r >> 2 );
				sS[( s_qg * 16 + 8 * ( qp >> 1 ) + m ) * PA_SS + s_kg * 16 + 8 * ( qp & 1 ) + n] = c[r];
			}
		}
		__syncthreads();

		// Online softmax of this tile for one query row per 8 lanes
		{
			float sv[8];
			float mx = -1e30f;
			#pragma unroll
			for ( int j = 0; j < 8; ++j ) {
				const bool ok = svalid && kb + sub * 8 + j <= spos;
				sv[j] = ok ? sS[srow * PA_SS + sub * 8 + j] * scale : -1e30f;
				mx = fmaxf( mx, sv[j] );
			}
			#pragma unroll
			for ( int mask = 1; mask < 8; mask <<= 1 ) mx = fmaxf( mx, __shfl_xor_sync( 0xffffffff, mx, mask ) );
			const float m_new = fmaxf( m_run, mx );
			const float alpha = __expf( m_run - m_new );
			float sum = 0.0f;
			half2 ph[4];
			#pragma unroll
			for ( int j = 0; j < 4; ++j ) {
				const float p0 = sv[2 * j] > -1e29f ? __expf( sv[2 * j] - m_new ) : 0.0f;
				const float p1 = sv[2 * j + 1] > -1e29f ? __expf( sv[2 * j + 1] - m_new ) : 0.0f;
				sum += p0 + p1;
				ph[j] = __floats2half2_rn( p0, p1 );
			}
			#pragma unroll
			for ( int mask = 1; mask < 8; mask <<= 1 ) sum += __shfl_xor_sync( 0xffffffff, sum, mask );
			l_run = l_run * alpha + sum;
			m_run = m_new;
			uint2* pd = reinterpret_cast<uint2*>( sP + srow * PA_VS + sub * 8 );
			pd[0] = make_uint2( *reinterpret_cast<uint32_t*>( &ph[0] ), *reinterpret_cast<uint32_t*>( &ph[1] ) );
			pd[1] = make_uint2( *reinterpret_cast<uint32_t*>( &ph[2] ), *reinterpret_cast<uint32_t*>( &ph[3] ) );
			if ( sub == 0 ) sA[srow] = alpha;
		}
		__syncthreads();

		// O = O * alpha + P V for this warp's 32 dimensions
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			#pragma unroll
			for ( int r = 0; r < 8; ++r ) {
				const int m = ( t & 1 ) + 2 * ( ( r >> 1 ) & 1 ) + 4 * hi;
				acc[i][r] *= sA[8 * i + m];
			}
		}
		#pragma unroll
		for ( int kk = 0; kk < PA_BK / 4; ++kk ) {
			const uint2 b = *reinterpret_cast<const uint2*>( v_row + 4 * kk );
			#pragma unroll
			for ( int i = 0; i < 4; ++i ) {
				const uint2 a = *reinterpret_cast<const uint2*>( p_row + 8 * i * PA_VS + 4 * kk );
				pa_mma( acc[i], a.x, a.y, b.x, b.y );
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
	for ( int i = 0; i < 4; ++i ) {
		#pragma unroll
		for ( int r = 0; r < 8; ++r ) {
			const int m = ( t & 1 ) + 2 * ( ( r >> 1 ) & 1 ) + 4 * hi;
			const int n = ( r & 1 ) + 2 * ( t >> 1 ) + 4 * ( r >> 2 );
			sO[( 8 * i + m ) * PA_KS + 32 * warp + qp + 4 * n] = acc[i][r] * sA[8 * i + m];
		}
	}
	__syncthreads();

	if ( nsplit > 1 ) {
		// Unnormalized partial (m, l, O) for the merge, in the layout of the chunked decode kernel
		for ( int u = tid; u < PA_BQ * HD; u += 256 ) {
			const int qq = u >> 8, d = u & 255;
			const int qi = q_base + qq;
			if ( qi >= count * G ) continue;
			float* ws = workspace + ( ( static_cast<size_t>( qi / G ) * num_q_heads + kvh * G + qi % G ) * nsplit + blockIdx.z ) * ( ATTN_MAX_HEAD_DIM + 2 );
			if ( d == 0 ) {
				ws[0] = sM[qq];
				ws[1] = sL[qq];
			}
			ws[2 + d] = sO[qq * PA_KS + d];
		}
		return;
	}

	if constexpr ( VQ == KVQ::Q4 ) {
		// Undo the per-64 rotation of V
		if ( tid < PA_BQ * 4 ) {
			float* x = sO + ( tid >> 2 ) * PA_KS + ( tid & 3 ) * 64;
			for ( int h = 1; h < 64; h <<= 1 ) {
				for ( int i = 0; i < 64; ++i ) {
					if ( ( i & h ) == 0 ) {
						const float a = x[i], b = x[i | h];
						x[i] = a + b;
						x[i | h] = a - b;
					}
				}
			}
			for ( int i = 0; i < 64; ++i ) x[i] *= 0.125f;
		}
		__syncthreads();
	}

	for ( int u = tid; u < PA_BQ * ( HD / 8 ); u += 256 ) {
		const int qq = u >> 5, c = u & 31;
		const int qi = q_base + qq;
		if ( qi >= count * G ) continue;
		const size_t off = ( static_cast<size_t>( qi / G ) * num_q_heads + kvh * G + qi % G ) * HD + c * 8;
		float sig[8];
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) sig[i] = 1.0f;
		if ( gate != nullptr ) {
			const uint4 graw = *reinterpret_cast<const uint4*>( gate + off );
			const half2* g2 = reinterpret_cast<const half2*>( &graw );
			#pragma unroll
			for ( int i = 0; i < 4; ++i ) {
				const float2 f = __half22float2( g2[i] );
				sig[2 * i] = 1.0f / ( 1.0f + __expf( -f.x ) );
				sig[2 * i + 1] = 1.0f / ( 1.0f + __expf( -f.y ) );
			}
		}
		const float* o = sO + qq * PA_KS + c * 8;
		uint4 res;
		half2* r2 = reinterpret_cast<half2*>( &res );
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) r2[i] = __floats2half2_rn( o[2 * i] * sig[2 * i], o[2 * i + 1] * sig[2 * i + 1] );
		*reinterpret_cast<uint4*>( out + off ) = res;
	}
}

template <int D>
__global__ void attn_reduce_kernel(
	const float* __restrict__ workspace,
	const half* __restrict__ gate,
	half* __restrict__ output,
	int num_q_heads,
	int n_chunks,
	int vrot
) {
	const int token_idx = blockIdx.y;
	const int q_head = blockIdx.x;
	const int tid = threadIdx.x;
	if ( q_head >= num_q_heads || tid >= D ) return;

	float m_acc = -1e30f;
	float l_acc = 0.0f;
	float o_acc = 0.0f;

	const float* ws_base = workspace + ( static_cast<size_t>( token_idx ) * num_q_heads + q_head ) * n_chunks * ( ATTN_MAX_HEAD_DIM + 2 );

	for ( int c = 0; c < n_chunks; ++c ) {
		const float* ws_c = ws_base + static_cast<size_t>( c ) * ( ATTN_MAX_HEAD_DIM + 2 );
		float m_c = ws_c[0];
		float l_c = ws_c[1];
		float o_c = ws_c[2 + tid];

		if ( l_c == 0.0f ) continue;

		float m_new = fmaxf( m_acc, m_c );
		float alpha = __expf( m_acc - m_new );
		float beta = __expf( m_c - m_new );

		o_acc = fmaf( alpha, o_acc, beta * o_c );
		l_acc = fmaf( alpha, l_acc, beta * l_c );
		m_acc = m_new;
	}

	float inv_l = ( l_acc > 0.0f ) ? ( 1.0f / l_acc ) : 0.0f;
	float val = o_acc * inv_l;
	if ( vrot ) {
		__shared__ float s_rot[D];
		val = kv_fwht64_block( val, tid, s_rot );
	}
	float sig = 1.0f;
	if ( gate != nullptr ) {
		size_t g_off = ( static_cast<size_t>( token_idx ) * num_q_heads + q_head ) * D + tid;
		float g_val = __half2float( gate[g_off] );
		sig = 1.0f / ( 1.0f + __expf( -g_val ) );
	}
	size_t out_off = ( static_cast<size_t>( token_idx ) * num_q_heads + q_head ) * D + tid;
	output[out_off] = __float2half_rn( val * sig );
}

} // namespace

void Attention::unpack_q_gate(
	const pod::TensorView& q_gate,
	pod::TensorView& q,
	pod::TensorView& gate,
	int num_q_heads,
	int head_dim,
	cudaStream_t stream
) {
	int M = ( q_gate.ndim >= 2 && q_gate.shape[0] > 0 ) ? q_gate.shape[0] : 1;
	dim3 grid( num_q_heads, M );
	dim3 block( head_dim );
	unpack_q_gate_kernel<<<grid, block, 0, stream>>>(
		q_gate.ptr<const half>(),
		q.ptr<half>(),
		gate.ptr<half>(),
		num_q_heads,
		head_dim,
		M
	);
}

namespace {

template <KVQ Q>
void launch_kv_store( void* data, void* scales, const half* src, size_t slot_stride, int layer_idx,
	const pod::BatchRows* d_rows, int count, int total_elems, int max_seq, cudaStream_t stream ) {
	constexpr int U = kv_unit<Q>();
	ENGINE_ASSERT( total_elems % U == 0, "KV store: row width must be a multiple of the quantization group" );
	const int units = ( total_elems / U ) * count;
	kv_store_kernel<Q><<<( units + 127 ) / 128, 128, 0, stream>>>( data, scales, src, slot_stride, layer_idx, d_rows, count, total_elems, max_seq );
}

void kv_store_dispatch( KVQ q, void* data, void* scales, const half* src, size_t slot_stride, int layer_idx,
	const pod::BatchRows* d_rows, int count, int total_elems, int max_seq, cudaStream_t stream ) {
	switch ( q ) {
		case KVQ::F16: launch_kv_store<KVQ::F16>( data, scales, src, slot_stride, layer_idx, d_rows, count, total_elems, max_seq, stream ); break;
		case KVQ::E5M2: launch_kv_store<KVQ::E5M2>( data, scales, src, slot_stride, layer_idx, d_rows, count, total_elems, max_seq, stream ); break;
		case KVQ::I8: launch_kv_store<KVQ::I8>( data, scales, src, slot_stride, layer_idx, d_rows, count, total_elems, max_seq, stream ); break;
		case KVQ::Q4: launch_kv_store<KVQ::Q4>( data, scales, src, slot_stride, layer_idx, d_rows, count, total_elems, max_seq, stream ); break;
		case KVQ::FP4: launch_kv_store<KVQ::FP4>( data, scales, src, slot_stride, layer_idx, d_rows, count, total_elems, max_seq, stream ); break;
	}
}

// Instantiated combinations: (F16, F16), (E5M2, E5M2), (I8, Q4), (FP4, FP4)
#define KV_DISPATCH( KQV, VQV, CALL ) \
	if ( kv.kq == KVQ::F16 && kv.vq == KVQ::F16 ) { constexpr KVQ KQV = KVQ::F16; constexpr KVQ VQV = KVQ::F16; CALL; } \
	else if ( kv.kq == KVQ::E5M2 && kv.vq == KVQ::E5M2 ) { constexpr KVQ KQV = KVQ::E5M2; constexpr KVQ VQV = KVQ::E5M2; CALL; } \
	else if ( kv.kq == KVQ::I8 && kv.vq == KVQ::Q4 ) { constexpr KVQ KQV = KVQ::I8; constexpr KVQ VQV = KVQ::Q4; CALL; } \
	else if ( kv.kq == KVQ::FP4 && kv.vq == KVQ::FP4 ) { constexpr KVQ KQV = KVQ::FP4; constexpr KVQ VQV = KVQ::FP4; CALL; } \
	else { ENGINE_ASSERT( false, "Attention: unsupported KV format pair" ); }

} // namespace

void Attention::store_kv_cache(
	const pod::KVRef& kv,
	const pod::TensorView& k,
	const pod::TensorView& v,
	size_t slot_stride,
	int layer_idx,
	const pod::BatchRows* d_rows,
	int count,
	int num_kv_heads,
	int head_dim,
	int max_seq,
	cudaStream_t stream
) {
	const int total_elems = num_kv_heads * head_dim;
	kv_store_dispatch( kv.kq, kv.k, kv.ks, k.ptr<const half>(), slot_stride, layer_idx, d_rows, count, total_elems, max_seq, stream );
	kv_store_dispatch( kv.vq, kv.v, kv.vs, v.ptr<const half>(), slot_stride, layer_idx, d_rows, count, total_elems, max_seq, stream );
}

void Attention::forward_decode(
	const pod::KVRef& kv,
	const pod::TensorView& q,
	const pod::TensorView& gate,
	pod::TensorView& out,
	size_t slot_stride,
	int layer_idx,
	const pod::BatchRows* d_rows,
	int count,
	int num_q_heads,
	int num_kv_heads,
	int head_dim,
	int max_seq,
	cudaStream_t stream
) {
	// Keys per chunk: one page, doubled until the window fits the workspace's chunk capacity
	int chunk_len = ATTN_PAGE_SIZE;
	while ( ( max_seq + chunk_len - 1 ) / chunk_len > ATTN_MAX_CHUNKS ) chunk_len *= 2;
	const float scale = 1.0f / sqrtf( static_cast<float>( head_dim ) );

	// Position-independent grid: one chunk per page of the full window (graph-capturable)
	const int n_chunks = MAX( 1, ( max_seq + chunk_len - 1 ) / chunk_len );
	float* workspace = attn_workspace_ptr();
	const half* gate_ptr = gate.data != nullptr ? gate.ptr<const half>() : nullptr;
	const size_t row_elems = static_cast<size_t>( num_q_heads ) * head_dim;
	ENGINE_ASSERT( kv.vq != KVQ::Q4 || head_dim % 64 == 0, "Attention::forward_decode: Q4 values need head_dim % 64 == 0" );
	const int vrot = kv.vq == KVQ::Q4 ? 1 : 0;

	// The partial-result workspace holds MAX_BATCH rows; larger counts (a GLM prefill chunk) run in row batches
	for ( int row0 = 0; row0 < count; row0 += pod::MAX_BATCH ) {
		const int M = MIN( pod::MAX_BATCH, count - row0 );
		const half* g = gate_ptr ? gate_ptr + row0 * row_elems : nullptr;
		half* o = out.ptr<half>() + row0 * row_elems;
		#define LAUNCH_DECODE() attn_chunked_paged_kernel<KQ_, VQ_><<<dim3( num_q_heads, M, n_chunks ), dim3( head_dim ), 0, stream>>>( \
			q.ptr<const half>() + row0 * row_elems, kv, g, o, workspace, slot_stride, layer_idx, d_rows, \
			num_q_heads, num_kv_heads, head_dim, max_seq, n_chunks, chunk_len, row0, scale )
		#define KQ_ KQV_
		#define VQ_ VQV_
		KV_DISPATCH( KQV_, VQV_, LAUNCH_DECODE() )
		#undef KQ_
		#undef VQ_
		#undef LAUNCH_DECODE
		if ( n_chunks > 1 ) {
			const dim3 red_grid( num_q_heads, M );
			const dim3 red_block( head_dim );
			if ( head_dim == 256 ) attn_reduce_kernel<256><<<red_grid, red_block, 0, stream>>>( workspace, g, o, num_q_heads, n_chunks, vrot );
			else if ( head_dim == 128 ) attn_reduce_kernel<128><<<red_grid, red_block, 0, stream>>>( workspace, g, o, num_q_heads, n_chunks, vrot );
			else attn_reduce_kernel<64><<<red_grid, red_block, 0, stream>>>( workspace, g, o, num_q_heads, n_chunks, vrot );
		}
	}
}

void Attention::forward_prefill(
	const pod::KVRef& kv,
	const pod::TensorView& q,
	const pod::TensorView& gate,
	pod::TensorView& out,
	size_t slot_stride,
	int layer_idx,
	const pod::BatchRows* d_rows,
	int count,
	int num_q_heads,
	int num_kv_heads,
	int head_dim,
	int max_seq,
	cudaStream_t stream,
	int key_hint
) {
	ENGINE_ASSERT( head_dim == 256 && num_q_heads % num_kv_heads == 0, "Attention::forward_prefill: head_dim 256 and GQA only" );
	const float scale = 1.0f / sqrtf( static_cast<float>( head_dim ) );
	const int G = num_q_heads / num_kv_heads;
	// Spread the key tiles over extra blocks when the query tiles alone leave the GPU under two waves
	const int base_blocks = num_kv_heads * ( ( count * G + PA_BQ - 1 ) / PA_BQ );
	int nsplit = 1;
	if ( base_blocks < 160 ) nsplit = MIN( ( 320 + base_blocks - 1 ) / base_blocks, 32 );
	if ( key_hint > 0 ) nsplit = MAX( 1, MIN( nsplit, key_hint / ( 4 * PA_BK ) ) );
	if ( const char* env = std::getenv( "EXL3_PREFILL_SPLIT" ); env && count <= pod::MAX_BATCH ) nsplit = MAX( 1, MIN( std::atoi( env ), 32 ) );
	ENGINE_ASSERT( static_cast<size_t>( count ) * num_q_heads * nsplit <= static_cast<size_t>( ATTN_MAX_ROW_HEADS ) * ATTN_MAX_CHUNKS, "Attention::forward_prefill: workspace" );
	float* workspace = attn_workspace_ptr();
	const dim3 grid( num_kv_heads, ( count * G + PF_QPB - 1 ) / PF_QPB );
	const dim3 grid_tc( num_kv_heads, ( count * G + PA_BQ - 1 ) / PA_BQ, nsplit );
	static const bool simt = std::getenv( "EXL3_PREFILL_SIMT" ) != nullptr;
	const half* gate_ptr = gate.data != nullptr ? gate.ptr<const half>() : nullptr;
	#define LAUNCH_PREFILL() do { \
		if ( simt ) { \
			attn_prefill_kernel<KQ_, VQ_><<<grid, 256, 0, stream>>>( \
				q.ptr<const half>(), kv, gate_ptr, out.ptr<half>(), slot_stride, layer_idx, d_rows, count, num_q_heads, num_kv_heads, max_seq, scale ); \
		} \
		else { \
			CUDA_CHECK( cudaFuncSetAttribute( attn_prefill_tc_kernel<KQ_, VQ_>, cudaFuncAttributeMaxDynamicSharedMemorySize, PA_SMEM ) ); \
			attn_prefill_tc_kernel<KQ_, VQ_><<<grid_tc, 256, PA_SMEM, stream>>>( \
				q.ptr<const half>(), kv, gate_ptr, out.ptr<half>(), slot_stride, layer_idx, d_rows, count, num_q_heads, num_kv_heads, max_seq, scale, workspace, nsplit ); \
		} \
	} while ( 0 )
	#define KQ_ KQV_
	#define VQ_ VQV_
	KV_DISPATCH( KQV_, VQV_, LAUNCH_PREFILL() )
	#undef KQ_
	#undef VQ_
	if ( !simt && nsplit > 1 ) {
		const int vrot = kv.vq == KVQ::Q4 ? 1 : 0;
		attn_reduce_kernel<256><<<dim3( num_q_heads, count ), 256, 0, stream>>>( workspace, gate_ptr, out.ptr<half>(), num_q_heads, nsplit, vrot );
	}
	#undef LAUNCH_PREFILL
}

} // namespace engine::modules
