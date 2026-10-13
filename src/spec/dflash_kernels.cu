#include "engine/spec/dflash.h"
#include "engine/modules/norm.h"
#include "engine/modules/swiglu.h"
#include "engine/modules/embedding.h"
#include "engine/modules/linear.h"
#include "../kernels/cuda/trellis/trellis_context.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <chrono>
#include <cstring>

namespace engine::spec {

namespace {

#define DYNCONV_THREADS 256

template <typename T> __device__ __forceinline__ float to_f( T v ) { return static_cast<float>( v ); }
template <> __device__ __forceinline__ float to_f( half v ) { return __half2float( v ); }
template <typename T> __device__ __forceinline__ T from_f( float v ) { return static_cast<T>( v ); }
template <> __device__ __forceinline__ half from_f( float v ) { return __float2half_rn( v ); }

// TD is the dynamic-coefficient type: FP32 everywhere in the engine.
template <typename TX, typename TD, typename TB, typename TO, bool accumulate>
__global__ __launch_bounds__(DYNCONV_THREADS)
void dflash2_dynconv_kernel(
	const TX* __restrict__ x,
	const TD* __restrict__ dyn,
	const TB* __restrict__ base,
	TO* __restrict__ out,
	const int seqlen,
	const int hidden,
	const int group_size,
	const int taps,
	const int64_t ds_b,
	const int64_t ds_t,
	const int64_t ds_k,
	const int64_t ds_g
) {
	int c = blockIdx.x * DYNCONV_THREADS + threadIdx.x;
	if ( c >= hidden ) return;
	int t = blockIdx.y;
	int b = blockIdx.z;
	int g = c / group_size;

	const TD* dyn_bt = dyn + b * ds_b + t * ds_t + g * ds_g;
	const TX* x_bt = x + ( static_cast<int64_t>( b ) * seqlen + t ) * hidden + c;

	float acc = 0.0f;
	#pragma unroll 4
	for ( int k = 0; k < taps; ++k ) {
		if ( k > t ) break;
		float w = to_f( base[k * hidden + c] ) + to_f( dyn_bt[k * ds_k] );
		acc += w * to_f( x_bt[-static_cast<int64_t>( k ) * hidden] );
	}

	TO* o = out + ( static_cast<int64_t>( b ) * seqlen + t ) * hidden + c;
	if constexpr ( accumulate ) {
		*o = from_f<TO>( to_f( *o ) + acc );
	}
	else {
		*o = from_f<TO>( acc );
	}
}

#define WALK_THREADS 256

// ---------------------------------------------------------------------------------------------------------
// Backbone building blocks.
//
// Plain FP16 GEMV, out[m, n] = sum_k w[n, k] * x[m, k] for m <= 8 rows. One warp per output column;
// lanes stride over k so the weight loads coalesce. Used for the drafter's two conv kernel projections
// and the selector's hidden projection, which are plain FP16 in the checkpoint rather than trellis.
// out[m][n] = sum_k w[n][k] * x[m][k], with w laid out [n, k] row-major (wcol = w + col * k).
//
// One warp owns one (row, col) pair, so m is unbounded. The previous shape -- one warp per column
// holding acc[8] for all 8 rows -- was fine for the 8-row query block but asserted at m > 8, which is
// exactly the context projection's shape: ctx_chunk_ is up to 256 rows, so any context longer than
// 8 rows hit the assert. The context path had therefore never executed.
__global__ __launch_bounds__(256)
void dflash_f16_gemv_nt_kernel(
	const half* __restrict__ w,
	const half* __restrict__ x,
	half* __restrict__ out,
	int n, int k, int m
) {
	const int warps_per_block = blockDim.x >> 5;
	const int warp_id = blockIdx.x * warps_per_block + ( threadIdx.x >> 5 );
	const int total = m * n;
	if ( warp_id >= total ) return;

	const int row = warp_id / n;
	const int col = warp_id - row * n;
	const int lane = threadIdx.x & 31;

	const half* wcol = w + static_cast<int64_t>( col ) * k;
	const half* xrow = x + static_cast<int64_t>( row ) * k;

	float acc = 0.0f;
	for ( int kk = lane; kk < k; kk += 32 )
		acc = fmaf( __half2float( wcol[kk] ), __half2float( xrow[kk] ), acc );
	#pragma unroll
	for ( int mask = 16; mask > 0; mask >>= 1 )
		acc += __shfl_xor_sync( 0xffffffff, acc, mask );

	if ( lane == 0 ) out[static_cast<int64_t>( row ) * n + col] = __float2half_rn( acc );
}

// Skinny variant for up to MR rows (the LM head at decode / verify): one warp per output column streams the weight
// row once with 128-bit loads and accumulates all rows, instead of re-reading it per row. k % 8 == 0.
template <int MR>
__global__ __launch_bounds__(256)
void dflash_f16_gemv_skinny_kernel(
	const half* __restrict__ w,
	const half* __restrict__ x,
	half* __restrict__ out,
	int n, int k, int m
) {
	const int col = blockIdx.x * ( blockDim.x >> 5 ) + ( threadIdx.x >> 5 );
	if ( col >= n ) return;
	const int lane = threadIdx.x & 31;
	const half* wc = w + static_cast<int64_t>( col ) * k;
	float acc[MR];
	#pragma unroll
	for ( int r = 0; r < MR; ++r ) acc[r] = 0.0f;
	for ( int k0 = lane * 8; k0 < k; k0 += 256 ) {
		const uint4 wv = __ldcs( reinterpret_cast<const uint4*>( wc + k0 ) );
		const half2* w2 = reinterpret_cast<const half2*>( &wv );
		#pragma unroll
		for ( int r = 0; r < MR; ++r ) {
			if ( r < m ) {
				const uint4 xv = __ldg( reinterpret_cast<const uint4*>( x + static_cast<int64_t>( r ) * k + k0 ) );
				const half2* x2 = reinterpret_cast<const half2*>( &xv );
				float a = acc[r];
				#pragma unroll
				for ( int j = 0; j < 4; ++j ) {
					const float2 wf = __half22float2( w2[j] ), xf = __half22float2( x2[j] );
					a = fmaf( wf.x, xf.x, a );
					a = fmaf( wf.y, xf.y, a );
				}
				acc[r] = a;
			}
		}
	}
	#pragma unroll
	for ( int r = 0; r < MR; ++r ) {
		float a = acc[r];
		#pragma unroll
		for ( int mask = 16; mask > 0; mask >>= 1 ) a += __shfl_xor_sync( 0xffffffff, a, mask );
		if ( lane == 0 && r < m ) out[static_cast<int64_t>( r ) * n + col] = __float2half_rn( a );
	}
}

// RMSNorm with the residual split the reference uses (qwen3_dflash2.py:294-299):
//   sum = x + residual_in;  normed = rmsnorm(sum) * (w + bias);  residual_out = sum
// One block per row. The residual is FP32 in and FP32 out: it accumulates across all layers and the
// reference keeps range there deliberately (dflash_sm70.py carries FP16 transport but rounds at the
// trained BF16 points, which needs the headroom). Pass a null residual_in for the first layer, where
// the reference sets residual = hidden and normalizes the embedding alone.
// Round an FP32 to BF16 with round-to-nearest-even, keeping the value in FP32 registers. The drafter
// is a BF16 model and vLLM's DFlashSM70RMSNorm emulates BF16 exactly this way on SM70, which has no
// native BF16. Matching the trained rounding points matters more than the extra mantissa bits.
__device__ __forceinline__ float dflash_bf16_rne( float v ) {
	unsigned int bits = __float_as_uint( v );
	bits = ( bits + 0x7FFFu + ( ( bits >> 16 ) & 1u ) ) & 0xFFFF0000u;
	return __uint_as_float( bits );
}

template <typename XT>
__global__ void dflash_rmsnorm_residual_kernel(
	const XT* __restrict__ x,
	const float* __restrict__ residual_in,
	const half* __restrict__ weight,
	half* __restrict__ normed_out,
	float* __restrict__ residual_out,
	int hidden_size,
	float eps,
	float constant_bias
) {
	const int row = blockIdx.x;
	const XT* row_src = x + static_cast<int64_t>( row ) * hidden_size;
	const float* row_res = ( residual_in != nullptr ) ? residual_in + static_cast<int64_t>( row ) * hidden_size : nullptr;
	half* row_dst = normed_out + static_cast<int64_t>( row ) * hidden_size;
	float* row_rop = residual_out + static_cast<int64_t>( row ) * hidden_size;
	const int tid = threadIdx.x;

	float local_sq = 0.0f;
	for ( int i = tid; i < hidden_size; i += blockDim.x ) {
		// The residual transport is BF16-rounded on the way in, matching DFlashSM70RMSNorm.
		const float val = dflash_bf16_rne( static_cast<float>( row_src[i] ) + ( row_res ? row_res[i] : 0.0f ) );
		row_rop[i] = val;
		local_sq = fmaf( val, val, local_sq );
	}

	#pragma unroll
	for ( int mask = 16; mask > 0; mask >>= 1 ) local_sq += __shfl_xor_sync( 0xffffffff, local_sq, mask );
	__shared__ float s_warp_sums[8];
	const int warp_id = tid >> 5;
	const int lane_id = tid & 31;
	if ( lane_id == 0 ) s_warp_sums[warp_id] = local_sq;
	__syncthreads();
	if ( warp_id == 0 ) {
		float b_sum = ( lane_id < ( blockDim.x >> 5 ) ) ? s_warp_sums[lane_id] : 0.0f;
		#pragma unroll
		for ( int mask = 4; mask > 0; mask >>= 1 ) b_sum += __shfl_xor_sync( 0xffffffff, b_sum, mask );
		if ( lane_id == 0 ) s_warp_sums[0] = rsqrtf( b_sum / static_cast<float>( hidden_size ) + eps );
	}
	__syncthreads();
	const float inv_rms = s_warp_sums[0];

	for ( int i = tid; i < hidden_size; i += blockDim.x ) {
		// Both the weight and the product are BF16-rounded before the FP16 cast, as the reference does:
		// `output = (normalized * weight).to(bfloat16).to(orig_dtype)`.
		const float w = ( weight != nullptr )
			? dflash_bf16_rne( __half2float( weight[i] ) + constant_bias )
			: 1.0f;
		row_dst[i] = __float2half_rn( dflash_bf16_rne( row_rop[i] * w * inv_rms ) );
	}
}

__global__ __launch_bounds__(WALK_THREADS)
void dflash2_selector_walk_kernel(
	const float* __restrict__ unary,        // [bsz, rows, k]
	const int* __restrict__ cands,          // [bsz, rows, k]
	const half* __restrict__ gate,          // [bsz, rows, rank]
	const half* __restrict__ pred_cb,       // [vocab, rank]
	const half* __restrict__ succ_cb,       // [vocab, rank]
	const int* __restrict__ anchor,         // [bsz]
	int* __restrict__ out,                  // [bsz, rows + 1]
	float* __restrict__ conf,               // [bsz, rows + 1] or nullptr
	const int rows,
	const int k,
	const int rank
) {
	extern __shared__ float smem[];
	float* a_g = smem;                      // [rank]: A[pred] * gate[i]
	float* scores = smem + rank;            // [k]
	__shared__ int s_pred;

	const int b = blockIdx.x;
	const int t = threadIdx.x;
	const int warp = t / 32, lane = t % 32;

	if ( t == 0 ) {
		s_pred = anchor[b];
		out[static_cast<int64_t>( b ) * ( rows + 1 )] = s_pred;
		if ( conf ) conf[static_cast<int64_t>( b ) * ( rows + 1 )] = 1.0f;
	}
	__syncthreads();

	for ( int i = 0; i < rows; ++i ) {
		const int pred = s_pred;
		const half* a_row = pred_cb + static_cast<int64_t>( pred ) * rank;
		const half* g_row = gate + ( static_cast<int64_t>( b ) * rows + i ) * rank;

		for ( int r = t; r < rank; r += WALK_THREADS ) {
			a_g[r] = __half2float( a_row[r] ) * __half2float( g_row[r] );
		}
		__syncthreads();

		const int* c_row = cands + ( static_cast<int64_t>( b ) * rows + i ) * k;
		const float* u_row = unary + ( static_cast<int64_t>( b ) * rows + i ) * k;

		for ( int c = warp; c < k; c += WALK_THREADS / 32 ) {
			const half* b_row = succ_cb + static_cast<int64_t>( c_row[c] ) * rank;
			float dot = 0.0f;
			for ( int r = lane; r < rank; r += 32 ) {
				dot += a_g[r] * __half2float( b_row[r] );
			}
			#pragma unroll
			for ( int offset = 16; offset > 0; offset /= 2 ) {
				dot += __shfl_xor_sync( 0xffffffff, dot, offset );
			}
			if ( lane == 0 ) scores[c] = u_row[c] + dot;
		}
		__syncthreads();

		if ( t == 0 ) {
			int best = 0;
			float best_score = scores[0];
			for ( int c = 1; c < k; ++c ) {
				if ( scores[c] > best_score ) {
					best_score = scores[c];
					best = c;
				}
			}
			s_pred = c_row[best];
			out[static_cast<int64_t>( b ) * ( rows + 1 ) + i + 1] = s_pred;
			if ( conf ) {
				float sum_exp = 0.0f;
				for ( int c = 0; c < k; ++c ) {
					sum_exp += expf( scores[c] - best_score );
				}
				conf[static_cast<int64_t>( b ) * ( rows + 1 ) + i + 1] = 1.0f / sum_exp;
			}
		}
		__syncthreads();
	}
}

#define TOPK_THREADS 256
static_assert( TOPK_THREADS * 16 == kDflashTopkTile, "kDflashTopkTile must match the top-k tile" );

template <int N, int K>
__device__ __forceinline__ void warp_topk( float (&x)[N], int (&id)[N], float* out_v, int* out_ix ) {
	const int lane = threadIdx.x % 32;
	float cur = -INFINITY;
	int arg = 0;
	#pragma unroll
	for ( int i = 0; i < N; ++i ) {
		if ( x[i] > cur ) {
			cur = x[i];
			arg = i;
		}
	}
	for ( int k = 0; k < K; ++k ) {
		float best = cur;
		int best_lane = lane;
		#pragma unroll
		for ( int offset = 16; offset > 0; offset /= 2 ) {
			float o_best = __shfl_xor_sync( 0xffffffff, best, offset );
			int o_lane = __shfl_xor_sync( 0xffffffff, best_lane, offset );
			if ( o_best > best || ( o_best == best && o_lane < best_lane ) ) {
				best = o_best;
				best_lane = o_lane;
			}
		}
		int best_id = -1;
		#pragma unroll
		for ( int i = 0; i < N; ++i ) {
			if ( i == arg ) best_id = id[i];
		}
		best_id = __shfl_sync( 0xffffffff, best_id, best_lane );
		if ( lane == 0 ) {
			out_v[k] = best;
			out_ix[k] = best_id;
		}
		if ( lane == best_lane ) {
			#pragma unroll
			for ( int i = 0; i < N; ++i ) {
				if ( i == arg ) x[i] = -INFINITY;
			}
			cur = -INFINITY;
			arg = 0;
			#pragma unroll
			for ( int i = 0; i < N; ++i ) {
				if ( x[i] > cur ) {
					cur = x[i];
					arg = i;
				}
			}
		}
	}
}

template <int N, int K>
__device__ __forceinline__ void block_topk( float (&x)[N], int (&id)[N], float* out_v, int* out_ix, float* s_wv, int* s_wix ) {
	constexpr int M = K / 4;
	const int t = threadIdx.x, warp = t / 32, lane = t % 32;
	warp_topk<N, K>( x, id, s_wv + warp * K, s_wix + warp * K );
	__syncthreads();
	if ( warp == 0 ) {
		float y[M];
		int yid[M];
		#pragma unroll
		for ( int i = 0; i < M; ++i ) {
			y[i] = s_wv[lane * M + i];
			yid[i] = s_wix[lane * M + i];
		}
		warp_topk<M, K>( y, yid, out_v, out_ix );
	}
	__syncthreads();
}

template <typename T, int K, int E>
__global__ __launch_bounds__(TOPK_THREADS)
void dflash2_topk_kernel(
	const T* __restrict__ logits,
	const int64_t stride_b,
	const int64_t stride_r,
	const int rows,
	const int splits,
	const int vocab,
	const float scale,
	const float softcap,
	float* __restrict__ part_v,
	int* __restrict__ part_ix,
	int* __restrict__ counters,
	float* __restrict__ values,
	int* __restrict__ indices,
	int index_offset
) {
	constexpr int VEC = 16 / sizeof( T );
	constexpr int LOADS = E / VEC;
	constexpr int SPAN = TOPK_THREADS * VEC;

	__shared__ float s_wv[( TOPK_THREADS / 32 ) * K];
	__shared__ int s_wix[( TOPK_THREADS / 32 ) * K];
	__shared__ bool s_last;

	const int row_id = blockIdx.x / splits;
	const int split = blockIdx.x % splits;
	const int b = row_id / rows, r = row_id % rows;
	const int t = threadIdx.x;
	const T* row = logits + b * stride_b + r * stride_r;
	const int c0 = split * ( TOPK_THREADS * E );

	float x[E];
	int id[E];
	#pragma unroll
	for ( int l = 0; l < LOADS; ++l ) {
		const int c = c0 + l * SPAN + t * VEC;
		if ( c + VEC <= vocab ) {
			int4 raw = *( (const int4*) ( row + c ) );
			const T* xs = (const T*) &raw;
			#pragma unroll
			for ( int i = 0; i < VEC; ++i ) {
				float xv = to_f( xs[i] ) * scale;
				if ( softcap > 0.0f ) xv = tanhf( xv / softcap ) * softcap;
				x[l * VEC + i] = xv;
				id[l * VEC + i] = c + i;
			}
		}
		else {
			#pragma unroll
			for ( int i = 0; i < VEC; ++i ) {
				float xv = -INFINITY;
				if ( c + i < vocab ) {
					xv = to_f( row[c + i] ) * scale;
					if ( softcap > 0.0f ) xv = tanhf( xv / softcap ) * softcap;
				}
				x[l * VEC + i] = xv;
				id[l * VEC + i] = c + i;
			}
		}
	}

	float* pv = part_v + ( static_cast<int64_t>( row_id ) * splits + split ) * K;
	int* pix = part_ix + ( static_cast<int64_t>( row_id ) * splits + split ) * K;
	block_topk<E, K>( x, id, pv, pix, s_wv, s_wix );

	__threadfence();
	if ( t == 0 ) {
		int arrived = atomicAdd( &counters[row_id], 1 );
		s_last = ( arrived == splits - 1 );
	}
	__syncthreads();
	if ( !s_last ) return;
	__threadfence();

	constexpr int M = K / 4;
	float y[M];
	int yid[M];
	const float* rv = part_v + static_cast<int64_t>( row_id ) * splits * K;
	const int* rix = part_ix + static_cast<int64_t>( row_id ) * splits * K;
	#pragma unroll
	for ( int i = 0; i < M; ++i ) {
		int j = t * M + i;
		bool ok = ( j < splits * K );
		y[i] = ok ? __ldcg( rv + j ) : -INFINITY;
		yid[i] = ok ? __ldcg( rix + j ) : -1;
	}
	__shared__ float o_v[K];
	__shared__ int o_ix[K];
	block_topk<M, K>( y, yid, o_v, o_ix, s_wv, s_wix );
	if ( t < K ) {
		int64_t o = static_cast<int64_t>( row_id ) * K + t;
		values[o] = o_v[t];
		indices[o] = index_offset + o_ix[t];
	}
	if ( t == 0 ) counters[row_id] = 0;
}

__global__ __launch_bounds__(256)
void dflash_qk_norm_rope_kernel(
	half* __restrict__ x,               // [rows, heads, head_dim]
	const half* __restrict__ weight,    // [head_dim]
	const float* __restrict__ rope_cos, // [max_pos, pairs]
	const float* __restrict__ rope_sin,
	int rows,
	int heads,
	int head_dim,
	int pairs,
	int pos0,
	float eps,
	float constant_bias
) {
	const int row = blockIdx.y;
	const int head = blockIdx.x;
	if ( row >= rows || head >= heads ) return;
	const int pos = pos0 + row;

	half* hrow = x + ( static_cast<int64_t>( row ) * heads + head ) * head_dim;
	const float* cos_p = rope_cos + static_cast<int64_t>( pos ) * pairs;
	const float* sin_p = rope_sin + static_cast<int64_t>( pos ) * pairs;
	const int tid = threadIdx.x;

	float local_sq = 0.0f;
	for ( int i = tid; i < head_dim; i += blockDim.x ) {
		const float v = __half2float( hrow[i] );
		local_sq = fmaf( v, v, local_sq );
	}
	#pragma unroll
	for ( int mask = 16; mask > 0; mask >>= 1 ) local_sq += __shfl_xor_sync( 0xffffffff, local_sq, mask );
	__shared__ float s_warp_sums[8];
	__shared__ float s_inv;
	const int warp_id = tid >> 5;
	const int lane_id = tid & 31;
	if ( lane_id == 0 ) s_warp_sums[warp_id] = local_sq;
	__syncthreads();
	if ( tid == 0 ) {
		float b_sum = 0.0f;
		#pragma unroll
		for ( int w = 0; w < ( blockDim.x >> 5 ); ++w ) b_sum += s_warp_sums[w];
		s_inv = rsqrtf( b_sum / static_cast<float>( head_dim ) + eps );
	}
	__syncthreads();
	const float inv_rms = s_inv;

	// Stage the normalized values: each element needs its rotation partner's *normalized* value, and
	// reading hrow directly would race with the partner writing its own rotated result.
	__shared__ float s_norm[128];
	for ( int i = tid; i < head_dim; i += blockDim.x ) {
		float v = __half2float( hrow[i] ) * inv_rms;
		if ( weight != nullptr ) v *= __half2float( weight[i] ) + constant_bias;
		s_norm[i] = v;
	}
	__syncthreads();

	for ( int i = tid; i < head_dim; i += blockDim.x ) {
		if ( i >= 2 * pairs ) {
			hrow[i] = __float2half_rn( s_norm[i] );
			continue;
		}
		// NeoX layout (rotate-half): element i pairs with i + pairs, not with its neighbour. The drafter
		// checkpoint declares no other layout and both reference implementations (exllamav3 RopeStyle.NEOX,
		// vLLM is_neox_style=True) default to this one; adjacent-pair rotation halves later-row acceptance.
		const int p = ( i < pairs ) ? i : i - pairs;
		const float c = cos_p[p];
		const float s = sin_p[p];
		hrow[i] = __float2half_rn( ( i < pairs ) ? ( s_norm[i] * c - s_norm[i + pairs] * s )
		                                         : ( s_norm[i] * c + s_norm[i - pairs] * s ) );
	}
}

// GQA small-query attention for the drafter block: `rows` queries (<= 8, the anchor plus its mask
// tokens) against the projected target context plus the block's own K/V. One block per (row, q_head),
// one warp per key so the K/V loads coalesce along head_dim. The block is bidirectional -- the
// checkpoint's is_causal is false and all five draft layers are sliding_attention, so upstream's
// block_window() returns (sliding_window - 1, sliding_window - 1) for every layer and every query row
// sees every other row (triton_paged.py:121-127 applies the q_abs <= k bound only under CAUSAL, which
// prepare_inputs sets to `is_causal is True` = false here). The caller narrows the context side to the
// sliding_window by choosing first_ctx/nctx.
//
// Two-pass softmax with the scores in shared memory. Recomputing nothing keeps the reduction order
// independent of how keys map to warps, which matters because the draft round is graph-replayed.
__global__ __launch_bounds__(128)
void dflash_attn_kernel(
	const half* __restrict__ q,       // [rows, q_heads, head_dim]
	const half* __restrict__ kcache,  // [slots, layers, kv_heads, head_dim]
	const half* __restrict__ vcache,
	half* __restrict__ out,           // [rows, q_heads, head_dim]
	int rows,
	int q_heads,
	int kv_heads,
	int head_dim,
	int nctx,        // context keys, logical slots [first_ctx, first_ctx + nctx), positions 0..anchor
	int first_ctx,
	int ring_slots,  // context KV slots form a ring of this length
	int ctx_layer,
	int stride_slot,
	int stride_layer,
	int stride_head,
	int own_slot,    // cache slot holding the block's own K/V (row r at own_slot + r)
	int own_first_row, // 1: rows 1..rows-1 are keys (anchor comes from the context). 0: rows 0..rows-1.
	float scale
) {
	extern __shared__ float smem[];
	const int row = blockIdx.y;
	const int head = blockIdx.x;
	if ( row >= rows || head >= q_heads ) return;
	// The block contributes rows own_first_row..rows-1. With own_first_row = 1 the anchor's position is
	// already covered by the context, so upstream's tap write replaces the transient row-0 K/V there
	// and row 0 is not a key; counting it again would attend to the anchor position twice.
	const int nblk = rows - own_first_row;
	const int nkeys = nctx + nblk;
	const int warps = blockDim.x >> 5;
	float* s_scores = smem;                              // [nkeys]
	float* s_acc = smem + nkeys;                         // [warps, head_dim], one row per warp

	const int warp0 = threadIdx.x >> 5;
	const int lane = threadIdx.x & 31;
	const int warp = threadIdx.x >> 5;
	const int g = head / ( q_heads / kv_heads );

	const half* qrow = q + ( static_cast<int64_t>( row ) * q_heads + head ) * head_dim;

	// Key kk maps to a context slot first, then to the block's own rows.
	auto ptr_of = [&] ( const half* base, int kk ) {
		int slot;
		if ( kk < nctx ) {
			// The context is a ring; one conditional subtract is enough since kk < nctx <= ring_slots.
			slot = first_ctx + kk;
			if ( slot >= ring_slots ) slot -= ring_slots;
		}
		else slot = own_slot + own_first_row + ( kk - nctx );
		return base + static_cast<int64_t>( slot ) * stride_slot
			+ static_cast<int64_t>( ctx_layer ) * stride_layer
			+ static_cast<int64_t>( g ) * stride_head;
	};

	// Pass 1: scores.
	for ( int kk = warp; kk < nkeys; kk += warps ) {
		const half* krow = ptr_of( kcache, kk );
		float dot = 0.0f;
		for ( int d = lane; d < head_dim; d += 32 )
			dot = fmaf( __half2float( qrow[d] ), __half2float( krow[d] ), dot );
		#pragma unroll
		for ( int mask = 16; mask > 0; mask >>= 1 ) dot += __shfl_xor_sync( 0xffffffff, dot, mask );
		__syncwarp();
		if ( lane == 0 ) s_scores[kk] = dot * scale;
	}
	__syncthreads();

	float m = -INFINITY;
	for ( int kk = threadIdx.x; kk < nkeys; kk += blockDim.x ) m = fmaxf( m, s_scores[kk] );
	#pragma unroll
	for ( int mask = 16; mask > 0; mask >>= 1 ) m = fmaxf( m, __shfl_xor_sync( 0xffffffff, m, mask ) );
	__shared__ float s_warp_max[4];
	__shared__ float s_max;
	if ( ( threadIdx.x & 31 ) == 0 ) s_warp_max[threadIdx.x >> 5] = m;
	__syncthreads();
	if ( threadIdx.x == 0 ) {
		float mm = s_warp_max[0];
		for ( int w = 1; w < warps; ++w ) mm = fmaxf( mm, s_warp_max[w] );
		s_max = mm;
	}
	__syncthreads();
	const float row_max = s_max;

	// Pass 2: exponentiate, blend the value vectors, and total the denominator.
	// One accumulator row per warp: the warps in a block accumulate into the same output vector, so
	// sharing one row would race.
	for ( int i = threadIdx.x; i < warps * head_dim; i += blockDim.x ) s_acc[i] = 0.0f;
	__syncthreads();
	for ( int kk = warp; kk < nkeys; kk += warps ) {
		const float p = expf( s_scores[kk] - row_max );
		// Volta has independent thread scheduling, so lanes of this warp need the value lane 0 just
		// stored before any of them reads it.
		__syncwarp();
		if ( lane == 0 ) s_scores[kk] = p;
		__syncwarp();
		const half* vrow = ptr_of( vcache, kk );
		float* acc_w = s_acc + static_cast<int64_t>( warp0 ) * head_dim;
		for ( int d = lane; d < head_dim; d += 32 )
			acc_w[d] = fmaf( p, __half2float( vrow[d] ), acc_w[d] );
	}
	__syncthreads();

	float l = 0.0f;
	for ( int kk = threadIdx.x; kk < nkeys; kk += blockDim.x ) l += s_scores[kk];
	#pragma unroll
	for ( int mask = 16; mask > 0; mask >>= 1 ) l += __shfl_xor_sync( 0xffffffff, l, mask );
	__shared__ float s_warp_sum[4];
	__shared__ float s_sum;
	if ( ( threadIdx.x & 31 ) == 0 ) s_warp_sum[threadIdx.x >> 5] = l;
	__syncthreads();
	if ( threadIdx.x == 0 ) {
		float ls = s_warp_sum[0];
		for ( int w = 1; w < warps; ++w ) ls += s_warp_sum[w];
		s_sum = ls;
	}
	__syncthreads();
	const float inv_l = 1.0f / s_sum;

	half* orow = out + ( static_cast<int64_t>( row ) * q_heads + head ) * head_dim;
	for ( int d = threadIdx.x; d < head_dim; d += blockDim.x ) {
		float a = 0.0f;
		for ( int w = 0; w < warps; ++w ) a += s_acc[static_cast<int64_t>( w ) * head_dim + d];
		orow[d] = __float2half_rn( a * inv_l );
	}
}

// Take one side out of a conv kernel projection's [rows, 2*taps*groups] output and reinterpret it as
// the [rows, taps, groups] the grouped conv indexes. The projection emits both sides of the sandwich
// contiguously, side 0 first.
__global__ void dflash_conv_side_split_kernel(
	float* __restrict__ side,          // [rows, taps, groups]
	const half* __restrict__ proj,    // [rows, 2*taps*groups]
	int rows, int taps, int groups, int side_idx
) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int total = rows * taps * groups;
	if ( idx >= total ) return;
	// Layout, from the reference: kernel_projection output reshapes to (rows, 2, taps, groups), and
	// _grouped_conv does delta.unsqueeze(-1) against base.view(1, taps, groups, group_size) -- so the
	// coefficient tensor is [rows, taps, groups] with groups innermost, and the source index inside a
	// row is side*taps*groups + tap*groups + group.
	//
	// Decomposing idx with taps innermost instead produced [rows, groups, taps], transposing every tap
	// against every group in the sandwich. Both indices are below; only the decomposition differs.
	const int g = idx % groups;
	const int t = ( idx / groups ) % taps;
	const int row = idx / ( taps * groups );
	side[idx] = __half2float( proj[static_cast<int64_t>( row ) * 2 * taps * groups + side_idx * taps * groups + t * groups + g] );
}

// Scatter a [n, heads*head_dim] projection into one layer of the ring-shaped context cache.
__global__ void dflash_cache_store_kernel(
	half* __restrict__ cache,
	const half* __restrict__ src,
	int n,
	int width,
	int slot0,
	int ring_slots,
	int layer,
	int stride_slot,
	int stride_layer
) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int total = n * width;
	if ( idx >= total ) return;
	const int row = idx / width;
	const int col = idx % width;
	int slot = slot0 + row;
	if ( slot >= ring_slots ) slot -= ring_slots;
	cache[static_cast<int64_t>( slot ) * stride_slot + static_cast<int64_t>( layer ) * stride_layer + col] = src[idx];
}

} // namespace

void dflash_conv_side_split(
	float* side,
	const half* proj,
	int rows,
	int taps,
	int groups,
	int side_idx,
	cudaStream_t stream
) {
	const int total = rows * taps * groups;
	const int threads = 256;
	dflash_conv_side_split_kernel<<<( total + threads - 1 ) / threads, threads, 0, stream >>>(
		side, proj, rows, taps, groups, side_idx );
}

void dflash_cache_store(
	half* cache,
	const half* src,
	int n,
	int heads,
	int head_dim,
	int slot0,
	int ring_slots,
	int layer,
	int stride_slot,
	int stride_layer,
	cudaStream_t stream
) {
	const int width = heads * head_dim;
	const int total = n * width;
	const int threads = 256;
	dflash_cache_store_kernel<<<( total + threads - 1 ) / threads, threads, 0, stream >>>(
		cache, src, n, width, slot0, ring_slots, layer, stride_slot, stride_layer );
}

void dflash_qk_norm_rope(
	half* x,
	const half* weight,
	const float* rope_cos,
	const float* rope_sin,
	int rows,
	int heads,
	int head_dim,
	int pairs,
	int pos0,
	float eps,
	float constant_bias,
	cudaStream_t stream
) {
	if ( rows <= 0 || heads <= 0 ) return;
	dim3 grid( heads, rows );
	dflash_qk_norm_rope_kernel<<<grid, 256, 0, stream >>>(
		x, weight, rope_cos, rope_sin, rows, heads, head_dim, pairs, pos0, eps, constant_bias );
}

void dflash_attn(
	const half* q, const half* kcache, const half* vcache, half* out,
	int rows, int q_heads, int kv_heads, int head_dim,
	int nctx, int first_ctx, int ring_slots, int ctx_layer,
	int stride_slot, int stride_layer, int stride_head,
	int own_slot,
	int own_first_row,
	cudaStream_t stream
) {
	if ( rows <= 0 || q_heads <= 0 ) return;
	const int nkeys = nctx + MAX( rows - own_first_row, 0 );
	const float scale = rsqrtf( static_cast<float>( head_dim ) );
	dim3 grid( q_heads, rows );
	const size_t smem = static_cast<size_t>( nkeys + ( 128 >> 5 ) * head_dim ) * sizeof( float );
	dflash_attn_kernel<<<grid, 128, smem, stream >>>(
		q, kcache, vcache, out, rows, q_heads, kv_heads, head_dim,
		nctx, first_ctx, ring_slots, ctx_layer, stride_slot, stride_layer, stride_head,
		own_slot, own_first_row, scale );
}

void dflash_f16_gemv_nt(
	const half* w, const half* x, half* out, int m, int n, int k, cudaStream_t stream
) {
	ENGINE_ASSERT( m >= 1 && n >= 1 && k >= 1, "dflash_f16_gemv_nt: degenerate shape" );
	const int threads = 256;
	const int warps = threads >> 5;
	if ( m <= 16 && k % 8 == 0 ) {
		const int blocks = ( n + warps - 1 ) / warps;
		if ( m <= 4 ) dflash_f16_gemv_skinny_kernel<4><<<blocks, threads, 0, stream>>>( w, x, out, n, k, m );
		else if ( m <= 8 ) dflash_f16_gemv_skinny_kernel<8><<<blocks, threads, 0, stream>>>( w, x, out, n, k, m );
		else dflash_f16_gemv_skinny_kernel<16><<<blocks, threads, 0, stream>>>( w, x, out, n, k, m );
		return;
	}
	const int64_t warps_needed = static_cast<int64_t>( m ) * n;
	dflash_f16_gemv_nt_kernel<<<static_cast<int>( ( warps_needed + warps - 1 ) / warps ), threads, 0, stream >>>(
		w, x, out, n, k, m );
}

void dflash_rmsnorm_residual(
	const pod::TensorView& x,
	const pod::TensorView& residual_in,
	const pod::TensorView& weight,
	pod::TensorView& normed_out,
	pod::TensorView& residual_out,
	int rows,
	float eps,
	float constant_bias,
	cudaStream_t stream
) {
	const int hidden = x.shape[1];
	// The sublayer output arrives as FP32 whenever it can exceed FP16's 65504 ceiling. The drafter's
	// post-attention conv multiplies o_proj, which already peaks past 15000, so a single element
	// overflowed to inf and the next dot product turned that into NaN across a whole row.
	if ( x.dtype == pod::DType::FP32 ) {
		dflash_rmsnorm_residual_kernel<float><<<rows, 256, 0, stream >>>(
			x.ptr<const float>(),
			( residual_in.data != nullptr ) ? residual_in.ptr<const float>() : nullptr,
			( weight.data != nullptr ) ? weight.ptr<const half>() : nullptr,
			normed_out.ptr<half>(),
			residual_out.ptr<float>(),
			hidden, eps, constant_bias );
		return;
	}
	dflash_rmsnorm_residual_kernel<half><<<rows, 256, 0, stream >>>(
		x.ptr<const half>(),
		( residual_in.data != nullptr ) ? residual_in.ptr<const float>() : nullptr,
		( weight.data != nullptr ) ? weight.ptr<const half>() : nullptr,
		normed_out.ptr<half>(),
		residual_out.ptr<float>(),
		hidden, eps, constant_bias );
}


// Widen an FP16 sublayer output before it can overflow. The drafter's residual is FP32, but the
// tensors that feed it were FP16, so a value past 65504 became inf before the norm ever saw it.
__global__ void dflash_h2f_kernel( float* __restrict__ out, const half* __restrict__ in, int n ) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < n ) out[i] = __half2float( in[i] );
}

void dflash_h2f( const half* in, float* out, int n, cudaStream_t stream ) {
	if ( n <= 0 ) return;
	const int threads = 256;
	dflash_h2f_kernel<<<( n + threads - 1 ) / threads, threads, 0, stream >>>( out, in, n );
}

// Copies a [rows, src_cols] slice into columns [col0, col0 + src_cols) of a [rows, dst_cols] matrix.
__global__ void dflash_place_cols_kernel( half* __restrict__ dst, const half* __restrict__ src,
	int rows, int dst_cols, int src_cols, int col0 ) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= rows * src_cols ) return;
	const int r = i / src_cols;
	const int c = i - r * src_cols;
	dst[static_cast<size_t>( r ) * dst_cols + col0 + c] = src[i];
}

void dflash_place_cols( half* dst, const half* src, int rows, int dst_cols, int src_cols, int col0, cudaStream_t stream ) {
	const int n = rows * src_cols;
	if ( n <= 0 ) return;
	const int threads = 256;
	dflash_place_cols_kernel<<<( n + threads - 1 ) / threads, threads, 0, stream >>>( dst, src, rows, dst_cols, src_cols, col0 );
}

void dflash2_dynconv(
	const pod::TensorView& x,
	const pod::TensorView& dyn,
	const pod::TensorView& base,
	pod::TensorView& out,
	int group_size,
	bool accumulate,
	cudaStream_t stream
) {
	int bsz = ( x.ndim == 3 ) ? x.shape[0] : 1;
	int seqlen = ( x.ndim == 3 ) ? x.shape[1] : x.shape[0];
	int hidden = x.shape[x.ndim - 1];
	int taps = ( base.ndim >= 2 ) ? base.shape[0] : 2;

	int groups = hidden / group_size;
	int64_t ds_b = static_cast<int64_t>( seqlen ) * taps * groups;
	int64_t ds_t = taps * groups;
	int64_t ds_k = groups;
	int64_t ds_g = 1;

	dim3 grid( ( hidden + DYNCONV_THREADS - 1 ) / DYNCONV_THREADS, seqlen, bsz );

	// The dynamic coefficients are FP32 in the engine (bb_coef_*_ are FP32, and dflash_conv_side_split
	// writes floats). Dispatching on x/out dtype alone picked the <half, half, half> instantiation and
	// called dyn.ptr<const half>(), reinterpreting FP32 coefficients as FP16 garbage.
	const bool x_f32 = ( x.dtype == pod::DType::FP32 );
	const bool o_f32 = ( out.dtype == pod::DType::FP32 );
	ENGINE_ASSERT( !( x_f32 ^ o_f32 ), "dflash2_dynconv: x and out must share a dtype" );
	ENGINE_ASSERT( base.dtype == pod::DType::FP16, "dflash2_dynconv: base kernel must be FP16" );
	if ( x_f32 ) {
		if ( accumulate ) {
			dflash2_dynconv_kernel<float, float, half, float, true><<<grid, DYNCONV_THREADS, 0, stream>>>(
				x.ptr<const float>(), dyn.ptr<const float>(), base.ptr<const half>(),
				out.ptr<float>(), seqlen, hidden, group_size, taps,
				ds_b, ds_t, ds_k, ds_g
			);
		}
		else {
			dflash2_dynconv_kernel<float, float, half, float, false><<<grid, DYNCONV_THREADS, 0, stream>>>(
				x.ptr<const float>(), dyn.ptr<const float>(), base.ptr<const half>(),
				out.ptr<float>(), seqlen, hidden, group_size, taps,
				ds_b, ds_t, ds_k, ds_g
			);
		}
	}
	else {
		if ( accumulate ) {
			dflash2_dynconv_kernel<half, float, half, half, true><<<grid, DYNCONV_THREADS, 0, stream>>>(
				x.ptr<const half>(), dyn.ptr<const float>(), base.ptr<const half>(),
				out.ptr<half>(), seqlen, hidden, group_size, taps,
				ds_b, ds_t, ds_k, ds_g
			);
		}
		else {
			dflash2_dynconv_kernel<half, float, half, half, false><<<grid, DYNCONV_THREADS, 0, stream>>>(
				x.ptr<const half>(), dyn.ptr<const float>(), base.ptr<const half>(),
				out.ptr<half>(), seqlen, hidden, group_size, taps,
				ds_b, ds_t, ds_k, ds_g
			);
		}
	}
}

void dflash2_selector_walk(
	const float* unary,
	const int* cands,
	const half* gate,
	const half* pred_cb,
	const half* succ_cb,
	const int* anchor,
	int* out_tokens,
	float* conf,
	int bsz,
	int rows,
	int k,
	int rank,
	cudaStream_t stream
) {
	if ( bsz <= 0 || rows <= 0 ) return;
	size_t smem = ( rank + k ) * sizeof( float );
	dim3 grid( bsz, 1 );
	dflash2_selector_walk_kernel<<<grid, WALK_THREADS, smem, stream>>>(
		unary, cands, gate, pred_cb, succ_cb, anchor, out_tokens, conf, rows, k, rank
	);
}

void dflash2_topk(
	const half* logits,
	int bsz,
	int rows,
	int vocab,
	int k,
	float* values,
	int* indices,
	float scale,
	float softcap,
	cudaStream_t stream
) {
	if ( bsz <= 0 || rows <= 0 ) return;
	int E = 16;
	// Each block covers TOPK_THREADS * E columns: LOADS = E/VEC loads of SPAN = TOPK_THREADS*VEC.
	int splits = ( vocab + TOPK_THREADS * E - 1 ) / ( TOPK_THREADS * E );
	int num_rows = bsz * rows;
	int64_t stride_b = static_cast<int64_t>( rows ) * vocab;
	int64_t stride_r = vocab;

	// In real execution, persistent scratch buffers from DFlashSpeculator are used.
	// For standalone invocation, we allocate static block scratch.
	static float* s_part_v = nullptr;
	static int* s_part_ix = nullptr;
	static int* s_counters = nullptr;
	static int s_alloc_capacity = 0;

	int required_capacity = num_rows * splits * k;
	if ( required_capacity > s_alloc_capacity ) {
		if ( s_part_v ) (void) cudaFree( s_part_v );
		if ( s_part_ix ) (void) cudaFree( s_part_ix );
		if ( s_counters ) (void) cudaFree( s_counters );
		(void) cudaMalloc( &s_part_v, required_capacity * sizeof( float ) );
		(void) cudaMalloc( &s_part_ix, required_capacity * sizeof( int ) );
		(void) cudaMalloc( &s_counters, num_rows * sizeof( int ) );
		(void) cudaMemset( s_counters, 0, num_rows * sizeof( int ) );
		s_alloc_capacity = required_capacity;
	}

	CUDA_CHECK( cudaMemsetAsync( s_counters, 0, num_rows * sizeof( int ), stream ) );

	dflash2_topk_kernel<half, 16, 16><<<num_rows * splits, TOPK_THREADS, 0, stream>>>(
		logits, stride_b, stride_r, rows, splits, vocab, scale, softcap,
		s_part_v, s_part_ix, s_counters, values, indices, 0
	);
}

void dflash2_topk_shard(
	const half* logits,
	int rows,
	int row_stride,
	int columns,
	int k,
	float* values,
	int* indices,
	int index_offset,
	float* part_v,
	int* part_ix,
	int* counters,
	cudaStream_t stream
) {
if ( rows <= 0 || columns <= 0 ) return;
	// Each block covers TOPK_THREADS * 16 columns; see dflash2_topk.
	const int splits = ( columns + TOPK_THREADS * 16 - 1 ) / ( TOPK_THREADS * 16 );
	// The kernel is two-phase with a completion counter per row; the finishing block resets it, so
	// zeroing once here is enough and keeps the call free of host-side state between replays.
	CUDA_CHECK( cudaMemsetAsync( counters, 0, rows * sizeof( int ), stream ) );

	dflash2_topk_kernel<half, 16, 16><<<rows * splits, TOPK_THREADS, 0, stream>>>(
		logits, static_cast<int64_t>( rows ) * row_stride, row_stride, rows, splits, columns,
		1.0f, 0.0f, part_v, part_ix, counters, values, indices, index_offset
	);
	(void) k;
}

// Merge the per-rank local top-k into one global top-k, per row.
//
// The local top-k of a row is a superset of the global top-k of the same row restricted to that
// shard, so merging tp_size * k candidates is exact, not a heuristic: no token outside the union
// can be in the global top-k. This is what removes gather_logits -- only k values and k indices per
// rank cross the interconnect instead of the whole vocabulary.
//
// Tie-break: higher value wins, then lower global token id. Deterministic and identical on every
// rank, which matters because the lattice walk and the rejection sampler must see one ordering.
template <int K_T>
__global__ void dflash2_topk_merge_kernel(
	const float* __restrict__ part_vals,   // [tp_size, rows, k]
	const int* __restrict__ part_ix,      // [tp_size, rows, k]
	const int tp_size,
	const int rows,
	const int k,
	float* __restrict__ out_vals,         // [rows, k]
	int* __restrict__ out_ix,            // [rows, k]
	const int part_rows
) {
	const int row = blockIdx.x;
	const int t = threadIdx.x;
	if ( row >= rows ) return;
	// dflash2_topk_shard always emits 16 partials per row whatever k was requested, so the rank stride
	// is rows * 16 and is independent of K_T.
	const int pstride = rows * 16;
	const float* pv = part_vals + static_cast<int64_t>( row ) * 16;
	const int* pix = part_ix + static_cast<int64_t>( row ) * 16;

	// One warp per row. The candidate set is tp_size * k entries, so lane l of the warp holds
	// MERGE_PER_LANE of them and warp_topk reduces the whole row in k rounds.
	constexpr int MERGE_PER_LANE = 4;   // 32 lanes * 4 = 128 = tp_size(4) * k(32) worst case
	if ( t < 32 ) {
		float x[MERGE_PER_LANE];
		int id[MERGE_PER_LANE];
		#pragma unroll
		for ( int i = 0; i < MERGE_PER_LANE; ++i ) {
			const int j = i * 32 + t;   // j enumerates the tp_size * k candidates
			const int d = j / k;
			const int pos = j % k;
			// rank d contributes its own top-k, so slot pos is live for pos < k.
			const bool live = ( d < tp_size ) && ( pos < k );
			x[i] = live ? pv[static_cast<int64_t>( d ) * pstride + pos] : -INFINITY;
			id[i] = live ? pix[static_cast<int64_t>( d ) * pstride + pos] : INT_MAX;
		}
		warp_topk<MERGE_PER_LANE, K_T>( x, id, out_vals + row * K_T, out_ix + row * K_T );
	}
}

void dflash2_topk_merge(
	const float* part_vals,
	const int* part_ix,
	int tp_size,
	int rows,
	int k,
	float* out_vals,
	int* out_ix,
	int part_rows,
	cudaStream_t stream
) {
	if ( rows <= 0 || k <= 0 ) return;
	ENGINE_ASSERT( tp_size * k <= 32 * 4, "dflash2_topk_merge: tp_size * k exceeds one warp's candidate capacity" );
	// The reduction count is a template parameter, so dispatch on the k values the engine uses.
	// k == 16 is the drafter's selector_top_k; the rest are exercised by the parity suite.
	switch ( k ) {
		case 1: dflash2_topk_merge_kernel<1><<<rows, 32, 0, stream>>>( part_vals, part_ix, tp_size, rows, k, out_vals, out_ix, part_rows ); break;
		case 4: dflash2_topk_merge_kernel<4><<<rows, 32, 0, stream>>>( part_vals, part_ix, tp_size, rows, k, out_vals, out_ix, part_rows ); break;
		case 8: dflash2_topk_merge_kernel<8><<<rows, 32, 0, stream>>>( part_vals, part_ix, tp_size, rows, k, out_vals, out_ix, part_rows ); break;
		case 16: dflash2_topk_merge_kernel<16><<<rows, 32, 0, stream>>>( part_vals, part_ix, tp_size, rows, k, out_vals, out_ix, part_rows ); break;
		case 32: dflash2_topk_merge_kernel<32><<<rows, 32, 0, stream>>>( part_vals, part_ix, tp_size, rows, k, out_vals, out_ix, part_rows ); break;
		default: ENGINE_ASSERT( false, "dflash2_topk_merge: unsupported k" );
	}
}

// NeoX RoPE tables for the drafter's own query positions. rope_dim comes from the config's dimension
// sections (Qwen3.8-27B drafter: [64,0,0,0] -> 64 rotary dims = 32 pairs out of head_dim 128).
__global__ void build_rope_kernel( float* cos_table, float* sin_table, int len, int pairs, float theta ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int total = len * pairs;
	if ( idx >= total ) return;
	const int pos = idx / pairs;
	const int i = idx % pairs;
	const float inv_freq = powf( theta, -2.0f * (float) i / (float) ( 2 * pairs ) );
	const float angle = (float) pos * inv_freq;
	cos_table[idx] = cosf( angle );
	sin_table[idx] = sinf( angle );
}

void build_rope_table(
	float* cos_table,
	float* sin_table,
	int len,
	int head_dim,
	float theta,
	const int dim_sections[4],
	cudaStream_t stream
) {
	// Sum the non-zero sections; a zero section means "not rotary" (pass-through dims).
	int rope_dim = 0;
	for ( int i = 0; i < 4; ++i ) rope_dim += dim_sections[i];
	if ( rope_dim <= 0 ) rope_dim = head_dim;
	rope_dim = MIN( rope_dim, head_dim );
	const int pairs = rope_dim / 2;
	const int total = len * pairs;
	const int threads = 256;
	build_rope_kernel<<<( total + threads - 1 ) / threads, threads, 0, stream >>>(
		cos_table, sin_table, len, pairs, theta );
}

} // namespace engine::spec

void engine::modules::LinearPlainGEMV::forward(
	const half* w,
	const half* x,
	half* out,
	int m,
	int n,
	int k,
	cudaStream_t stream
) {
	if ( m >= 128 ) {
		// Many rows (an unquantized vision tower, prefill chunks): the tensor-core HGEMM; the skinny GEMV reads the whole weight once per few rows
		int device = 0;
		(void) cudaGetDevice( &device );
		cublasHandle_t h = engine::modules::trellis_get_cublas( device );
		cublasSetStream( h, stream );
		const float one = 1.0f, zero = 0.0f;
		const cublasStatus_t st = cublasGemmEx( h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one, w, CUDA_R_16F, k, x, CUDA_R_16F, k, &zero, out, CUDA_R_16F, n, CUBLAS_COMPUTE_32F,
			CUBLAS_GEMM_DEFAULT_TENSOR_OP );
		ENGINE_ASSERT( st == CUBLAS_STATUS_SUCCESS, "LinearPlainGEMV: cublasGemmEx failed" );
		return;
	}
	engine::spec::dflash_f16_gemv_nt( w, x, out, m, n, k, stream );
}
