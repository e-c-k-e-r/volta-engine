#include "engine/modules/attention.h"
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>

namespace engine::modules {

namespace {

#define ATTN_PAGE_SIZE 256
#define ATTN_MAX_CHUNKS 16

__device__ float g_attn_workspace[8 * 16 * ATTN_MAX_CHUNKS * ( 128 + 2 )];

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

__global__ void store_kv_cache_kernel(
	half* __restrict__ k_cache,
	half* __restrict__ v_cache,
	const half* __restrict__ k,
	const half* __restrict__ v,
	int layer_idx,
	int pos,
	int num_kv_heads,
	int head_dim,
	int max_seq,
	int M
) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_elems = num_kv_heads * head_dim;
	if ( tid < total_elems * M ) {
		int m = tid / total_elems;
		int elem = tid % total_elems;
		size_t offset = ( static_cast<size_t>( layer_idx ) * max_seq + ( pos + m ) ) * total_elems + elem;
		k_cache[offset] = k[tid];
		v_cache[offset] = v[tid];
	}
}

__global__ void attn_chunked_paged_kernel(
	const half* __restrict__ q,
	const half* __restrict__ k_cache,
	const half* __restrict__ v_cache,
	const half* __restrict__ gate,
	half* __restrict__ out,
	float* __restrict__ workspace,
	int layer_idx,
	int pos,
	int num_q_heads,
	int num_kv_heads,
	int head_dim,
	int max_seq,
	int n_chunks,
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

	int cur_pos = pos + m;
	int total_tokens = min( cur_pos + 1, max_seq );

	int kv_start = chunk * ATTN_PAGE_SIZE;
	int kv_end = min( kv_start + ATTN_PAGE_SIZE, total_tokens );

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
			size_t kv_offset = ( static_cast<size_t>( layer_idx ) * max_seq + t ) * ( num_kv_heads * head_dim ) + kv_head * head_dim + tid;
			float my_k = __half2float( k_cache[kv_offset] );
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

			float my_v = __half2float( v_cache[kv_offset] );
			float score = s_score;
			if ( t == kv_start ) {
				m_reg = score;
				l_reg = 1.0f;
				o_dim = my_v;
			}
			else {
				float m_new = fmaxf( m_reg, score );
				float alpha = expf( m_reg - m_new );
				float exp_s = expf( score - m_new );
				o_dim = alpha * o_dim + exp_s * my_v;
				l_reg = alpha * l_reg + exp_s;
				m_reg = m_new;
			}
		}
	}

	if ( n_chunks == 1 ) {
		float inv_l = ( l_reg > 0.0f ) ? ( 1.0f / l_reg ) : 0.0f;
		float sig = 1.0f;
		if ( gate != nullptr ) {
			size_t g_off = ( static_cast<size_t>( m ) * num_q_heads + head ) * head_dim + tid;
			float g_val = __half2float( gate[g_off] );
			sig = 1.0f / ( 1.0f + __expf( -g_val ) );
		}
		out_head[tid] = __float2half_rn( o_dim * inv_l * sig );
	}
	else {
		float* ws = workspace + ( ( static_cast<size_t>( m ) * num_q_heads + head ) * n_chunks + chunk ) * ( head_dim + 2 );
		if ( tid == 0 ) {
			ws[0] = m_reg;
			ws[1] = l_reg;
		}
		ws[2 + tid] = o_dim;
	}
}

template <int D>
__global__ void attn_reduce_kernel(
	const float* __restrict__ workspace,
	const half* __restrict__ gate,
	half* __restrict__ output,
	int num_q_heads,
	int n_chunks
) {
	const int token_idx = blockIdx.y;
	const int q_head = blockIdx.x;
	const int tid = threadIdx.x;
	if ( q_head >= num_q_heads || tid >= D ) return;

	float m_acc = -1e30f;
	float l_acc = 0.0f;
	float o_acc = 0.0f;

	const float* ws_base = workspace + ( static_cast<size_t>( token_idx ) * num_q_heads + q_head ) * n_chunks * ( D + 2 );

	for ( int c = 0; c < n_chunks; ++c ) {
		const float* ws_c = ws_base + static_cast<size_t>( c ) * ( D + 2 );
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
	float sig = 1.0f;
	if ( gate != nullptr ) {
		size_t g_off = ( static_cast<size_t>( token_idx ) * num_q_heads + q_head ) * D + tid;
		float g_val = __half2float( gate[g_off] );
		sig = 1.0f / ( 1.0f + __expf( -g_val ) );
	}
	size_t out_off = ( static_cast<size_t>( token_idx ) * num_q_heads + q_head ) * D + tid;
	output[out_off] = __float2half_rn( o_acc * inv_l * sig );
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
	unpack_q_gate_kernel<<<dim3( num_q_heads, M ), dim3( head_dim ), 0, stream>>>(
		q_gate.ptr<const half>(),
		q.ptr<half>(),
		gate.ptr<half>(),
		num_q_heads,
		head_dim,
		M
	);
}

void Attention::store_kv_cache(
	half* k_cache,
	half* v_cache,
	const pod::TensorView& k,
	const pod::TensorView& v,
	int layer_idx,
	int pos,
	int num_kv_heads,
	int head_dim,
	int max_seq,
	cudaStream_t stream
) {
	int M = ( k.ndim >= 2 && k.shape[0] > 0 ) ? k.shape[0] : 1;
	int total_elems = num_kv_heads * head_dim;
	int total_work = total_elems * M;
	int threads = 256;
	int blocks = ( total_work + threads - 1 ) / threads;
	store_kv_cache_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
		k_cache,
		v_cache,
		k.ptr<const half>(),
		v.ptr<const half>(),
		layer_idx,
		pos,
		num_kv_heads,
		head_dim,
		max_seq,
		M
	);
}

void Attention::forward_decode(
	const pod::TensorView& q,
	const half* k_cache,
	const half* v_cache,
	const pod::TensorView& gate,
	pod::TensorView& out,
	int layer_idx,
	int pos,
	int num_q_heads,
	int num_kv_heads,
	int head_dim,
	int max_seq,
	cudaStream_t stream
) {
	int M = ( q.ndim >= 2 && q.shape[0] > 0 ) ? q.shape[0] : 1;
	float scale = 1.0f / sqrtf( static_cast<float>( head_dim ) );

	int cur_max_tokens = std::min( pos + M, max_seq );
	int n_chunks = ( cur_max_tokens + ATTN_PAGE_SIZE - 1 ) / ATTN_PAGE_SIZE;
	if ( n_chunks < 1 ) n_chunks = 1;
	if ( n_chunks > ATTN_MAX_CHUNKS ) n_chunks = ATTN_MAX_CHUNKS;

	dim3 grid( num_q_heads, M, n_chunks );
	dim3 block( head_dim );
	size_t smem = 0;

	attn_chunked_paged_kernel<<<grid, block, smem, stream>>>(
		q.ptr<const half>(),
		k_cache,
		v_cache,
		gate.ptr<const half>(),
		out.ptr<half>(),
		g_attn_workspace,
		layer_idx,
		pos,
		num_q_heads,
		num_kv_heads,
		head_dim,
		max_seq,
		n_chunks,
		scale
	);

	if ( n_chunks > 1 ) {
		dim3 red_grid( num_q_heads, M );
		dim3 red_block( head_dim );
		attn_reduce_kernel<128><<<red_grid, red_block, 0, stream>>>(
			g_attn_workspace,
			gate.ptr<const half>(),
			out.ptr<half>(),
			num_q_heads,
			n_chunks
		);
	}
}

} // namespace engine::modules
