#include "engine/modules/attention.h"
#include <cuda_fp16.h>
#include <cmath>

namespace engine::modules {

namespace {

__global__ void unpack_q_gate_kernel(
	const half* __restrict__ q_gate,
	half* __restrict__ q,
	half* __restrict__ gate,
	int num_q_heads,
	int head_dim
) {
	int head = blockIdx.x;
	int tid = threadIdx.x;
	if ( head >= num_q_heads || tid >= head_dim ) return;

	int src_offset = head * 2 * head_dim;
	int dst_offset = head * head_dim + tid;

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
	int max_seq
) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_elems = num_kv_heads * head_dim;
	if ( tid < total_elems ) {
		size_t offset = ( static_cast<size_t>( layer_idx ) * max_seq + pos ) * total_elems + tid;
		k_cache[offset] = k[tid];
		v_cache[offset] = v[tid];
	}
}

__global__ void full_attn_decode_kernel(
	const half* __restrict__ q,
	const half* __restrict__ k_cache,
	const half* __restrict__ v_cache,
	const half* __restrict__ gate,
	half* __restrict__ out,
	int layer_idx,
	int pos,
	int num_q_heads,
	int num_kv_heads,
	int head_dim,
	int max_seq
) {
	int head = blockIdx.x;
	int tid = threadIdx.x;
	if ( head >= num_q_heads || tid >= head_dim ) return;

	int heads_per_kv = num_q_heads / num_kv_heads;
	if ( heads_per_kv < 1 ) heads_per_kv = 1;
	int kv_head = head / heads_per_kv;

	const half* q_head = q + head * head_dim;
	half* out_head = out + head * head_dim;

	float my_q = __half2float( q_head[tid] );
	float scale = 1.0f / sqrtf( static_cast<float>( head_dim ) );

	int num_tokens = pos + 1;
	if ( num_tokens > max_seq ) num_tokens = max_seq;

	extern __shared__ float s_scores[];
	__shared__ float s_warp_sums[8];
	__shared__ float s_inv_sum;

	int warp_id = tid >> 5;
	int lane_id = tid & 31;
	int num_warps = ( head_dim + 31 ) >> 5;

	// 1. Compute dot product Q . K for all past tokens
	for ( int t = 0; t < num_tokens; ++t ) {
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
				s_scores[t] = b_sum * scale;
			}
		}
		__syncthreads();
	}

	// 2. Softmax: max, exp, sum, normalize
	if ( tid == 0 ) {
		float max_val = -1e30f;
		for ( int t = 0; t < num_tokens; ++t ) {
			if ( s_scores[t] > max_val ) {
				max_val = s_scores[t];
			}
		}

		float sum_exp = 0.0f;
		for ( int t = 0; t < num_tokens; ++t ) {
			float val = expf( s_scores[t] - max_val );
			s_scores[t] = val;
			sum_exp += val;
		}
		s_inv_sum = 1.0f / ( sum_exp + 1e-9f );
	}
	__syncthreads();

	for ( int t = tid; t < num_tokens; t += blockDim.x ) {
		s_scores[t] *= s_inv_sum;
	}
	__syncthreads();

	// 3. Accumulate V weighted by softmax scores
	float o_dim = 0.0f;
	for ( int t = 0; t < num_tokens; ++t ) {
		size_t kv_offset = ( static_cast<size_t>( layer_idx ) * max_seq + t ) * ( num_kv_heads * head_dim ) + kv_head * head_dim + tid;
		float my_v = __half2float( v_cache[kv_offset] );
		o_dim = fmaf( s_scores[t], my_v, o_dim );
	}

	// 4. Output gate: sigmoid(gate)
	float g_val = ( gate != nullptr ) ? __half2float( gate[head * head_dim + tid] ) : 0.0f;
	float sig_g = 1.0f / ( 1.0f + __expf( -g_val ) );
	out_head[tid] = __float2half_rn( o_dim * sig_g );
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
	unpack_q_gate_kernel<<<dim3( num_q_heads ), dim3( head_dim ), 0, stream>>>(
		q_gate.ptr<const half>(),
		q.ptr<half>(),
		gate.ptr<half>(),
		num_q_heads,
		head_dim
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
	int total_elems = num_kv_heads * head_dim;
	int threads = 256;
	int blocks = ( total_elems + threads - 1 ) / threads;
	store_kv_cache_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
		k_cache,
		v_cache,
		k.ptr<const half>(),
		v.ptr<const half>(),
		layer_idx,
		pos,
		num_kv_heads,
		head_dim,
		max_seq
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
	size_t smem_bytes = static_cast<size_t>( max_seq ) * sizeof( float );
	full_attn_decode_kernel<<<dim3( num_q_heads ), dim3( head_dim ), smem_bytes, stream>>>(
		q.ptr<const half>(),
		k_cache,
		v_cache,
		gate.ptr<const half>(),
		out.ptr<half>(),
		layer_idx,
		pos,
		num_q_heads,
		num_kv_heads,
		head_dim,
		max_seq
	);
}

} // namespace engine::modules
