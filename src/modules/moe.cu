#include "engine/modules/moe.h"
#include <cuda_fp16.h>
#include <cmath>

namespace engine::modules {

namespace {

__global__ __launch_bounds__( 288 )
void router_topk_kernel(
	const half* __restrict__ x,
	const half* __restrict__ router_weights,
	const half* __restrict__ e_score_bias,
	int* __restrict__ topk_indices,
	half* __restrict__ topk_weights,
	int hidden_size,
	int num_experts,
	int top_k,
	float routed_scaling_factor
) {
	int expert = threadIdx.x;
	if ( expert >= num_experts ) return;

	float logit = 0.0f;
	#pragma unroll 4
	for ( int d = 0; d < hidden_size; ++d ) {
		float x_val = __half2float( x[d] );
		float w_val = __half2float( router_weights[d * num_experts + expert] );
		logit = fmaf( x_val, w_val, logit );
	}

	float raw_score = 1.0f / ( 1.0f + __expf( -logit ) );
	float select_score = raw_score;
	if ( e_score_bias != nullptr ) {
		select_score += __half2float( e_score_bias[expert] );
	}

	__shared__ float s_raw[288];
	__shared__ float s_select[288];
	__shared__ int s_indices[288];
	s_raw[expert] = raw_score;
	s_select[expert] = select_score;
	s_indices[expert] = expert;
	__syncthreads();

	if ( threadIdx.x == 0 ) {
		#pragma unroll
		for ( int i = 0; i < top_k; ++i ) {
			int max_idx = i;
			float max_val = s_select[i];
			for ( int j = i + 1; j < num_experts; ++j ) {
				if ( s_select[j] > max_val ) {
					max_val = s_select[j];
					max_idx = j;
				}
			}
			float tmp_s = s_select[i];
			s_select[i] = s_select[max_idx];
			s_select[max_idx] = tmp_s;

			int tmp_i = s_indices[i];
			s_indices[i] = s_indices[max_idx];
			s_indices[max_idx] = tmp_i;
		}

		float sum_scores = 0.0f;
		#pragma unroll
		for ( int i = 0; i < top_k; ++i ) {
			sum_scores += s_raw[s_indices[i]];
		}

		float inv_sum = routed_scaling_factor / ( sum_scores + 1e-9f );
		#pragma unroll
		for ( int i = 0; i < top_k; ++i ) {
			topk_indices[i] = s_indices[i];
			topk_weights[i] = __float2half_rn( s_raw[s_indices[i]] * inv_sum );
		}
	}
}

__global__ void moe_combine_kernel(
	const half* __restrict__ expert_parts,
	const half* __restrict__ topk_weights,
	const half* __restrict__ shared_out,
	half* __restrict__ output,
	int hidden_size,
	int top_k
) {
	int d = blockIdx.x * blockDim.x + threadIdx.x;
	if ( d >= hidden_size ) return;

	float sum = ( shared_out != nullptr ) ? __half2float( shared_out[d] ) : 0.0f;
	#pragma unroll
	for ( int k = 0; k < top_k; ++k ) {
		float w = __half2float( topk_weights[k] );
		float part = __half2float( expert_parts[k * hidden_size + d] );
		sum = fmaf( part, w, sum );
	}
	output[d] = __float2half_rn( sum );
}

__global__ void moe_accumulate_kernel(
	half* __restrict__ accum_out,
	const half* __restrict__ expert_out,
	half weight,
	int hidden_size
) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ( tid < hidden_size ) {
		float cur = __half2float( accum_out[tid] );
		float val = __half2float( expert_out[tid] );
		float w = __half2float( weight );
		accum_out[tid] = __float2half_rn( fmaf( val, w, cur ) );
	}
}

} // namespace

void MoERouter::route_topk(
	const pod::TensorView& x,
	const pod::TensorView& router_weights,
	const pod::TensorView& e_score_bias,
	int* d_topk_indices,
	half* d_topk_weights,
	int hidden_size,
	int num_experts,
	int top_k,
	float routed_scaling_factor,
	cudaStream_t stream
) {
	router_topk_kernel<<<dim3( 1 ), dim3( num_experts ), 0, stream>>>(
		x.ptr<const half>(),
		router_weights.ptr<const half>(),
		e_score_bias.ptr<const half>(),
		d_topk_indices,
		d_topk_weights,
		hidden_size,
		num_experts,
		top_k,
		routed_scaling_factor
	);
}

void MoERouter::combine(
	const half* expert_parts,
	const half* topk_weights,
	const half* shared_out,
	half* output,
	int hidden_size,
	int top_k,
	cudaStream_t stream
) {
	int threads = 256;
	int blocks = ( hidden_size + threads - 1 ) / threads;
	moe_combine_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
		expert_parts,
		topk_weights,
		shared_out,
		output,
		hidden_size,
		top_k
	);
}

void MoERouter::accumulate(
	half* accum_out,
	const half* expert_out,
	half weight,
	int hidden_size,
	cudaStream_t stream
) {
	int threads = 256;
	int blocks = ( hidden_size + threads - 1 ) / threads;
	moe_accumulate_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
		accum_out, expert_out, weight, hidden_size
	);
}

void MoERouter::accumulate(
	pod::TensorView& accum_out,
	const pod::TensorView& expert_out,
	half weight,
	int hidden_size,
	cudaStream_t stream
) {
	accumulate( accum_out.ptr<half>(), expert_out.ptr<const half>(), weight, hidden_size, stream );
}

} // namespace engine::modules
