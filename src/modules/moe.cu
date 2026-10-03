#include "engine/modules/moe.h"
#include <cuda_fp16.h>
#include <cmath>

namespace engine::modules {

namespace {

__global__ __launch_bounds__( 288 )
void router_topk_kernel(
	const half* __restrict__ x,
	const half* __restrict__ router_weights,
	int* __restrict__ topk_indices,
	half* __restrict__ topk_weights,
	int hidden_size,
	int num_experts,
	int top_k
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

	__shared__ float s_logits[288];
	__shared__ int s_indices[288];
	s_logits[expert] = logit;
	s_indices[expert] = expert;
	__syncthreads();

	if ( threadIdx.x == 0 ) {
		#pragma unroll
		for ( int i = 0; i < top_k; ++i ) {
			int max_idx = i;
			float max_val = s_logits[i];
			for ( int j = i + 1; j < num_experts; ++j ) {
				if ( s_logits[j] > max_val ) {
					max_val = s_logits[j];
					max_idx = j;
				}
			}
			float tmp_v = s_logits[i];
			s_logits[i] = s_logits[max_idx];
			s_logits[max_idx] = tmp_v;

			int tmp_i = s_indices[i];
			s_indices[i] = s_indices[max_idx];
			s_indices[max_idx] = tmp_i;
		}

		float max_topk = s_logits[0];
		float sum_exp = 0.0f;
		#pragma unroll
		for ( int i = 0; i < top_k; ++i ) {
			float p = expf( s_logits[i] - max_topk );
			s_logits[i] = p;
			sum_exp += p;
		}

		float inv_sum = 1.0f / ( sum_exp + 1e-9f );
		#pragma unroll
		for ( int i = 0; i < top_k; ++i ) {
			topk_indices[i] = s_indices[i];
			topk_weights[i] = __float2half_rn( s_logits[i] * inv_sum );
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

} // namespace

void MoERouter::route_topk(
	const pod::TensorView& x,
	const pod::TensorView& router_weights,
	int* d_topk_indices,
	half* d_topk_weights,
	int hidden_size,
	int num_experts,
	int top_k,
	cudaStream_t stream
) {
	router_topk_kernel<<<dim3( 1 ), dim3( num_experts ), 0, stream>>>(
		x.ptr<const half>(),
		router_weights.ptr<const half>(),
		d_topk_indices,
		d_topk_weights,
		hidden_size,
		num_experts,
		top_k
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

} // namespace engine::modules
