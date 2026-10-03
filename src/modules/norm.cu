#include "engine/modules/norm.h"
#include <cuda_fp16.h>
#include <cmath>

namespace engine::modules {

namespace {

__global__ void rms_norm_kernel(
	const half* __restrict__ src,
	half* __restrict__ dst,
	const half* __restrict__ weight,
	int hidden_size,
	float eps,
	float constant_bias
) {
	int tid = threadIdx.x;
	float local_sq = 0.0f;

	for ( int i = tid; i < hidden_size; i += blockDim.x ) {
		float val = __half2float( src[i] );
		local_sq = fmaf( val, val, local_sq );
	}

	#pragma unroll
	for ( int mask = 16; mask > 0; mask >>= 1 ) {
		local_sq += __shfl_xor_sync( 0xffffffff, local_sq, mask );
	}

	__shared__ float s_warp_sums[8];
	int warp_id = tid >> 5;
	int lane_id = tid & 31;
	if ( lane_id == 0 ) {
		s_warp_sums[warp_id] = local_sq;
	}
	__syncthreads();

	if ( warp_id == 0 ) {
		float b_sum = ( lane_id < 8 ) ? s_warp_sums[lane_id] : 0.0f;
		#pragma unroll
		for ( int mask = 4; mask > 0; mask >>= 1 ) {
			b_sum += __shfl_xor_sync( 0xffffffff, b_sum, mask );
		}
		if ( lane_id == 0 ) {
			s_warp_sums[0] = rsqrtf( b_sum / static_cast<float>( hidden_size ) + eps );
		}
	}
	__syncthreads();

	float inv_rms = s_warp_sums[0];

	for ( int i = tid; i < hidden_size; i += blockDim.x ) {
		float val = __half2float( src[i] );
		float w = ( weight != nullptr ) ? ( __half2float( weight[i] ) + constant_bias ) : 1.0f;
		dst[i] = __float2half_rn( val * w * inv_rms );
	}
}

__global__ void rms_norm_heads_kernel(
	half* __restrict__ qk,
	const half* __restrict__ weight,
	int num_heads,
	int head_dim,
	float eps,
	float constant_bias
) {
	int head = blockIdx.x;
	int tid = threadIdx.x;
	if ( head >= num_heads || tid >= head_dim ) return;

	half* ptr = qk + head * head_dim;
	float val = __half2float( ptr[tid] );
	float sq = val * val;

	#pragma unroll
	for ( int mask = 16; mask > 0; mask >>= 1 ) {
		sq += __shfl_xor_sync( 0xffffffff, sq, mask );
	}

	__shared__ float s_warp_sums[8];
	int warp_id = tid >> 5;
	int lane_id = tid & 31;
	if ( lane_id == 0 ) {
		s_warp_sums[warp_id] = sq;
	}
	__syncthreads();

	int num_warps = ( head_dim + 31 ) >> 5;
	if ( warp_id == 0 ) {
		float b_sum = ( lane_id < num_warps ) ? s_warp_sums[lane_id] : 0.0f;
		#pragma unroll
		for ( int mask = 4; mask > 0; mask >>= 1 ) {
			b_sum += __shfl_xor_sync( 0xffffffff, b_sum, mask );
		}
		if ( lane_id == 0 ) {
			s_warp_sums[0] = rsqrtf( b_sum / static_cast<float>( head_dim ) + eps );
		}
	}
	__syncthreads();

	float inv_rms = s_warp_sums[0];
	float w = ( weight != nullptr ) ? ( __half2float( weight[tid] ) + constant_bias ) : 1.0f;
	ptr[tid] = __float2half_rn( val * w * inv_rms );
}

__global__ void add_residual_kernel(
	half* __restrict__ hidden,
	const half* __restrict__ residual,
	int hidden_size
) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ( tid < hidden_size ) {
		float h = __half2float( hidden[tid] );
		float r = __half2float( residual[tid] );
		hidden[tid] = __float2half_rn( h + r );
	}
}

__global__ void argmax_kernel(
	const half* __restrict__ logits,
	int* __restrict__ out_idx,
	int numel
) {
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	float max_val = -1e30f;
	int max_idx = -1;

	for ( int i = gid; i < numel; i += blockDim.x * gridDim.x ) {
		float val = __half2float( logits[i] );
		if ( val > max_val ) {
			max_val = val;
			max_idx = i;
		}
	}

	#pragma unroll
	for ( int mask = 16; mask > 0; mask >>= 1 ) {
		float other_val = __shfl_xor_sync( 0xffffffff, max_val, mask );
		int other_idx = __shfl_xor_sync( 0xffffffff, max_idx, mask );
		if ( other_val > max_val ) {
			max_val = other_val;
			max_idx = other_idx;
		}
	}

	__shared__ float s_max_vals[32];
	__shared__ int s_max_idxs[32];
	int warp_id = tid >> 5;
	int lane_id = tid & 31;

	if ( lane_id == 0 ) {
		s_max_vals[warp_id] = max_val;
		s_max_idxs[warp_id] = max_idx;
	}
	__syncthreads();

	if ( warp_id == 0 ) {
		float b_max = ( lane_id < ( blockDim.x >> 5 ) ) ? s_max_vals[lane_id] : -1e30f;
		int b_idx = ( lane_id < ( blockDim.x >> 5 ) ) ? s_max_idxs[lane_id] : -1;
		#pragma unroll
		for ( int mask = 16; mask > 0; mask >>= 1 ) {
			float other_val = __shfl_xor_sync( 0xffffffff, b_max, mask );
			int other_idx = __shfl_xor_sync( 0xffffffff, b_idx, mask );
			if ( other_val > b_max ) {
				b_max = other_val;
				b_idx = other_idx;
			}
		}
		if ( lane_id == 0 ) {
			out_idx[0] = b_idx;
		}
	}
}

} // namespace

void RMSNorm::forward(
	const pod::TensorView& x,
	const pod::TensorView& weight,
	pod::TensorView& out,
	float eps,
	float constant_bias,
	cudaStream_t stream
) {
	int hidden_size = x.shape[x.ndim - 1];
	rms_norm_kernel<<<dim3( 1 ), dim3( 256 ), 0, stream>>>(
		x.ptr<const half>(),
		out.ptr<half>(),
		weight.ptr<const half>(),
		hidden_size,
		eps,
		constant_bias
	);
}

void RMSNorm::forward_heads(
	pod::TensorView& qk,
	const pod::TensorView& weight,
	int num_heads,
	int head_dim,
	float eps,
	float constant_bias,
	cudaStream_t stream
) {
	rms_norm_heads_kernel<<<dim3( num_heads ), dim3( head_dim ), 0, stream>>>(
		qk.ptr<half>(),
		weight.ptr<const half>(),
		num_heads,
		head_dim,
		eps,
		constant_bias
	);
}

void Elementwise::add_residual(
	pod::TensorView& hidden,
	const pod::TensorView& residual,
	cudaStream_t stream
) {
	int hidden_size = hidden.shape[hidden.ndim - 1];
	int threads = 256;
	int blocks = ( hidden_size + threads - 1 ) / threads;
	add_residual_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
		hidden.ptr<half>(),
		residual.ptr<const half>(),
		hidden_size
	);
}

void Elementwise::argmax(
	const pod::TensorView& logits,
	int* d_out_idx,
	int numel,
	cudaStream_t stream
) {
	argmax_kernel<<<dim3( 1 ), dim3( 256 ), 0, stream>>>(
		logits.ptr<const half>(),
		d_out_idx,
		numel
	);
}

} // namespace engine::modules
