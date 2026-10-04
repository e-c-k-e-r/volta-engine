#include "engine/modules/mhc.h"
#include <cuda_fp16.h>
#include <cmath>

namespace engine::modules {

namespace {

__global__ void init_streams_kernel(
	const half* __restrict__ in,
	float* __restrict__ streams,
	int hidden_size
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx >= hidden_size ) return;
	float val = __half2float( in[idx] );
	#pragma unroll
	for ( int h = 0; h < 4; ++h ) {
		streams[h * hidden_size + idx] = val;
	}
}

__global__ void mhc_calc_inv_rms_kernel(
	const float* __restrict__ streams,
	float* __restrict__ inv_rms,
	int total_elements
) {
	int tid = threadIdx.x;
	float sum_sq = 0.0f;
	for ( int i = tid; i < total_elements; i += blockDim.x ) {
		float val = streams[i];
		sum_sq = fmaf( val, val, sum_sq );
	}
	#pragma unroll
	for ( int mask = 16; mask > 0; mask >>= 1 ) {
		sum_sq += __shfl_xor_sync( 0xffffffff, sum_sq, mask );
	}
	__shared__ float s_warp_sums[8];
	int warp_id = tid >> 5;
	int lane_id = tid & 31;
	if ( lane_id == 0 ) {
		s_warp_sums[warp_id] = sum_sq;
	}
	__syncthreads();
	if ( warp_id == 0 ) {
		float b_sum = ( lane_id < ( blockDim.x >> 5 ) ) ? s_warp_sums[lane_id] : 0.0f;
		#pragma unroll
		for ( int mask = 4; mask > 0; mask >>= 1 ) {
			b_sum += __shfl_xor_sync( 0xffffffff, b_sum, mask );
		}
		if ( lane_id == 0 ) {
			*inv_rms = rsqrtf( b_sum / static_cast<float>( total_elements ) + 1e-5f );
		}
	}
}

__global__ void mhc_gemv_kernel(
	const float* __restrict__ streams,
	const float* __restrict__ inv_rms_ptr,
	const float* __restrict__ fn,
	float* __restrict__ mix_logits,
	int total_elements
) {
	int k = blockIdx.x; // 0 .. 23
	int tid = threadIdx.x;
	float inv_rms = *inv_rms_ptr;
	const float* fn_row = fn + k * total_elements;

	float dot = 0.0f;
	for ( int j = tid; j < total_elements; j += blockDim.x ) {
		float v = streams[j] * inv_rms;
		dot = fmaf( v, fn_row[j], dot );
	}
	#pragma unroll
	for ( int mask = 16; mask > 0; mask >>= 1 ) {
		dot += __shfl_xor_sync( 0xffffffff, dot, mask );
	}
	__shared__ float s_warp_sums[8];
	int warp_id = tid >> 5;
	int lane_id = tid & 31;
	if ( lane_id == 0 ) {
		s_warp_sums[warp_id] = dot;
	}
	__syncthreads();
	if ( warp_id == 0 ) {
		float b_sum = ( lane_id < ( blockDim.x >> 5 ) ) ? s_warp_sums[lane_id] : 0.0f;
		#pragma unroll
		for ( int mask = 4; mask > 0; mask >>= 1 ) {
			b_sum += __shfl_xor_sync( 0xffffffff, b_sum, mask );
		}
		if ( lane_id == 0 ) {
			mix_logits[k] = b_sum;
		}
	}
}

__global__ void mhc_sinkhorn_kernel(
	const float* __restrict__ mix_logits,
	const float* __restrict__ base,
	const float* __restrict__ scale,
	float* __restrict__ pre_out,
	float* __restrict__ post_out,
	float* __restrict__ comb_out,
	int sinkhorn_iters,
	float eps
) {
	if ( threadIdx.x != 0 ) return;

	float s_pre = scale[0];
	float s_post = scale[1];
	float s_comb = scale[2];

	#pragma unroll
	for ( int h = 0; h < 4; ++h ) {
		float val = mix_logits[h] * s_pre + base[h];
		pre_out[h] = 1.0f / ( 1.0f + __expf( -val ) ) + eps;
	}

	#pragma unroll
	for ( int h = 0; h < 4; ++h ) {
		float val = mix_logits[4 + h] * s_post + base[4 + h];
		post_out[h] = 2.0f / ( 1.0f + __expf( -val ) );
	}

	float comb[4][4];
	for ( int i = 0; i < 4; ++i ) {
		float max_v = -1e20f;
		for ( int j = 0; j < 4; ++j ) {
			float val = mix_logits[8 + i * 4 + j] * s_comb + base[8 + i * 4 + j];
			comb[i][j] = val;
			if ( val > max_v ) max_v = val;
		}
		float sum_exp = 0.0f;
		for ( int j = 0; j < 4; ++j ) {
			comb[i][j] = __expf( comb[i][j] - max_v );
			sum_exp += comb[i][j];
		}
		float inv_sum = 1.0f / sum_exp;
		for ( int j = 0; j < 4; ++j ) {
			comb[i][j] = comb[i][j] * inv_sum + eps;
		}
	}

	// Initial column normalization
	for ( int j = 0; j < 4; ++j ) {
		float col_sum = comb[0][j] + comb[1][j] + comb[2][j] + comb[3][j] + eps;
		float inv_col = 1.0f / col_sum;
		for ( int i = 0; i < 4; ++i ) {
			comb[i][j] *= inv_col;
		}
	}

	// Sinkhorn iterations
	for ( int it = 0; it < sinkhorn_iters - 1; ++it ) {
		// Row normalization
		for ( int i = 0; i < 4; ++i ) {
			float row_sum = comb[i][0] + comb[i][1] + comb[i][2] + comb[i][3] + eps;
			float inv_row = 1.0f / row_sum;
			for ( int j = 0; j < 4; ++j ) {
				comb[i][j] *= inv_row;
			}
		}
		// Column normalization
		for ( int j = 0; j < 4; ++j ) {
			float col_sum = comb[0][j] + comb[1][j] + comb[2][j] + comb[3][j] + eps;
			float inv_col = 1.0f / col_sum;
			for ( int i = 0; i < 4; ++i ) {
				comb[i][j] *= inv_col;
			}
		}
	}

	for ( int i = 0; i < 4; ++i ) {
		for ( int j = 0; j < 4; ++j ) {
			comb_out[i * 4 + j] = comb[i][j];
		}
	}
}

__global__ void mhc_collapse_kernel(
	const float* __restrict__ streams,
	const float* __restrict__ pre,
	half* __restrict__ collapsed_out,
	int hidden_size
) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if ( j >= hidden_size ) return;

	float p0 = pre[0];
	float p1 = pre[1];
	float p2 = pre[2];
	float p3 = pre[3];

	float sum = p0 * streams[0 * hidden_size + j]
	          + p1 * streams[1 * hidden_size + j]
	          + p2 * streams[2 * hidden_size + j]
	          + p3 * streams[3 * hidden_size + j];

	collapsed_out[j] = __float2half_rn( sum );
}

__global__ void mhc_expand_and_update_kernel(
	float* __restrict__ streams_out,
	const float* __restrict__ streams_in,
	const half* __restrict__ sublayer_out,
	const float* __restrict__ post,
	const float* __restrict__ comb,
	int hidden_size
) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if ( j >= hidden_size ) return;

	float y = __half2float( sublayer_out[j] );
	float r0 = streams_in[0 * hidden_size + j];
	float r1 = streams_in[1 * hidden_size + j];
	float r2 = streams_in[2 * hidden_size + j];
	float r3 = streams_in[3 * hidden_size + j];

	#pragma unroll
	for ( int h = 0; h < 4; ++h ) {
		float post_h = post[h];
		float c0 = comb[0 * 4 + h];
		float c1 = comb[1 * 4 + h];
		float c2 = comb[2 * 4 + h];
		float c3 = comb[3 * 4 + h];

		float res_mix = c0 * r0 + c1 * r1 + c2 * r2 + c3 * r3;
		float new_val = fmaf( post_h, y, res_mix );
		streams_out[h * hidden_size + j] = new_val;
	}
}

__global__ void mhc_hyperhead_mean_kernel(
	const float* __restrict__ streams,
	half* __restrict__ final_hidden,
	int hidden_size
) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if ( j >= hidden_size ) return;

	float sum = streams[0 * hidden_size + j]
	          + streams[1 * hidden_size + j]
	          + streams[2 * hidden_size + j]
	          + streams[3 * hidden_size + j];

	final_hidden[j] = __float2half_rn( sum * 0.25f );
}

} // namespace

void HyperConnection::init_streams(
	const pod::TensorView& in,
	pod::TensorView& streams,
	int hidden_size,
	cudaStream_t stream
) {
	init_streams( in.ptr<const half>(), streams.ptr<float>(), hidden_size, stream );
}

void HyperConnection::init_streams(
	const half* in,
	float* streams,
	int hidden_size,
	cudaStream_t stream
) {
	int threads = 256;
	int blocks = ( hidden_size + threads - 1 ) / threads;
	init_streams_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
		in, streams, hidden_size
	);
}

void HyperConnection::collapse(
	const pod::TensorView& streams,
	const pod::TensorView& fn,
	const pod::TensorView& base,
	const pod::TensorView& scale,
	pod::TensorView& collapsed_out,
	pod::TensorView& post_out,
	pod::TensorView& comb_out,
	pod::TensorView& inv_rms_buf,
	pod::TensorView& mix_logits_buf,
	pod::TensorView& pre_buf,
	int hidden_size,
	int sinkhorn_iters,
	float eps,
	cudaStream_t stream
) {
	collapse(
		streams.ptr<const float>(),
		fn.ptr<const float>(),
		base.ptr<const float>(),
		scale.ptr<const float>(),
		collapsed_out.ptr<half>(),
		post_out.ptr<float>(),
		comb_out.ptr<float>(),
		inv_rms_buf.ptr<float>(),
		mix_logits_buf.ptr<float>(),
		pre_buf.ptr<float>(),
		hidden_size,
		sinkhorn_iters,
		eps,
		stream
	);
}

void HyperConnection::collapse(
	const float* streams,
	const float* fn,
	const float* base,
	const float* scale,
	half* collapsed_out,
	float* post_out,
	float* comb_out,
	float* inv_rms_buf,
	float* mix_logits_buf,
	float* pre_buf,
	int hidden_size,
	int sinkhorn_iters,
	float eps,
	cudaStream_t stream
) {
	int total_elements = 4 * hidden_size;
	mhc_calc_inv_rms_kernel<<<dim3( 1 ), dim3( 256 ), 0, stream>>>(
		streams, inv_rms_buf, total_elements
	);
	mhc_gemv_kernel<<<dim3( 24 ), dim3( 256 ), 0, stream>>>(
		streams, inv_rms_buf, fn, mix_logits_buf, total_elements
	);
	mhc_sinkhorn_kernel<<<dim3( 1 ), dim3( 32 ), 0, stream>>>(
		mix_logits_buf, base, scale, pre_buf, post_out, comb_out, sinkhorn_iters, eps
	);
	int threads = 256;
	int blocks = ( hidden_size + threads - 1 ) / threads;
	mhc_collapse_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
		streams, pre_buf, collapsed_out, hidden_size
	);
}

void HyperConnection::update(
	pod::TensorView& streams,
	const pod::TensorView& sublayer_out,
	const pod::TensorView& post,
	const pod::TensorView& comb,
	int hidden_size,
	cudaStream_t stream
) {
	update(
		streams.ptr<float>(),
		streams.ptr<const float>(),
		sublayer_out.ptr<const half>(),
		post.ptr<const float>(),
		comb.ptr<const float>(),
		hidden_size,
		stream
	);
}

void HyperConnection::update(
	pod::TensorView& streams_out,
	const pod::TensorView& streams_in,
	const pod::TensorView& sublayer_out,
	const pod::TensorView& post,
	const pod::TensorView& comb,
	int hidden_size,
	cudaStream_t stream
) {
	update(
		streams_out.ptr<float>(),
		streams_in.ptr<const float>(),
		sublayer_out.ptr<const half>(),
		post.ptr<const float>(),
		comb.ptr<const float>(),
		hidden_size,
		stream
	);
}

void HyperConnection::update(
	float* streams,
	const half* sublayer_out,
	const float* post,
	const float* comb,
	int hidden_size,
	cudaStream_t stream
) {
	update( streams, streams, sublayer_out, post, comb, hidden_size, stream );
}

void HyperConnection::update(
	float* streams_out,
	const float* streams_in,
	const half* sublayer_out,
	const float* post,
	const float* comb,
	int hidden_size,
	cudaStream_t stream
) {
	int threads = 256;
	int blocks = ( hidden_size + threads - 1 ) / threads;
	mhc_expand_and_update_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
		streams_out, streams_in, sublayer_out, post, comb, hidden_size
	);
}

void HyperHead::forward(
	const pod::TensorView& streams,
	pod::TensorView& final_hidden,
	int hidden_size,
	bool is_mean,
	cudaStream_t stream
) {
	if ( is_mean ) {
		mean( streams, final_hidden, hidden_size, stream );
	}
}

void HyperHead::mean(
	const pod::TensorView& streams,
	pod::TensorView& final_hidden,
	int hidden_size,
	cudaStream_t stream
) {
	mean( streams.ptr<const float>(), final_hidden.ptr<half>(), hidden_size, stream );
}

void HyperHead::mean(
	const float* streams,
	half* final_hidden,
	int hidden_size,
	cudaStream_t stream
) {
	int threads = 256;
	int blocks = ( hidden_size + threads - 1 ) / threads;
	mhc_hyperhead_mean_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
		streams, final_hidden, hidden_size
	);
}

} // namespace engine::modules
