#include "engine/modules/mla.h"
#include <cuda_fp16.h>
#include <cmath>

namespace engine::modules {

namespace {

__inline__ __device__ float warp_reduce_sum( float val ) {
	#pragma unroll
	for ( int offset = 16; offset > 0; offset >>= 1 ) {
		val += __shfl_down_sync( 0xffffffff, val, offset );
	}
	return val;
}

__global__ void store_mla_cache_kernel(
	half* __restrict__ latent_cache,
	half* __restrict__ rope_cache,
	const half* __restrict__ k_lat,
	const half* __restrict__ k_pe,
	int layer_idx,
	int pos,
	int mla_dc,
	int mla_dr,
	int max_seq
) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t slot = static_cast<size_t>( layer_idx ) * max_seq + pos;

	if ( tid < mla_dc && k_lat != nullptr ) {
		latent_cache[slot * mla_dc + tid] = k_lat[tid];
	}
	if ( tid < mla_dr && k_pe != nullptr ) {
		rope_cache[slot * mla_dr + tid] = k_pe[tid];
	}
}

__global__ __launch_bounds__( 256 )
void dsa_mla_decode_kernel(
	const half* __restrict__ q_lat,
	const half* __restrict__ q_pe,
	const half* __restrict__ latent_cache,
	const half* __restrict__ rope_cache,
	half* __restrict__ out,
	int layer_idx,
	int current_pos,
	int dsa_top_k,
	int mla_dc,
	int mla_dr,
	int max_seq,
	int out_dim
) {
	int tid = threadIdx.x;
	int wid = tid >> 5;
	int lane = tid & 31;

	__shared__ float s_scores[128];
	__shared__ float s_inv_sum;

	int num_keys = ( current_pos < dsa_top_k ) ? ( current_pos + 1 ) : dsa_top_k;
	if ( num_keys <= 0 ) num_keys = 1;
	if ( num_keys > 128 ) num_keys = 128;

	float scale = 1.0f / sqrtf( static_cast<float>( mla_dc + mla_dr ) );

	// 1. Compute dot product Q . K for each sparse token position across warps
	for ( int k = wid; k < num_keys; k += 8 ) {
		int pos = ( current_pos >= dsa_top_k ) ? ( current_pos - dsa_top_k + 1 + k ) : k;
		size_t slot = static_cast<size_t>( layer_idx ) * max_seq + pos;

		const half* k_lat = latent_cache + slot * mla_dc;
		const half* k_pe = rope_cache + slot * mla_dr;

		float dot_lat = 0.0f;
		for ( int d = lane; d < mla_dc; d += 32 ) {
			float q_val = ( q_lat != nullptr ) ? __half2float( q_lat[d] ) : 0.0f;
			dot_lat = fmaf( q_val, __half2float( k_lat[d] ), dot_lat );
		}

		float dot_pe = 0.0f;
		if ( q_pe != nullptr && mla_dr > 0 ) {
			for ( int d = lane; d < mla_dr; d += 32 ) {
				dot_pe = fmaf( __half2float( q_pe[d] ), __half2float( k_pe[d] ), dot_pe );
			}
		}

		float total_dot = warp_reduce_sum( dot_lat + dot_pe );
		if ( lane == 0 ) {
			s_scores[k] = total_dot * scale;
		}
	}
	__syncthreads();

	// 2. Softmax normalization
	if ( tid == 0 ) {
		float max_val = s_scores[0];
		for ( int k = 1; k < num_keys; ++k ) {
			if ( s_scores[k] > max_val ) {
				max_val = s_scores[k];
			}
		}

		float sum_exp = 0.0f;
		for ( int k = 0; k < num_keys; ++k ) {
			float p = expf( s_scores[k] - max_val );
			s_scores[k] = p;
			sum_exp += p;
		}
		s_inv_sum = 1.0f / ( sum_exp + 1e-9f );
	}
	__syncthreads();

	for ( int k = tid; k < num_keys; k += blockDim.x ) {
		s_scores[k] *= s_inv_sum;
	}
	__syncthreads();

	// 3. Output reduction into latent dimensions
	for ( int d = tid; d < out_dim; d += blockDim.x ) {
		if ( d < mla_dc ) {
			float acc = 0.0f;
			for ( int k = 0; k < num_keys; ++k ) {
				int pos = ( current_pos >= dsa_top_k ) ? ( current_pos - dsa_top_k + 1 + k ) : k;
				size_t slot = static_cast<size_t>( layer_idx ) * max_seq + pos;

				const half* v_lat = latent_cache + slot * mla_dc;
				acc = fmaf( s_scores[k], __half2float( v_lat[d] ), acc );
			}
			out[d] = __float2half_rn( acc );
		}
		else {
			out[d] = 0;
		}
	}
}

} // namespace

void MLADSA::store_cache(
	half* latent_cache,
	half* rope_cache,
	const half* k_lat,
	const half* k_pe,
	int layer_idx,
	int pos,
	int mla_dc,
	int mla_dr,
	int max_seq,
	cudaStream_t stream
) {
	int max_dim = ( mla_dc > mla_dr ) ? mla_dc : mla_dr;
	int threads = 256;
	int blocks = ( max_dim + threads - 1 ) / threads;
	store_mla_cache_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
		latent_cache,
		rope_cache,
		k_lat,
		k_pe,
		layer_idx,
		pos,
		mla_dc,
		mla_dr,
		max_seq
	);
}

void MLADSA::forward_decode(
	const pod::TensorView& q_lat,
	const pod::TensorView& q_pe,
	const half* latent_cache,
	const half* rope_cache,
	pod::TensorView& out,
	int layer_idx,
	int current_pos,
	int dsa_top_k,
	int mla_dc,
	int mla_dr,
	int max_seq,
	int out_dim,
	cudaStream_t stream
) {
	dsa_mla_decode_kernel<<<dim3( 1 ), dim3( 256 ), 0, stream>>>(
		q_lat.ptr<const half>(),
		q_pe.ptr<const half>(),
		latent_cache,
		rope_cache,
		out.ptr<half>(),
		layer_idx,
		current_pos,
		dsa_top_k,
		mla_dc,
		mla_dr,
		max_seq,
		out_dim
	);
}

} // namespace engine::modules
