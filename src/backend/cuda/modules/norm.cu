#include "engine/modules/norm.h"
#include "engine/backend/cuda/reduce.cuh"
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
	int row = blockIdx.x;
	const half* row_src = src + static_cast<int64_t>( row ) * hidden_size;
	half* row_dst = dst + static_cast<int64_t>( row ) * hidden_size;
	int tid = threadIdx.x;
	float local_sq = 0.0f;

	// Element i belongs to thread i % blockDim.x and is accumulated in ascending order, exactly as the plain loop
	// did, so the result is bit-identical; but all of a thread's loads are issued before any is consumed (the row
	// stays in registers for the second pass) instead of one dependent global load per step.
	constexpr int kMaxPer = 24;
	float v[kMaxPer];
	const int per = ( hidden_size + static_cast<int>( blockDim.x ) - 1 ) / static_cast<int>( blockDim.x );
	if ( per <= kMaxPer ) {
		#pragma unroll
		for ( int k = 0; k < kMaxPer; ++k ) {
			const int i = tid + k * blockDim.x;
			v[k] = ( i < hidden_size ) ? __half2float( row_src[i] ) : 0.0f;
		}
		#pragma unroll
		for ( int k = 0; k < kMaxPer; ++k ) {
			if ( tid + k * blockDim.x < hidden_size ) local_sq = fmaf( v[k], v[k], local_sq );
		}
	}
	else {
		for ( int i = tid; i < hidden_size; i += blockDim.x ) {
			float val = __half2float( row_src[i] );
			local_sq = fmaf( val, val, local_sq );
		}
	}

	float inv_rms = rsqrtf( engine::math::block_sum<8>( local_sq ) / static_cast<float>( hidden_size ) + eps );

	if ( per <= kMaxPer ) {
		float w[kMaxPer];
		#pragma unroll
		for ( int k = 0; k < kMaxPer; ++k ) {
			const int i = tid + k * blockDim.x;
			w[k] = ( weight != nullptr && i < hidden_size ) ? ( __half2float( weight[i] ) + constant_bias ) : 1.0f;
		}
		#pragma unroll
		for ( int k = 0; k < kMaxPer; ++k ) {
			const int i = tid + k * blockDim.x;
			if ( i < hidden_size ) row_dst[i] = __float2half_rn( v[k] * w[k] * inv_rms );
		}
	}
	else {
		for ( int i = tid; i < hidden_size; i += blockDim.x ) {
			float val = __half2float( row_src[i] );
			float w = ( weight != nullptr ) ? ( __half2float( weight[i] ) + constant_bias ) : 1.0f;
			row_dst[i] = __float2half_rn( val * w * inv_rms );
		}
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
	// blockIdx.y = row: rows are [num_heads * head_dim] contiguous
	int head = blockIdx.x;
	int tid = threadIdx.x;
	if ( head >= num_heads || tid >= head_dim ) return;

	half* ptr = qk + ( static_cast<size_t>( blockIdx.y ) * num_heads + head ) * head_dim;
	float val = __half2float( ptr[tid] );
	float sq = val * val;

	float inv_rms = rsqrtf( engine::math::block_sum<8>( sq ) / static_cast<float>( head_dim ) + eps );
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

// Does (ov, oi) beat (v, i)? Larger value wins; on equal values the lower index wins (ties resolve to the first maximum, as the CPU and
// Vulkan references do), and an empty slot (index -1) loses to any real one.
__device__ __forceinline__ bool argmax_better( float ov, int oi, float v, int i ) {
	return ov > v || ( ov == v && oi >= 0 && ( i < 0 || oi < i ) );
}

// Block-wide argmax of the per-thread (val, idx) pairs: a butterfly inside each warp, then the warp results through shared memory. True on
// thread 0, where val / idx hold the block's winner.
__device__ __forceinline__ bool block_argmax( float& max_val, int& max_idx ) {
	const int tid = threadIdx.x;
	#pragma unroll
	for ( int mask = 16; mask > 0; mask >>= 1 ) {
		float other_val = __shfl_xor_sync( 0xffffffff, max_val, mask );
		int other_idx = __shfl_xor_sync( 0xffffffff, max_idx, mask );
		if ( argmax_better( other_val, other_idx, max_val, max_idx ) ) {
			max_val = other_val;
			max_idx = other_idx;
		}
	}

	__shared__ float s_max_vals[32];
	__shared__ int s_max_idxs[32];
	const int warp_id = tid >> 5;
	const int lane_id = tid & 31;

	if ( lane_id == 0 ) {
		s_max_vals[warp_id] = max_val;
		s_max_idxs[warp_id] = max_idx;
	}
	__syncthreads();

	if ( warp_id != 0 ) return false;
	float b_max = ( lane_id < ( blockDim.x >> 5 ) ) ? s_max_vals[lane_id] : -1e30f;
	int b_idx = ( lane_id < ( blockDim.x >> 5 ) ) ? s_max_idxs[lane_id] : -1;
	#pragma unroll
	for ( int mask = 16; mask > 0; mask >>= 1 ) {
		float other_val = __shfl_xor_sync( 0xffffffff, b_max, mask );
		int other_idx = __shfl_xor_sync( 0xffffffff, b_idx, mask );
		if ( argmax_better( other_val, other_idx, b_max, b_idx ) ) {
			b_max = other_val;
			b_idx = other_idx;
		}
	}
	max_val = b_max;
	max_idx = b_idx;
	return lane_id == 0;
}

__global__ void argmax_kernel(
	const half* __restrict__ logits,
	int* __restrict__ out_idx,
	int numel
) {
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

	if ( block_argmax( max_val, max_idx ) ) out_idx[0] = max_idx;
}

__global__ void batched_argmax_kernel(
	const half* __restrict__ logits,
	int* __restrict__ out_idx,
	float* __restrict__ out_val,
	int numel,
	int batch_size,
	int index_offset,
	int row_stride
) {
	int row = blockIdx.x;
	if ( row >= batch_size ) return;

	const half* row_logits = logits + static_cast<size_t>( row ) * row_stride;
	int tid = threadIdx.x;

	float max_val = -1e30f;
	int max_idx = -1;

	for ( int i = tid; i < numel; i += blockDim.x ) {
		float val = __half2float( row_logits[i] );
		if ( val > max_val ) {
			max_val = val;
			max_idx = i;
		}
	}

	if ( block_argmax( max_val, max_idx ) ) {
		out_idx[row] = max_idx < 0 ? -1 : max_idx + index_offset;
		if ( out_val ) out_val[row] = max_val;
	}
}

#if defined(USE_ROCM)
#include "../../../kernels/rocm/rms_norm_vec.cuh"
#endif

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
	int rows = 1;
	for ( int i = 0; i < x.ndim - 1; ++i ) {
		rows *= x.shape[i];
	}
#if defined(USE_ROCM)
	if ( hidden_size % 8 == 0 && hidden_size <= 8192 && ( reinterpret_cast<uintptr_t>( x.data ) & 15 ) == 0 && ( reinterpret_cast<uintptr_t>( out.data ) & 15 ) == 0
		&& ( weight.data == nullptr || ( reinterpret_cast<uintptr_t>( weight.data ) & 15 ) == 0 ) ) {
		rms_norm_vec_kernel<<<dim3( rows ), dim3( 256 ), 0, stream>>>( x.ptr<const half>(), out.ptr<half>(), weight.ptr<const half>(), hidden_size, eps, constant_bias );
		return;
	}
#endif
	rms_norm_kernel<<<dim3( rows ), dim3( 256 ), 0, stream>>>(
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
	const int rows = ( qk.ndim >= 2 && qk.shape[0] > 0 ) ? qk.shape[0] : 1;
	rms_norm_heads_kernel<<<dim3( num_heads, rows ), dim3( head_dim ), 0, stream>>>(
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
	int rows = 1;
	for ( int i = 0; i < hidden.ndim - 1; ++i ) {
		rows *= hidden.shape[i];
	}
	int total_elements = rows * hidden_size;
	int threads = 256;
	int blocks = ( total_elements + threads - 1 ) / threads;
	add_residual_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
		hidden.ptr<half>(),
		residual.ptr<const half>(),
		total_elements
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

void Elementwise::batched_argmax(
	const pod::TensorView& logits,
	int* d_out_idx,
	int numel,
	int batch_size,
	cudaStream_t stream
) {
	batched_argmax_kernel<<<dim3( batch_size ), dim3( 512 ), 0, stream>>>(
		logits.ptr<const half>(),
		d_out_idx,
		nullptr,
		numel,
		batch_size,
		0,
		logits.shape[1]
	);
}

void Elementwise::batched_argmax_range(
	const pod::TensorView& logits,
	int* d_out_idx,
	float* d_out_val,
	int columns,
	int batch_size,
	int index_offset,
	cudaStream_t stream
) {
	// A vocabulary shard carries pad columns past `columns`; they decode to 0.0 and would win against
	// real negative logits, so the scan stops at the real width and the index is offset to the shard's
	// slice of the global vocabulary.
	ENGINE_ASSERT( columns > 0 && columns <= logits.shape[1], "batched_argmax_range: columns outside the row" );
	batched_argmax_kernel<<<dim3( batch_size ), dim3( 512 ), 0, stream>>>(
		logits.ptr<const half>(),
		d_out_idx,
		d_out_val,
		columns,
		batch_size,
		index_offset,
		logits.shape[1]
	);
}

} // namespace engine::modules
