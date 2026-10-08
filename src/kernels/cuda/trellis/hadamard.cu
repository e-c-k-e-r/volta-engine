#include <cuda_fp16.h>
#include "hadamard.cuh"
#include "hadamard_inner.cuh"
#include "util.cuh"

template <bool pre_scale, bool post_scale>
__global__ __launch_bounds__( 32 )
void had_hf_r_128_kernel(
	const half* __restrict__ input_ptr,
	half* __restrict__ output_ptr,
	const half* __restrict__ scale,
	const float r_scale
) {
	input_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
	output_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
	had_hf_r_128_inner<pre_scale, post_scale>( input_ptr, output_ptr, scale, r_scale );
}

void had_r_128(
	const half* input,
	half* output,
	const half* pre_scale,
	const half* post_scale,
	float scale,
	int rows,
	int cols,
	cudaStream_t stream
) {
	int blocks = cols / 128;
	dim3 blockDim( 32 );
	dim3 gridDim( rows, blocks );

	if ( pre_scale != nullptr ) {
		had_hf_r_128_kernel<true, false><<<gridDim, blockDim, 0, stream>>>( input, output, pre_scale, scale );
	}
	else if ( post_scale != nullptr ) {
		had_hf_r_128_kernel<false, true><<<gridDim, blockDim, 0, stream>>>( input, output, post_scale, scale );
	}
	else {
		had_hf_r_128_kernel<false, false><<<gridDim, blockDim, 0, stream>>>( input, output, nullptr, scale );
	}
}
