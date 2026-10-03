#include "engine/modules/swiglu.h"
#include <cuda_fp16.h>
#include <cmath>

namespace engine::modules {

namespace {

__global__ void swiglu_kernel(
	const half* __restrict__ gate,
	const half* __restrict__ up,
	half* __restrict__ out,
	int numel
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < numel ) {
		float g = __half2float( gate[idx] );
		float u = __half2float( up[idx] );
		float silu_g = g / ( 1.0f + __expf( -g ) );
		out[idx] = __float2half_rn( silu_g * u );
	}
}

} // namespace

void SwiGLU::forward(
	const pod::TensorView& gate,
	const pod::TensorView& up,
	pod::TensorView& out,
	cudaStream_t stream
) {
	int numel = gate.numel();
	int threads = 256;
	int blocks = ( numel + threads - 1 ) / threads;
	swiglu_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
		gate.ptr<const half>(),
		up.ptr<const half>(),
		out.ptr<half>(),
		numel
	);
}

} // namespace engine::modules
