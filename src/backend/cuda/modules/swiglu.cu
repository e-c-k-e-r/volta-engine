#include "engine/modules/swiglu.h"
#include "engine/backend/cuda/activation.cuh"
#include <cuda_fp16.h>
#include <cmath>

namespace engine::modules {

namespace {

__global__ void swiglu_kernel(
	const half* __restrict__ gate,
	const half* __restrict__ up,
	half* __restrict__ out,
	int numel,
	float limit
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < numel ) {
		float g = __half2float( gate[idx] );
		float u = __half2float( up[idx] );
		if ( limit > 0.0f ) {
			g = fminf( g, limit );
			u = fmaxf( -limit, fminf( u, limit ) );
		}
		float silu_g = engine::math::silu( g );
		out[idx] = __float2half_rn( silu_g * u );
	}
}

__global__ void swiglu_fused_kernel(
	const half* __restrict__ gate_up,
	half* __restrict__ out,
	int inter_dim,
	int total_out_numel,
	float limit
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx < total_out_numel ) {
		int m = idx / inter_dim;
		int i = idx % inter_dim;
		int src_offset = m * ( 2 * inter_dim );
		float g = __half2float( gate_up[src_offset + i] );
		float u = __half2float( gate_up[src_offset + inter_dim + i] );
		if ( limit > 0.0f ) {
			g = fminf( g, limit );
			u = fmaxf( -limit, fminf( u, limit ) );
		}
		float silu_g = engine::math::silu( g );
		out[idx] = __float2half_rn( silu_g * u );
	}
}

} // namespace

void SwiGLU::forward(
	const pod::TensorView& gate,
	const pod::TensorView& up,
	pod::TensorView& out,
	cudaStream_t stream,
	float limit
) {
	int numel = gate.numel();
	int threads = 256;
	int blocks = ( numel + threads - 1 ) / threads;
	swiglu_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
		gate.ptr<const half>(),
		up.ptr<const half>(),
		out.ptr<half>(),
		numel,
		limit
	);
}

void SwiGLU::forward_fused(
	const pod::TensorView& gate_up,
	pod::TensorView& out,
	cudaStream_t stream,
	float limit
) {
	int total_out_numel = out.numel();
	int inter_dim = ( out.ndim >= 2 ) ? out.shape[1] : total_out_numel;
	int threads = 256;
	int blocks = ( total_out_numel + threads - 1 ) / threads;
	swiglu_fused_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
		gate_up.ptr<const half>(),
		out.ptr<half>(),
		inter_dim,
		total_out_numel,
		limit
	);
}

} // namespace engine::modules
