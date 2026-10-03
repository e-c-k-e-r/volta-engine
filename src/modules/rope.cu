#include "engine/modules/rope.h"
#include <cuda_fp16.h>
#include <cmath>

namespace engine::modules {

namespace {

__global__ void rope_kernel(
	half* __restrict__ q,
	half* __restrict__ k,
	int num_q_heads,
	int num_kv_heads,
	int head_dim,
	int rot_dim,
	int pos,
	float freq_base
) {
	int tid = threadIdx.x; // pair index [0 .. 31]
	if ( tid >= rot_dim / 2 ) return;

	float theta = static_cast<float>( pos ) * powf( freq_base, -( 2.0f * tid / static_cast<float>( rot_dim ) ) );
	float cos_val = __cosf( theta );
	float sin_val = __sinf( theta );
	int half_rot = rot_dim / 2;

	for ( int h = 0; h < num_q_heads; ++h ) {
		half* q_head = q + h * head_dim;
		float a = __half2float( q_head[tid] );
		float b = __half2float( q_head[tid + half_rot] );
		q_head[tid] = __float2half_rn( a * cos_val - b * sin_val );
		q_head[tid + half_rot] = __float2half_rn( a * sin_val + b * cos_val );
	}

	for ( int h = 0; h < num_kv_heads; ++h ) {
		half* k_head = k + h * head_dim;
		float a = __half2float( k_head[tid] );
		float b = __half2float( k_head[tid + half_rot] );
		k_head[tid] = __float2half_rn( a * cos_val - b * sin_val );
		k_head[tid + half_rot] = __float2half_rn( a * sin_val + b * cos_val );
	}
}

} // namespace

void RoPE::forward(
	pod::TensorView& q,
	pod::TensorView& k,
	int num_q_heads,
	int num_kv_heads,
	int head_dim,
	int rot_dim,
	int pos,
	float freq_base,
	cudaStream_t stream
) {
	rope_kernel<<<dim3( 1 ), dim3( rot_dim / 2 ), 0, stream>>>(
		q.ptr<half>(),
		k.ptr<half>(),
		num_q_heads,
		num_kv_heads,
		head_dim,
		rot_dim,
		pos,
		freq_base
	);
}

} // namespace engine::modules
