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
	int base_pos,
	int m_tokens,
	float freq_base
) {
	int tid = threadIdx.x; // pair index [0 .. 31]
	int m = blockIdx.y;    // token index [0 .. m_tokens - 1]
	if ( m >= m_tokens || tid >= rot_dim / 2 ) return;

	int pos = base_pos + m;
	float theta = static_cast<float>( pos ) * powf( freq_base, -( 2.0f * tid / static_cast<float>( rot_dim ) ) );
	float cos_val = __cosf( theta );
	float sin_val = __sinf( theta );
	int half_rot = rot_dim / 2;

	half* q_m = q + static_cast<size_t>( m ) * num_q_heads * head_dim;
	half* k_m = k + static_cast<size_t>( m ) * num_kv_heads * head_dim;

	#pragma unroll
	for ( int h = 0; h < num_q_heads; ++h ) {
		half* q_head = q_m + h * head_dim;
		float a = __half2float( q_head[tid] );
		float b = __half2float( q_head[tid + half_rot] );
		q_head[tid] = __float2half_rn( a * cos_val - b * sin_val );
		q_head[tid + half_rot] = __float2half_rn( a * sin_val + b * cos_val );
	}

	#pragma unroll
	for ( int h = 0; h < num_kv_heads; ++h ) {
		half* k_head = k_m + h * head_dim;
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
	int m = ( q.ndim >= 1 && q.shape[0] > 0 ) ? q.shape[0] : 1;
	rope_kernel<<<dim3( 1, m ), dim3( rot_dim / 2 ), 0, stream>>>(
		q.ptr<half>(),
		k.ptr<half>(),
		num_q_heads,
		num_kv_heads,
		head_dim,
		rot_dim,
		pos,
		m,
		freq_base
	);
}

} // namespace engine::modules
