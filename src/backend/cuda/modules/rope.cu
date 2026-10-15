#include "engine/modules/rope.h"
#include "engine/backend/cuda/rope_math.cuh"
#include <cuda_fp16.h>
#include <cmath>

namespace engine::modules {

namespace {

// Interleaved: channel pair (2i, 2i+1) as in GLM4V / MRoPE; otherwise the halves pair (i, i + rot/2) as in Llama / Qwen.
template <bool Interleaved>
__global__ void rope_kernel(
	half* __restrict__ q,
	half* __restrict__ k,
	int num_q_heads,
	int num_kv_heads,
	int head_dim,
	int rot_dim,
	const pod::BatchRows* __restrict__ rows,
	float freq_base
) {
	int tid = threadIdx.x; // pair index [0 .. rot_dim / 2)
	int m = blockIdx.y;    // row index (grid sized by host row count)
	if ( tid >= rot_dim / 2 ) return;

	int pos = rows->pos[m];
	if ( rows->mrope ) {
		// Qwen3-VL interleaved M-RoPE: frequency i takes the height position at i % 3 == 1 (i < 33), the width position at
		// i % 3 == 2 (i < 30) and the temporal position otherwise
		const int comp = ( tid % 3 == 1 && tid < 33 ) ? 1 : ( ( tid % 3 == 2 && tid < 30 ) ? 2 : 0 );
		pos = rows->rope_pos[comp][m];
	}
	float cos_val, sin_val;
	engine::math::rope_cos_sin<true>( pos, tid, rot_dim, freq_base, cos_val, sin_val );
	int half_rot = rot_dim / 2;

	half* q_m = q + static_cast<size_t>( m ) * num_q_heads * head_dim;
	half* k_m = k + static_cast<size_t>( m ) * num_kv_heads * head_dim;

	#pragma unroll
	for ( int h = 0; h < num_q_heads; ++h ) {
		half* q_head = q_m + h * head_dim;
		const int i0 = Interleaved ? 2 * tid : tid, i1 = Interleaved ? 2 * tid + 1 : tid + half_rot;
		float a = __half2float( q_head[i0] );
		float b = __half2float( q_head[i1] );
		q_head[i0] = __float2half_rn( a * cos_val - b * sin_val );
		q_head[i1] = __float2half_rn( a * sin_val + b * cos_val );
	}

	#pragma unroll
	for ( int h = 0; h < num_kv_heads; ++h ) {
		half* k_head = k_m + h * head_dim;
		const int i0 = Interleaved ? 2 * tid : tid, i1 = Interleaved ? 2 * tid + 1 : tid + half_rot;
		float a = __half2float( k_head[i0] );
		float b = __half2float( k_head[i1] );
		k_head[i0] = __float2half_rn( a * cos_val - b * sin_val );
		k_head[i1] = __float2half_rn( a * sin_val + b * cos_val );
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
	const pod::BatchRows* d_rows,
	int count,
	float freq_base,
	cudaStream_t stream,
	bool interleaved
) {
	auto kernel = interleaved ? rope_kernel<true> : rope_kernel<false>;
	kernel<<<dim3( 1, count ), dim3( rot_dim / 2 ), 0, stream>>>(
		q.ptr<half>(),
		k.ptr<half>(),
		num_q_heads,
		num_kv_heads,
		head_dim,
		rot_dim,
		d_rows,
		freq_base
	);
}

} // namespace engine::modules
