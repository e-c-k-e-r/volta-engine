#include "engine/modules/gdn.h"
#include <cuda_fp16.h>
#include <cmath>

namespace engine::modules {

namespace {

__device__ __forceinline__ float warp_sum( float value ) {
	#pragma unroll
	for ( int offset = 16; offset > 0; offset >>= 1 ) {
		value += __shfl_xor_sync( 0xffffffff, value, offset, 32 );
	}
	return value;
}

__global__ void project_gates_kernel(
	const half* __restrict__ norm_hidden,
	const half* __restrict__ in_proj_a,
	const half* __restrict__ in_proj_b,
	const half* __restrict__ A_log,
	const half* __restrict__ dt_bias,
	float* __restrict__ gate_out,
	float* __restrict__ beta_out,
	int num_v_heads,
	int hidden_size
) {
	int head = blockIdx.x;
	int tid = threadIdx.x;
	if ( head >= num_v_heads ) return;

	const half* a_row = in_proj_a + head * hidden_size;
	const half* b_row = in_proj_b + head * hidden_size;

	float dot_a = 0.0f;
	float dot_b = 0.0f;

	for ( int i = tid; i < hidden_size; i += blockDim.x ) {
		float x_val = __half2float( norm_hidden[i] );
		dot_a = fmaf( __half2float( a_row[i] ), x_val, dot_a );
		dot_b = fmaf( __half2float( b_row[i] ), x_val, dot_b );
	}

	dot_a = warp_sum( dot_a );
	dot_b = warp_sum( dot_b );

	__shared__ float s_a[8];
	__shared__ float s_b[8];
	int warp_id = tid >> 5;
	int lane_id = tid & 31;

	if ( lane_id == 0 ) {
		s_a[warp_id] = dot_a;
		s_b[warp_id] = dot_b;
	}
	__syncthreads();

	if ( warp_id == 0 ) {
		int num_warps = ( blockDim.x + 31 ) >> 5;
		float total_a = ( lane_id < num_warps ) ? s_a[lane_id] : 0.0f;
		float total_b = ( lane_id < num_warps ) ? s_b[lane_id] : 0.0f;

		#pragma unroll
		for ( int mask = 4; mask > 0; mask >>= 1 ) {
			total_a += __shfl_xor_sync( 0xffffffff, total_a, mask );
			total_b += __shfl_xor_sync( 0xffffffff, total_b, mask );
		}

		if ( lane_id == 0 ) {
			float beta_val = 1.0f / ( 1.0f + __expf( -total_b ) );
			float dt_val = ( dt_bias != nullptr ) ? __half2float( dt_bias[head] ) : 0.0f;
			float a_val = ( A_log != nullptr ) ? __half2float( A_log[head] ) : -2.0f;

			float val = total_a + dt_val;
			float sp = ( val > 20.0f ) ? val : log1pf( expf( val ) );
			float gate_val = -expf( a_val ) * sp;

			gate_out[head] = gate_val;
			beta_out[head] = beta_val;
		}
	}
}

__global__ void causal_conv1d_silu_kernel(
	half* __restrict__ qkv,
	float* __restrict__ history,
	const half* __restrict__ weights,
	int channels
) {
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if ( c >= channels ) return;

	float v0 = history[c * 3 + 0];
	float v1 = history[c * 3 + 1];
	float v2 = history[c * 3 + 2];
	float in_val = __half2float( qkv[c] );

	float w0 = ( weights != nullptr ) ? __half2float( weights[c * 4 + 0] ) : 0.0f;
	float w1 = ( weights != nullptr ) ? __half2float( weights[c * 4 + 1] ) : 0.0f;
	float w2 = ( weights != nullptr ) ? __half2float( weights[c * 4 + 2] ) : 0.0f;
	float w3 = ( weights != nullptr ) ? __half2float( weights[c * 4 + 3] ) : 1.0f;

	float sum = v0 * w0 + v1 * w1 + v2 * w2 + in_val * w3;
	float silu = sum / ( 1.0f + __expf( -sum ) );

	qkv[c] = __float2half_rn( silu );

	history[c * 3 + 0] = v1;
	history[c * 3 + 1] = v2;
	history[c * 3 + 2] = in_val;
}

__global__ void l2_norm_heads_kernel(
	half* __restrict__ qkv,
	int num_k_heads,
	int head_dim
) {
	int block = blockIdx.x; // 0 .. num_k_heads * 2 - 1
	int tid = threadIdx.x;
	if ( tid >= head_dim ) return;

	half* ptr = qkv + block * head_dim;
	float val = __half2float( ptr[tid] );
	float sq = val * val;

	#pragma unroll
	for ( int mask = 16; mask > 0; mask >>= 1 ) {
		sq += __shfl_xor_sync( 0xffffffff, sq, mask );
	}

	__shared__ float s_warp_sums[4];
	int warp_id = tid >> 5;
	int lane_id = tid & 31;
	if ( lane_id == 0 ) {
		s_warp_sums[warp_id] = sq;
	}
	__syncthreads();

	if ( warp_id == 0 ) {
		float b_sum = ( lane_id < 4 ) ? s_warp_sums[lane_id] : 0.0f;
		#pragma unroll
		for ( int mask = 2; mask > 0; mask >>= 1 ) {
			b_sum += __shfl_xor_sync( 0xffffffff, b_sum, mask );
		}
		if ( lane_id == 0 ) {
			s_warp_sums[0] = rsqrtf( b_sum + 1e-6f );
		}
	}
	__syncthreads();

	float inv_l2 = s_warp_sums[0];
	ptr[tid] = __float2half_rn( val * inv_l2 );
}

__global__ void __launch_bounds__( 128, 2 )
recurrent_step_kernel(
	float* __restrict__ state,
	const half* __restrict__ qkv,
	const float* __restrict__ gate,
	const float* __restrict__ beta,
	half* __restrict__ output,
	int h_k,
	int h_v,
	float scale
) {
	const int head = blockIdx.x;
	const int lane = threadIdx.x;
	const int col = blockIdx.z * blockDim.y + threadIdx.y;
	const int q_head = head / ( h_v / h_k );

	const half* q_ptr = qkv + q_head * 128;
	const half* k_ptr = qkv + h_k * 128 + q_head * 128;
	const half* v_ptr = qkv + 2 * h_k * 128 + head * 128;

	float s_shard[4], k_reg[4], q_reg[4];
	#pragma unroll
	for ( int r = 0; r < 4; ++r ) {
		const int i = r * 32 + lane;
		s_shard[r] = state[( size_t( head ) * 128 + i ) * 128 + col];
		k_reg[r] = __half2float( k_ptr[i] );
		q_reg[r] = __half2float( q_ptr[i] );
	}

	const float g_val = expf( gate[head] );
	float kv_shard = 0.0f;
	#pragma unroll
	for ( int r = 0; r < 4; ++r ) {
		kv_shard += s_shard[r] * k_reg[r];
	}
	const float kv_col = warp_sum( kv_shard );
	const float v_val = __half2float( v_ptr[col] );
	const float delta_col = ( v_val - g_val * kv_col ) * beta[head];

	float attn_partial = 0.0f;
	#pragma unroll
	for ( int r = 0; r < 4; ++r ) {
		s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
		attn_partial += s_shard[r] * q_reg[r];
	}
	const float attn_col = warp_sum( attn_partial );

	if ( lane == 0 ) {
		output[head * 128 + col] = __float2half_rn( attn_col * scale );
	}

	#pragma unroll
	for ( int r = 0; r < 4; ++r ) {
		const int i = r * 32 + lane;
		state[( size_t( head ) * 128 + i ) * 128 + col] = s_shard[r];
	}
}

__global__ void gated_rmsnorm_kernel(
	half* __restrict__ attn_out,
	const half* __restrict__ z,
	const half* __restrict__ norm_weight,
	int num_v_heads,
	int head_dim
) {
	int head = blockIdx.x;
	int tid = threadIdx.x;
	if ( head >= num_v_heads || tid >= head_dim ) return;

	half* head_out = attn_out + head * head_dim;
	float val = __half2float( head_out[tid] );
	float sq = val * val;

	#pragma unroll
	for ( int mask = 16; mask > 0; mask >>= 1 ) {
		sq += __shfl_xor_sync( 0xffffffff, sq, mask );
	}

	__shared__ float s_warp_sums[4];
	int warp_id = tid >> 5;
	int lane_id = tid & 31;
	if ( lane_id == 0 ) {
		s_warp_sums[warp_id] = sq;
	}
	__syncthreads();

	if ( warp_id == 0 ) {
		float b_sum = ( lane_id < 4 ) ? s_warp_sums[lane_id] : 0.0f;
		#pragma unroll
		for ( int mask = 2; mask > 0; mask >>= 1 ) {
			b_sum += __shfl_xor_sync( 0xffffffff, b_sum, mask );
		}
		if ( lane_id == 0 ) {
			s_warp_sums[0] = rsqrtf( b_sum / static_cast<float>( head_dim ) + 1e-6f );
		}
	}
	__syncthreads();

	float inv_rms = s_warp_sums[0];
	float gamma = ( norm_weight != nullptr ) ? __half2float( norm_weight[tid] ) : 1.0f;
	float z_val = ( z != nullptr ) ? __half2float( z[head * head_dim + tid] ) : 0.0f;
	float silu_z = z_val / ( 1.0f + __expf( -z_val ) );

	head_out[tid] = __float2half_rn( val * inv_rms * gamma * silu_z );
}

} // namespace

void GDN::project_gates(
	const pod::TensorView& norm_hidden,
	const pod::TensorView& in_proj_a,
	const pod::TensorView& in_proj_b,
	const pod::TensorView& A_log,
	const pod::TensorView& dt_bias,
	pod::TensorView& gate_out,
	pod::TensorView& beta_out,
	int num_v_heads,
	int hidden_size,
	cudaStream_t stream
) {
	project_gates_kernel<<<dim3( num_v_heads ), dim3( 256 ), 0, stream>>>(
		norm_hidden.ptr<const half>(),
		in_proj_a.ptr<const half>(),
		in_proj_b.ptr<const half>(),
		A_log.ptr<const half>(),
		dt_bias.ptr<const half>(),
		gate_out.ptr<float>(),
		beta_out.ptr<float>(),
		num_v_heads,
		hidden_size
	);
}

void GDN::causal_conv1d_silu(
	pod::TensorView& qkv,
	pod::TensorView& conv_history,
	const pod::TensorView& conv_weights,
	int channels,
	cudaStream_t stream
) {
	int threads = 256;
	int blocks = ( channels + threads - 1 ) / threads;
	causal_conv1d_silu_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
		qkv.ptr<half>(),
		conv_history.ptr<float>(),
		conv_weights.ptr<const half>(),
		channels
	);
}

void GDN::l2_norm_heads(
	pod::TensorView& qkv,
	int num_k_heads,
	int head_dim,
	cudaStream_t stream
) {
	int total_blocks = num_k_heads * 2;
	l2_norm_heads_kernel<<<dim3( total_blocks ), dim3( head_dim ), 0, stream>>>(
		qkv.ptr<half>(),
		num_k_heads,
		head_dim
	);
}

void GDN::recurrent_step(
	float* gdn_state,
	const pod::TensorView& qkv,
	const pod::TensorView& gate,
	const pod::TensorView& beta,
	pod::TensorView& attn_out,
	int num_k_heads,
	int num_v_heads,
	int head_dim,
	cudaStream_t stream
) {
	float scale = 1.0f / sqrtf( static_cast<float>( head_dim ) );
	recurrent_step_kernel<<<dim3( num_v_heads, 1, head_dim / 4 ), dim3( 32, 4 ), 0, stream>>>(
		gdn_state,
		qkv.ptr<const half>(),
		gate.ptr<const float>(),
		beta.ptr<const float>(),
		attn_out.ptr<half>(),
		num_k_heads,
		num_v_heads,
		scale
	);
}

void GDN::gated_rmsnorm(
	pod::TensorView& attn_out,
	const pod::TensorView& z,
	const pod::TensorView& norm_weight,
	int num_v_heads,
	int head_dim,
	cudaStream_t stream
) {
	gated_rmsnorm_kernel<<<dim3( num_v_heads ), dim3( head_dim ), 0, stream>>>(
		attn_out.ptr<half>(),
		z.ptr<const half>(),
		norm_weight.ptr<const half>(),
		num_v_heads,
		head_dim
	);
}

} // namespace engine::modules
