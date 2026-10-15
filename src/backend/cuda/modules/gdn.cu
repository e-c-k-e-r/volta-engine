#include "engine/modules/gdn.h"
#include "engine/backend/cuda/reduce.cuh"
#include <cuda_fp16.h>
#include <cmath>

namespace engine::modules {

namespace {

using engine::math::warp_sum;

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
	// Rows ride blockIdx.y: row r reads norm_hidden[r] and writes gate/beta[r].
	norm_hidden += static_cast<size_t>( blockIdx.y ) * hidden_size;
	gate_out += static_cast<size_t>( blockIdx.y ) * num_v_heads;
	beta_out += static_cast<size_t>( blockIdx.y ) * num_v_heads;

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
	int channels,
	const pod::SlotRef slot,
	float* __restrict__ snap,
	const int* __restrict__ snap_on
) {
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if ( c >= channels ) return;
	history = slot.apply( history );

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

	// Speculative verify: also publish the post-row history so the accepted prefix can be committed by index.
	if ( snap && *snap_on ) {
		snap[c * 3 + 0] = v1;
		snap[c * 3 + 1] = v2;
		snap[c * 3 + 2] = in_val;
	}
}

// Multi-row conv: one thread per channel walks the rows in order, keeping the 3-tap history in
// registers across consecutive rows of the same slot (same arithmetic as the single-row kernel).
__global__ void causal_conv1d_silu_rows_kernel(
	half* __restrict__ qkv,
	float* __restrict__ history,
	const half* __restrict__ weights,
	int channels,
	int rows,
	const pod::BatchRows* __restrict__ d_rows,
	size_t slot_stride,
	float* __restrict__ snap,
	size_t snap_stride,
	int snap_rows,
	const int* __restrict__ snap_on
) {
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if ( c >= channels ) return;

	float w0 = ( weights != nullptr ) ? __half2float( weights[c * 4 + 0] ) : 0.0f;
	float w1 = ( weights != nullptr ) ? __half2float( weights[c * 4 + 1] ) : 0.0f;
	float w2 = ( weights != nullptr ) ? __half2float( weights[c * 4 + 2] ) : 0.0f;
	float w3 = ( weights != nullptr ) ? __half2float( weights[c * 4 + 3] ) : 1.0f;
	const bool do_snap = snap && *snap_on;

	int cur = -1;
	float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f;
	for ( int m = 0; m < rows; ++m ) {
		const int slot = d_rows->slot[m];
		if ( slot != cur ) {
			if ( cur >= 0 ) {
				float* h = history + static_cast<size_t>( cur ) * slot_stride;
				h[c * 3 + 0] = v0;
				h[c * 3 + 1] = v1;
				h[c * 3 + 2] = v2;
			}
			const float* h = history + static_cast<size_t>( slot ) * slot_stride;
			v0 = h[c * 3 + 0];
			v1 = h[c * 3 + 1];
			v2 = h[c * 3 + 2];
			cur = slot;
		}
		float in_val = __half2float( qkv[static_cast<size_t>( m ) * channels + c] );
		float sum = v0 * w0 + v1 * w1 + v2 * w2 + in_val * w3;
		float silu = sum / ( 1.0f + __expf( -sum ) );
		qkv[static_cast<size_t>( m ) * channels + c] = __float2half_rn( silu );
		v0 = v1;
		v1 = v2;
		v2 = in_val;
		if ( do_snap && m < snap_rows ) {
			float* sp = snap + m * snap_stride;
			sp[c * 3 + 0] = v0;
			sp[c * 3 + 1] = v1;
			sp[c * 3 + 2] = v2;
		}
	}
	float* h = history + static_cast<size_t>( cur ) * slot_stride;
	h[c * 3 + 0] = v0;
	h[c * 3 + 1] = v1;
	h[c * 3 + 2] = v2;
}

// Prefill conv: the 4-tap FIR is over *inputs*, so rows are independent given the previous three inputs. Each thread
// takes one channel and a tile of rows, reads from src and writes to dst (distinct buffers: a tile reads the three
// rows before its own, which a neighbouring tile would otherwise already have overwritten). One slot per chunk.
#define CONV_TILE_ROWS 16
__global__ void causal_conv1d_silu_tiled_kernel(
	const half* __restrict__ src,
	half* __restrict__ dst,
	const float* __restrict__ history,
	const half* __restrict__ weights,
	int channels,
	int rows,
	const pod::BatchRows* __restrict__ d_rows,
	size_t slot_stride
) {
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	if ( c >= channels ) return;
	const int r0 = blockIdx.y * CONV_TILE_ROWS;
	const int r1 = min( rows, r0 + CONV_TILE_ROWS );
	const float* h = history + static_cast<size_t>( d_rows->slot[0] ) * slot_stride + c * 3;

	const float w0 = ( weights != nullptr ) ? __half2float( weights[c * 4 + 0] ) : 0.0f;
	const float w1 = ( weights != nullptr ) ? __half2float( weights[c * 4 + 1] ) : 0.0f;
	const float w2 = ( weights != nullptr ) ? __half2float( weights[c * 4 + 2] ) : 0.0f;
	const float w3 = ( weights != nullptr ) ? __half2float( weights[c * 4 + 3] ) : 1.0f;

	// x[t - 3], x[t - 2], x[t - 1] for t = r0; indices below zero come from the saved history (oldest first).
	float v[3];
	#pragma unroll
	for ( int k = 0; k < 3; ++k ) {
		const int idx = r0 - 3 + k;
		v[k] = ( idx >= 0 ) ? __half2float( src[static_cast<size_t>( idx ) * channels + c] ) : h[3 + idx];
	}
	float v0 = v[0], v1 = v[1], v2 = v[2];
	for ( int t = r0; t < r1; ++t ) {
		const float in_val = __half2float( src[static_cast<size_t>( t ) * channels + c] );
		const float sum = v0 * w0 + v1 * w1 + v2 * w2 + in_val * w3;
		dst[static_cast<size_t>( t ) * channels + c] = __float2half_rn( sum / ( 1.0f + __expf( -sum ) ) );
		v0 = v1;
		v1 = v2;
		v2 = in_val;
	}
}

// After the tiles: the history for the next chunk is the last three inputs.
__global__ void causal_conv1d_history_kernel(
	const half* __restrict__ src,
	float* __restrict__ history,
	int channels,
	int rows,
	const pod::BatchRows* __restrict__ d_rows,
	size_t slot_stride
) {
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	if ( c >= channels ) return;
	float* h = history + static_cast<size_t>( d_rows->slot[0] ) * slot_stride + c * 3;
	float nv[3];
	#pragma unroll
	for ( int k = 0; k < 3; ++k ) {
		const int idx = rows - 3 + k;
		nv[k] = ( idx >= 0 ) ? __half2float( src[static_cast<size_t>( idx ) * channels + c] ) : h[3 + idx];
	}
	h[0] = nv[0];
	h[1] = nv[1];
	h[2] = nv[2];
}

__global__ void l2_norm_heads_kernel(
	half* __restrict__ qkv,
	int num_k_heads,
	int head_dim,
	bool scale_q,
	size_t row_stride
) {
	int block = blockIdx.x; // 0 .. num_k_heads * 2 - 1
	int tid = threadIdx.x;
	if ( tid >= head_dim ) return;

	half* ptr = qkv + blockIdx.y * row_stride + block * head_dim;
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
	if ( scale_q && block < num_k_heads ) {
		inv_l2 *= 1.0f / sqrtf( static_cast<float>( head_dim ) );
	}
	ptr[tid] = __float2half_rn( val * inv_l2 );
}

// GDN state layout: [v_head][col][row], row fastest. A warp's lanes walk the row index, so this makes the state
// reads, writes and snapshot writes fully coalesced (the [head][row][col] layout touched 32 sectors for 128 bytes).
__global__ void __launch_bounds__( 128, 2 )
recurrent_step_kernel(
	float* __restrict__ state,
	const half* __restrict__ qkv,
	const float* __restrict__ gate,
	const float* __restrict__ beta,
	half* __restrict__ output,
	int h_k,
	int h_v,
	float scale,
	const pod::SlotRef slot,
	float* __restrict__ snap,
	const int* __restrict__ snap_on
) {
	state = slot.apply( state );
	const bool do_snap = snap && *snap_on;
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
		s_shard[r] = state[( size_t( head ) * 128 + col ) * 128 + i];
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
		state[( size_t( head ) * 128 + col ) * 128 + i] = s_shard[r];
		if ( do_snap ) {
			snap[( size_t( head ) * 128 + col ) * 128 + i] = s_shard[r];
		}
	}
}

// Multi-row recurrence: the 128x128 per-head state stays in registers across consecutive rows of the
// same slot and touches memory once per slot run instead of once per row. Per-row arithmetic is that of
// recurrent_step_kernel.
__global__ void __launch_bounds__( 128, 2 )
recurrent_step_rows_kernel(
	float* __restrict__ state,
	const half* __restrict__ qkv,
	const float* __restrict__ gate,
	const float* __restrict__ beta,
	half* __restrict__ output,
	int h_k,
	int h_v,
	float scale,
	int rows,
	const pod::BatchRows* __restrict__ d_rows,
	size_t slot_stride,
	float* __restrict__ snap,
	size_t snap_stride,
	int snap_rows,
	const int* __restrict__ snap_on
) {
	const int head = blockIdx.x;
	const int lane = threadIdx.x;
	const int col = blockIdx.z * blockDim.y + threadIdx.y;
	const int q_head = head / ( h_v / h_k );
	const size_t qkv_stride = static_cast<size_t>( 2 * h_k + h_v ) * 128;
	const bool do_snap = snap && *snap_on;

	float s_shard[4] = {};
	int cur = -1;
	for ( int m = 0; m < rows; ++m ) {
		const int slot = d_rows->slot[m];
		if ( slot != cur ) {
			if ( cur >= 0 ) {
				float* st = state + static_cast<size_t>( cur ) * slot_stride;
				#pragma unroll
				for ( int r = 0; r < 4; ++r ) {
					st[( size_t( head ) * 128 + col ) * 128 + r * 32 + lane] = s_shard[r];
				}
			}
			const float* st = state + static_cast<size_t>( slot ) * slot_stride;
			#pragma unroll
			for ( int r = 0; r < 4; ++r ) {
				s_shard[r] = st[( size_t( head ) * 128 + col ) * 128 + r * 32 + lane];
			}
			cur = slot;
		}

		const half* q_ptr = qkv + m * qkv_stride + q_head * 128;
		const half* k_ptr = qkv + m * qkv_stride + h_k * 128 + q_head * 128;
		const half* v_ptr = qkv + m * qkv_stride + 2 * h_k * 128 + head * 128;
		float k_reg[4], q_reg[4];
		#pragma unroll
		for ( int r = 0; r < 4; ++r ) {
			const int i = r * 32 + lane;
			k_reg[r] = __half2float( k_ptr[i] );
			q_reg[r] = __half2float( q_ptr[i] );
		}

		const float g_val = expf( gate[m * h_v + head] );
		float kv_shard = 0.0f;
		#pragma unroll
		for ( int r = 0; r < 4; ++r ) {
			kv_shard += s_shard[r] * k_reg[r];
		}
		const float kv_col = warp_sum( kv_shard );
		const float v_val = __half2float( v_ptr[col] );
		const float delta_col = ( v_val - g_val * kv_col ) * beta[m * h_v + head];

		float attn_partial = 0.0f;
		#pragma unroll
		for ( int r = 0; r < 4; ++r ) {
			s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
			attn_partial += s_shard[r] * q_reg[r];
		}
		const float attn_col = warp_sum( attn_partial );

		if ( lane == 0 ) {
			output[m * static_cast<size_t>( h_v ) * 128 + head * 128 + col] = __float2half_rn( attn_col * scale );
		}
		if ( do_snap && m < snap_rows ) {
			float* sp = snap + m * snap_stride;
			#pragma unroll
			for ( int r = 0; r < 4; ++r ) {
				sp[( size_t( head ) * 128 + col ) * 128 + r * 32 + lane] = s_shard[r];
			}
		}
	}
	float* st = state + static_cast<size_t>( cur ) * slot_stride;
	#pragma unroll
	for ( int r = 0; r < 4; ++r ) {
		st[( size_t( head ) * 128 + col ) * 128 + r * 32 + lane] = s_shard[r];
	}
}

__global__ void gated_rmsnorm_kernel(
	half* __restrict__ attn_out,
	const half* __restrict__ z,
	const half* __restrict__ norm_weight,
	int num_v_heads,
	int head_dim,
	bool use_sigmoid
) {
	int head = blockIdx.x;
	int tid = threadIdx.x;
	if ( head >= num_v_heads || tid >= head_dim ) return;
	// Rows ride blockIdx.y with a packed [rows, num_v_heads * head_dim] layout for attn_out and z.
	const size_t row_off = static_cast<size_t>( blockIdx.y ) * num_v_heads * head_dim;
	attn_out += row_off;
	if ( z != nullptr ) z += row_off;

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
	float gate_val;
	if ( use_sigmoid ) {
		gate_val = 1.0f / ( 1.0f + __expf( -z_val ) );
	}
	else {
		gate_val = z_val / ( 1.0f + __expf( -z_val ) );
	}

	head_out[tid] = __float2half_rn( val * inv_rms * gamma * gate_val );
}

__global__ void project_kda_stage1_kernel(
	const half* __restrict__ norm_hidden,
	const half* __restrict__ b_proj,
	const half* __restrict__ f_a_proj,
	const half* __restrict__ g_a_proj,
	float* __restrict__ beta_out,
	half* __restrict__ f_a_buf,
	half* __restrict__ g_a_buf,
	int hidden_size,
	int num_heads
) {
	int row = blockIdx.x;
	if ( row >= num_heads + 256 ) return;
	norm_hidden += static_cast<size_t>( blockIdx.y ) * hidden_size;
	beta_out += blockIdx.y * num_heads;
	f_a_buf += blockIdx.y * 128;
	g_a_buf += blockIdx.y * 128;

	const half* weight_row = nullptr;
	if ( row < num_heads ) {
		if ( b_proj == nullptr ) return;
		weight_row = b_proj + row * hidden_size;
	}
	else if ( row < num_heads + 128 ) {
		if ( f_a_proj == nullptr ) return;
		weight_row = f_a_proj + ( row - num_heads ) * hidden_size;
	}
	else {
		if ( g_a_proj == nullptr ) return;
		weight_row = g_a_proj + ( row - num_heads - 128 ) * hidden_size;
	}

	int tid = threadIdx.x;
	float sum = 0.0f;

	const float4* x4 = reinterpret_cast<const float4*>( norm_hidden );
	const float4* w4 = reinterpret_cast<const float4*>( weight_row );
	int num_float4 = hidden_size / 8; // 4096 / 8 = 512

	for ( int i = tid; i < num_float4; i += blockDim.x ) {
		float4 vx = x4[i];
		float4 vw = w4[i];
		const half2* hx = reinterpret_cast<const half2*>( &vx );
		const half2* hw = reinterpret_cast<const half2*>( &vw );
		#pragma unroll
		for ( int j = 0; j < 4; ++j ) {
			float2 fx = __half22float2( hx[j] );
			float2 fw = __half22float2( hw[j] );
			sum = fmaf( fx.x, fw.x, sum );
			sum = fmaf( fx.y, fw.y, sum );
		}
	}

	#pragma unroll
	for ( int mask = 16; mask > 0; mask >>= 1 ) {
		sum += __shfl_xor_sync( 0xffffffff, sum, mask );
	}

	__shared__ float s_warp_sums[8];
	int warp_id = tid >> 5;
	int lane_id = tid & 31;

	if ( lane_id == 0 ) {
		s_warp_sums[warp_id] = sum;
	}
	__syncthreads();

	if ( warp_id == 0 ) {
		float b_sum = ( lane_id < ( blockDim.x >> 5 ) ) ? s_warp_sums[lane_id] : 0.0f;
		#pragma unroll
		for ( int mask = 4; mask > 0; mask >>= 1 ) {
			b_sum += __shfl_xor_sync( 0xffffffff, b_sum, mask );
		}
		if ( lane_id == 0 ) {
			if ( row < num_heads ) {
				beta_out[row] = 1.0f / ( 1.0f + __expf( -b_sum ) );
			}
			else if ( row < num_heads + 128 ) {
				f_a_buf[row - num_heads] = __float2half_rn( b_sum );
			}
			else {
				g_a_buf[row - num_heads - 128] = __float2half_rn( b_sum );
			}
		}
	}
}

__global__ void project_kda_stage2_kernel(
	const half* __restrict__ f_a_buf,
	const half* __restrict__ g_a_buf,
	const half* __restrict__ f_b_proj,
	const half* __restrict__ g_b_proj,
	const half* __restrict__ A_log,
	const half* __restrict__ dt_bias,
	float* __restrict__ g_decay_out,
	half* __restrict__ z_out,
	int num_channels
) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if ( j >= num_channels ) return;
	f_a_buf += blockIdx.y * 128;
	g_a_buf += blockIdx.y * 128;
	g_decay_out += static_cast<size_t>( blockIdx.y ) * num_channels;
	z_out += static_cast<size_t>( blockIdx.y ) * num_channels;

	int h = j / 128;

	if ( f_b_proj != nullptr ) {
		const float4* fa4 = reinterpret_cast<const float4*>( f_a_buf );
		const float4* fb4 = reinterpret_cast<const float4*>( f_b_proj + j * 128 );
		float f_val = 0.0f;
		#pragma unroll
		for ( int i = 0; i < 16; ++i ) {
			float4 vfa = fa4[i];
			float4 vfb = fb4[i];
			const half2* hfa = reinterpret_cast<const half2*>( &vfa );
			const half2* hfb = reinterpret_cast<const half2*>( &vfb );
			#pragma unroll
			for ( int k = 0; k < 4; ++k ) {
				float2 f1 = __half22float2( hfa[k] );
				float2 f2 = __half22float2( hfb[k] );
				f_val = fmaf( f1.x, f2.x, f_val );
				f_val = fmaf( f1.y, f2.y, f_val );
			}
		}
		float dt = ( dt_bias != nullptr ) ? __half2float( dt_bias[j] ) : 0.0f;
		float gf = f_val + dt;
		float a_val = ( A_log != nullptr ) ? __half2float( A_log[h] ) : 0.0f;
		float decay = __expf( a_val );
		float g_val = -5.0f / ( 1.0f + __expf( -decay * gf ) );
		g_decay_out[j] = g_val;
	}

	if ( g_b_proj != nullptr ) {
		const float4* ga4 = reinterpret_cast<const float4*>( g_a_buf );
		const float4* gb4 = reinterpret_cast<const float4*>( g_b_proj + j * 128 );
		float z_val = 0.0f;
		#pragma unroll
		for ( int i = 0; i < 16; ++i ) {
			float4 vga = ga4[i];
			float4 vgb = gb4[i];
			const half2* hga = reinterpret_cast<const half2*>( &vga );
			const half2* hgb = reinterpret_cast<const half2*>( &vgb );
			#pragma unroll
			for ( int k = 0; k < 4; ++k ) {
				float2 g1 = __half22float2( hga[k] );
				float2 g2 = __half22float2( hgb[k] );
				z_val = fmaf( g1.x, g2.x, z_val );
				z_val = fmaf( g1.y, g2.y, z_val );
			}
		}
		z_out[j] = __float2half_rn( z_val );
	}
}

__global__ void kda_recurrent_step_kernel(
	float* __restrict__ state,
	const half* __restrict__ qkv,
	const float* __restrict__ g_decay,
	const float* __restrict__ beta,
	half* __restrict__ output,
	int num_heads,
	int head_dim
) {
	int h = blockIdx.x;
	int c = threadIdx.x;
	if ( h >= num_heads || c >= head_dim ) return;

	float* head_state = state + static_cast<size_t>( h ) * head_dim * head_dim;
	const half* q_ptr = qkv + h * head_dim;
	const half* k_ptr = qkv + num_heads * head_dim + h * head_dim;
	const half* v_ptr = qkv + 2 * num_heads * head_dim + h * head_dim;
	const float* head_g = g_decay + h * head_dim;
	float beta_h = beta[h];

	// 1. Compute kv_mem from S_{t-1} and decay state
	float kv_mem = 0.0f;
	#pragma unroll 4
	for ( int r = 0; r < head_dim; ++r ) {
		float s_prev = head_state[r * head_dim + c];
		kv_mem = fmaf( s_prev, __half2float( k_ptr[r] ), kv_mem );
		float g_exp = __expf( head_g[r] );
		head_state[r * head_dim + c] = s_prev * g_exp;
	}

	// 2. Compute delta for column c
	float v_val = __half2float( v_ptr[c] );
	float delta = ( v_val - kv_mem ) * beta_h;

	// 3. Update state and compute attention output for column c
	float attn_out = 0.0f;
	#pragma unroll 4
	for ( int r = 0; r < head_dim; ++r ) {
		float s = fmaf( __half2float( k_ptr[r] ), delta, head_state[r * head_dim + c] );
		head_state[r * head_dim + c] = s;
		attn_out = fmaf( s, __half2float( q_ptr[r] ), attn_out );
	}

	output[h * head_dim + c] = __float2half_rn( attn_out );
}

__global__ void kda_recurrent_step_and_norm_kernel(
	float* __restrict__ state,
	const half* __restrict__ qkv,
	const float* __restrict__ g_decay,
	const float* __restrict__ beta,
	const half* __restrict__ z,
	const half* __restrict__ o_norm,
	half* __restrict__ output,
	int num_heads,
	int head_dim,
	float eps,
	const pod::SlotRef slot
) {
	int h = blockIdx.x;
	int c = threadIdx.x;
	if ( h >= num_heads || c >= head_dim ) return;
	state = slot.apply( state );

	__shared__ float s_g_exp[128];
	__shared__ float s_k[128];
	__shared__ float s_q[128];

	float* head_state = state + static_cast<size_t>( h ) * head_dim * head_dim;
	const half* q_ptr = qkv + h * head_dim;
	const half* k_ptr = qkv + num_heads * head_dim + h * head_dim;
	const half* v_ptr = qkv + 2 * num_heads * head_dim + h * head_dim;
	const float* head_g = g_decay + h * head_dim;
	float beta_h = beta[h];

	s_g_exp[c] = __expf( head_g[c] );
	s_k[c] = __half2float( k_ptr[c] );
	s_q[c] = __half2float( q_ptr[c] );
	__syncthreads();

	// 1. Decay state and compute kv_mem for column c
	float kv_mem = 0.0f;
	#pragma unroll 4
	for ( int r = 0; r < head_dim; ++r ) {
		float g_exp = s_g_exp[r];
		float s = head_state[r * head_dim + c] * g_exp;
		head_state[r * head_dim + c] = s;
		kv_mem = fmaf( s, s_k[r], kv_mem );
	}

	// 2. Compute delta for column c: beta_t * (v_t - S_{t-1} k_t)
	float v_val = __half2float( v_ptr[c] );
	float delta = ( v_val - kv_mem ) * beta_h;

	// 3. Update state and compute attention output for column c: y = S_t * q_t
	float y_core = 0.0f;
	#pragma unroll 4
	for ( int r = 0; r < head_dim; ++r ) {
		float s = fmaf( s_k[r], delta, head_state[r * head_dim + c] );
		head_state[r * head_dim + c] = s;
		y_core = fmaf( s, s_q[r], y_core );
	}

	// 4. In-register Gated RMSNorm across 128 threads of this head
	float sq = y_core * y_core;
	#pragma unroll
	for ( int mask = 16; mask > 0; mask >>= 1 ) {
		sq += __shfl_xor_sync( 0xffffffff, sq, mask );
	}

	__shared__ float s_warp_sums[4];
	int warp_id = c >> 5;
	int lane_id = c & 31;
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
			s_warp_sums[0] = rsqrtf( b_sum / static_cast<float>( head_dim ) + eps );
		}
	}
	__syncthreads();

	float inv_rms = s_warp_sums[0];
	float gamma = ( o_norm != nullptr ) ? __half2float( o_norm[c] ) : 1.0f;
	float z_val = ( z != nullptr ) ? __half2float( z[h * head_dim + c] ) : 0.0f;
	float gate_val = 1.0f / ( 1.0f + __expf( -z_val ) );

	output[h * head_dim + c] = __float2half_rn( y_core * inv_rms * gamma * gate_val );
}


// Multi-row KDA step: thread c owns value column c of one head; its 128-element state column stays in registers
// across consecutive rows of one slot. Per-row arithmetic and summation order are those of
// kda_recurrent_step_and_norm_kernel (bit-identical to stepping the rows one at a time).
__global__ void __launch_bounds__( 128 )
kda_recurrent_rows_kernel(
	float* __restrict__ state,
	const half* __restrict__ qkv,
	const float* __restrict__ g_decay,
	const float* __restrict__ beta,
	const half* __restrict__ z,
	const half* __restrict__ o_norm,
	half* __restrict__ output,
	int num_heads,
	float eps,
	int rows,
	const pod::BatchRows* __restrict__ d_rows,
	size_t slot_stride,
	float* __restrict__ snap,
	size_t snap_stride,
	int snap_rows,
	const int* __restrict__ snap_on
) {
	const int h = blockIdx.x;
	const int c = threadIdx.x;
	const int channels = num_heads * 128;
	const bool do_snap = snap && *snap_on;

	__shared__ float s_g_exp[128];
	__shared__ float s_k[128];
	__shared__ float s_q[128];
	__shared__ float s_warp_sums[4];

	float s[128];
	int cur = -1;
	for ( int m = 0; m < rows; ++m ) {
		const int slot = d_rows->slot[m];
		if ( slot != cur ) {
			if ( cur >= 0 ) {
				float* st = state + static_cast<size_t>( cur ) * slot_stride + static_cast<size_t>( h ) * 128 * 128;
				#pragma unroll
				for ( int r = 0; r < 128; ++r ) st[r * 128 + c] = s[r];
			}
			const float* st = state + static_cast<size_t>( slot ) * slot_stride + static_cast<size_t>( h ) * 128 * 128;
			#pragma unroll
			for ( int r = 0; r < 128; ++r ) s[r] = st[r * 128 + c];
			cur = slot;
		}

		const half* q_ptr = qkv + static_cast<size_t>( m ) * 3 * channels + h * 128;
		const half* k_ptr = q_ptr + channels;
		const half* v_ptr = q_ptr + 2 * channels;
		s_g_exp[c] = __expf( g_decay[static_cast<size_t>( m ) * channels + h * 128 + c] );
		s_k[c] = __half2float( k_ptr[c] );
		s_q[c] = __half2float( q_ptr[c] );
		__syncthreads();

		float kv_mem = 0.0f;
		#pragma unroll
		for ( int r = 0; r < 128; ++r ) {
			s[r] = s[r] * s_g_exp[r];
			kv_mem = fmaf( s[r], s_k[r], kv_mem );
		}
		const float delta = ( __half2float( v_ptr[c] ) - kv_mem ) * beta[m * num_heads + h];

		float y_core = 0.0f;
		#pragma unroll
		for ( int r = 0; r < 128; ++r ) {
			s[r] = fmaf( s_k[r], delta, s[r] );
			y_core = fmaf( s[r], s_q[r], y_core );
		}

		if ( do_snap && m < snap_rows ) {
			float* sp = snap + m * snap_stride + static_cast<size_t>( h ) * 128 * 128;
			#pragma unroll
			for ( int r = 0; r < 128; ++r ) sp[r * 128 + c] = s[r];
		}

		float sq = y_core * y_core;
		#pragma unroll
		for ( int mask = 16; mask > 0; mask >>= 1 ) {
			sq += __shfl_xor_sync( 0xffffffff, sq, mask );
		}
		const int warp_id = c >> 5;
		const int lane_id = c & 31;
		if ( lane_id == 0 ) s_warp_sums[warp_id] = sq;
		__syncthreads();
		if ( warp_id == 0 ) {
			float b_sum = ( lane_id < 4 ) ? s_warp_sums[lane_id] : 0.0f;
			#pragma unroll
			for ( int mask = 2; mask > 0; mask >>= 1 ) {
				b_sum += __shfl_xor_sync( 0xffffffff, b_sum, mask );
			}
			if ( lane_id == 0 ) s_warp_sums[0] = rsqrtf( b_sum / 128.0f + eps );
		}
		__syncthreads();

		const float inv_rms = s_warp_sums[0];
		const float gamma = ( o_norm != nullptr ) ? __half2float( o_norm[c] ) : 1.0f;
		const float z_val = ( z != nullptr ) ? __half2float( z[static_cast<size_t>( m ) * channels + h * 128 + c] ) : 0.0f;
		const float gate_val = 1.0f / ( 1.0f + __expf( -z_val ) );
		output[static_cast<size_t>( m ) * channels + h * 128 + c] = __float2half_rn( y_core * inv_rms * gamma * gate_val );
		__syncthreads();
	}
	float* st = state + static_cast<size_t>( cur ) * slot_stride + static_cast<size_t>( h ) * 128 * 128;
	#pragma unroll
	for ( int r = 0; r < 128; ++r ) st[r * 128 + c] = s[r];
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
	cudaStream_t stream,
	int rows
) {
	project_gates_kernel<<<dim3( num_v_heads, rows ), dim3( 256 ), 0, stream>>>(
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
	cudaStream_t stream,
	const pod::SlotRef& slot,
	float* snap,
	const int* snap_on
) {
	int threads = 256;
	int blocks = ( channels + threads - 1 ) / threads;
	causal_conv1d_silu_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
		qkv.ptr<half>(),
		conv_history.ptr<float>(),
		conv_weights.ptr<const half>(),
		channels,
		slot,
		snap,
		snap_on
	);
}

void GDN::l2_norm_heads(
	pod::TensorView& qkv,
	int num_k_heads,
	int head_dim,
	cudaStream_t stream,
	bool scale_q,
	int rows,
	size_t row_stride
) {
	int total_blocks = num_k_heads * 2;
	l2_norm_heads_kernel<<<dim3( total_blocks, rows ), dim3( head_dim ), 0, stream>>>(
		qkv.ptr<half>(),
		num_k_heads,
		head_dim,
		scale_q,
		row_stride
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
	cudaStream_t stream,
	const pod::SlotRef& slot,
	float* snap,
	const int* snap_on
) {
	float scale = 1.0f / sqrtf( static_cast<float>( head_dim ) );
	const int m_tokens = ( qkv.ndim >= 1 && qkv.shape[0] > 0 ) ? qkv.shape[0] : 1;

	if ( m_tokens == 1 ) {
		recurrent_step_kernel<<<dim3( num_v_heads, 1, head_dim / 4 ), dim3( 32, 4 ), 0, stream>>>(
			gdn_state,
			qkv.ptr<const half>(),
			gate.ptr<const float>(),
			beta.ptr<const float>(),
			attn_out.ptr<half>(),
			num_k_heads,
			num_v_heads,
			scale,
			slot,
			snap,
			snap_on
		);
	}
	else {
		size_t qkv_stride = ( 2 * num_k_heads + num_v_heads ) * head_dim;
		size_t gate_stride = num_v_heads;
		size_t beta_stride = num_v_heads;
		size_t out_stride = num_v_heads * head_dim;

		for ( int m = 0; m < m_tokens; ++m ) {
			const half* qkv_m = qkv.ptr<const half>() + m * qkv_stride;
			const float* gate_m = gate.ptr<const float>() + m * gate_stride;
			const float* beta_m = beta.ptr<const float>() + m * beta_stride;
			half* out_m = attn_out.ptr<half>() + m * out_stride;
			pod::SlotRef slot_m = slot;
			slot_m.row = slot.row + m;

			recurrent_step_kernel<<<dim3( num_v_heads, 1, head_dim / 4 ), dim3( 32, 4 ), 0, stream>>>(
				gdn_state,
				qkv_m,
				gate_m,
				beta_m,
				out_m,
				num_k_heads,
				num_v_heads,
				scale,
				slot_m,
				snap,
				snap_on
			);
		}
	}
}

void GDN::causal_conv1d_silu_rows(
	pod::TensorView& qkv,
	float* history,
	const pod::TensorView& conv_weights,
	int channels,
	int rows,
	const pod::BatchRows* d_rows,
	size_t slot_stride,
	float* snap,
	size_t snap_stride,
	int snap_rows,
	const int* snap_on,
	cudaStream_t stream
) {
	int threads = 256;
	int blocks = ( channels + threads - 1 ) / threads;
	causal_conv1d_silu_rows_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
		qkv.ptr<half>(), history, conv_weights.ptr<const half>(), channels, rows, d_rows, slot_stride,
		snap, snap_stride, snap_rows, snap_on
	);
}

void GDN::recurrent_step_rows(
	float* gdn_state,
	const pod::TensorView& qkv,
	const pod::TensorView& gate,
	const pod::TensorView& beta,
	pod::TensorView& attn_out,
	int num_k_heads,
	int num_v_heads,
	int head_dim,
	int rows,
	const pod::BatchRows* d_rows,
	size_t slot_stride,
	float* snap,
	size_t snap_stride,
	int snap_rows,
	const int* snap_on,
	cudaStream_t stream
) {
	const float scale = 1.0f / sqrtf( static_cast<float>( head_dim ) );
	recurrent_step_rows_kernel<<<dim3( num_v_heads, 1, head_dim / 4 ), dim3( 32, 4 ), 0, stream>>>(
		gdn_state, qkv.ptr<const half>(), gate.ptr<const float>(), beta.ptr<const float>(), attn_out.ptr<half>(),
		num_k_heads, num_v_heads, scale, rows, d_rows, slot_stride, snap, snap_stride, snap_rows, snap_on
	);
}

void GDN::causal_conv1d_silu_tiled(
	const pod::TensorView& src,
	pod::TensorView& dst,
	float* history,
	const pod::TensorView& conv_weights,
	int channels,
	int rows,
	const pod::BatchRows* d_rows,
	size_t slot_stride,
	cudaStream_t stream
) {
	const int threads = 128;
	const int blocks = ( channels + threads - 1 ) / threads;
	const int tiles = ( rows + CONV_TILE_ROWS - 1 ) / CONV_TILE_ROWS;
	causal_conv1d_silu_tiled_kernel<<<dim3( blocks, tiles ), dim3( threads ), 0, stream>>>(
		src.ptr<const half>(), dst.ptr<half>(), history, conv_weights.ptr<const half>(), channels, rows, d_rows, slot_stride );
	causal_conv1d_history_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
		src.ptr<const half>(), history, channels, rows, d_rows, slot_stride );
}

void GDN::project_kda_gates(
	const pod::TensorView& norm_hidden,
	const pod::TensorView& b_proj,
	const pod::TensorView& f_a_proj,
	const pod::TensorView& f_b_proj,
	const pod::TensorView& g_a_proj,
	const pod::TensorView& g_b_proj,
	const pod::TensorView& A_log,
	const pod::TensorView& dt_bias,
	pod::TensorView& beta_out,
	pod::TensorView& g_decay_out,
	pod::TensorView& z_out,
	pod::TensorView& f_a_buf,
	pod::TensorView& g_a_buf,
	int num_heads,
	int head_dim,
	int hidden_size,
	cudaStream_t stream,
	int rows
) {
	(void) head_dim;
	project_kda_stage1_kernel<<<dim3( num_heads + 256, rows ), dim3( 256 ), 0, stream>>>(
		norm_hidden.ptr<const half>(),
		b_proj.ptr<const half>(),
		f_a_proj.ptr<const half>(),
		g_a_proj.ptr<const half>(),
		beta_out.ptr<float>(),
		f_a_buf.ptr<half>(),
		g_a_buf.ptr<half>(),
		hidden_size,
		num_heads
	);

	int num_channels = num_heads * head_dim; // 2048
	int threads = 256;
	int blocks = ( num_channels + threads - 1 ) / threads;
	project_kda_stage2_kernel<<<dim3( blocks, rows ), dim3( threads ), 0, stream>>>(
		f_a_buf.ptr<const half>(),
		g_a_buf.ptr<const half>(),
		f_b_proj.ptr<const half>(),
		g_b_proj.ptr<const half>(),
		A_log.ptr<const half>(),
		dt_bias.ptr<const half>(),
		g_decay_out.ptr<float>(),
		z_out.ptr<half>(),
		num_channels
	);
}

void GDN::recurrent_step_kda(
	float* gdn_state,
	const pod::TensorView& qkv,
	const pod::TensorView& g_decay,
	const pod::TensorView& beta,
	pod::TensorView& attn_out,
	int num_heads,
	int head_dim,
	cudaStream_t stream
) {
	kda_recurrent_step_kernel<<<dim3( num_heads ), dim3( head_dim ), 0, stream>>>(
		gdn_state,
		qkv.ptr<const half>(),
		g_decay.ptr<const float>(),
		beta.ptr<const float>(),
		attn_out.ptr<half>(),
		num_heads,
		head_dim
	);
}

void GDN::recurrent_step_kda_fused(
	float* gdn_state,
	const pod::TensorView& qkv,
	const pod::TensorView& g_decay,
	const pod::TensorView& beta,
	const pod::TensorView& z,
	const pod::TensorView& o_norm,
	pod::TensorView& attn_out,
	int num_heads,
	int head_dim,
	float eps,
	cudaStream_t stream,
	const pod::SlotRef& slot
) {
	kda_recurrent_step_and_norm_kernel<<<dim3( num_heads ), dim3( head_dim ), 0, stream>>>(
		gdn_state,
		qkv.ptr<const half>(),
		g_decay.ptr<const float>(),
		beta.ptr<const float>(),
		z.ptr<const half>(),
		o_norm.ptr<const half>(),
		attn_out.ptr<half>(),
		num_heads,
		head_dim,
		eps,
		slot
	);
}

void GDN::gated_rmsnorm(
	pod::TensorView& attn_out,
	const pod::TensorView& z,
	const pod::TensorView& norm_weight,
	int num_v_heads,
	int head_dim,
	cudaStream_t stream,
	bool use_sigmoid,
	int rows
) {
	gated_rmsnorm_kernel<<<dim3( num_v_heads, rows ), dim3( head_dim ), 0, stream>>>(
		attn_out.ptr<half>(),
		z.ptr<const half>(),
		norm_weight.ptr<const half>(),
		num_v_heads,
		head_dim,
		use_sigmoid
	);
}

void GDN::recurrent_step_kda_rows(
	float* gdn_state,
	const pod::TensorView& qkv,
	const pod::TensorView& g_decay,
	const pod::TensorView& beta,
	const pod::TensorView& z,
	const pod::TensorView& o_norm,
	pod::TensorView& attn_out,
	int num_heads,
	int head_dim,
	float eps,
	int rows,
	const pod::BatchRows* d_rows,
	size_t slot_stride,
	float* snap,
	size_t snap_stride,
	int snap_rows,
	const int* snap_on,
	cudaStream_t stream
) {
	ENGINE_ASSERT( head_dim == 128, "GDN::recurrent_step_kda_rows: head_dim must be 128" );
	kda_recurrent_rows_kernel<<<dim3( num_heads ), dim3( 128 ), 0, stream>>>(
		gdn_state, qkv.ptr<const half>(), g_decay.ptr<const float>(), beta.ptr<const float>(), z.ptr<const half>(),
		o_norm.ptr<const half>(), attn_out.ptr<half>(), num_heads, eps, rows, d_rows, slot_stride,
		snap, snap_stride, snap_rows, snap_on
	);
}

} // namespace engine::modules
