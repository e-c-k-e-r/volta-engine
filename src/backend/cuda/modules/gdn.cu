#include "engine/modules/gdn.h"
#include "engine/backend/cuda/activation.cuh"
#include "engine/backend/cuda/reduce.cuh"
#include "engine/runtime/options.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>

namespace engine::modules {

namespace {

using engine::math::warp_sum;
namespace wmma = nvcuda::wmma;

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
			float beta_val = engine::math::sigmoid( total_b );
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
	float silu = engine::math::silu( sum );

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
		float silu = engine::math::silu( sum );
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
		dst[static_cast<size_t>( t ) * channels + c] = __float2half_rn( engine::math::silu( sum ) );
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

	float inv_l2 = rsqrtf( engine::math::block_sum<4>( sq ) + 1e-6f );
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

	float inv_rms = rsqrtf( engine::math::block_sum<4>( sq ) / static_cast<float>( head_dim ) + 1e-6f );
	float gamma = ( norm_weight != nullptr ) ? __half2float( norm_weight[tid] ) : 1.0f;
	float z_val = ( z != nullptr ) ? __half2float( z[head * head_dim + tid] ) : 0.0f;
	float gate_val;
	if ( use_sigmoid ) {
		gate_val = engine::math::sigmoid( z_val );
	}
	else {
		gate_val = engine::math::silu( z_val );
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

	sum = engine::math::warp_sum( sum );

	__shared__ float s_warp_sums[8];
	int warp_id = tid >> 5;
	int lane_id = tid & 31;

	if ( lane_id == 0 ) {
		s_warp_sums[warp_id] = sum;
	}
	__syncthreads();

	if ( warp_id == 0 ) {
		float b_sum = ( lane_id < ( blockDim.x >> 5 ) ) ? s_warp_sums[lane_id] : 0.0f;
		b_sum = engine::math::warp_sum<4>( b_sum );
		if ( lane_id == 0 ) {
			if ( row < num_heads ) {
				beta_out[row] = engine::math::sigmoid( b_sum );
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

// Prefill forms of the two projection kernels above: RB rows share each load of a weight row (the one-row kernels re-read all the weights per
// row, which bounds them at 512 rows). Every row keeps the one-row kernels' arithmetic and reduction order, so the results are bit-identical.
template <int RB>
__global__ void project_kda_stage1_rows_kernel(
	const half* __restrict__ norm_hidden,
	const half* __restrict__ b_proj,
	const half* __restrict__ f_a_proj,
	const half* __restrict__ g_a_proj,
	float* __restrict__ beta_out,
	half* __restrict__ f_a_buf,
	half* __restrict__ g_a_buf,
	int hidden_size,
	int num_heads,
	int rows
) {
	int row = blockIdx.x;
	if ( row >= num_heads + 256 ) return;
	const int r0 = blockIdx.y * RB;

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
	float sum[RB];
	#pragma unroll
	for ( int b = 0; b < RB; ++b ) sum[b] = 0.0f;

	const float4* w4 = reinterpret_cast<const float4*>( weight_row );
	int num_float4 = hidden_size / 8;

	for ( int i = tid; i < num_float4; i += blockDim.x ) {
		const float4 vw = w4[i];
		const half2* hw = reinterpret_cast<const half2*>( &vw );
		#pragma unroll
		for ( int b = 0; b < RB; ++b ) {
			float4 vx = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
			if ( r0 + b < rows ) vx = reinterpret_cast<const float4*>( norm_hidden + static_cast<size_t>( r0 + b ) * hidden_size )[i];
			const half2* hx = reinterpret_cast<const half2*>( &vx );
			#pragma unroll
			for ( int j = 0; j < 4; ++j ) {
				float2 fx = __half22float2( hx[j] );
				float2 fw = __half22float2( hw[j] );
				sum[b] = fmaf( fx.x, fw.x, sum[b] );
				sum[b] = fmaf( fx.y, fw.y, sum[b] );
			}
		}
	}

	__shared__ float s_warp_sums[RB][8];
	int warp_id = tid >> 5;
	int lane_id = tid & 31;
	#pragma unroll
	for ( int b = 0; b < RB; ++b ) {
		const float t = engine::math::warp_sum( sum[b] );
		if ( lane_id == 0 ) s_warp_sums[b][warp_id] = t;
	}
	__syncthreads();

	if ( warp_id == 0 ) {
		#pragma unroll
		for ( int b = 0; b < RB; ++b ) {
			float b_sum = ( lane_id < ( blockDim.x >> 5 ) ) ? s_warp_sums[b][lane_id] : 0.0f;
			b_sum = engine::math::warp_sum<4>( b_sum );
			if ( lane_id == 0 && r0 + b < rows ) {
				const int rr = r0 + b;
				if ( row < num_heads ) {
					beta_out[rr * num_heads + row] = engine::math::sigmoid( b_sum );
				}
				else if ( row < num_heads + 128 ) {
					f_a_buf[rr * 128 + row - num_heads] = __float2half_rn( b_sum );
				}
				else {
					g_a_buf[rr * 128 + row - num_heads - 128] = __float2half_rn( b_sum );
				}
			}
		}
	}
}

template <int RB>
__global__ void project_kda_stage2_rows_kernel(
	const half* __restrict__ f_a_buf,
	const half* __restrict__ g_a_buf,
	const half* __restrict__ f_b_proj,
	const half* __restrict__ g_b_proj,
	const half* __restrict__ A_log,
	const half* __restrict__ dt_bias,
	float* __restrict__ g_decay_out,
	half* __restrict__ z_out,
	int num_channels,
	int rows
) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if ( j >= num_channels ) return;
	const int r0 = blockIdx.y * RB;
	int h = j / 128;

	if ( f_b_proj != nullptr ) {
		float f_val[RB];
		#pragma unroll
		for ( int b = 0; b < RB; ++b ) f_val[b] = 0.0f;
		const float4* fb4 = reinterpret_cast<const float4*>( f_b_proj + j * 128 );
		#pragma unroll
		for ( int i = 0; i < 16; ++i ) {
			const float4 vfb = fb4[i];
			const half2* hfb = reinterpret_cast<const half2*>( &vfb );
			#pragma unroll
			for ( int b = 0; b < RB; ++b ) {
				float4 vfa = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
				if ( r0 + b < rows ) vfa = reinterpret_cast<const float4*>( f_a_buf + static_cast<size_t>( r0 + b ) * 128 )[i];
				const half2* hfa = reinterpret_cast<const half2*>( &vfa );
				#pragma unroll
				for ( int k = 0; k < 4; ++k ) {
					float2 f1 = __half22float2( hfa[k] );
					float2 f2 = __half22float2( hfb[k] );
					f_val[b] = fmaf( f1.x, f2.x, f_val[b] );
					f_val[b] = fmaf( f1.y, f2.y, f_val[b] );
				}
			}
		}
		float dt = ( dt_bias != nullptr ) ? __half2float( dt_bias[j] ) : 0.0f;
		float a_val = ( A_log != nullptr ) ? __half2float( A_log[h] ) : 0.0f;
		float decay = __expf( a_val );
		#pragma unroll
		for ( int b = 0; b < RB; ++b ) {
			if ( r0 + b >= rows ) continue;
			float gf = f_val[b] + dt;
			g_decay_out[static_cast<size_t>( r0 + b ) * num_channels + j] = -5.0f / ( 1.0f + __expf( -decay * gf ) );
		}
	}

	if ( g_b_proj != nullptr ) {
		float z_val[RB];
		#pragma unroll
		for ( int b = 0; b < RB; ++b ) z_val[b] = 0.0f;
		const float4* gb4 = reinterpret_cast<const float4*>( g_b_proj + j * 128 );
		#pragma unroll
		for ( int i = 0; i < 16; ++i ) {
			const float4 vgb = gb4[i];
			const half2* hgb = reinterpret_cast<const half2*>( &vgb );
			#pragma unroll
			for ( int b = 0; b < RB; ++b ) {
				float4 vga = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
				if ( r0 + b < rows ) vga = reinterpret_cast<const float4*>( g_a_buf + static_cast<size_t>( r0 + b ) * 128 )[i];
				const half2* hga = reinterpret_cast<const half2*>( &vga );
				#pragma unroll
				for ( int k = 0; k < 4; ++k ) {
					float2 g1 = __half22float2( hga[k] );
					float2 g2 = __half22float2( hgb[k] );
					z_val[b] = fmaf( g1.x, g2.x, z_val[b] );
					z_val[b] = fmaf( g1.y, g2.y, z_val[b] );
				}
			}
		}
		#pragma unroll
		for ( int b = 0; b < RB; ++b ) {
			if ( r0 + b < rows ) z_out[static_cast<size_t>( r0 + b ) * num_channels + j] = __float2half_rn( z_val[b] );
		}
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
	float inv_rms = rsqrtf( engine::math::block_sum<4>( sq ) / static_cast<float>( head_dim ) + eps );
	float gamma = ( o_norm != nullptr ) ? __half2float( o_norm[c] ) : 1.0f;
	float z_val = ( z != nullptr ) ? __half2float( z[h * head_dim + c] ) : 0.0f;
	float gate_val = engine::math::sigmoid( z_val );

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
	const int* __restrict__ snap_on,
	unsigned char* __restrict__ journal,
	size_t journal_row_stride,
	int journal_rows
) {
	const int h = blockIdx.x;
	const int c = threadIdx.x;
	const int channels = num_heads * 128;
	const bool do_snap = snap && *snap_on;
	// Journal mode (speculative verify): each row's k, v, decay and beta are recorded and the slot state in memory is left at its value before
	// the block; kda_replay_kernel applies the accepted rows afterwards with the same arithmetic
	const bool jmode = journal && *snap_on;

	__shared__ float s_g_exp[128];
	__shared__ float s_k[128];
	__shared__ float s_q[128];

	float s[128];
	int cur = -1;
	for ( int m = 0; m < rows; ++m ) {
		const int slot = d_rows->slot[m];
		if ( slot != cur ) {
			if ( cur >= 0 && !jmode ) {
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
		if ( jmode && m < journal_rows ) {
			unsigned char* rec = journal + static_cast<size_t>( m ) * journal_row_stride;
			reinterpret_cast<half*>( rec )[h * 128 + c] = k_ptr[c];
			reinterpret_cast<half*>( rec + static_cast<size_t>( channels ) * 2 )[h * 128 + c] = v_ptr[c];
			reinterpret_cast<float*>( rec + static_cast<size_t>( channels ) * 4 )[h * 128 + c] = g_decay[static_cast<size_t>( m ) * channels + h * 128 + c];
			if ( c == 0 ) reinterpret_cast<float*>( rec + static_cast<size_t>( channels ) * 8 )[h] = beta[m * num_heads + h];
		}
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
		const float inv_rms = rsqrtf( engine::math::block_sum<4>( sq ) / 128.0f + eps );
		const float gamma = ( o_norm != nullptr ) ? __half2float( o_norm[c] ) : 1.0f;
		const float z_val = ( z != nullptr ) ? __half2float( z[static_cast<size_t>( m ) * channels + h * 128 + c] ) : 0.0f;
		const float gate_val = engine::math::sigmoid( z_val );
		output[static_cast<size_t>( m ) * channels + h * 128 + c] = __float2half_rn( y_core * inv_rms * gamma * gate_val );
		__syncthreads();
	}
	if ( !jmode ) {
		float* st = state + static_cast<size_t>( cur ) * slot_stride + static_cast<size_t>( h ) * 128 * 128;
		#pragma unroll
		for ( int r = 0; r < 128; ++r ) st[r * 128 + c] = s[r];
	}
}

// Applies rows [first_row, first_row + nrows) of a journal to one slot's state, for every linear layer (blockIdx.y): the same update, in the same
// order and with the same fused operations, as kda_recurrent_rows_kernel, so the result is its state after those rows bit for bit
__global__ void __launch_bounds__( 128 )
kda_replay_kernel(
	float* __restrict__ state,
	size_t slot_stride,
	size_t layer_stride,
	const unsigned char* __restrict__ journal,
	size_t journal_layer_stride,
	size_t journal_row_stride,
	int slot,
	int first_row,
	int nrows,
	int num_heads
) {
	const int h = blockIdx.x;
	const int lin = blockIdx.y;
	const int c = threadIdx.x;
	const int channels = num_heads * 128;
	__shared__ float s_g_exp[128];
	__shared__ float s_k[128];

	float* st = state + static_cast<size_t>( slot ) * slot_stride + static_cast<size_t>( lin ) * layer_stride + static_cast<size_t>( h ) * 128 * 128;
	float s[128];
	#pragma unroll
	for ( int r = 0; r < 128; ++r ) s[r] = st[r * 128 + c];
	for ( int i = 0; i < nrows; ++i ) {
		const unsigned char* rec = journal + static_cast<size_t>( lin ) * journal_layer_stride + static_cast<size_t>( first_row + i ) * journal_row_stride;
		s_g_exp[c] = __expf( reinterpret_cast<const float*>( rec + static_cast<size_t>( channels ) * 4 )[h * 128 + c] );
		s_k[c] = __half2float( reinterpret_cast<const half*>( rec )[h * 128 + c] );
		__syncthreads();
		float kv_mem = 0.0f;
		#pragma unroll
		for ( int r = 0; r < 128; ++r ) {
			s[r] = s[r] * s_g_exp[r];
			kv_mem = fmaf( s[r], s_k[r], kv_mem );
		}
		const float delta = ( __half2float( reinterpret_cast<const half*>( rec + static_cast<size_t>( channels ) * 2 )[h * 128 + c] ) - kv_mem )
			* reinterpret_cast<const float*>( rec + static_cast<size_t>( channels ) * 8 )[h];
		#pragma unroll
		for ( int r = 0; r < 128; ++r ) s[r] = fmaf( s_k[r], delta, s[r] );
		__syncthreads();
	}
	#pragma unroll
	for ( int r = 0; r < 128; ++r ) st[r * 128 + c] = s[r];
}


// ---- Chunked delta rule (the prefill form) ----
// Per v-head and 64-row chunk, with gamma_t the chunk's inclusive cumulative log decay and G_t = exp( gamma_t ), S0 the state entering
// the chunk ([v][k]) and T = ( I + A )^-1, A_tj = beta_t exp( gamma_t - gamma_j ) k_t.k_j (j < t):
//   Delta = U - W S0^T, W = T diag( beta G ) K, U = T diag( beta ) V      (the rows' deltas of the recurrence)
//   S = G_last S0 + ( D Delta )^T K, D_j = exp( gamma_last - gamma_j )
//   O = G Q S0^T + P Delta, P_tj = exp( gamma_t - gamma_j ) q_t.k_j (j <= t)
// gdn_chunk_prep_kernel computes P, W, U and gamma of every chunk at once (T in 16 x 16 blocks); gdn_chunk_scan_kernel walks the chunks with a 32 x 128 tile of
// the state in registers (the next chunk's operands loading meanwhile) and keeps each chunk's S0 and Delta; gdn_chunk_out_kernel then
// computes O for every chunk at once. On the state's path a tensor-core product with an FP32 operand splits it into FP16 hi + lo (dropping
// lo x lo), which keeps the recurrence near FP32; the output's products take FP16 operands (the output is FP16). A partial last chunk reads zero-padded copies of its K and Q rows (rows past the end count for nothing: beta, gate,
// k, q and v are 0 there).
constexpr int GC = 64;           // chunk rows
constexpr int GKP = 136;         // shared pitch (halves) of 128-wide rows
constexpr int GTP = 72;          // shared pitch (halves) of a 64 x 64 operand
constexpr int GSV = 32;          // state rows (v) a scan block owns, and v columns an output block computes
constexpr int GEP = 40;          // shared pitch (halves) of a 64 x 32 operand
constexpr int GPP = 36;          // shared pitch (floats) of a 32-wide product
constexpr int GQP = 144;         // shared pitch (halves) of the output block's Q rows (the P halves fit there after)
constexpr size_t GSCAN_SMEM = 3 * GC * GKP * sizeof( half ) + 2 * GSV * GKP * sizeof( half ) + GC * GPP * sizeof( float ) + 2 * GSV * GTP * sizeof( half );
constexpr size_t GPW = GC * GC + 2 * GC * 128;                                              // halves of P and W (hi, lo) per head and chunk
constexpr size_t GSC = 128 * 128;                                                            // halves of S0
constexpr size_t GDD = GC * 128;                                                             // halves of Delta
static_assert( GC * GPP >= 8 * 256 && GC * GQP >= GC * GTP && 4 * 16 * GPP * sizeof( float ) >= GSV * GKP * sizeof( half ), "scan / output regions" );

using AccF = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
using FragAR = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>;
using FragBR = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>;
using FragBC = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>;

__device__ __forceinline__ void split_hl( float x, half& hi, half& lo ) {
	hi = __float2half_rn( x );
	lo = __float2half_rn( x - __half2float( hi ) );
}

// Eight floats to two 16-byte stores of their FP16 hi and lo halves
__device__ __forceinline__ void split8( const float* x, half* hi, half* lo ) {
	alignas( 16 ) half h[8], l[8];
	#pragma unroll
	for ( int i = 0; i < 8; ++i ) split_hl( x[i], h[i], l[i] );
	*reinterpret_cast<uint4*>( hi ) = *reinterpret_cast<const uint4*>( h );
	*reinterpret_cast<uint4*>( lo ) = *reinterpret_cast<const uint4*>( l );
}

// Lower-triangle tile ( ti, tj ) of 10 (tj <= ti)
__device__ __forceinline__ void tri_tile( int tile, int& ti, int& tj ) {
	ti = 0;
	while ( ( ti + 1 ) * ( ti + 2 ) / 2 <= tile ) ++ti;
	tj = tile - ti * ( ti + 1 ) / 2;
}

// Scratch of one call: P and W (hi, lo) at pw + ( c * h_v + h ) * GPW, W then U (FP32) at uu + ( c * h_v + h ) * 2 * 64 * 128, gamma at gam + ( c * h_v + h ) * 64,
// S0 at sc + ( c * h_v + h ) * GSC, Delta at dd + ( c * h_v + h ) * GDD, the padded K / Q of K head kh at pad + kh * 2 * 64 * 128
struct ChunkScratch {
	half* pw;
	float* uu;
	float* gam;
	half* sc;
	half* dd;
	half* pad;
};

// One block per (v-head, chunk). The lower-triangular 64 x 64 matrices (A, T and T's scaled copies) live as their ten 16 x 16 blocks,
// block ( i, j ) (j <= i) at index i ( i + 1 ) / 2 + j
__device__ __forceinline__ int tri_index( int i, int j ) { return i * ( i + 1 ) / 2 + j; }

// T = ( I + A )^-1 for a strictly lower-triangular 64 x 64 A held as its ten blocks (A's diagonal blocks are read below their diagonal only), by a
// block of 256 threads: the diagonal blocks by forward substitution (a column a thread), then the blocks below them a diagonal at a time,
// T_ij = -T_ii sum_{j <= k < i} A_ik T_kj. fm holds 3 x 256 floats of partial products; returns with the block synchronized.
__device__ __forceinline__ void solve_unit_lower( const float* fa, float* ft, float* fm, int tid ) {
	if ( tid < GC ) {
		const int bb = tid >> 4, col = tid & 15;
		const float* ad = fa + tri_index( bb, bb ) * 256;
		float x[16];
		#pragma unroll
		for ( int i = 0; i < 16; ++i ) {
			float s0 = 0.0f, s1 = 0.0f;
			#pragma unroll
			for ( int j = 0; j < i; ++j ) {
				if ( j & 1 ) s1 = fmaf( ad[i * 16 + j], x[j], s1 );
				else s0 = fmaf( ad[i * 16 + j], x[j], s0 );
			}
			x[i] = i < col ? 0.0f : i == col ? 1.0f : -( s0 + s1 );
		}
		float* td = ft + tri_index( bb, bb ) * 256;
		#pragma unroll
		for ( int i = 0; i < 16; ++i ) td[i * 16 + col] = x[i];
	}
	__syncthreads();
	for ( int d = 1; d < 4; ++d ) {
		const int nb = 4 - d;
		for ( int e = tid; e < nb * 256; e += 256 ) {
			const int j = e >> 8, i = j + d, r = ( e >> 4 ) & 15, cc = e & 15;
			float s0 = 0.0f, s1 = 0.0f;
			for ( int k = j; k < i; ++k ) {
				const float* ar = fa + tri_index( i, k ) * 256 + r * 16;
				const float* tc = ft + tri_index( k, j ) * 256 + cc;
				#pragma unroll
				for ( int l = 0; l < 16; l += 2 ) {
					s0 = fmaf( ar[l], tc[l * 16], s0 );
					s1 = fmaf( ar[l + 1], tc[( l + 1 ) * 16], s1 );
				}
			}
			fm[e] = s0 + s1;
		}
		__syncthreads();
		for ( int e = tid; e < nb * 256; e += 256 ) {
			const int j = e >> 8, i = j + d, r = ( e >> 4 ) & 15, cc = e & 15;
			const float* tr = ft + tri_index( i, i ) * 256 + r * 16;
			const float* mc = fm + j * 256 + cc;
			float s0 = 0.0f, s1 = 0.0f;
			#pragma unroll
			for ( int l = 0; l < 16; l += 2 ) {
				s0 = fmaf( tr[l], mc[l * 16], s0 );
				s1 = fmaf( tr[l + 1], mc[( l + 1 ) * 16], s1 );
			}
			ft[tri_index( i, j ) * 256 + r * 16 + cc] = -( s0 + s1 );
		}
		__syncthreads();
	}
}

__global__ void __launch_bounds__( 256, 2 ) gdn_chunk_prep_kernel(
	const half* __restrict__ qkv,
	const float* __restrict__ gate,
	const float* __restrict__ beta,
	ChunkScratch cs,
	int h_k,
	int h_v,
	int rows
) {
	__shared__ __align__( 32 ) half s_r1[GC * GKP];                   // K, then V
	__shared__ __align__( 32 ) unsigned char s_r2[GC * GKP * 2];      // Q, then A (blocks) and the solve's partial products, then T's scaled copy (hi, lo)
	__shared__ __align__( 32 ) float s_r3[10 * 256];                  // per-warp product staging, then T (blocks)
	__shared__ float s_g[GC], s_b[GC];
	half* ks = s_r1;
	half* vs = s_r1;
	half* qs = reinterpret_cast<half*>( s_r2 );
	float* fa = reinterpret_cast<float*>( s_r2 );
	float* fm = fa + 10 * 256;
	half* txh = reinterpret_cast<half*>( s_r2 );
	half* txl = txh + 10 * 256;
	float* ft = s_r3;
	const int h = blockIdx.x, c = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
	const int ratio = h_v / h_k, kh = h / ratio;
	const int r0 = c * GC, n = min( GC, rows - r0 );
	const size_t stride = static_cast<size_t>( 2 * h_k + h_v ) * 128;
	const size_t hc = static_cast<size_t>( c ) * h_v + h;
	half* pw = cs.pw + hc * GPW;
	const int er = lane >> 1, ec = ( lane & 1 ) * 8;   // a lane's 8 elements of a 16 x 16 tile

	{
		uint4 q[4], k[4];
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			const int x = tid + i * 256, t = x >> 4, v = ( x & 15 ) * 8;
			q[i] = k[i] = make_uint4( 0, 0, 0, 0 );
			if ( t < n ) {
				const half* row = qkv + static_cast<size_t>( r0 + t ) * stride + kh * 128 + v;
				q[i] = *reinterpret_cast<const uint4*>( row );
				k[i] = *reinterpret_cast<const uint4*>( row + h_k * 128 );
			}
		}
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			const int x = tid + i * 256, t = x >> 4, v = ( x & 15 ) * 8;
			*reinterpret_cast<uint4*>( qs + t * GKP + v ) = q[i];
			*reinterpret_cast<uint4*>( ks + t * GKP + v ) = k[i];
		}
	}
	if ( tid < GC ) {
		const bool in = tid < n;
		s_g[tid] = in ? gate[static_cast<size_t>( r0 + tid ) * h_v + h] : 0.0f;
		s_b[tid] = in ? beta[static_cast<size_t>( r0 + tid ) * h_v + h] : 0.0f;
	}
	__syncthreads();
	if ( tid == 0 ) {
		float a = 0.0f;
		for ( int t = 0; t < GC; ++t ) {
			a += s_g[t];
			s_g[t] = a;
		}
	}
	if ( n < GC && h % ratio == 0 ) {
		half* pk = cs.pad + static_cast<size_t>( kh ) * 2 * GC * 128;
		for ( int i = tid; i < GC * 16; i += 256 ) {
			const int t = i >> 4, v = ( i & 15 ) * 8;
			*reinterpret_cast<uint4*>( pk + t * 128 + v ) = *reinterpret_cast<const uint4*>( ks + t * GKP + v );
			*reinterpret_cast<uint4*>( pk + GC * 128 + t * 128 + v ) = *reinterpret_cast<const uint4*>( qs + t * GKP + v );
		}
	}
	// P's tiles above the diagonal are zero
	for ( int x = tid; x < 6 * 32; x += 256 ) {
		const int tile = x >> 5, l = x & 31;
		const int ti = tile < 3 ? 0 : tile < 5 ? 1 : 2, tj = tile < 3 ? tile + 1 : tile < 5 ? tile - 1 : 3;
		*reinterpret_cast<uint4*>( pw + ( ti * 16 + ( l >> 1 ) ) * GC + tj * 16 + ( l & 1 ) * 8 ) = make_uint4( 0, 0, 0, 0 );
	}
	__syncthreads();

	// P = exp( gamma_t - gamma_j ) q_t.k_j (j <= t) to the scratch, a tile at a time through the warp's staging
	float* z = s_r3 + warp * 256;
	for ( int tile = warp; tile < 10; tile += 8 ) {
		int ti, tj;
		tri_tile( tile, ti, tj );
		AccF acc;
		wmma::fill_fragment( acc, 0.0f );
		#pragma unroll
		for ( int kk = 0; kk < 8; ++kk ) {
			FragAR a;
			FragBC b;
			wmma::load_matrix_sync( a, qs + ti * 16 * GKP + kk * 16, GKP );
			wmma::load_matrix_sync( b, ks + tj * 16 * GKP + kk * 16, GKP );
			wmma::mma_sync( acc, a, b, acc );
		}
		wmma::store_matrix_sync( z, acc, 16, wmma::mem_row_major );
		__syncwarp();
		const int t = ti * 16 + er;
		alignas( 16 ) half ph[8];
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) {
			const int j = tj * 16 + ec + i;
			ph[i] = __float2half_rn( j <= t ? expf( s_g[t] - s_g[j] ) * z[er * 16 + ec + i] : 0.0f );
		}
		*reinterpret_cast<uint4*>( pw + t * GC + tj * 16 + ec ) = *reinterpret_cast<const uint4*>( ph );
		__syncwarp();
	}
	if ( tid < GC ) cs.gam[hc * GC + tid] = s_g[tid];
	__syncthreads();

	// A = beta_t exp( gamma_t - gamma_j ) k_t.k_j (j < t), into its blocks over Q
	for ( int tile = warp; tile < 10; tile += 8 ) {
		int ti, tj;
		tri_tile( tile, ti, tj );
		AccF acc;
		wmma::fill_fragment( acc, 0.0f );
		#pragma unroll
		for ( int kk = 0; kk < 8; ++kk ) {
			FragAR a;
			FragBC b;
			wmma::load_matrix_sync( a, ks + ti * 16 * GKP + kk * 16, GKP );
			wmma::load_matrix_sync( b, ks + tj * 16 * GKP + kk * 16, GKP );
			wmma::mma_sync( acc, a, b, acc );
		}
		wmma::store_matrix_sync( z, acc, 16, wmma::mem_row_major );
		__syncwarp();
		const int t = ti * 16 + er;
		float* ab = fa + tile * 256 + er * 16 + ec;
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) {
			const int j = tj * 16 + ec + i;
			ab[i] = j < t ? s_b[t] * expf( s_g[t] - s_g[j] ) * z[er * 16 + ec + i] : 0.0f;
		}
		__syncwarp();
	}
	__syncthreads();

	solve_unit_lower( fa, ft, fm, tid );

	// W = T diag( beta G ) K and U = T diag( beta ) V (FP32; T is lower triangular): warp w takes row tile w & 3, column tiles 4 ( w >> 2 ) ..
	auto scale_t = [&]( bool with_g ) {
		#pragma unroll
		for ( int blk = 0; blk < 10; ++blk ) {
			const int bj = blk == 0 ? 0 : blk < 3 ? blk - 1 : blk < 6 ? blk - 3 : blk - 6;   // tri_tile's column
			const int e = blk * 256 + tid, j = bj * 16 + ( tid & 15 );
			float x = ft[e] * s_b[j];
			if ( with_g ) x *= expf( s_g[j] );
			split_hl( x, txh[e], txl[e] );
		}
	};
	const int rt = warp & 3, ct = ( warp >> 2 ) * 4;
	auto product = [&]( const half* b_src, float* out ) {
		for ( int tn = ct; tn < ct + 4; ++tn ) {
			AccF acc;
			wmma::fill_fragment( acc, 0.0f );
			for ( int k = 0; k <= rt; ++k ) {
				FragAR ah, al;
				FragBR b;
				wmma::load_matrix_sync( ah, txh + tri_index( rt, k ) * 256, 16 );
				wmma::load_matrix_sync( al, txl + tri_index( rt, k ) * 256, 16 );
				wmma::load_matrix_sync( b, b_src + k * 16 * GKP + tn * 16, GKP );
				wmma::mma_sync( acc, ah, b, acc );
				wmma::mma_sync( acc, al, b, acc );
			}
			wmma::store_matrix_sync( out + rt * 16 * 128 + tn * 16, acc, 128, wmma::mem_row_major );
		}
	};
	float* wf = cs.uu + hc * 2 * GC * 128;
	scale_t( true );
	__syncthreads();
	product( ks, wf );
	__syncthreads();
	scale_t( false );
	for ( int i = tid; i < GC * 16; i += 256 ) {
		const int t = i >> 4, v = ( i & 15 ) * 8;
		uint4 x = make_uint4( 0, 0, 0, 0 );
		if ( t < n ) x = *reinterpret_cast<const uint4*>( qkv + static_cast<size_t>( r0 + t ) * stride + ( 2 * h_k + h ) * 128 + v );
		*reinterpret_cast<uint4*>( vs + t * GKP + v ) = x;
	}
	__syncthreads();
	product( vs, wf + GC * 128 );
	// W (written above by this block) to its hi / lo halves
	half* wh = pw + GC * GC;
	for ( int x = tid; x < GC * 128 / 8; x += 256 ) {
		float w[8];
		*reinterpret_cast<float4*>( w ) = *reinterpret_cast<const float4*>( wf + x * 8 );
		*reinterpret_cast<float4*>( w + 4 ) = *reinterpret_cast<const float4*>( wf + x * 8 + 4 );
		split8( w, wh + x * 8, wh + GC * 128 + x * 8 );
	}
}

// One block per (v-head, 32 state rows): the chunks in order, the operands of the next chunk loading into registers meanwhile. Warp w holds
// k columns 16 w .. 16 w + 15 of both 16-row halves of the state tile; for Delta it takes row tile w & 3 of v half w >> 2.
__global__ void __launch_bounds__( 256, 1 ) gdn_chunk_scan_kernel(
	float* __restrict__ state,
	const half* __restrict__ qkv,
	ChunkScratch cs,
	int h_k,
	int h_v,
	int rows,
	const pod::BatchRows* __restrict__ d_rows,
	size_t slot_stride
) {
	extern __shared__ __align__( 32 ) unsigned char gdn_smem[];
	half* s_wh = reinterpret_cast<half*>( gdn_smem );        // W (hi, lo), [t][k]
	half* s_wl = s_wh + GC * GKP;
	half* s_k = s_wl + GC * GKP;                              // K, [t][k]
	half* s_sh = s_k + GC * GKP;                              // S0 (hi, lo), [v][k]
	half* s_sl = s_sh + GSV * GKP;
	float* s_p = reinterpret_cast<float*>( s_sl + GSV * GKP ); // W S0^T, then per-warp staging
	half* s_eh = reinterpret_cast<half*>( s_p + GC * GPP );   // ( D Delta )^T (hi, lo), [v][t]
	half* s_el = s_eh + GSV * GTP;
	__shared__ float s_dt[2][GC], s_gl[2];
	const int h = blockIdx.x, d0 = blockIdx.y * GSV, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
	const int ratio = h_v / h_k, kh = h / ratio, tr = warp & 3, ds = warp >> 2;
	const size_t stride = static_cast<size_t>( 2 * h_k + h_v ) * 128;
	const int chunks = ( rows + GC - 1 ) / GC;
	float* st = state + static_cast<size_t>( d_rows->slot[0] ) * slot_stride + ( static_cast<size_t>( h ) * 128 + d0 ) * 128;
	auto hcoff = [&]( int c ) { return static_cast<size_t>( c ) * h_v + h; };

	uint4 rw[8], rk[4];
	float4 ru[2];
	const int ut = tid >> 2, uj = ( tid & 3 ) * 8;            // the thread's 8 elements of Delta
	auto fetch_w = [&]( int c ) {
		const half* w = cs.pw + hcoff( c ) * GPW + GC * GC;
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) rw[i] = *reinterpret_cast<const uint4*>( w + ( tid + i * 256 ) * 8 );
	};
	auto store_w = [&]() {
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) {
			const int x = ( tid + i * 256 ) * 8, m = x >> 13, o = ( ( x >> 7 ) & 63 ) * GKP + ( x & 127 );
			*reinterpret_cast<uint4*>( ( m ? s_wl : s_wh ) + o ) = rw[i];
		}
	};
	auto fetch_k = [&]( int c ) {
		const int n = min( GC, rows - c * GC );
		const half* kp = n < GC ? cs.pad + static_cast<size_t>( kh ) * 2 * GC * 128 : qkv + static_cast<size_t>( c * GC ) * stride + ( h_k + kh ) * 128;
		const size_t ld = n < GC ? 128 : stride;
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			const int x = tid + i * 256;
			rk[i] = *reinterpret_cast<const uint4*>( kp + ( x >> 4 ) * ld + ( x & 15 ) * 8 );
		}
	};
	auto store_k = [&]() {
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			const int x = tid + i * 256;
			*reinterpret_cast<uint4*>( s_k + ( x >> 4 ) * GKP + ( x & 15 ) * 8 ) = rk[i];
		}
	};
	auto fetch_u = [&]( int c ) {
		const float* u = cs.uu + hcoff( c ) * 2 * GC * 128 + GC * 128 + ut * 128 + d0 + uj;
		ru[0] = *reinterpret_cast<const float4*>( u );
		ru[1] = *reinterpret_cast<const float4*>( u + 4 );
	};
	auto scalars = [&]( int c, int b ) {
		if ( tid < GC ) {
			const float* g = cs.gam + hcoff( c ) * GC;
			s_dt[b][tid] = expf( g[GC - 1] - g[tid] );
			if ( tid == 0 ) s_gl[b] = expf( g[GC - 1] );
		}
	};
	// The warp's columns of both state halves into s_sh / s_sl
	AccF S[2];
	auto split_state = [&]() {
		float* z = s_p + warp * 256;
		const int r = lane >> 1, k0 = warp * 16 + ( lane & 1 ) * 8;
		#pragma unroll
		for ( int f = 0; f < 2; ++f ) {
			wmma::store_matrix_sync( z, S[f], 16, wmma::mem_row_major );
			__syncwarp();
			split8( z + r * 16 + ( lane & 1 ) * 8, s_sh + ( f * 16 + r ) * GKP + k0, s_sl + ( f * 16 + r ) * GKP + k0 );
			__syncwarp();
		}
	};

	#pragma unroll
	for ( int f = 0; f < 2; ++f ) wmma::load_matrix_sync( S[f], st + f * 16 * 128 + warp * 16, 128, wmma::mem_row_major );
	fetch_w( 0 );
	fetch_k( 0 );
	fetch_u( 0 );
	split_state();
	store_w();
	scalars( 0, 0 );
	__syncthreads();

	for ( int c = 0; c < chunks; ++c ) {
		const int b = c & 1, n = min( GC, rows - c * GC );
		const size_t hc = hcoff( c );
		// This chunk's S0 to the scratch (the output kernel's), its K into shared memory; the next chunk's W and K start loading
		{
			half* sc = cs.sc + hc * GSC + static_cast<size_t>( d0 ) * 128;
			#pragma unroll
			for ( int i = 0; i < 2; ++i ) {
				const int x = tid + i * 256, r = x >> 4, v = ( x & 15 ) * 8;
				*reinterpret_cast<uint4*>( sc + r * 128 + v ) = *reinterpret_cast<const uint4*>( s_sh + r * GKP + v );
			}
		}
		store_k();
		if ( c + 1 < chunks ) {
			fetch_w( c + 1 );
			fetch_k( c + 1 );
		}

		// W S0^T, row tile tr of v half ds
		{
			AccF a1, a2, a3;
			wmma::fill_fragment( a1, 0.0f );
			wmma::fill_fragment( a2, 0.0f );
			wmma::fill_fragment( a3, 0.0f );
			#pragma unroll
			for ( int kk = 0; kk < 8; ++kk ) {
				FragAR ah, al;
				FragBC bh, bl;
				wmma::load_matrix_sync( ah, s_wh + tr * 16 * GKP + kk * 16, GKP );
				wmma::load_matrix_sync( al, s_wl + tr * 16 * GKP + kk * 16, GKP );
				wmma::load_matrix_sync( bh, s_sh + ds * 16 * GKP + kk * 16, GKP );
				wmma::load_matrix_sync( bl, s_sl + ds * 16 * GKP + kk * 16, GKP );
				wmma::mma_sync( a1, ah, bh, a1 );
				wmma::mma_sync( a2, ah, bl, a2 );
				wmma::mma_sync( a3, al, bh, a3 );
			}
			#pragma unroll
			for ( int i = 0; i < a1.num_elements; ++i ) a1.x[i] += a2.x[i] + a3.x[i];
			wmma::store_matrix_sync( s_p + tr * 16 * GPP + ds * 16, a1, GPP, wmma::mem_row_major );
		}
		__syncthreads();

		// Delta = U - W S0^T (to the scratch, hi / lo) and D Delta (hi / lo); the next chunk's W into shared memory, its U loading
		{
			float dv[8];
			const float* u = reinterpret_cast<const float*>( ru );
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) dv[i] = ut < n ? u[i] - s_p[ut * GPP + uj + i] : 0.0f;
			alignas( 16 ) half dh[8];
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) dh[i] = __float2half_rn( dv[i] );
			*reinterpret_cast<uint4*>( cs.dd + hc * GDD + ut * 128 + d0 + uj ) = *reinterpret_cast<const uint4*>( dh );
			const float dt = s_dt[b][ut];
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) split_hl( dv[i] * dt, s_eh[( uj + i ) * GTP + ut], s_el[( uj + i ) * GTP + ut] );
		}
		if ( c + 1 < chunks ) {
			store_w();
			fetch_u( c + 1 );
		}
		__syncthreads();

		// S = G_last S0 + ( D Delta )^T K over the warp's 16 k columns
		{
			const float gl = s_gl[b];
			#pragma unroll
			for ( int f = 0; f < 2; ++f ) {
				#pragma unroll
				for ( int i = 0; i < S[f].num_elements; ++i ) S[f].x[i] *= gl;
			}
			#pragma unroll
			for ( int k = 0; k < 4; ++k ) {
				FragBR kf;
				wmma::load_matrix_sync( kf, s_k + k * 16 * GKP + warp * 16, GKP );
				#pragma unroll
				for ( int f = 0; f < 2; ++f ) {
					FragAR eh, el;
					wmma::load_matrix_sync( eh, s_eh + f * 16 * GTP + k * 16, GTP );
					wmma::load_matrix_sync( el, s_el + f * 16 * GTP + k * 16, GTP );
					wmma::mma_sync( S[f], eh, kf, S[f] );
					wmma::mma_sync( S[f], el, kf, S[f] );
				}
			}
			split_state();
		}
		if ( c + 1 < chunks ) scalars( c + 1, b ^ 1 );
		__syncthreads();
	}
	#pragma unroll
	for ( int f = 0; f < 2; ++f ) wmma::store_matrix_sync( st + f * 16 * 128 + warp * 16, S[f], 128, wmma::mem_row_major );
}

// One block per (v-head, chunk, 32 v columns): O = G Q S0^T + P Delta, rows of the chunk to the output
__global__ void __launch_bounds__( 128 ) gdn_chunk_out_kernel(
	const half* __restrict__ qkv,
	ChunkScratch cs,
	half* __restrict__ output,
	int h_k,
	int h_v,
	float scale,
	int rows
) {
	__shared__ __align__( 32 ) half s_x[GC * GQP];               // Q [t][k], then P [t][j]
	__shared__ __align__( 32 ) float s_s[4 * 16 * GPP];          // S0 [v][k] (halves), then per-warp staging
	__shared__ __align__( 32 ) half s_d[GC * GEP];               // Delta [t][v]
	const int h = blockIdx.x, c = blockIdx.y, v0 = blockIdx.z * GSV, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, r16 = warp * 16;
	const int ratio = h_v / h_k, kh = h / ratio;
	const int r0 = c * GC, n = min( GC, rows - r0 );
	const size_t stride = static_cast<size_t>( 2 * h_k + h_v ) * 128;
	const size_t hc = static_cast<size_t>( c ) * h_v + h;
	const half* qp = n < GC ? cs.pad + static_cast<size_t>( kh ) * 2 * GC * 128 + GC * 128 : qkv + static_cast<size_t>( r0 ) * stride + kh * 128;
	const size_t ld = n < GC ? 128 : stride;
	const half* pw = cs.pw + hc * GPW;
	const half* sc = cs.sc + hc * GSC + static_cast<size_t>( v0 ) * 128;
	const half* dd = cs.dd + hc * GDD + v0;
	half* ss = reinterpret_cast<half*>( s_s );
	uint4 rp[4];
	{
		uint4 rq[8], rs[4], rd[2];
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) {
			const int x = tid + i * 128;
			rq[i] = *reinterpret_cast<const uint4*>( qp + ( x >> 4 ) * ld + ( x & 15 ) * 8 );
		}
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			const int x = tid + i * 128;
			rs[i] = *reinterpret_cast<const uint4*>( sc + ( x >> 4 ) * 128 + ( x & 15 ) * 8 );
			rp[i] = *reinterpret_cast<const uint4*>( pw + ( x >> 3 ) * GC + ( x & 7 ) * 8 );
		}
		#pragma unroll
		for ( int i = 0; i < 2; ++i ) {
			const int x = tid + i * 128;
			rd[i] = *reinterpret_cast<const uint4*>( dd + ( x >> 2 ) * 128 + ( x & 3 ) * 8 );
		}
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) {
			const int x = tid + i * 128;
			*reinterpret_cast<uint4*>( s_x + ( x >> 4 ) * GQP + ( x & 15 ) * 8 ) = rq[i];
		}
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			const int x = tid + i * 128;
			*reinterpret_cast<uint4*>( ss + ( x >> 4 ) * GKP + ( x & 15 ) * 8 ) = rs[i];
		}
		#pragma unroll
		for ( int i = 0; i < 2; ++i ) {
			const int x = tid + i * 128;
			*reinterpret_cast<uint4*>( s_d + ( x >> 2 ) * GEP + ( x & 3 ) * 8 ) = rd[i];
		}
	}
	__syncthreads();

	AccF o[2];
	#pragma unroll
	for ( int s = 0; s < 2; ++s ) wmma::fill_fragment( o[s], 0.0f );
	#pragma unroll
	for ( int kk = 0; kk < 8; ++kk ) {
		FragAR a;
		wmma::load_matrix_sync( a, s_x + r16 * GQP + kk * 16, GQP );
		#pragma unroll
		for ( int s = 0; s < 2; ++s ) {
			FragBC b;
			wmma::load_matrix_sync( b, ss + ( s * 16 ) * GKP + kk * 16, GKP );
			wmma::mma_sync( o[s], a, b, o[s] );
		}
	}
	__syncthreads();
	#pragma unroll
	for ( int i = 0; i < 4; ++i ) {
		const int x = tid + i * 128;
		*reinterpret_cast<uint4*>( s_x + ( x >> 3 ) * GTP + ( x & 7 ) * 8 ) = rp[i];
	}
	// G Q S0^T, row-scaled in the warp's staging, becomes the accumulator P Delta adds to
	float* z = s_s + warp * 16 * GPP;
	#pragma unroll
	for ( int s = 0; s < 2; ++s ) wmma::store_matrix_sync( z + s * 16, o[s], GPP, wmma::mem_row_major );
	__syncwarp();
	const float* g = cs.gam + hc * GC;
	const int er = lane >> 1, ec = ( lane & 1 ) * 16;
	{
		const float gt = expf( g[r16 + er] );
		#pragma unroll
		for ( int i = 0; i < 16; ++i ) z[er * GPP + ec + i] *= gt;
	}
	__syncwarp();
	#pragma unroll
	for ( int s = 0; s < 2; ++s ) wmma::load_matrix_sync( o[s], z + s * 16, GPP, wmma::mem_row_major );
	__syncthreads();
	for ( int k = 0; k <= warp; ++k ) {
		FragAR a;
		wmma::load_matrix_sync( a, s_x + r16 * GTP + k * 16, GTP );
		#pragma unroll
		for ( int s = 0; s < 2; ++s ) {
			FragBR b;
			wmma::load_matrix_sync( b, s_d + k * 16 * GEP + s * 16, GEP );
			wmma::mma_sync( o[s], a, b, o[s] );
		}
	}
	#pragma unroll
	for ( int s = 0; s < 2; ++s ) wmma::store_matrix_sync( z + s * 16, o[s], GPP, wmma::mem_row_major );
	__syncwarp();
	const int t = r16 + er;
	if ( t < n ) {
		alignas( 16 ) half out[16];
		#pragma unroll
		for ( int i = 0; i < 16; ++i ) out[i] = __float2half_rn( z[er * GPP + ec + i] * scale );
		half* op = output + static_cast<size_t>( r0 + t ) * h_v * 128 + h * 128 + v0 + ec;
		*reinterpret_cast<uint4*>( op ) = *reinterpret_cast<const uint4*>( out );
		*reinterpret_cast<uint4*>( op + 8 ) = *reinterpret_cast<const uint4*>( out + 8 );
	}
}

// ---- Chunked KDA (the prefill form of kda_recurrent_rows_kernel) ----
// Per head and 64-row chunk, with gamma_t the chunk's inclusive cumulative log decay of each key channel and S0 the state entering the chunk ([k][v]):
//   A_ts = beta_t sum_r k_tr k_sr exp( gamma_tr - gamma_sr ) (s < t), T = ( I + A )^-1
//   Delta = U - W S0, W = T diag( beta ) ( exp( gamma ) o K ), U = T diag( beta ) V
//   S = diag( exp( gamma_last ) ) S0 + Kbar^T Delta, Kbar_s = exp( gamma_last - gamma_s ) o k_s
//   Y = Qt S0 + P Delta, Qt_t = exp( gamma_t ) o q_t, P_ts = sum_r q_tr k_sr exp( gamma_tr - gamma_sr ) (s <= t)
// then each row's gated RMS norm. Every decay factor is the exponential of a difference that is at most 0, whatever the gate's range.
// kda_chunk_prep_kernel computes A and P in FP32 (each pair's decays exactly), T as the GDN form does, then W, U, Kbar and Qt; the scan and
// the output kernels follow the GDN form's, with the decay per key channel (the scan's state tile is the transpose of the KDA layout, [v][k]).
// The state's path splits the FP32 operands of its tensor-core products into FP16 hi + lo, as there; the output's products take FP16 operands.
constexpr int KQP = 36;          // shared pitch (floats) of the prep's 64 x 32 quarters
constexpr int KHP = 72;          // shared pitch (halves) of a 64 x 64 operand half
constexpr int KYP = 132;         // shared pitch (floats) of the output block's rows
constexpr size_t KPM = GC * GC, KQT = GC * 128, KKB = 2 * GC * 128, KKT = 128 * 128, KWH = 2 * GC * 128, KDD = GC * 128;   // halves per head and chunk
constexpr size_t KUU = GC * 128, KDL = 128;                                                                                // floats per head and chunk
constexpr size_t KSCAN_SMEM = 4 * GC * GKP * sizeof( half ) + 2 * GSV * GKP * sizeof( half ) + GC * GPP * sizeof( float );
constexpr size_t KOUT_SMEM = ( 2 * GC * GKP + 128 * GKP + GC * GTP ) * sizeof( half );
static_assert( GC * GPP * sizeof( float ) >= 2 * GSV * GTP * sizeof( half ) && GC * KYP * sizeof( float ) <= ( GC + 128 ) * GKP * sizeof( half ), "KDA scan / output regions" );

// Scratch of one call, per head and chunk ( c * heads + h ): P, Qt, Kbar (hi, lo), Ktilde = exp( gamma ) o K (hi, lo; the prep's) whose room then holds
// S0 ([v][k], the scan's), W (hi, lo), Delta, U and exp( gamma_last )
struct KdaScratch {
	half* pm;
	half* qt;
	half* kb;
	half* kt;
	half* wh;
	half* dd;
	float* uu;
	float* dl;
};

// One block per (head, chunk). A and P are summed over the four quarters of the key channels, with each quarter's K, Q and log2 gamma in shared
// memory: warp item ii is rows 4 tg .. 4 tg + 3 against keys s = 32 sh + lane (pairs above the diagonal are computed and dropped)
__global__ void __launch_bounds__( 256, 2 ) kda_chunk_prep_kernel(
	const half* __restrict__ qkv,
	const float* __restrict__ g_decay,
	const float* __restrict__ beta,
	KdaScratch ks,
	int heads,
	int rows
) {
	constexpr float LOG2E = 1.4426950408889634f;
	__shared__ __align__( 32 ) unsigned char s_raw[41984];
	__shared__ float s_b[GC], s_tot[8][32];
	float* kf = reinterpret_cast<float*>( s_raw );              // a quarter of K, Q and log2 gamma, [t][r]
	float* qf = kf + GC * KQP;
	float* gm = qf + GC * KQP;
	float* fa = reinterpret_cast<float*>( s_raw );              // then A (blocks), the solve's partial products and T (blocks)
	float* fm = fa + 10 * 256;
	float* ft = fm + 3 * 256;
	half* txh = reinterpret_cast<half*>( s_raw + 23552 );       // then T diag( beta ) (hi, lo; blocks), a half of Ktilde (hi, lo) or of V, per-warp staging
	half* txl = txh + 10 * 256;
	half* oh = reinterpret_cast<half*>( s_raw );
	half* ol = oh + GC * KHP;
	float* stg = reinterpret_cast<float*>( s_raw + 33792 );
	const int h = blockIdx.x, c = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
	const int channels = heads * 128, r0 = c * GC, n = min( GC, rows - r0 );
	const size_t stride = static_cast<size_t>( 3 ) * channels, hc = static_cast<size_t>( c ) * heads + h;
	const int lt = tid >> 2, lj = ( tid & 3 ) * 8;              // the thread's 8 elements of a quarter
	if ( tid < GC ) s_b[tid] = tid < n ? beta[static_cast<size_t>( r0 + tid ) * heads + h] : 0.0f;

	float acc_a[3][4], acc_p[3][4];
	#pragma unroll
	for ( int ii = 0; ii < 3; ++ii ) {
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) acc_a[ii][i] = acc_p[ii][i] = 0.0f;
	}
	for ( int qq = 0; qq < 4; ++qq ) {
		const int cq = qq * 32;
		{
			uint4 k8 = make_uint4( 0, 0, 0, 0 ), q8 = k8;
			float4 g0 = make_float4( 0.0f, 0.0f, 0.0f, 0.0f ), g1 = g0;
			if ( lt < n ) {
				const half* row = qkv + static_cast<size_t>( r0 + lt ) * stride + h * 128 + cq + lj;
				q8 = *reinterpret_cast<const uint4*>( row );
				k8 = *reinterpret_cast<const uint4*>( row + channels );
				const float* gr = g_decay + static_cast<size_t>( r0 + lt ) * channels + h * 128 + cq + lj;
				g0 = *reinterpret_cast<const float4*>( gr );
				g1 = *reinterpret_cast<const float4*>( gr + 4 );
			}
			__syncthreads();   // the previous quarter's readers are done
			const half* kh = reinterpret_cast<const half*>( &k8 );
			const half* qh = reinterpret_cast<const half*>( &q8 );
			float* kd = kf + lt * KQP + lj;
			float* qd = qf + lt * KQP + lj;
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) {
				kd[i] = __half2float( kh[i] );
				qd[i] = __half2float( qh[i] );
			}
			*reinterpret_cast<float4*>( gm + lt * KQP + lj ) = g0;
			*reinterpret_cast<float4*>( gm + lt * KQP + lj + 4 ) = g1;
		}
		__syncthreads();
		// gamma: warp w sums rows 8 w .. 8 w + 7 of channel `lane`, then adds the totals of the segments before it
		{
			float* gp = gm + warp * 8 * KQP + lane;
			float a = 0.0f, x[8];
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) {
				a += gp[i * KQP];
				x[i] = a;
			}
			s_tot[warp][lane] = a;
			__syncthreads();
			float pre = 0.0f;
			for ( int w = 0; w < warp; ++w ) pre += s_tot[w][lane];
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) gp[i * KQP] = ( x[i] + pre ) * LOG2E;
		}
		__syncthreads();
		// This quarter of Kbar and Ktilde (hi, lo), Qt and exp( gamma_last )
		{
			const float* kr = kf + lt * KQP + lj;
			const float* qr = qf + lt * KQP + lj;
			const float* gr = gm + lt * KQP + lj;
			const float* gl = gm + ( GC - 1 ) * KQP + lj;
			float xb[8], xt[8];
			alignas( 16 ) half xq[8];
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) {
				const float e = exp2f( gr[i] );
				xb[i] = kr[i] * exp2f( fminf( gl[i] - gr[i], 0.0f ) );
				xt[i] = kr[i] * e;
				xq[i] = __float2half_rn( qr[i] * e );
			}
			const size_t o = static_cast<size_t>( lt ) * 128 + cq + lj;
			split8( xb, ks.kb + hc * KKB + o, ks.kb + hc * KKB + GC * 128 + o );
			split8( xt, ks.kt + hc * KKT + o, ks.kt + hc * KKT + GC * 128 + o );
			*reinterpret_cast<uint4*>( ks.qt + hc * KQT + o ) = *reinterpret_cast<const uint4*>( xq );
			if ( tid < 32 ) ks.dl[hc * KDL + cq + tid] = exp2f( gm[( GC - 1 ) * KQP + tid] );
		}
		#pragma unroll
		for ( int ii = 0; ii < 3; ++ii ) {
			const int it = warp + ii * 8, tg = it < 16 ? it : it - 8, s = ( it < 16 ? 0 : 32 ) + lane;
			const float* gs = gm + s * KQP;
			const float* kss = kf + s * KQP;
			#pragma unroll 2
			for ( int r = 0; r < 32; r += 4 ) {
				const float4 g4 = *reinterpret_cast<const float4*>( gs + r );
				const float4 k4 = *reinterpret_cast<const float4*>( kss + r );
				#pragma unroll
				for ( int i = 0; i < 4; ++i ) {
					const int t = tg * 4 + i;
					const float4 gt = *reinterpret_cast<const float4*>( gm + t * KQP + r );
					const float4 kt = *reinterpret_cast<const float4*>( kf + t * KQP + r );
					const float4 qt = *reinterpret_cast<const float4*>( qf + t * KQP + r );
					float e = exp2f( fminf( gt.x - g4.x, 0.0f ) ) * k4.x;
					acc_a[ii][i] = fmaf( kt.x, e, acc_a[ii][i] );
					acc_p[ii][i] = fmaf( qt.x, e, acc_p[ii][i] );
					e = exp2f( fminf( gt.y - g4.y, 0.0f ) ) * k4.y;
					acc_a[ii][i] = fmaf( kt.y, e, acc_a[ii][i] );
					acc_p[ii][i] = fmaf( qt.y, e, acc_p[ii][i] );
					e = exp2f( fminf( gt.z - g4.z, 0.0f ) ) * k4.z;
					acc_a[ii][i] = fmaf( kt.z, e, acc_a[ii][i] );
					acc_p[ii][i] = fmaf( qt.z, e, acc_p[ii][i] );
					e = exp2f( fminf( gt.w - g4.w, 0.0f ) ) * k4.w;
					acc_a[ii][i] = fmaf( kt.w, e, acc_a[ii][i] );
					acc_p[ii][i] = fmaf( qt.w, e, acc_p[ii][i] );
				}
			}
		}
	}
	__syncthreads();

	// A (beta_t times the sum, s < t) into its blocks, P (s <= t) to the scratch; P's entries t < 32 <= s are zero
	#pragma unroll
	for ( int ii = 0; ii < 3; ++ii ) {
		const int it = warp + ii * 8, tg = it < 16 ? it : it - 8, s = ( it < 16 ? 0 : 32 ) + lane;
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			const int t = tg * 4 + i;
			if ( ( t >> 4 ) >= ( s >> 4 ) ) fa[tri_index( t >> 4, s >> 4 ) * 256 + ( t & 15 ) * 16 + ( s & 15 )] = s < t ? s_b[t] * acc_a[ii][i] : 0.0f;
			ks.pm[hc * KPM + t * GC + s] = __float2half_rn( s <= t ? acc_p[ii][i] : 0.0f );
		}
	}
	for ( int x = tid; x < 32 * 32; x += 256 ) ks.pm[hc * KPM + ( x >> 5 ) * GC + 32 + ( x & 31 )] = __float2half_rn( 0.0f );
	__syncthreads();
	solve_unit_lower( fa, ft, fm, tid );

	// T diag( beta ) (hi, lo)
	#pragma unroll
	for ( int blk = 0; blk < 10; ++blk ) {
		const int bj = blk == 0 ? 0 : blk < 3 ? blk - 1 : blk < 6 ? blk - 3 : blk - 6;   // tri_tile's column
		const int e = blk * 256 + tid;
		split_hl( ft[e] * s_b[bj * 16 + ( tid & 15 )], txh[e], txl[e] );
	}
	__syncthreads();

	// W = T diag( beta ) Ktilde (hi, lo, to the scratch) and U = T diag( beta ) V (FP32), 64 channels at a time: warp w takes row tile w & 3 and
	// column tiles 2 ( w >> 2 ), + 1 of the half
	const int rt = warp & 3, cn0 = ( warp >> 2 ) * 2, er = lane >> 1, ec = ( lane & 1 ) * 8;
	float* z = stg + warp * 256;
	for ( int hh = 0; hh < 2; ++hh ) {
		const int c0 = hh * 64;
		{
			uint4 x[4];
			#pragma unroll
			for ( int i = 0; i < 4; ++i ) {
				const int e = tid + i * 256, m = e >> 9, t = ( e >> 3 ) & 63, v = ( e & 7 ) * 8;
				x[i] = *reinterpret_cast<const uint4*>( ks.kt + hc * KKT + m * GC * 128 + t * 128 + c0 + v );
			}
			#pragma unroll
			for ( int i = 0; i < 4; ++i ) {
				const int e = tid + i * 256, m = e >> 9, t = ( e >> 3 ) & 63, v = ( e & 7 ) * 8;
				*reinterpret_cast<uint4*>( ( m ? ol : oh ) + t * KHP + v ) = x[i];
			}
		}
		__syncthreads();
		for ( int cn = cn0; cn < cn0 + 2; ++cn ) {
			AccF acc;
			wmma::fill_fragment( acc, 0.0f );
			for ( int k = 0; k <= rt; ++k ) {
				FragAR ah, al;
				FragBR bh, bl;
				wmma::load_matrix_sync( ah, txh + tri_index( rt, k ) * 256, 16 );
				wmma::load_matrix_sync( al, txl + tri_index( rt, k ) * 256, 16 );
				wmma::load_matrix_sync( bh, oh + k * 16 * KHP + cn * 16, KHP );
				wmma::load_matrix_sync( bl, ol + k * 16 * KHP + cn * 16, KHP );
				wmma::mma_sync( acc, ah, bh, acc );
				wmma::mma_sync( acc, ah, bl, acc );
				wmma::mma_sync( acc, al, bh, acc );
			}
			wmma::store_matrix_sync( z, acc, 16, wmma::mem_row_major );
			__syncwarp();
			const size_t o = static_cast<size_t>( rt * 16 + er ) * 128 + c0 + cn * 16 + ec;
			split8( z + er * 16 + ec, ks.wh + hc * KWH + o, ks.wh + hc * KWH + GC * 128 + o );
			__syncwarp();
		}
		__syncthreads();
		{
			uint4 x[2];
			#pragma unroll
			for ( int i = 0; i < 2; ++i ) {
				const int e = tid + i * 256, t = e >> 3, v = ( e & 7 ) * 8;
				x[i] = make_uint4( 0, 0, 0, 0 );
				if ( t < n ) x[i] = *reinterpret_cast<const uint4*>( qkv + static_cast<size_t>( r0 + t ) * stride + 2 * channels + h * 128 + c0 + v );
			}
			#pragma unroll
			for ( int i = 0; i < 2; ++i ) {
				const int e = tid + i * 256, t = e >> 3, v = ( e & 7 ) * 8;
				*reinterpret_cast<uint4*>( oh + t * KHP + v ) = x[i];
			}
		}
		__syncthreads();
		for ( int cn = cn0; cn < cn0 + 2; ++cn ) {
			AccF acc;
			wmma::fill_fragment( acc, 0.0f );
			for ( int k = 0; k <= rt; ++k ) {
				FragAR ah, al;
				FragBR b;
				wmma::load_matrix_sync( ah, txh + tri_index( rt, k ) * 256, 16 );
				wmma::load_matrix_sync( al, txl + tri_index( rt, k ) * 256, 16 );
				wmma::load_matrix_sync( b, oh + k * 16 * KHP + cn * 16, KHP );
				wmma::mma_sync( acc, ah, b, acc );
				wmma::mma_sync( acc, al, b, acc );
			}
			wmma::store_matrix_sync( ks.uu + hc * KUU + rt * 16 * 128 + c0 + cn * 16, acc, 128, wmma::mem_row_major );
		}
		__syncthreads();
	}
}

// One block per (head, 32 value columns): the chunks in order with the state tile ([v][k]) in registers, the next chunk's operands loading meanwhile;
// keeps each chunk's S0 and Delta for the output kernel. Warp w holds k columns 16 w .. 16 w + 15 of both 16-row halves of the tile; for Delta it
// takes row tile w & 3 of v half w >> 2.
__global__ void __launch_bounds__( 256, 1 ) kda_chunk_scan_kernel(
	float* __restrict__ state,
	KdaScratch ks,
	int heads,
	int rows,
	const pod::BatchRows* __restrict__ d_rows,
	size_t slot_stride
) {
	extern __shared__ __align__( 32 ) unsigned char kda_smem[];
	half* s_wh = reinterpret_cast<half*>( kda_smem );          // W (hi, lo), [t][k]
	half* s_wl = s_wh + GC * GKP;
	half* s_kh = s_wl + GC * GKP;                               // Kbar (hi, lo), [t][k]
	half* s_kl = s_kh + GC * GKP;
	half* s_sh = s_kl + GC * GKP;                               // S0 (hi, lo), [v][k]
	half* s_sl = s_sh + GSV * GKP;
	float* s_x = reinterpret_cast<float*>( s_sl + GSV * GKP );  // W S0^T, then Delta^T (hi, lo; [v][t]), then per-warp staging
	half* s_eh = reinterpret_cast<half*>( s_x );
	half* s_el = s_eh + GSV * GTP;
	__shared__ float s_dl[2][128];
	const int h = blockIdx.x, d0 = blockIdx.y * GSV, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, tr = warp & 3, ds = warp >> 2;
	const int chunks = ( rows + GC - 1 ) / GC;
	// Element ( v, k ) of the tile is at st[k * 128 + v - d0]
	float* st = state + static_cast<size_t>( d_rows->slot[0] ) * slot_stride + static_cast<size_t>( h ) * 128 * 128 + d0;
	auto hcoff = [&]( int c ) { return static_cast<size_t>( c ) * heads + h; };

	uint4 rw[8], rk[8];
	float4 ru[2];
	float rd = 0.0f;
	const int ut = tid >> 2, uj = ( tid & 3 ) * 8;              // the thread's 8 elements of Delta
	auto fetch = [&]( uint4* r, const half* src ) {
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) r[i] = *reinterpret_cast<const uint4*>( src + ( tid + i * 256 ) * 8 );
	};
	auto store = [&]( const uint4* r, half* hi, half* lo ) {
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) {
			const int x = ( tid + i * 256 ) * 8, m = x >> 13, o = ( ( x >> 7 ) & 63 ) * GKP + ( x & 127 );
			*reinterpret_cast<uint4*>( ( m ? lo : hi ) + o ) = r[i];
		}
	};
	auto fetch_u = [&]( int c ) {
		const float* u = ks.uu + hcoff( c ) * KUU + ut * 128 + d0 + uj;
		ru[0] = *reinterpret_cast<const float4*>( u );
		ru[1] = *reinterpret_cast<const float4*>( u + 4 );
	};
	auto fetch_d = [&]( int c ) {
		if ( tid < 128 ) rd = ks.dl[hcoff( c ) * KDL + tid];
	};
	AccF S[2];
	// The warp's columns of both halves of the state into s_sh / s_sl; with dl, the state then takes the decay of the chunk it enters
	auto split_state = [&]( const float* dl ) {
		float* z = s_x + warp * 256;
		const int r = lane >> 1, c0 = ( lane & 1 ) * 8, k0 = warp * 16 + c0;
		#pragma unroll
		for ( int f = 0; f < 2; ++f ) {
			wmma::store_matrix_sync( z, S[f], 16, wmma::mem_row_major );
			__syncwarp();
			float* zr = z + r * 16 + c0;
			split8( zr, s_sh + ( f * 16 + r ) * GKP + k0, s_sl + ( f * 16 + r ) * GKP + k0 );
			if ( dl != nullptr ) {
				#pragma unroll
				for ( int i = 0; i < 8; ++i ) zr[i] *= dl[k0 + i];
				__syncwarp();
				wmma::load_matrix_sync( S[f], z, 16, wmma::mem_row_major );
			}
			__syncwarp();
		}
	};

	#pragma unroll
	for ( int f = 0; f < 2; ++f ) wmma::load_matrix_sync( S[f], st + warp * 16 * 128 + f * 16, 128, wmma::mem_col_major );
	fetch( rw, ks.wh + hcoff( 0 ) * KWH );
	fetch( rk, ks.kb + hcoff( 0 ) * KKB );
	fetch_u( 0 );
	fetch_d( 0 );
	if ( tid < 128 ) s_dl[0][tid] = rd;
	__syncthreads();
	split_state( s_dl[0] );
	store( rw, s_wh, s_wl );
	__syncthreads();

	for ( int c = 0; c < chunks; ++c ) {
		const int b = c & 1, n = min( GC, rows - c * GC );
		const size_t hc = hcoff( c );
		const bool more = c + 1 < chunks;
		// This chunk's S0 to the scratch (the output kernel's), its Kbar into shared memory; the next chunk's W, Kbar and decay start loading
		{
			half* sc = ks.kt + hc * KKT + static_cast<size_t>( d0 ) * 128;
			#pragma unroll
			for ( int i = 0; i < 2; ++i ) {
				const int x = tid + i * 256, r = x >> 4, v = ( x & 15 ) * 8;
				*reinterpret_cast<uint4*>( sc + r * 128 + v ) = *reinterpret_cast<const uint4*>( s_sh + r * GKP + v );
			}
		}
		store( rk, s_kh, s_kl );
		if ( more ) {
			fetch( rw, ks.wh + hcoff( c + 1 ) * KWH );
			fetch( rk, ks.kb + hcoff( c + 1 ) * KKB );
			fetch_d( c + 1 );
		}

		// W S0^T, row tile tr of v half ds
		{
			AccF a1, a2, a3;
			wmma::fill_fragment( a1, 0.0f );
			wmma::fill_fragment( a2, 0.0f );
			wmma::fill_fragment( a3, 0.0f );
			#pragma unroll
			for ( int kk = 0; kk < 8; ++kk ) {
				FragAR ah, al;
				FragBC bh, bl;
				wmma::load_matrix_sync( ah, s_wh + tr * 16 * GKP + kk * 16, GKP );
				wmma::load_matrix_sync( al, s_wl + tr * 16 * GKP + kk * 16, GKP );
				wmma::load_matrix_sync( bh, s_sh + ds * 16 * GKP + kk * 16, GKP );
				wmma::load_matrix_sync( bl, s_sl + ds * 16 * GKP + kk * 16, GKP );
				wmma::mma_sync( a1, ah, bh, a1 );
				wmma::mma_sync( a2, ah, bl, a2 );
				wmma::mma_sync( a3, al, bh, a3 );
			}
			#pragma unroll
			for ( int i = 0; i < a1.num_elements; ++i ) a1.x[i] += a2.x[i] + a3.x[i];
			wmma::store_matrix_sync( s_x + tr * 16 * GPP + ds * 16, a1, GPP, wmma::mem_row_major );
		}
		__syncthreads();

		// Delta = U - W S0^T (to the scratch, FP16) and Delta^T (hi, lo) over W S0^T's room; the next chunk's W into shared memory, its U loading
		float dv[8];
		{
			const float* u = reinterpret_cast<const float*>( ru );
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) dv[i] = ut < n ? u[i] - s_x[ut * GPP + uj + i] : 0.0f;
		}
		__syncthreads();
		{
			alignas( 16 ) half dh[8];
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) dh[i] = __float2half_rn( dv[i] );
			*reinterpret_cast<uint4*>( ks.dd + hc * KDD + ut * 128 + d0 + uj ) = *reinterpret_cast<const uint4*>( dh );
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) split_hl( dv[i], s_eh[( uj + i ) * GTP + ut], s_el[( uj + i ) * GTP + ut] );
		}
		if ( more ) {
			store( rw, s_wh, s_wl );
			fetch_u( c + 1 );
			if ( tid < 128 ) s_dl[b ^ 1][tid] = rd;
		}
		__syncthreads();

		// S = diag( exp( gamma_last ) ) S0 (taken by split_state) + Kbar^T Delta over the warp's 16 k columns
		#pragma unroll
		for ( int k = 0; k < 4; ++k ) {
			FragBR kh, kl;
			wmma::load_matrix_sync( kh, s_kh + k * 16 * GKP + warp * 16, GKP );
			wmma::load_matrix_sync( kl, s_kl + k * 16 * GKP + warp * 16, GKP );
			#pragma unroll
			for ( int f = 0; f < 2; ++f ) {
				FragAR eh, el;
				wmma::load_matrix_sync( eh, s_eh + f * 16 * GTP + k * 16, GTP );
				wmma::load_matrix_sync( el, s_el + f * 16 * GTP + k * 16, GTP );
				wmma::mma_sync( S[f], eh, kh, S[f] );
				wmma::mma_sync( S[f], eh, kl, S[f] );
				wmma::mma_sync( S[f], el, kh, S[f] );
			}
		}
		__syncthreads();
		split_state( more ? s_dl[b ^ 1] : nullptr );
		__syncthreads();
	}
	#pragma unroll
	for ( int f = 0; f < 2; ++f ) wmma::store_matrix_sync( st + warp * 16 * 128 + f * 16, S[f], 128, wmma::mem_col_major );
}

// One block per (head, chunk): Y = Qt S0 + P Delta, then each row's gated RMS norm to the output. Warp w computes row tile w & 3 of value half w >> 2.
__global__ void __launch_bounds__( 256, 1 ) kda_chunk_out_kernel(
	KdaScratch ks,
	const half* __restrict__ z,
	const half* __restrict__ o_norm,
	half* __restrict__ output,
	int heads,
	float eps,
	int rows
) {
	extern __shared__ __align__( 32 ) unsigned char kda_out_smem[];
	half* s_q = reinterpret_cast<half*>( kda_out_smem );        // Qt [t][k]
	half* s_s = s_q + GC * GKP;                                  // S0 [v][k]
	half* s_d = s_s + 128 * GKP;                                 // Delta [t][v]
	half* s_p = s_d + GC * GKP;                                  // P [t][s]
	float* s_y = reinterpret_cast<float*>( kda_out_smem );       // then Y [t][v]
	const int h = blockIdx.x, c = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, rt = warp & 3, vh = warp >> 2;
	const int channels = heads * 128, r0 = c * GC, n = min( GC, rows - r0 );
	const size_t hc = static_cast<size_t>( c ) * heads + h;
	{
		const half* qt = ks.qt + hc * KQT;
		const half* sc = ks.kt + hc * KKT;
		const half* dd = ks.dd + hc * KDD;
		const half* pm = ks.pm + hc * KPM;
		uint4 rq[4], rs[8], rd[4], rp[2];
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			rq[i] = *reinterpret_cast<const uint4*>( qt + ( tid + i * 256 ) * 8 );
			rd[i] = *reinterpret_cast<const uint4*>( dd + ( tid + i * 256 ) * 8 );
		}
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) rs[i] = *reinterpret_cast<const uint4*>( sc + ( tid + i * 256 ) * 8 );
		#pragma unroll
		for ( int i = 0; i < 2; ++i ) rp[i] = *reinterpret_cast<const uint4*>( pm + ( tid + i * 256 ) * 8 );
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			const int x = tid + i * 256, o = ( x >> 4 ) * GKP + ( x & 15 ) * 8;
			*reinterpret_cast<uint4*>( s_q + o ) = rq[i];
			*reinterpret_cast<uint4*>( s_d + o ) = rd[i];
		}
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) {
			const int x = tid + i * 256;
			*reinterpret_cast<uint4*>( s_s + ( x >> 4 ) * GKP + ( x & 15 ) * 8 ) = rs[i];
		}
		#pragma unroll
		for ( int i = 0; i < 2; ++i ) {
			const int x = tid + i * 256;
			*reinterpret_cast<uint4*>( s_p + ( x >> 3 ) * GTP + ( x & 7 ) * 8 ) = rp[i];
		}
	}
	__syncthreads();

	AccF o[4];
	#pragma unroll
	for ( int j = 0; j < 4; ++j ) wmma::fill_fragment( o[j], 0.0f );
	#pragma unroll
	for ( int kk = 0; kk < 8; ++kk ) {
		FragAR a;
		wmma::load_matrix_sync( a, s_q + rt * 16 * GKP + kk * 16, GKP );
		#pragma unroll
		for ( int j = 0; j < 4; ++j ) {
			FragBC b;
			wmma::load_matrix_sync( b, s_s + ( vh * 64 + j * 16 ) * GKP + kk * 16, GKP );
			wmma::mma_sync( o[j], a, b, o[j] );
		}
	}
	for ( int k = 0; k <= rt; ++k ) {
		FragAR a;
		wmma::load_matrix_sync( a, s_p + rt * 16 * GTP + k * 16, GTP );
		#pragma unroll
		for ( int j = 0; j < 4; ++j ) {
			FragBR b;
			wmma::load_matrix_sync( b, s_d + k * 16 * GKP + vh * 64 + j * 16, GKP );
			wmma::mma_sync( o[j], a, b, o[j] );
		}
	}
	__syncthreads();
	#pragma unroll
	for ( int j = 0; j < 4; ++j ) wmma::store_matrix_sync( s_y + rt * 16 * KYP + vh * 64 + j * 16, o[j], KYP, wmma::mem_row_major );
	__syncthreads();

	// The gated RMS norm, four threads a row: thread q of row t takes columns 4 q + 16 i .. + 3
	const int t = tid >> 2, q = tid & 3;
	float y[32], ss = 0.0f;
	#pragma unroll
	for ( int i = 0; i < 8; ++i ) {
		const float4 v = *reinterpret_cast<const float4*>( s_y + t * KYP + q * 4 + i * 16 );
		y[i * 4] = v.x;
		y[i * 4 + 1] = v.y;
		y[i * 4 + 2] = v.z;
		y[i * 4 + 3] = v.w;
		ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	}
	ss += __shfl_xor_sync( 0xffffffffu, ss, 1 );
	ss += __shfl_xor_sync( 0xffffffffu, ss, 2 );
	const float inv_rms = rsqrtf( ss / 128.0f + eps );
	if ( t < n ) {
		const size_t base = static_cast<size_t>( r0 + t ) * channels + h * 128;
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) {
			const int col = q * 4 + i * 16;
			alignas( 8 ) half zz[4], gg[4], out[4];
			*reinterpret_cast<uint2*>( zz ) = z != nullptr ? *reinterpret_cast<const uint2*>( z + base + col ) : make_uint2( 0, 0 );
			if ( o_norm != nullptr ) *reinterpret_cast<uint2*>( gg ) = *reinterpret_cast<const uint2*>( o_norm + col );
			#pragma unroll
			for ( int e = 0; e < 4; ++e ) {
				const float gamma = o_norm != nullptr ? __half2float( gg[e] ) : 1.0f;
				out[e] = __float2half_rn( y[i * 4 + e] * inv_rms * gamma * engine::math::sigmoid( __half2float( zz[e] ) ) );
			}
			*reinterpret_cast<uint2*>( output + base + col ) = *reinterpret_cast<const uint2*>( out );
		}
	}
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

// ROCm: the rows kernel (the scan kernel's shared memory is above RDNA's 64 KB a workgroup)
size_t GDN::recurrent_step_chunked_scratch_bytes( int rows, int num_k_heads, int num_v_heads ) {
#if defined(USE_ROCM)
	return 0;
#endif
	const size_t hc = static_cast<size_t>( rows + GC - 1 ) / GC * num_v_heads;
	return hc * ( ( GPW + GSC + GDD ) * sizeof( half ) + ( 2 * GC * 128 + GC ) * sizeof( float ) ) + static_cast<size_t>( num_k_heads ) * 2 * GC * 128 * sizeof( half );
}

void GDN::recurrent_step_chunked(
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
	void* scratch,
	cudaStream_t stream
) {
#if defined(USE_ROCM)
	recurrent_step_rows( gdn_state, qkv, gate, beta, attn_out, num_k_heads, num_v_heads, head_dim, rows, d_rows, slot_stride, nullptr, 0, 0, nullptr, stream );
	return;
#endif
	ENGINE_ASSERT( head_dim == 128, "GDN::recurrent_step_chunked: the kernels are specialized for head_dim 128" );
	ENGINE_ASSERT( num_v_heads % num_k_heads == 0, "GDN::recurrent_step_chunked: v heads must be a multiple of k heads" );
	const int chunks = ( rows + GC - 1 ) / GC;
	const size_t hc = static_cast<size_t>( chunks ) * num_v_heads;
	ChunkScratch cs;
	cs.pw = static_cast<half*>( scratch );
	cs.sc = cs.pw + hc * GPW;
	cs.dd = cs.sc + hc * GSC;
	cs.pad = cs.dd + hc * GDD;
	cs.uu = reinterpret_cast<float*>( cs.pad + static_cast<size_t>( num_k_heads ) * 2 * GC * 128 );
	cs.gam = cs.uu + hc * 2 * GC * 128;
	static bool attr[ENGINE_NUM_GPUS] = {};
	int dev = 0;
	cudaGetDevice( &dev );
	if ( !attr[dev] ) {
		CUDA_CHECK( cudaFuncSetAttribute( gdn_chunk_scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>( GSCAN_SMEM ) ) );
		CUDA_CHECK( cudaFuncSetAttribute( gdn_chunk_prep_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100 ) );
		CUDA_CHECK( cudaFuncSetAttribute( gdn_chunk_out_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100 ) );
		attr[dev] = true;
	}
	const half* q = qkv.ptr<const half>();
	gdn_chunk_prep_kernel<<<dim3( num_v_heads, chunks ), 256, 0, stream>>>( q, gate.ptr<const float>(), beta.ptr<const float>(), cs, num_k_heads, num_v_heads, rows );
	gdn_chunk_scan_kernel<<<dim3( num_v_heads, 128 / GSV ), 256, GSCAN_SMEM, stream>>>( gdn_state, q, cs, num_k_heads, num_v_heads, rows, d_rows, slot_stride );
	gdn_chunk_out_kernel<<<dim3( num_v_heads, chunks, 128 / GSV ), 128, 0, stream>>>( q, cs, attn_out.ptr<half>(), num_k_heads, num_v_heads,
		1.0f / sqrtf( static_cast<float>( head_dim ) ), rows );
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
	if ( rows > pod::MAX_BATCH && runtime::options().kda_rows ) {
		constexpr int RB = 8;
		project_kda_stage1_rows_kernel<RB><<<dim3( num_heads + 256, ( rows + RB - 1 ) / RB ), dim3( 256 ), 0, stream>>>(
			norm_hidden.ptr<const half>(), b_proj.ptr<const half>(), f_a_proj.ptr<const half>(), g_a_proj.ptr<const half>(),
			beta_out.ptr<float>(), f_a_buf.ptr<half>(), g_a_buf.ptr<half>(), hidden_size, num_heads, rows );
		const int channels = num_heads * head_dim;
		project_kda_stage2_rows_kernel<RB><<<dim3( ( channels + 255 ) / 256, ( rows + RB - 1 ) / RB ), dim3( 256 ), 0, stream>>>(
			f_a_buf.ptr<const half>(), g_a_buf.ptr<const half>(), f_b_proj.ptr<const half>(), g_b_proj.ptr<const half>(),
			A_log.ptr<const half>(), dt_bias.ptr<const half>(), g_decay_out.ptr<float>(), z_out.ptr<half>(), channels, rows );
		return;
	}
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
		snap, snap_stride, snap_rows, snap_on, nullptr, 0, 0
	);
}

// ROCm: the rows kernel (as recurrent_step_chunked)
size_t GDN::recurrent_step_kda_chunked_scratch_bytes( int rows, int num_heads ) {
#if defined(USE_ROCM)
	return 0;
#endif
	const size_t hc = static_cast<size_t>( rows + GC - 1 ) / GC * num_heads;
	return hc * ( ( KPM + KQT + KKB + KKT + KWH + KDD ) * sizeof( half ) + ( KUU + KDL ) * sizeof( float ) );
}

void GDN::recurrent_step_kda_chunked(
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
	void* scratch,
	cudaStream_t stream
) {
#if defined(USE_ROCM)
	recurrent_step_kda_rows( gdn_state, qkv, g_decay, beta, z, o_norm, attn_out, num_heads, head_dim, eps, rows, d_rows, slot_stride, nullptr, 0, 0, nullptr, stream );
	return;
#endif
	ENGINE_ASSERT( head_dim == 128, "GDN::recurrent_step_kda_chunked: the kernels are specialized for head_dim 128" );
	const int chunks = ( rows + GC - 1 ) / GC;
	const size_t hc = static_cast<size_t>( chunks ) * num_heads;
	KdaScratch ks;
	ks.pm = static_cast<half*>( scratch );
	ks.qt = ks.pm + hc * KPM;
	ks.kb = ks.qt + hc * KQT;
	ks.kt = ks.kb + hc * KKB;
	ks.wh = ks.kt + hc * KKT;
	ks.dd = ks.wh + hc * KWH;
	ks.uu = reinterpret_cast<float*>( ks.dd + hc * KDD );
	ks.dl = ks.uu + hc * KUU;
	static bool attr[ENGINE_NUM_GPUS] = {};
	int dev = 0;
	cudaGetDevice( &dev );
	if ( !attr[dev] ) {
		CUDA_CHECK( cudaFuncSetAttribute( kda_chunk_scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>( KSCAN_SMEM ) ) );
		CUDA_CHECK( cudaFuncSetAttribute( kda_chunk_out_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>( KOUT_SMEM ) ) );
		CUDA_CHECK( cudaFuncSetAttribute( kda_chunk_prep_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100 ) );
		attr[dev] = true;
	}
	kda_chunk_prep_kernel<<<dim3( num_heads, chunks ), 256, 0, stream>>>( qkv.ptr<const half>(), g_decay.ptr<const float>(), beta.ptr<const float>(), ks, num_heads, rows );
	kda_chunk_scan_kernel<<<dim3( num_heads, 128 / GSV ), 256, KSCAN_SMEM, stream>>>( gdn_state, ks, num_heads, rows, d_rows, slot_stride );
	kda_chunk_out_kernel<<<dim3( num_heads, chunks ), 256, KOUT_SMEM, stream>>>( ks, z.ptr<const half>(), o_norm.ptr<const half>(), attn_out.ptr<half>(), num_heads, eps, rows );
}

bool GDN::kda_journal_supported() { return true; }

size_t GDN::kda_journal_row_bytes( int num_heads ) {
	return static_cast<size_t>( num_heads ) * 128 * 8 + static_cast<size_t>( num_heads ) * sizeof( float );   // k, v (half), decay (float), beta
}

void GDN::recurrent_step_kda_rows_journal(
	float* gdn_state, const pod::TensorView& qkv, const pod::TensorView& g_decay, const pod::TensorView& beta, const pod::TensorView& z,
	const pod::TensorView& o_norm, pod::TensorView& attn_out, int num_heads, int head_dim, float eps, int rows, const pod::BatchRows* d_rows,
	size_t slot_stride, unsigned char* journal, size_t journal_row_stride, int journal_rows, const int* journal_on, backend::stream_t stream
) {
	ENGINE_ASSERT( head_dim == 128, "GDN::recurrent_step_kda_rows_journal: head_dim must be 128" );
	kda_recurrent_rows_kernel<<<dim3( num_heads ), dim3( 128 ), 0, stream>>>(
		gdn_state, qkv.ptr<const half>(), g_decay.ptr<const float>(), beta.ptr<const float>(), z.ptr<const half>(),
		o_norm.ptr<const half>(), attn_out.ptr<half>(), num_heads, eps, rows, d_rows, slot_stride,
		nullptr, 0, 0, journal_on, journal, journal_row_stride, journal_rows
	);
}

void GDN::kda_replay( float* gdn_state, size_t slot_stride, size_t layer_stride, int layers, int slot, const unsigned char* journal,
	size_t journal_layer_stride, size_t journal_row_stride, int first_row, int nrows, int num_heads, backend::stream_t stream ) {
	if ( nrows <= 0 ) return;
	kda_replay_kernel<<<dim3( num_heads, layers ), dim3( 128 ), 0, stream>>>(
		gdn_state, slot_stride, layer_stride, journal, journal_layer_stride, journal_row_stride, slot, first_row, nrows, num_heads );
}

} // namespace engine::modules
