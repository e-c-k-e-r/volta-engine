#include "engine/modules/attention.h"
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>

namespace engine::modules {

namespace {

#define ATTN_PAGE_SIZE 256
#define ATTN_MAX_CHUNKS 32
#define ATTN_MAX_HEAD_DIM 256
#define ATTN_MAX_ROW_HEADS ( pod::MAX_BATCH * 16 )

__device__ float g_attn_workspace[ATTN_MAX_ROW_HEADS * ATTN_MAX_CHUNKS * ( ATTN_MAX_HEAD_DIM + 2 )];

// Device address of g_attn_workspace on the current GPU. The bare symbol in host code is the
// host shadow, not a usable kernel argument; resolve once per device.
float* attn_workspace_ptr() {
	static float* ptrs[ENGINE_NUM_GPUS] = {};
	int dev = 0;
	CUDA_CHECK( cudaGetDevice( &dev ) );
	ENGINE_ASSERT( dev >= 0 && dev < ENGINE_NUM_GPUS, "attn_workspace_ptr: device out of range" );
	if ( !ptrs[dev] ) {
		void* p = nullptr;
		CUDA_CHECK( cudaGetSymbolAddress( &p, g_attn_workspace ) );
		ptrs[dev] = static_cast<float*>( p );
	}
	return ptrs[dev];
}

__global__ void unpack_q_gate_kernel(
	const half* __restrict__ q_gate,
	half* __restrict__ q,
	half* __restrict__ gate,
	int num_q_heads,
	int head_dim,
	int M
) {
	int head = blockIdx.x;
	int m = blockIdx.y;
	int tid = threadIdx.x;
	if ( head >= num_q_heads || m >= M || tid >= head_dim ) return;

	size_t src_offset = ( static_cast<size_t>( m ) * num_q_heads + head ) * 2 * head_dim;
	size_t dst_offset = ( static_cast<size_t>( m ) * num_q_heads + head ) * head_dim + tid;

	q[dst_offset] = q_gate[src_offset + tid];
	gate[dst_offset] = q_gate[src_offset + head_dim + tid];
}

__global__ void store_kv_cache_kernel(
	half* __restrict__ k_cache,
	half* __restrict__ v_cache,
	const half* __restrict__ k,
	const half* __restrict__ v,
	size_t slot_stride,
	int layer_idx,
	const pod::BatchRows* __restrict__ rows,
	int count,
	int num_kv_heads,
	int head_dim,
	int max_seq
) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_elems = num_kv_heads * head_dim;
	if ( tid < total_elems * count ) {
		int m = tid / total_elems;
		int elem = tid % total_elems;
		size_t offset = static_cast<size_t>( rows->slot[m] ) * slot_stride
			+ ( static_cast<size_t>( layer_idx ) * max_seq + rows->pos[m] ) * total_elems + elem;
		k_cache[offset] = k[tid];
		v_cache[offset] = v[tid];
	}
}

__global__ void attn_chunked_paged_kernel(
	const half* __restrict__ q,
	const half* __restrict__ k_cache,
	const half* __restrict__ v_cache,
	const half* __restrict__ gate,
	half* __restrict__ out,
	float* __restrict__ workspace,
	size_t slot_stride,
	int layer_idx,
	const pod::BatchRows* __restrict__ rows,
	int num_q_heads,
	int num_kv_heads,
	int head_dim,
	int max_seq,
	int n_chunks,
	float scale
) {
	int head = blockIdx.x;
	int m = blockIdx.y;
	int chunk = blockIdx.z;
	int tid = threadIdx.x;
	if ( head >= num_q_heads || tid >= head_dim ) return;

	int heads_per_kv = num_q_heads / num_kv_heads;
	if ( heads_per_kv < 1 ) heads_per_kv = 1;
	int kv_head = head / heads_per_kv;

	// Chunks past this row's context write an empty (l = 0) partial that the reduce skips, so
	// the grid can be sized by max_seq and stay valid across positions (graph replay)
	int cur_pos = rows->pos[m];
	int total_tokens = min( cur_pos + 1, max_seq );
	k_cache += static_cast<size_t>( rows->slot[m] ) * slot_stride;
	v_cache += static_cast<size_t>( rows->slot[m] ) * slot_stride;

	int kv_start = chunk * ATTN_PAGE_SIZE;
	int kv_end = min( kv_start + ATTN_PAGE_SIZE, total_tokens );

	const half* q_head = q + ( static_cast<size_t>( m ) * num_q_heads + head ) * head_dim;
	half* out_head = out + ( static_cast<size_t>( m ) * num_q_heads + head ) * head_dim;

	float my_q = __half2float( q_head[tid] );

	__shared__ float s_warp_sums[8];
	__shared__ float s_score;

	int warp_id = tid >> 5;
	int lane_id = tid & 31;
	int num_warps = ( head_dim + 31 ) >> 5;

	float m_reg = -1e30f;
	float l_reg = 0.0f;
	float o_dim = 0.0f;

	if ( kv_start < total_tokens ) {
		for ( int t = kv_start; t < kv_end; ++t ) {
			size_t kv_offset = ( static_cast<size_t>( layer_idx ) * max_seq + t ) * ( num_kv_heads * head_dim ) + kv_head * head_dim + tid;
			float my_k = __half2float( k_cache[kv_offset] );
			float dot = my_q * my_k;

			#pragma unroll
			for ( int mask = 16; mask > 0; mask >>= 1 ) {
				dot += __shfl_xor_sync( 0xffffffff, dot, mask );
			}

			if ( lane_id == 0 ) {
				s_warp_sums[warp_id] = dot;
			}
			__syncthreads();

			if ( warp_id == 0 ) {
				float b_sum = ( lane_id < num_warps ) ? s_warp_sums[lane_id] : 0.0f;
				#pragma unroll
				for ( int mask = 4; mask > 0; mask >>= 1 ) {
					b_sum += __shfl_xor_sync( 0xffffffff, b_sum, mask );
				}
				if ( lane_id == 0 ) {
					s_score = b_sum * scale;
				}
			}
			__syncthreads();

			float my_v = __half2float( v_cache[kv_offset] );
			float score = s_score;

			float m_new = fmaxf( m_reg, score );
			float alpha = expf( m_reg - m_new );
			float exp_s = expf( score - m_new );
			o_dim = alpha * o_dim + exp_s * my_v;
			l_reg = alpha * l_reg + exp_s;
			m_reg = m_new;
		}
	}

	if ( n_chunks == 1 ) {
		float inv_l = ( l_reg > 0.0f ) ? ( 1.0f / l_reg ) : 0.0f;
		float sig = 1.0f;
		if ( gate != nullptr ) {
			size_t g_off = ( static_cast<size_t>( m ) * num_q_heads + head ) * head_dim + tid;
			float g_val = __half2float( gate[g_off] );
			sig = 1.0f / ( 1.0f + __expf( -g_val ) );
		}
		out_head[tid] = __float2half_rn( o_dim * inv_l * sig );
	}
	else {
		float* ws = workspace + ( ( static_cast<size_t>( m ) * num_q_heads + head ) * n_chunks + chunk ) * ( ATTN_MAX_HEAD_DIM + 2 );
		if ( tid == 0 ) {
			ws[0] = m_reg;
			ws[1] = l_reg;
		}
		ws[2 + tid] = o_dim;
	}
}

template <int D>
__global__ void attn_reduce_kernel(
	const float* __restrict__ workspace,
	const half* __restrict__ gate,
	half* __restrict__ output,
	int num_q_heads,
	int n_chunks
) {
	const int token_idx = blockIdx.y;
	const int q_head = blockIdx.x;
	const int tid = threadIdx.x;
	if ( q_head >= num_q_heads || tid >= D ) return;

	float m_acc = -1e30f;
	float l_acc = 0.0f;
	float o_acc = 0.0f;

	const float* ws_base = workspace + ( static_cast<size_t>( token_idx ) * num_q_heads + q_head ) * n_chunks * ( ATTN_MAX_HEAD_DIM + 2 );

	for ( int c = 0; c < n_chunks; ++c ) {
		const float* ws_c = ws_base + static_cast<size_t>( c ) * ( ATTN_MAX_HEAD_DIM + 2 );
		float m_c = ws_c[0];
		float l_c = ws_c[1];
		float o_c = ws_c[2 + tid];

		if ( l_c == 0.0f ) continue;

		float m_new = fmaxf( m_acc, m_c );
		float alpha = __expf( m_acc - m_new );
		float beta = __expf( m_c - m_new );

		o_acc = fmaf( alpha, o_acc, beta * o_c );
		l_acc = fmaf( alpha, l_acc, beta * l_c );
		m_acc = m_new;
	}

	float inv_l = ( l_acc > 0.0f ) ? ( 1.0f / l_acc ) : 0.0f;
	float sig = 1.0f;
	if ( gate != nullptr ) {
		size_t g_off = ( static_cast<size_t>( token_idx ) * num_q_heads + q_head ) * D + tid;
		float g_val = __half2float( gate[g_off] );
		sig = 1.0f / ( 1.0f + __expf( -g_val ) );
	}
	size_t out_off = ( static_cast<size_t>( token_idx ) * num_q_heads + q_head ) * D + tid;
	output[out_off] = __float2half_rn( o_acc * inv_l * sig );
}

} // namespace

void Attention::unpack_q_gate(
	const pod::TensorView& q_gate,
	pod::TensorView& q,
	pod::TensorView& gate,
	int num_q_heads,
	int head_dim,
	cudaStream_t stream
) {
	int M = ( q_gate.ndim >= 2 && q_gate.shape[0] > 0 ) ? q_gate.shape[0] : 1;
	dim3 grid( num_q_heads, M );
	dim3 block( head_dim );
	unpack_q_gate_kernel<<<grid, block, 0, stream>>>(
		q_gate.ptr<const half>(),
		q.ptr<half>(),
		gate.ptr<half>(),
		num_q_heads,
		head_dim,
		M
	);
}

void Attention::store_kv_cache(
	half* k_cache,
	half* v_cache,
	const pod::TensorView& k,
	const pod::TensorView& v,
	size_t slot_stride,
	int layer_idx,
	const pod::BatchRows* d_rows,
	int count,
	int num_kv_heads,
	int head_dim,
	int max_seq,
	cudaStream_t stream
) {
	int total_elems = num_kv_heads * head_dim;
	int total_work = total_elems * count;
	int threads = 256;
	int blocks = ( total_work + threads - 1 ) / threads;
	store_kv_cache_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
		k_cache,
		v_cache,
		k.ptr<const half>(),
		v.ptr<const half>(),
		slot_stride,
		layer_idx,
		d_rows,
		count,
		num_kv_heads,
		head_dim,
		max_seq
	);
}

void Attention::forward_decode(
	const pod::TensorView& q,
	const half* k_cache,
	const half* v_cache,
	const pod::TensorView& gate,
	pod::TensorView& out,
	size_t slot_stride,
	int layer_idx,
	const pod::BatchRows* d_rows,
	int count,
	int num_q_heads,
	int num_kv_heads,
	int head_dim,
	int max_seq,
	cudaStream_t stream
) {
	const int M = count;
	ENGINE_ASSERT( M * num_q_heads <= ATTN_MAX_ROW_HEADS, "Attention::forward_decode: workspace too small for rows x heads" );
	ENGINE_ASSERT( max_seq <= ATTN_MAX_CHUNKS * ATTN_PAGE_SIZE, "Attention::forward_decode: max_seq exceeds chunk capacity" );
	float scale = 1.0f / sqrtf( static_cast<float>( head_dim ) );

	// Position-independent grid: one chunk per page of the full window (graph-capturable)
	int n_chunks = MAX( 1, ( max_seq + ATTN_PAGE_SIZE - 1 ) / ATTN_PAGE_SIZE );
	float* workspace = attn_workspace_ptr();

	dim3 grid( num_q_heads, M, n_chunks );
	dim3 block( head_dim );
	size_t smem = 0;

	attn_chunked_paged_kernel<<<grid, block, smem, stream>>>(
		q.ptr<const half>(),
		k_cache,
		v_cache,
		gate.ptr<const half>(),
		out.ptr<half>(),
		workspace,
		slot_stride,
		layer_idx,
		d_rows,
		num_q_heads,
		num_kv_heads,
		head_dim,
		max_seq,
		n_chunks,
		scale
	);

	if ( n_chunks > 1 ) {
		dim3 red_grid( num_q_heads, M );
		dim3 red_block( head_dim );
		if ( head_dim == 256 ) {
			attn_reduce_kernel<256><<<red_grid, red_block, 0, stream>>>(
				workspace,
				gate.ptr<const half>(),
				out.ptr<half>(),
				num_q_heads,
				n_chunks
			);
		}
		else if ( head_dim == 128 ) {
			attn_reduce_kernel<128><<<red_grid, red_block, 0, stream>>>(
				workspace,
				gate.ptr<const half>(),
				out.ptr<half>(),
				num_q_heads,
				n_chunks
			);
		}
		else {
			attn_reduce_kernel<64><<<red_grid, red_block, 0, stream>>>(
				workspace,
				gate.ptr<const half>(),
				out.ptr<half>(),
				num_q_heads,
				n_chunks
			);
		}
	}
}

} // namespace engine::modules
