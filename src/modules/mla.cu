#include "engine/modules/mla.h"
#include "engine/modules/attention.h"
#include <cuda_fp16.h>
#include <cmath>

namespace engine::modules {

namespace {

__inline__ __device__ float warp_reduce_sum( float val ) {
	#pragma unroll
	for ( int offset = 16; offset > 0; offset >>= 1 ) {
		val += __shfl_down_sync( 0xffffffff, val, offset );
	}
	return val;
}

__global__ void store_mla_cache_kernel(
	half* __restrict__ latent_cache,
	half* __restrict__ rope_cache,
	const half* __restrict__ k_lat,
	const half* __restrict__ k_pe,
	int layer_idx,
	int pos,
	int mla_dc,
	int mla_dr,
	int max_seq
) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t slot = static_cast<size_t>( layer_idx ) * max_seq + pos;

	if ( tid < mla_dc && k_lat != nullptr ) {
		latent_cache[slot * mla_dc + tid] = k_lat[tid];
	}
	if ( tid < mla_dr && k_pe != nullptr ) {
		rope_cache[slot * mla_dr + tid] = k_pe[tid];
	}
}

__global__ __launch_bounds__( 256 )
void dsa_mla_decode_kernel(
	const half* __restrict__ q_lat,
	const half* __restrict__ q_pe,
	const half* __restrict__ latent_cache,
	const half* __restrict__ rope_cache,
	half* __restrict__ out,
	int layer_idx,
	int current_pos,
	int dsa_top_k,
	int mla_dc,
	int mla_dr,
	int max_seq,
	int out_dim
) {
	int tid = threadIdx.x;
	int wid = tid >> 5;
	int lane = tid & 31;

	__shared__ float s_scores[128];
	__shared__ float s_inv_sum;

	int num_keys = ( current_pos < dsa_top_k ) ? ( current_pos + 1 ) : dsa_top_k;
	if ( num_keys <= 0 ) num_keys = 1;
	if ( num_keys > 128 ) num_keys = 128;

	float scale = 1.0f / sqrtf( static_cast<float>( mla_dc + mla_dr ) );

	// 1. Compute dot product Q . K for each sparse token position across warps
	for ( int k = wid; k < num_keys; k += 8 ) {
		int pos = ( current_pos >= dsa_top_k ) ? ( current_pos - dsa_top_k + 1 + k ) : k;
		size_t slot = static_cast<size_t>( layer_idx ) * max_seq + pos;

		const half* k_lat = latent_cache + slot * mla_dc;
		const half* k_pe = rope_cache + slot * mla_dr;

		float dot_lat = 0.0f;
		for ( int d = lane; d < mla_dc; d += 32 ) {
			float q_val = ( q_lat != nullptr ) ? __half2float( q_lat[d] ) : 0.0f;
			dot_lat = fmaf( q_val, __half2float( k_lat[d] ), dot_lat );
		}

		float dot_pe = 0.0f;
		if ( q_pe != nullptr && mla_dr > 0 ) {
			for ( int d = lane; d < mla_dr; d += 32 ) {
				dot_pe = fmaf( __half2float( q_pe[d] ), __half2float( k_pe[d] ), dot_pe );
			}
		}

		float total_dot = warp_reduce_sum( dot_lat + dot_pe );
		if ( lane == 0 ) {
			s_scores[k] = total_dot * scale;
		}
	}
	__syncthreads();

	// 2. Softmax normalization
	if ( tid == 0 ) {
		float max_val = s_scores[0];
		for ( int k = 1; k < num_keys; ++k ) {
			if ( s_scores[k] > max_val ) {
				max_val = s_scores[k];
			}
		}

		float sum_exp = 0.0f;
		for ( int k = 0; k < num_keys; ++k ) {
			float p = expf( s_scores[k] - max_val );
			s_scores[k] = p;
			sum_exp += p;
		}
		s_inv_sum = 1.0f / ( sum_exp + 1e-9f );
	}
	__syncthreads();

	for ( int k = tid; k < num_keys; k += blockDim.x ) {
		s_scores[k] *= s_inv_sum;
	}
	__syncthreads();

	// 3. Output reduction into latent dimensions
	for ( int d = tid; d < out_dim; d += blockDim.x ) {
		if ( d < mla_dc ) {
			float acc = 0.0f;
			for ( int k = 0; k < num_keys; ++k ) {
				int pos = ( current_pos >= dsa_top_k ) ? ( current_pos - dsa_top_k + 1 + k ) : k;
				size_t slot = static_cast<size_t>( layer_idx ) * max_seq + pos;

				const half* v_lat = latent_cache + slot * mla_dc;
				acc = fmaf( s_scores[k], __half2float( v_lat[d] ), acc );
			}
			out[d] = __float2half_rn( acc );
		}
		else {
			out[d] = 0;
		}
	}
}

__global__ void mla_kv_b_gemv_unpack_kernel(
	const half* __restrict__ W,
	const half* __restrict__ x,
	half* __restrict__ k_out,
	half* __restrict__ v_out,
	int num_heads,
	int head_dim
) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int total_rows = num_heads * ( head_dim * 2 );
	if ( row >= total_rows ) return;
	const int m = blockIdx.y;
	const int out_stride = num_heads * head_dim;
	k_out += static_cast<size_t>( m ) * out_stride;
	v_out += static_cast<size_t>( m ) * out_stride;

	const half2* W2 = reinterpret_cast<const half2*>( W + row * 512 );
	const half2* x2 = reinterpret_cast<const half2*>( x + static_cast<size_t>( m ) * 512 );

	float sum = 0.0f;
	#pragma unroll 16
	for ( int i = 0; i < 256; ++i ) {
		float2 w = __half22float2( W2[i] );
		float2 xv = __half22float2( x2[i] );
		sum += w.x * xv.x + w.y * xv.y;
	}

	int h = row / ( head_dim * 2 );
	int rem = row % ( head_dim * 2 );
	if ( rem < head_dim ) {
		k_out[h * head_dim + rem] = __float2half_rn( sum );
	}
	else {
		v_out[h * head_dim + ( rem - head_dim )] = __float2half_rn( sum );
	}
}

__global__ void mla_store_kv_cache_kernel(
	half* __restrict__ k_cache,
	half* __restrict__ v_cache,
	const half* __restrict__ k,
	const half* __restrict__ v,
	int mla_layer_idx,
	int pos,
	int total_elems,
	int max_seq,
	const pod::SlotRef slot
) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int m = blockIdx.y;
	if ( slot.rows ) pos = slot.rows->pos[slot.row + m];
	k_cache += slot.rows ? static_cast<size_t>( slot.rows->slot[slot.row + m] ) * slot.stride : 0;
	v_cache += slot.rows ? static_cast<size_t>( slot.rows->slot[slot.row + m] ) * slot.stride : 0;
	k += static_cast<size_t>( m ) * total_elems;
	v += static_cast<size_t>( m ) * total_elems;
	if ( tid < total_elems ) {
		size_t offset = ( static_cast<size_t>( mla_layer_idx ) * max_seq + pos ) * total_elems + tid;
		k_cache[offset] = k[tid];
		v_cache[offset] = v[tid];
	}
}

__global__ __launch_bounds__( 256 )
void mla_attend_decode_kernel(
	const half* __restrict__ q,
	const half* __restrict__ k_cache,
	const half* __restrict__ v_cache,
	half* __restrict__ out,
	int mla_layer_idx,
	int pos,
	int num_heads,
	int head_dim,
	int max_seq,
	const pod::SlotRef slot
) {
	int head = blockIdx.x;
	int tid = threadIdx.x;
	if ( head >= num_heads || tid >= head_dim ) return;
	const int m = blockIdx.y;
	if ( slot.rows ) {
		pos = slot.rows->pos[slot.row + m];
		const size_t base = static_cast<size_t>( slot.rows->slot[slot.row + m] ) * slot.stride;
		k_cache += base;
		v_cache += base;
	}

	int total_elems = num_heads * head_dim;
	const half* q_head = q + static_cast<size_t>( m ) * total_elems + head * head_dim;
	half* out_head = out + static_cast<size_t>( m ) * total_elems + head * head_dim;

	float my_q = __half2float( q_head[tid] );
	float scale = 1.0f / sqrtf( static_cast<float>( head_dim ) );

	int num_tokens = pos + 1;
	if ( num_tokens > max_seq ) num_tokens = max_seq;

	extern __shared__ float s_scores[];
	__shared__ float s_warp_sums[8];
	__shared__ float s_inv_sum;

	int warp_id = tid >> 5;
	int lane_id = tid & 31;
	int num_warps = ( head_dim + 31 ) >> 5;

	// 1. Compute dot product Q . K for all past tokens
	for ( int t = 0; t < num_tokens; ++t ) {
		size_t kv_offset = ( static_cast<size_t>( mla_layer_idx ) * max_seq + t ) * total_elems + head * head_dim + tid;
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
				s_scores[t] = b_sum * scale;
			}
		}
		__syncthreads();
	}

	// 2. Softmax: max, exp, sum, normalize
	if ( tid == 0 ) {
		float max_val = -1e30f;
		for ( int t = 0; t < num_tokens; ++t ) {
			if ( s_scores[t] > max_val ) {
				max_val = s_scores[t];
			}
		}

		float sum_exp = 0.0f;
		for ( int t = 0; t < num_tokens; ++t ) {
			float val = expf( s_scores[t] - max_val );
			s_scores[t] = val;
			sum_exp += val;
		}
		s_inv_sum = 1.0f / ( sum_exp + 1e-9f );
	}
	__syncthreads();

	for ( int t = tid; t < num_tokens; t += blockDim.x ) {
		s_scores[t] *= s_inv_sum;
	}
	__syncthreads();

	// 3. Accumulate V weighted by softmax scores
	float o_dim = 0.0f;
	for ( int t = 0; t < num_tokens; ++t ) {
		size_t kv_offset = ( static_cast<size_t>( mla_layer_idx ) * max_seq + t ) * total_elems + head * head_dim + tid;
		float my_v = __half2float( v_cache[kv_offset] );
		o_dim = fmaf( s_scores[t], my_v, o_dim );
	}

	out_head[tid] = __float2half_rn( o_dim );
}

} // namespace

void MLADSA::store_cache(
	half* latent_cache,
	half* rope_cache,
	const half* k_lat,
	const half* k_pe,
	int layer_idx,
	int pos,
	int mla_dc,
	int mla_dr,
	int max_seq,
	cudaStream_t stream
) {
	int max_dim = ( mla_dc > mla_dr ) ? mla_dc : mla_dr;
	int threads = 256;
	int blocks = ( max_dim + threads - 1 ) / threads;
	store_mla_cache_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
		latent_cache,
		rope_cache,
		k_lat,
		k_pe,
		layer_idx,
		pos,
		mla_dc,
		mla_dr,
		max_seq
	);
}

void MLADSA::forward_decode(
	const pod::TensorView& q_lat,
	const pod::TensorView& q_pe,
	const half* latent_cache,
	const half* rope_cache,
	pod::TensorView& out,
	int layer_idx,
	int current_pos,
	int dsa_top_k,
	int mla_dc,
	int mla_dr,
	int max_seq,
	int out_dim,
	cudaStream_t stream
) {
	dsa_mla_decode_kernel<<<dim3( 1 ), dim3( 256 ), 0, stream>>>(
		q_lat.ptr<const half>(),
		q_pe.ptr<const half>(),
		latent_cache,
		rope_cache,
		out.ptr<half>(),
		layer_idx,
		current_pos,
		dsa_top_k,
		mla_dc,
		mla_dr,
		max_seq,
		out_dim
	);
}

void MLA::project_kv_b(
	const pod::TensorView& kv_norm,
	const pod::TensorView& kv_b_proj,
	pod::TensorView& k_out,
	pod::TensorView& v_out,
	int num_heads,
	int head_dim,
	cudaStream_t stream,
	int rows
) {
	int total_rows = num_heads * ( head_dim * 2 );
	int threads = 256;
	int blocks = ( total_rows + threads - 1 ) / threads;
	mla_kv_b_gemv_unpack_kernel<<<dim3( blocks, rows ), dim3( threads ), 0, stream>>>(
		kv_b_proj.ptr<const half>(),
		kv_norm.ptr<const half>(),
		k_out.ptr<half>(),
		v_out.ptr<half>(),
		num_heads,
		head_dim
	);
}

void MLA::store_kv_cache(
	const pod::KVRef& kv,
	const pod::TensorView& k,
	const pod::TensorView& v,
	int mla_layer_idx,
	int pos,
	int total_elems,
	int max_seq,
	cudaStream_t stream,
	const pod::SlotRef& slot,
	int rows
) {
	if ( slot.rows ) {
		// Device-resolved positions: the shared quantizing store (FP16 included)
		ENGINE_ASSERT( slot.row == 0, "MLA::store_kv_cache: row offset unsupported" );
		const int num_heads = total_elems / 256;
		Attention::store_kv_cache( kv, k, v, slot.stride, mla_layer_idx, slot.rows, rows, num_heads, 256, max_seq, stream );
		return;
	}
	ENGINE_ASSERT( kv.kq == pod::KVQ::F16 && kv.vq == pod::KVQ::F16, "MLA::store_kv_cache: quantized formats need device-resolved positions" );
	int threads = 256;
	int blocks = ( total_elems + threads - 1 ) / threads;
	mla_store_kv_cache_kernel<<<dim3( blocks, rows ), dim3( threads ), 0, stream>>>(
		static_cast<half*>( kv.k ),
		static_cast<half*>( kv.v ),
		k.ptr<const half>(),
		v.ptr<const half>(),
		mla_layer_idx,
		pos,
		total_elems,
		max_seq,
		slot
	);
}

void MLA::forward_decode(
	const pod::TensorView& q,
	const pod::KVRef& kv,
	pod::TensorView& out,
	int mla_layer_idx,
	int pos,
	int num_heads,
	int head_dim,
	int max_seq,
	cudaStream_t stream,
	const pod::SlotRef& slot,
	int rows
) {
	// Device-resolved positions: split-KV decode attention (chunks over the window, merged by a reduce), no score
	// buffer, so the window is not bounded by shared memory. Every head is its own KV head (full-expanded cache).
	if ( slot.rows ) {
		ENGINE_ASSERT( slot.row == 0, "MLA::forward_decode: row offset unsupported" );
		pod::TensorView no_gate{};
		Attention::forward_decode( kv, q, no_gate, out, slot.stride, mla_layer_idx, slot.rows, rows,
			num_heads, num_heads, head_dim, max_seq, stream );
		return;
	}
	ENGINE_ASSERT( kv.kq == pod::KVQ::F16 && kv.vq == pod::KVQ::F16, "MLA::forward_decode: quantized formats need device-resolved positions" );
	const half* k_cache = static_cast<const half*>( kv.k );
	const half* v_cache = static_cast<const half*>( kv.v );
	// Device-resolved position (slot.rows set): size scores for the full window so the
	// launch is position-independent; otherwise size for the host-provided position
	int num_tokens = slot.rows ? max_seq : MIN( pos + 1, max_seq );
	size_t smem_bytes = static_cast<size_t>( num_tokens ) * sizeof( float );
	ENGINE_ASSERT( smem_bytes <= 48 * 1024, "MLA::forward_decode: score buffer exceeds default smem" );

	mla_attend_decode_kernel<<<dim3( num_heads, rows ), dim3( head_dim ), smem_bytes, stream>>>(
		q.ptr<const half>(),
		k_cache,
		v_cache,
		out.ptr<half>(),
		mla_layer_idx,
		pos,
		num_heads,
		head_dim,
		max_seq,
		slot
	);
}

} // namespace engine::modules
