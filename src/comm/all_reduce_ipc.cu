#include "engine/core/comm.h"
#include <cuda_fp16.h>
#include <cstdio>

namespace engine::comm {

#define IPC_SLOT_BYTES (256 * 1024)

__device__ __forceinline__ uint32_t ldg_acquire_sys_u32( const uint32_t* p ) {
	uint32_t v;
	asm volatile ( "ld.global.acquire.sys.u32 %0, [%1];" : "=r"( v ) : "l"( p ) : "memory" );
	return v;
}

__device__ __forceinline__ void stg_release_sys_u32( uint32_t* p, uint32_t v ) {
	asm volatile ( "st.global.release.sys.u32 [%0], %1;" :: "l"( p ), "r"( v ) : "memory" );
}

__global__ __launch_bounds__( 512 )
void all_reduce_rms_norm_4ring_kernel(
	int rank,
	const void* __restrict__ p0,
	const void* __restrict__ p1,
	const void* __restrict__ p2,
	const void* __restrict__ p3,
	uint32_t* __restrict__ f0,
	uint32_t* __restrict__ f1,
	uint32_t* __restrict__ f2,
	uint32_t* __restrict__ f3,
	uint32_t seq,
	half* __restrict__ x,
	half* __restrict__ residual,
	const half* __restrict__ weight,
	int hidden_size,
	float eps
) {
	int row = blockIdx.x;
	size_t row_elems = static_cast<size_t>( hidden_size );
	size_t n4 = row_elems / 8; // 8 half elements per 128-bit uint4

	const void* peer_slots[4] = { p0, p1, p2, p3 };
	uint32_t* flags[4] = { f0, f1, f2, f3 };

	int stage = seq & 3;
	uint8_t* my_staging = static_cast<uint8_t*>( const_cast<void*>( peer_slots[rank] ) ) + ( (size_t) stage * 4 + rank ) * IPC_SLOT_BYTES;
	uint4* my_slot = reinterpret_cast<uint4*>( my_staging ) + row * n4;
	const uint4* src_x = ( (const uint4*) ( x + row * row_elems ) );

	// Step 1: publish local X slice to my staging slot
	for ( size_t i = threadIdx.x; i < n4; i += blockDim.x ) {
		my_slot[i] = src_x[i];
	}

	__syncthreads();

	// Step 2: synchronization across 4 ranks using system-scope flags
	if ( threadIdx.x == 0 ) {
		stg_release_sys_u32( flags[rank] + stage * 4 + rank, seq );
	}
	__syncwarp();

	if ( threadIdx.x < 4 ) {
		int p = threadIdx.x;
		if ( p != rank ) {
			uint32_t* flag_ptr = flags[p] + stage * 4 + p;
			int spins = 0;
			while ( ldg_acquire_sys_u32( flag_ptr ) < seq ) {
				if ( ++spins > 64 ) {
					__nanosleep( 32 );
				}
			}
		}
	}
	__syncthreads();

	// Step 3: contention-free ring accumulation across 4 peers
	const uint4* r_ptrs[4];
	#pragma unroll
	for ( int offset = 0; offset < 4; ++offset ) {
		int p = ( rank + offset ) & 3;
		uint8_t* p_staging = static_cast<uint8_t*>( const_cast<void*>( peer_slots[p] ) ) + ( (size_t) stage * 4 + p ) * IPC_SLOT_BYTES;
		r_ptrs[offset] = reinterpret_cast<const uint4*>( p_staging ) + row * n4;
	}

	float local_sum_sq = 0.0f;
	half* row_res = residual + row * row_elems;
	half* row_x = x + row * row_elems;

	for ( size_t i = threadIdx.x; i < n4; i += blockDim.x ) {
		uint4 v0 = r_ptrs[0][i];
		uint4 v1 = r_ptrs[1][i];
		uint4 v2 = r_ptrs[2][i];
		uint4 v3 = r_ptrs[3][i];

		__half h0[8], h1[8], h2[8], h3[8];
		*reinterpret_cast<uint4*>( h0 ) = v0;
		*reinterpret_cast<uint4*>( h1 ) = v1;
		*reinterpret_cast<uint4*>( h2 ) = v2;
		*reinterpret_cast<uint4*>( h3 ) = v3;

		uint4 vr = *( (const uint4*) ( row_res + i * 8 ) );
		__half hr[8];
		*reinterpret_cast<uint4*>( hr ) = vr;

		__half h_new_res[8];
		#pragma unroll
		for ( int k = 0; k < 8; ++k ) {
			float sum = __half2float( h0[k] ) + __half2float( h1[k] ) + __half2float( h2[k] ) + __half2float( h3[k] );
			float res_val = sum + __half2float( hr[k] );
			h_new_res[k] = __float2half_rn( res_val );
			local_sum_sq = fmaf( res_val, res_val, local_sum_sq );
		}

		*( (uint4*) ( row_res + i * 8 ) ) = *reinterpret_cast<uint4*>( h_new_res );
	}

	// Step 4: block-level reduction of sum_sq
	#pragma unroll
	for ( int mask = 16; mask > 0; mask >>= 1 ) {
		local_sum_sq += __shfl_xor_sync( 0xffffffff, local_sum_sq, mask );
	}

	__shared__ float s_block_sum_sq;
	__shared__ float s_warp_sums[8];
	int warp_id = threadIdx.x >> 5;
	int lane_id = threadIdx.x & 31;

	if ( lane_id == 0 ) {
		s_warp_sums[warp_id] = local_sum_sq;
	}
	__syncthreads();

	if ( warp_id == 0 ) {
		float warp_sum = ( lane_id < 8 ) ? s_warp_sums[lane_id] : 0.0f;
		#pragma unroll
		for ( int mask = 4; mask > 0; mask >>= 1 ) {
			warp_sum += __shfl_xor_sync( 0xffffffff, warp_sum, mask );
		}
		if ( lane_id == 0 ) {
			s_block_sum_sq = rsqrtf( warp_sum / static_cast<float>( hidden_size ) + eps );
		}
	}
	__syncthreads();

	float inv_rms = s_block_sum_sq;

	// Step 5: normalize residual with weight and write to X
	for ( size_t i = threadIdx.x; i < n4; i += blockDim.x ) {
		uint4 vr = *( (const uint4*) ( row_res + i * 8 ) );
		uint4 vw = *( (const uint4*) ( weight + i * 8 ) );
		__half hr[8], hw[8], hx[8];
		*reinterpret_cast<uint4*>( hr ) = vr;
		*reinterpret_cast<uint4*>( hw ) = vw;

		#pragma unroll
		for ( int k = 0; k < 8; ++k ) {
			float r_val = __half2float( hr[k] );
			float w_val = __half2float( hw[k] );
			hx[k] = __float2half_rn( r_val * w_val * inv_rms );
		}

		*( (uint4*) ( row_x + i * 8 ) ) = *reinterpret_cast<uint4*>( hx );
	}
}

__global__ __launch_bounds__( 256 )
void all_reduce_4ring_kernel(
	int rank,
	const void* __restrict__ p0,
	const void* __restrict__ p1,
	const void* __restrict__ p2,
	const void* __restrict__ p3,
	uint32_t* __restrict__ f0,
	uint32_t* __restrict__ f1,
	uint32_t* __restrict__ f2,
	uint32_t* __restrict__ f3,
	uint32_t seq,
	half* __restrict__ x,
	int hidden_size
) {
	int row = blockIdx.x;
	size_t row_elems = static_cast<size_t>( hidden_size );
	size_t n4 = row_elems / 8;

	const void* peer_slots[4] = { p0, p1, p2, p3 };
	uint32_t* flags[4] = { f0, f1, f2, f3 };

	int stage = seq & 3;
	uint8_t* my_staging = static_cast<uint8_t*>( const_cast<void*>( peer_slots[rank] ) ) + ( (size_t) stage * 4 + rank ) * IPC_SLOT_BYTES;
	uint4* my_slot = reinterpret_cast<uint4*>( my_staging ) + row * n4;
	const uint4* src_x = ( (const uint4*) ( x + row * row_elems ) );

	for ( size_t i = threadIdx.x; i < n4; i += blockDim.x ) {
		my_slot[i] = src_x[i];
	}

	__threadfence_system();
	__syncthreads();

	if ( threadIdx.x == 0 ) {
		stg_release_sys_u32( flags[rank] + stage * 4 + rank, seq );
		for ( int p = 0; p < 4; ++p ) {
			if ( p == rank ) continue;
			while ( ldg_acquire_sys_u32( flags[p] + stage * 4 + p ) < seq ) {
				__nanosleep( 32 );
			}
		}
	}
	__syncthreads();

	const uint4* r_ptrs[4];
	#pragma unroll
	for ( int offset = 0; offset < 4; ++offset ) {
		int p = ( rank + offset ) & 3;
		uint8_t* p_staging = static_cast<uint8_t*>( const_cast<void*>( peer_slots[p] ) ) + ( (size_t) stage * 4 + p ) * IPC_SLOT_BYTES;
		r_ptrs[offset] = reinterpret_cast<const uint4*>( p_staging ) + row * n4;
	}

	half* row_x = x + row * row_elems;
	for ( size_t i = threadIdx.x; i < n4; i += blockDim.x ) {
		uint4 v0 = r_ptrs[0][i];
		uint4 v1 = r_ptrs[1][i];
		uint4 v2 = r_ptrs[2][i];
		uint4 v3 = r_ptrs[3][i];

		__half h0[8], h1[8], h2[8], h3[8], h_sum[8];
		*reinterpret_cast<uint4*>( h0 ) = v0;
		*reinterpret_cast<uint4*>( h1 ) = v1;
		*reinterpret_cast<uint4*>( h2 ) = v2;
		*reinterpret_cast<uint4*>( h3 ) = v3;

		#pragma unroll
		for ( int k = 0; k < 8; ++k ) {
			float s = __half2float( h0[k] ) + __half2float( h1[k] ) + __half2float( h2[k] ) + __half2float( h3[k] );
			h_sum[k] = __float2half_rn( s );
		}

		*( (uint4*) ( row_x + i * 8 ) ) = *reinterpret_cast<uint4*>( h_sum );
	}
}

void CollectiveRing::init(
	int local_rank,
	void* const peer_staging[ENGINE_NUM_GPUS],
	void* const peer_flags[ENGINE_NUM_GPUS]
) {
	rank = local_rank;
	seq = 1;
	for ( int p = 0; p < ENGINE_NUM_GPUS; ++p ) {
		peer_ptrs[p] = peer_staging[p];
		sync_flags[p] = peer_flags[p];
	}
}

void CollectiveRing::all_reduce_rms_norm(
	pod::TensorView& x,
	pod::TensorView& residual,
	const pod::TensorView& weight,
	float eps,
	cudaStream_t stream
) {
	int rows = x.shape[0];
	int hidden_size = x.shape[1];

	ENGINE_ASSERT( rows >= 1 && rows <= 16, "all_reduce_rms_norm: rows must be 1..16" );
	ENGINE_ASSERT( hidden_size % 128 == 0, "all_reduce_rms_norm: hidden_size must be multiple of 128" );

	half* x_ptr = x.ptr<half>();
	half* res_ptr = residual.ptr<half>();
	const half* w_ptr = weight.ptr<const half>();

	all_reduce_rms_norm_4ring_kernel<<<dim3( rows ), dim3( 512 ), 0, stream>>>(
		rank,
		peer_ptrs[0],
		peer_ptrs[1],
		peer_ptrs[2],
		peer_ptrs[3],
		static_cast<uint32_t*>( sync_flags[0] ),
		static_cast<uint32_t*>( sync_flags[1] ),
		static_cast<uint32_t*>( sync_flags[2] ),
		static_cast<uint32_t*>( sync_flags[3] ),
		seq++,
		x_ptr,
		res_ptr,
		w_ptr,
		hidden_size,
		eps
	);
}

void CollectiveRing::all_reduce(
	pod::TensorView& x,
	cudaStream_t stream
) {
	int rows = x.shape[0];
	int hidden_size = x.shape[1];

	ENGINE_ASSERT( rows >= 1 && rows <= 16, "all_reduce: rows must be 1..16" );
	ENGINE_ASSERT( hidden_size % 128 == 0, "all_reduce: hidden_size must be multiple of 128" );

	half* x_ptr = x.ptr<half>();

	all_reduce_4ring_kernel<<<dim3( rows ), dim3( 256 ), 0, stream>>>(
		rank,
		peer_ptrs[0],
		peer_ptrs[1],
		peer_ptrs[2],
		peer_ptrs[3],
		static_cast<uint32_t*>( sync_flags[0] ),
		static_cast<uint32_t*>( sync_flags[1] ),
		static_cast<uint32_t*>( sync_flags[2] ),
		static_cast<uint32_t*>( sync_flags[3] ),
		seq++,
		x_ptr,
		hidden_size
	);
}

void init_4gpu_mesh(
	CollectiveRing rings[ENGINE_NUM_GPUS],
	void* staging_ptrs[ENGINE_NUM_GPUS],
	void* flag_ptrs[ENGINE_NUM_GPUS],
	size_t staging_bytes
) {
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		CUDA_CHECK( cudaSetDevice( d ) );
		for ( int peer = 0; peer < ENGINE_NUM_GPUS; ++peer ) {
			if ( peer != d ) {
				cudaDeviceEnablePeerAccess( peer, 0 );
			}
		}
		CUDA_CHECK( cudaMalloc( &staging_ptrs[d], staging_bytes ) );
		CUDA_CHECK( cudaMemset( staging_ptrs[d], 0, staging_bytes ) );

		CUDA_CHECK( cudaMalloc( &flag_ptrs[d], 4096 ) );
		CUDA_CHECK( cudaMemset( flag_ptrs[d], 0, 4096 ) );
	}

	for ( int r = 0; r < ENGINE_NUM_GPUS; ++r ) {
		rings[r].init( r, staging_ptrs, flag_ptrs );
	}
}

void destroy_4gpu_mesh(
	void* staging_ptrs[ENGINE_NUM_GPUS],
	void* flag_ptrs[ENGINE_NUM_GPUS]
) {
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		if ( staging_ptrs[d] != nullptr ) {
			CUDA_CHECK( cudaSetDevice( d ) );
			CUDA_CHECK( cudaFree( staging_ptrs[d] ) );
			staging_ptrs[d] = nullptr;
		}
		if ( flag_ptrs[d] != nullptr ) {
			CUDA_CHECK( cudaSetDevice( d ) );
			CUDA_CHECK( cudaFree( flag_ptrs[d] ) );
			flag_ptrs[d] = nullptr;
		}
	}
}

} // namespace engine::comm
