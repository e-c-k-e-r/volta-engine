#include "engine/core/comm.h"
#include <cuda_fp16.h>
#include <cstdio>

namespace engine::comm {

// One staging slot per (stage, rank). Sized for a 512-row prefill chunk at hidden 5120 (5 MiB); 16 slots per device.
#define IPC_SLOT_BYTES ( 6 * 1024 * 1024 )

__device__ __forceinline__ uint32_t ldg_acquire_sys_u32( const uint32_t* p ) {
	uint32_t v;
	asm volatile ( "ld.global.acquire.sys.u32 %0, [%1];" : "=r"( v ) : "l"( p ) : "memory" );
	return v;
}

__device__ __forceinline__ void stg_release_sys_u32( uint32_t* p, uint32_t v ) {
	asm volatile ( "st.global.release.sys.u32 [%0], %1;" :: "l"( p ), "r"( v ) : "memory" );
}

// Flag page layout (per rank, uint32 words): [stage(4)][row(512)][rank(4)] publish flags,
// then a device-resident sequence counter and a block-retire counter. Sequence numbers live
// on the device so launches carry no host-incremented arguments and replay inside CUDA graphs.
#define AR_MAX_ROWS 512
#define AR_SEQ_WORD ( 4 * AR_MAX_ROWS * 4 )
#define AR_RETIRE_WORD ( AR_SEQ_WORD + 1 )

__device__ __forceinline__ uint32_t* ar_flag( uint32_t* base, uint32_t stage, int row, int p ) {
	return base + ( stage * AR_MAX_ROWS + row ) * 4 + p;
}

__device__ __forceinline__ uint32_t ar_next_seq( const uint32_t* local_flags ) {
	return *reinterpret_cast<const volatile uint32_t*>( local_flags + AR_SEQ_WORD ) + 1;
}

// Last block to finish publishes seq as consumed; the next launch (stream-ordered) reads seq + 1
__device__ __forceinline__ void ar_retire( uint32_t* local_flags, uint32_t seq ) {
	__syncthreads();
	if ( threadIdx.x == 0 ) {
		if ( atomicAdd( local_flags + AR_RETIRE_WORD, 1u ) == gridDim.x - 1 ) {
			local_flags[AR_RETIRE_WORD] = 0;
			local_flags[AR_SEQ_WORD] = seq;
			__threadfence();
		}
	}
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
	half* __restrict__ x,
	half* __restrict__ residual,
	const half* __restrict__ weight,
	int hidden_size,
	float eps,
	float constant_bias = 1.0f,
	half* __restrict__ out = nullptr
) {
	int row = blockIdx.x;
	size_t row_elems = static_cast<size_t>( hidden_size );
	size_t n4 = row_elems / 8; // 8 half elements per 128-bit uint4

	const void* peer_slots[4] = { p0, p1, p2, p3 };
	uint32_t* flags[4] = { f0, f1, f2, f3 };

	const uint32_t seq = ar_next_seq( flags[rank] );
	const uint32_t stage = seq & 3;
	uint8_t* my_staging = static_cast<uint8_t*>( const_cast<void*>( peer_slots[rank] ) ) + ( (size_t) stage * 4 + rank ) * IPC_SLOT_BYTES;
	uint4* my_slot = reinterpret_cast<uint4*>( my_staging ) + row * n4;
	const uint4* src_x = ( (const uint4*) ( x + row * row_elems ) );

	// Step 1: publish local X slice to my staging slot
	for ( size_t i = threadIdx.x; i < n4; i += blockDim.x ) {
		my_slot[i] = src_x[i];
	}

	__threadfence_system();
	__syncthreads();

	// Step 2: per-row synchronization across the 4 ranks (the kernel is 4-rank only) using system-scope flags
	if ( threadIdx.x == 0 ) {
		stg_release_sys_u32( ar_flag( flags[rank], stage, row, rank ), seq );
	}
	__syncwarp();

	if ( threadIdx.x < 4 ) {
		int p = threadIdx.x;
		if ( p != rank ) {
			uint32_t* flag_ptr = ar_flag( flags[p], stage, row, p );
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

	// Step 5: normalize residual with weight and write to out (or x if out is null)
	half* dst_row = ( out != nullptr ) ? ( out + row * row_elems ) : row_x;
	for ( size_t i = threadIdx.x; i < n4; i += blockDim.x ) {
		uint4 vr = *( (const uint4*) ( row_res + i * 8 ) );
		uint4 vw = *( (const uint4*) ( weight + i * 8 ) );
		__half hr[8], hw[8], hx[8];
		*reinterpret_cast<uint4*>( hr ) = vr;
		*reinterpret_cast<uint4*>( hw ) = vw;

		#pragma unroll
		for ( int k = 0; k < 8; ++k ) {
			float r_val = __half2float( hr[k] );
			float w_val = ( weight != nullptr ) ? ( __half2float( hw[k] ) + constant_bias ) : 1.0f;
			hx[k] = __float2half_rn( r_val * w_val * inv_rms );
		}

		*( (uint4*) ( dst_row + i * 8 ) ) = *reinterpret_cast<uint4*>( hx );
	}

	ar_retire( flags[rank], seq );
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
	half* __restrict__ x,
	int hidden_size
) {
	int row = blockIdx.x;
	size_t row_elems = static_cast<size_t>( hidden_size );
	size_t n4 = row_elems / 8;

	const void* peer_slots[4] = { p0, p1, p2, p3 };
	uint32_t* flags[4] = { f0, f1, f2, f3 };

	const uint32_t seq = ar_next_seq( flags[rank] );
	const uint32_t stage = seq & 3;
	uint8_t* my_staging = static_cast<uint8_t*>( const_cast<void*>( peer_slots[rank] ) ) + ( (size_t) stage * 4 + rank ) * IPC_SLOT_BYTES;
	uint4* my_slot = reinterpret_cast<uint4*>( my_staging ) + row * n4;
	const uint4* src_x = ( (const uint4*) ( x + row * row_elems ) );

	for ( size_t i = threadIdx.x; i < n4; i += blockDim.x ) {
		my_slot[i] = src_x[i];
	}

	__threadfence_system();
	__syncthreads();

	if ( threadIdx.x == 0 ) {
		stg_release_sys_u32( ar_flag( flags[rank], stage, row, rank ), seq );
		for ( int p = 0; p < 4; ++p ) {
			if ( p == rank ) continue;
			while ( ldg_acquire_sys_u32( ar_flag( flags[p], stage, row, p ) ) < seq ) {
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

	ar_retire( flags[rank], seq );
}

void CollectiveRing::init(
	int local_rank,
	void* const peer_staging[ENGINE_NUM_GPUS],
	void* const peer_flags[ENGINE_NUM_GPUS]
) {
	rank = local_rank;
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
	cudaStream_t stream,
	float constant_bias,
	pod::TensorView* out
) {
	int rows = x.shape[0];
	int hidden_size = x.shape[1];

	ENGINE_ASSERT( rows >= 1 && rows <= AR_MAX_ROWS && static_cast<size_t>( rows ) * hidden_size * sizeof( half ) <= IPC_SLOT_BYTES, "all_reduce_rms_norm: rows out of range for the staging slot" );
	ENGINE_ASSERT( hidden_size % 128 == 0, "all_reduce_rms_norm: hidden_size must be multiple of 128" );

	half* x_ptr = x.ptr<half>();
	half* res_ptr = residual.ptr<half>();
	const half* w_ptr = weight.ptr<const half>();
	half* out_ptr = ( out != nullptr ) ? out->ptr<half>() : nullptr;

	all_reduce_rms_norm_4ring_kernel<<<dim3( rows ), dim3( 256 ), 0, stream>>>(
		rank,
		peer_ptrs[0],
		peer_ptrs[1],
		peer_ptrs[2],
		peer_ptrs[3],
		static_cast<uint32_t*>( sync_flags[0] ),
		static_cast<uint32_t*>( sync_flags[1] ),
		static_cast<uint32_t*>( sync_flags[2] ),
		static_cast<uint32_t*>( sync_flags[3] ),
		x_ptr,
		res_ptr,
		w_ptr,
		hidden_size,
		eps,
		constant_bias,
		out_ptr
	);
}

void CollectiveRing::all_reduce(
	pod::TensorView& x,
	cudaStream_t stream
) {
	int rows = x.shape[0];
	int hidden_size = x.shape[1];

	ENGINE_ASSERT( rows >= 1 && rows <= AR_MAX_ROWS && static_cast<size_t>( rows ) * hidden_size * sizeof( half ) <= IPC_SLOT_BYTES, "all_reduce: rows out of range for the staging slot" );
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
				cudaGetLastError();
			}
		}
		ENGINE_ASSERT( staging_bytes >= kMeshStagingBytes, "init_4gpu_mesh: staging too small for 16 slots" );
		CUDA_CHECK( cudaMalloc( &staging_ptrs[d], staging_bytes ) );
		CUDA_CHECK( cudaMemset( staging_ptrs[d], 0, staging_bytes ) );

		CUDA_CHECK( cudaMalloc( &flag_ptrs[d], 65536 ) );
		CUDA_CHECK( cudaMemset( flag_ptrs[d], 0, 65536 ) );
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
