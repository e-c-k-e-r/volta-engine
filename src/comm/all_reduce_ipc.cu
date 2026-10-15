#include "engine/core/comm.h"
#include "engine/backend/cuda/reduce.cuh"
#include <cuda_fp16.h>
#include <cstdio>

namespace engine::comm {

// One staging slot per (stage, rank). Sized for a 512-row prefill chunk at hidden 5120 (5 MiB); 16 slots per device.
#define IPC_SLOT_BYTES ( 6 * 1024 * 1024 )

#if defined(USE_ROCM)
__device__ __forceinline__ uint32_t ldg_acquire_sys_u32( const uint32_t* p ) { return __hip_atomic_load( p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM ); }
__device__ __forceinline__ void stg_release_sys_u32( uint32_t* p, uint32_t v ) { __hip_atomic_store( p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM ); }
#else
__device__ __forceinline__ uint32_t ldg_acquire_sys_u32( const uint32_t* p ) {
	uint32_t v;
	asm volatile ( "ld.global.acquire.sys.u32 %0, [%1];" : "=r"( v ) : "l"( p ) : "memory" );
	return v;
}

__device__ __forceinline__ void stg_release_sys_u32( uint32_t* p, uint32_t v ) {
	asm volatile ( "st.global.release.sys.u32 [%0], %1;" :: "l"( p ), "r"( v ) : "memory" );
}
#endif

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

template <int NR>
__global__ __launch_bounds__( 512 )
void all_reduce_rms_norm_ring_kernel(
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

	// Step 2: per-row synchronization across the NR ranks using system-scope flags
	if ( threadIdx.x == 0 ) {
		stg_release_sys_u32( ar_flag( flags[rank], stage, row, rank ), seq );
	}
	__syncwarp();

	if ( threadIdx.x < NR ) {
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

	// Step 3: contention-free ring accumulation across NR peers
	const uint4* r_ptrs[NR];
	#pragma unroll
	for ( int offset = 0; offset < NR; ++offset ) {
		int p = ( rank + offset ) % NR;
		uint8_t* p_staging = static_cast<uint8_t*>( const_cast<void*>( peer_slots[p] ) ) + ( (size_t) stage * 4 + p ) * IPC_SLOT_BYTES;
		r_ptrs[offset] = reinterpret_cast<const uint4*>( p_staging ) + row * n4;
	}

	float local_sum_sq = 0.0f;
	half* row_res = residual + row * row_elems;
	half* row_x = x + row * row_elems;

	for ( size_t i = threadIdx.x; i < n4; i += blockDim.x ) {
		__half hp[NR][8];
		#pragma unroll
		for ( int r = 0; r < NR; ++r ) *reinterpret_cast<uint4*>( hp[r] ) = r_ptrs[r][i];

		uint4 vr = *( (const uint4*) ( row_res + i * 8 ) );
		__half hr[8];
		*reinterpret_cast<uint4*>( hr ) = vr;

		__half h_new_res[8];
		#pragma unroll
		for ( int k = 0; k < 8; ++k ) {
			float sum = __half2float( hp[0][k] );
			#pragma unroll
			for ( int r = 1; r < NR; ++r ) sum += __half2float( hp[r][k] );
			float res_val = sum + __half2float( hr[k] );
			h_new_res[k] = __float2half_rn( res_val );
			local_sum_sq = fmaf( res_val, res_val, local_sum_sq );
		}

		*( (uint4*) ( row_res + i * 8 ) ) = *reinterpret_cast<uint4*>( h_new_res );
	}

	// Step 4: block-level reduction of sum_sq
	local_sum_sq = engine::math::warp_sum( local_sum_sq );

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
		warp_sum = engine::math::warp_sum<4>( warp_sum );
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

template <int NR>
__global__ __launch_bounds__( 256 )
void all_reduce_ring_kernel(
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
		for ( int p = 0; p < NR; ++p ) {
			if ( p == rank ) continue;
			while ( ldg_acquire_sys_u32( ar_flag( flags[p], stage, row, p ) ) < seq ) {
				__nanosleep( 32 );
			}
		}
	}
	__syncthreads();

	const uint4* r_ptrs[NR];
	#pragma unroll
	for ( int offset = 0; offset < NR; ++offset ) {
		int p = ( rank + offset ) % NR;
		uint8_t* p_staging = static_cast<uint8_t*>( const_cast<void*>( peer_slots[p] ) ) + ( (size_t) stage * 4 + p ) * IPC_SLOT_BYTES;
		r_ptrs[offset] = reinterpret_cast<const uint4*>( p_staging ) + row * n4;
	}

	half* row_x = x + row * row_elems;
	for ( size_t i = threadIdx.x; i < n4; i += blockDim.x ) {
		__half hp[NR][8], h_sum[8];
		#pragma unroll
		for ( int r = 0; r < NR; ++r ) *reinterpret_cast<uint4*>( hp[r] ) = r_ptrs[r][i];

		#pragma unroll
		for ( int k = 0; k < 8; ++k ) {
			float s = __half2float( hp[0][k] );
			#pragma unroll
			for ( int r = 1; r < NR; ++r ) s += __half2float( hp[r][k] );
			h_sum[k] = __float2half_rn( s );
		}

		*( (uint4*) ( row_x + i * 8 ) ) = *reinterpret_cast<uint4*>( h_sum );
	}

	ar_retire( flags[rank], seq );
}

// Last block to finish publishes seq as consumed (a grid of `blocks` blocks)
__device__ __forceinline__ void ar_retire_n( uint32_t* local_flags, uint32_t seq, unsigned blocks ) {
	__syncthreads();
	if ( threadIdx.x == 0 ) {
		if ( atomicAdd( local_flags + AR_RETIRE_WORD, 1u ) == blocks - 1 ) {
			local_flags[AR_RETIRE_WORD] = 0;
			local_flags[AR_SEQ_WORD] = seq;
			__threadfence();
		}
	}
}

// all_reduce_ring_kernel followed by the Flash-Next gated add and the per-stream norm for the next site. Grid (rows, hc): the block (row, s) reduces the
// whole row of y (every block of a row publishes the same bytes to the staging slot, so the flags are idempotent) and then owns stream s alone: it adds
// post[row, s] * y to the fp32 stream, reduces its sum of squares and writes the normed stream. y itself is not written back (it is dead after the add).
// The summation of the squares differs in order from GatedResidual::norm; the streams are updated identically.
constexpr int AR_GN_CHUNKS = 2;   // 8-value chunks per thread: hidden <= 2 * 256 * 8 = 4096
template <int NR>
__global__ __launch_bounds__( 256 )
void all_reduce_gated_norm_kernel(
	int rank,
	const void* __restrict__ p0, const void* __restrict__ p1, const void* __restrict__ p2, const void* __restrict__ p3,
	uint32_t* __restrict__ f0, uint32_t* __restrict__ f1, uint32_t* __restrict__ f2, uint32_t* __restrict__ f3,
	const half* __restrict__ y,
	float* __restrict__ x,
	const float* __restrict__ post,
	const half* __restrict__ w,
	float* __restrict__ normed,
	int hidden_size,
	int hc,
	float eps,
	const half* __restrict__ addend,
	const float* __restrict__ addend_gate
) {
	const int row = blockIdx.x, s = blockIdx.y;
	const size_t row_elems = static_cast<size_t>( hidden_size );
	const size_t n4 = row_elems / 8;

	const void* peer_slots[4] = { p0, p1, p2, p3 };
	uint32_t* flags[4] = { f0, f1, f2, f3 };
	const uint32_t seq = ar_next_seq( flags[rank] );
	const uint32_t stage = seq & 3;
	uint8_t* my_staging = static_cast<uint8_t*>( const_cast<void*>( peer_slots[rank] ) ) + ( (size_t) stage * 4 + rank ) * IPC_SLOT_BYTES;
	uint4* my_slot = reinterpret_cast<uint4*>( my_staging ) + row * n4;
	const uint4* src_y = ( (const uint4*) ( y + row * row_elems ) );
	if ( addend == nullptr ) {
		for ( size_t i = threadIdx.x; i < n4; i += blockDim.x ) my_slot[i] = src_y[i];
	}
	else {
		const uint4* src_a = reinterpret_cast<const uint4*>( addend + row * row_elems );
		const float g = addend_gate[row];
		for ( size_t i = threadIdx.x; i < n4; i += blockDim.x ) {
			uint4 yv = src_y[i];
			const uint4 av = src_a[i];
			__half* yh = reinterpret_cast<__half*>( &yv );
			const __half* ah = reinterpret_cast<const __half*>( &av );
			#pragma unroll
			for ( int k = 0; k < 8; ++k ) yh[k] = __float2half_rn( __half2float( yh[k] ) + g * __half2float( ah[k] ) );
			my_slot[i] = yv;
		}
	}
	__threadfence_system();
	__syncthreads();
	if ( threadIdx.x == 0 ) {
		stg_release_sys_u32( ar_flag( flags[rank], stage, row, rank ), seq );
		for ( int p = 0; p < NR; ++p ) {
			if ( p == rank ) continue;
			while ( ldg_acquire_sys_u32( ar_flag( flags[p], stage, row, p ) ) < seq ) __nanosleep( 32 );
		}
	}
	__syncthreads();

	const uint4* r_ptrs[NR];
	#pragma unroll
	for ( int offset = 0; offset < NR; ++offset ) {
		const int p = ( rank + offset ) % NR;
		uint8_t* p_staging = static_cast<uint8_t*>( const_cast<void*>( peer_slots[p] ) ) + ( (size_t) stage * 4 + p ) * IPC_SLOT_BYTES;
		r_ptrs[offset] = reinterpret_cast<const uint4*>( p_staging ) + row * n4;
	}

	const float pv = post[static_cast<size_t>( row ) * hc + s];
	float xv[AR_GN_CHUNKS][8];
	float sq = 0.0f;
	float* xs_row = x + ( static_cast<size_t>( row ) * hc + s ) * row_elems;
	#pragma unroll
	for ( int c = 0; c < AR_GN_CHUNKS; ++c ) {
		const size_t i = threadIdx.x + static_cast<size_t>( c ) * blockDim.x;
		if ( i >= n4 ) continue;
		__half hp[NR][8];
		#pragma unroll
		for ( int r = 0; r < NR; ++r ) *reinterpret_cast<uint4*>( hp[r] ) = r_ptrs[r][i];
		float* xs = xs_row + i * 8;
		const float4 a0 = *reinterpret_cast<const float4*>( xs ), a1 = *reinterpret_cast<const float4*>( xs + 4 );
		float v[8] = { a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w };
		#pragma unroll
		for ( int k = 0; k < 8; ++k ) {
			float sum = __half2float( hp[0][k] );
			#pragma unroll
			for ( int r = 1; r < NR; ++r ) sum += __half2float( hp[r][k] );
			const float yk = __half2float( __float2half_rn( sum ) );   // y is stored as fp16 between the reduction and the add
			v[k] += pv * yk;
			xv[c][k] = v[k];
			sq += v[k] * v[k];
		}
		*reinterpret_cast<float4*>( xs ) = make_float4( v[0], v[1], v[2], v[3] );
		*reinterpret_cast<float4*>( xs + 4 ) = make_float4( v[4], v[5], v[6], v[7] );
	}
	sq = engine::math::warp_sum( sq );
	__shared__ float s_part[8];
	__shared__ float s_inv;
	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	if ( lane == 0 ) s_part[warp] = sq;
	__syncthreads();
	if ( threadIdx.x == 0 ) {
		float total = 0.0f;
		#pragma unroll
		for ( int wi = 0; wi < 8; ++wi ) total += s_part[wi];
		s_inv = rsqrtf( total / hidden_size + eps );
	}
	__syncthreads();
	float* ns_row = normed + ( static_cast<size_t>( row ) * hc + s ) * row_elems;
	#pragma unroll
	for ( int c = 0; c < AR_GN_CHUNKS; ++c ) {
		const size_t i = threadIdx.x + static_cast<size_t>( c ) * blockDim.x;
		if ( i >= n4 ) continue;
		__half hw[8];
		*reinterpret_cast<uint4*>( hw ) = *reinterpret_cast<const uint4*>( w + static_cast<size_t>( s ) * row_elems + i * 8 );
		float o[8];
		#pragma unroll
		for ( int k = 0; k < 8; ++k ) o[k] = xv[c][k] * s_inv * ( 1.0f + __half2float( hw[k] ) );
		*reinterpret_cast<float4*>( ns_row + i * 8 ) = make_float4( o[0], o[1], o[2], o[3] );
		*reinterpret_cast<float4*>( ns_row + i * 8 + 4 ) = make_float4( o[4], o[5], o[6], o[7] );
	}
	ar_retire_n( flags[rank], seq, gridDim.x * gridDim.y );
}

// all_reduce_ring_kernel followed by the mHC post-step (HyperConnection::update with a comb matrix): x[r, h, :] = post[r, h] * y[r, :] + sum_g comb[r, g, h] *
// x[r, g, :] over the fp32 streams x [rows, 4, hidden], in the same fma order as hc_apply_kernel. Grid (rows, column groups): every block publishes the whole row
// (idempotent, as in all_reduce_gated_norm_kernel), then updates its own columns of all four streams, so the in-place update never crosses a block. y is not
// written back.
constexpr int AR_HC_GROUPS = 4;
template <int NR>
__global__ __launch_bounds__( 256 )
void all_reduce_hc_apply_kernel(
	int rank,
	const void* __restrict__ p0, const void* __restrict__ p1, const void* __restrict__ p2, const void* __restrict__ p3,
	uint32_t* __restrict__ f0, uint32_t* __restrict__ f1, uint32_t* __restrict__ f2, uint32_t* __restrict__ f3,
	const half* __restrict__ y,
	float* __restrict__ x,
	const float* __restrict__ post,
	const float* __restrict__ comb,
	int hidden_size
) {
	const int row = blockIdx.x;
	const size_t row_elems = static_cast<size_t>( hidden_size );
	const size_t n4 = row_elems / 8;

	const void* peer_slots[4] = { p0, p1, p2, p3 };
	uint32_t* flags[4] = { f0, f1, f2, f3 };
	const uint32_t seq = ar_next_seq( flags[rank] );
	const uint32_t stage = seq & 3;
	uint8_t* my_staging = static_cast<uint8_t*>( const_cast<void*>( peer_slots[rank] ) ) + ( (size_t) stage * 4 + rank ) * IPC_SLOT_BYTES;
	uint4* my_slot = reinterpret_cast<uint4*>( my_staging ) + row * n4;
	const uint4* src_y = ( (const uint4*) ( y + row * row_elems ) );
	for ( size_t i = threadIdx.x; i < n4; i += blockDim.x ) my_slot[i] = src_y[i];
	__threadfence_system();
	__syncthreads();
	if ( threadIdx.x == 0 ) {
		stg_release_sys_u32( ar_flag( flags[rank], stage, row, rank ), seq );
		for ( int p = 0; p < NR; ++p ) {
			if ( p == rank ) continue;
			while ( ldg_acquire_sys_u32( ar_flag( flags[p], stage, row, p ) ) < seq ) __nanosleep( 32 );
		}
	}
	__syncthreads();

	const uint4* r_ptrs[NR];
	#pragma unroll
	for ( int offset = 0; offset < NR; ++offset ) {
		const int p = ( rank + offset ) % NR;
		uint8_t* p_staging = static_cast<uint8_t*>( const_cast<void*>( peer_slots[p] ) ) + ( (size_t) stage * 4 + p ) * IPC_SLOT_BYTES;
		r_ptrs[offset] = reinterpret_cast<const uint4*>( p_staging ) + row * n4;
	}

	float post_r[4], comb_r[4][4];
	#pragma unroll
	for ( int h = 0; h < 4; ++h ) {
		post_r[h] = post[static_cast<size_t>( row ) * 4 + h];
		#pragma unroll
		for ( int g = 0; g < 4; ++g ) comb_r[h][g] = comb[( static_cast<size_t>( row ) * 4 + h ) * 4 + g];
	}
	const size_t per = ( n4 + gridDim.y - 1 ) / gridDim.y;
	const size_t u_end = min( n4, ( blockIdx.y + 1 ) * per );
	float* x_row = x + static_cast<size_t>( row ) * 4 * row_elems;
	for ( size_t i = blockIdx.y * per + threadIdx.x; i < u_end; i += blockDim.x ) {
		__half hp[NR][8];
		#pragma unroll
		for ( int r = 0; r < NR; ++r ) *reinterpret_cast<uint4*>( hp[r] ) = r_ptrs[r][i];
		float yk[8];
		#pragma unroll
		for ( int k = 0; k < 8; ++k ) {
			float sum = __half2float( hp[0][k] );
			#pragma unroll
			for ( int r = 1; r < NR; ++r ) sum += __half2float( hp[r][k] );
			yk[k] = __half2float( __float2half_rn( sum ) );   // y is stored as fp16 between the reduction and the update
		}
		float xv[4][8];
		#pragma unroll
		for ( int g = 0; g < 4; ++g ) {
			const float4 a0 = *reinterpret_cast<const float4*>( x_row + g * row_elems + i * 8 );
			const float4 a1 = *reinterpret_cast<const float4*>( x_row + g * row_elems + i * 8 + 4 );
			xv[g][0] = a0.x; xv[g][1] = a0.y; xv[g][2] = a0.z; xv[g][3] = a0.w;
			xv[g][4] = a1.x; xv[g][5] = a1.y; xv[g][6] = a1.z; xv[g][7] = a1.w;
		}
		#pragma unroll
		for ( int h = 0; h < 4; ++h ) {
			float o[8];
			#pragma unroll
			for ( int k = 0; k < 8; ++k ) {
				o[k] = post_r[h] * yk[k];
				#pragma unroll
				for ( int g = 0; g < 4; ++g ) o[k] = fmaf( comb_r[g][h], xv[g][k], o[k] );
			}
			*reinterpret_cast<float4*>( x_row + h * row_elems + i * 8 ) = make_float4( o[0], o[1], o[2], o[3] );
			*reinterpret_cast<float4*>( x_row + h * row_elems + i * 8 + 4 ) = make_float4( o[4], o[5], o[6], o[7] );
		}
	}
	ar_retire_n( flags[rank], seq, gridDim.x * gridDim.y );
}

void CollectiveRing::init(
	int local_rank,
	int ranks,
	void* const peer_staging[ENGINE_NUM_GPUS],
	void* const peer_flags[ENGINE_NUM_GPUS]
) {
	ENGINE_ASSERT( ranks == 2 || ranks == 4, "CollectiveRing::init: 2 or 4 ranks" );
	rank = local_rank;
	nranks = ranks;
	for ( int p = 0; p < ENGINE_NUM_GPUS; ++p ) {
		peer_ptrs[p] = ( p < ranks ) ? peer_staging[p] : nullptr;
		sync_flags[p] = ( p < ranks ) ? peer_flags[p] : nullptr;
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

	auto kernel = ( nranks == 2 ) ? all_reduce_rms_norm_ring_kernel<2> : all_reduce_rms_norm_ring_kernel<4>;
	kernel<<<dim3( rows ), dim3( 256 ), 0, stream>>>(
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

	auto kernel = ( nranks == 2 ) ? all_reduce_ring_kernel<2> : all_reduce_ring_kernel<4>;
	kernel<<<dim3( rows ), dim3( 256 ), 0, stream>>>(
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

void CollectiveRing::all_reduce_gated_norm(
	pod::TensorView& y,
	float* streams,
	const float* post,
	const backend::half_t* norm_w,
	float* normed,
	int hc,
	float eps,
	cudaStream_t stream,
	const backend::half_t* addend,
	const float* addend_gate
) {
	const int rows = y.shape[0];
	const int hidden_size = y.shape[1];
	ENGINE_ASSERT( hc == 4 && rows >= 1 && rows <= AR_MAX_ROWS && static_cast<size_t>( rows ) * hidden_size * sizeof( half ) <= IPC_SLOT_BYTES, "all_reduce_gated_norm: rows out of range" );
	ENGINE_ASSERT( hidden_size % 128 == 0 && hidden_size / 8 <= AR_GN_CHUNKS * 256, "all_reduce_gated_norm: hidden_size" );
	auto kernel = ( nranks == 2 ) ? all_reduce_gated_norm_kernel<2> : all_reduce_gated_norm_kernel<4>;
	kernel<<<dim3( rows, hc ), dim3( 256 ), 0, stream>>>(
		rank, peer_ptrs[0], peer_ptrs[1], peer_ptrs[2], peer_ptrs[3],
		static_cast<uint32_t*>( sync_flags[0] ), static_cast<uint32_t*>( sync_flags[1] ), static_cast<uint32_t*>( sync_flags[2] ), static_cast<uint32_t*>( sync_flags[3] ),
		y.ptr<const half>(), streams, post, reinterpret_cast<const half*>( norm_w ), normed, hidden_size, hc, eps,
		reinterpret_cast<const half*>( addend ), addend_gate );
}

void CollectiveRing::all_reduce_hc_apply(
	pod::TensorView& y,
	float* streams,
	const float* post,
	const float* comb,
	cudaStream_t stream
) {
	const int rows = y.shape[0];
	const int hidden_size = y.shape[1];
	ENGINE_ASSERT( rows >= 1 && rows <= AR_MAX_ROWS && static_cast<size_t>( rows ) * hidden_size * sizeof( half ) <= IPC_SLOT_BYTES, "all_reduce_hc_apply: rows out of range" );
	ENGINE_ASSERT( hidden_size % 128 == 0, "all_reduce_hc_apply: hidden_size" );
	auto kernel = ( nranks == 2 ) ? all_reduce_hc_apply_kernel<2> : all_reduce_hc_apply_kernel<4>;
	kernel<<<dim3( rows, AR_HC_GROUPS ), dim3( 256 ), 0, stream>>>(
		rank, peer_ptrs[0], peer_ptrs[1], peer_ptrs[2], peer_ptrs[3],
		static_cast<uint32_t*>( sync_flags[0] ), static_cast<uint32_t*>( sync_flags[1] ), static_cast<uint32_t*>( sync_flags[2] ), static_cast<uint32_t*>( sync_flags[3] ),
		y.ptr<const half>(), streams, post, comb, hidden_size );
}

void init_4gpu_mesh(
	CollectiveRing rings[ENGINE_NUM_GPUS],
	void* staging_ptrs[ENGINE_NUM_GPUS],
	void* flag_ptrs[ENGINE_NUM_GPUS],
	size_t staging_bytes,
	int ranks
) {
	for ( int d = 0; d < ranks; ++d ) {
		CUDA_CHECK( cudaSetDevice( d ) );
		for ( int peer = 0; peer < ranks; ++peer ) {
			if ( peer != d ) {
				(void) cudaDeviceEnablePeerAccess( peer, 0 );
				(void) cudaGetLastError();
			}
		}
		ENGINE_ASSERT( staging_bytes >= kMeshStagingBytes, "init_4gpu_mesh: staging too small for 16 slots" );
		CUDA_CHECK( cudaMalloc( &staging_ptrs[d], staging_bytes ) );
		CUDA_CHECK( cudaMemset( staging_ptrs[d], 0, staging_bytes ) );

		CUDA_CHECK( cudaMalloc( &flag_ptrs[d], 65536 ) );
		CUDA_CHECK( cudaMemset( flag_ptrs[d], 0, 65536 ) );
	}

	for ( int r = 0; r < ranks; ++r ) {
		rings[r].init( r, ranks, staging_ptrs, flag_ptrs );
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
