#include <cuda_fp16.h>
#include "reconstruct.cuh"
#include "util.cuh"
#include "ptx.cuh"
#include "exl3_dq_ext.cuh"
#include "hadamard_inner.cuh"
#include "engine/modules/trellis_bits.h"
#include <array>

// The 8 tiles from tile0 (gains applied, unit scales, before the rotations) into tile: 16 rows of 8 tiles' 8 column pairs; the block's 256
// threads, synchronized after
template <int K, int cb>
__device__ __forceinline__ void decode_strip( half2 ( &tile )[16][8][8], const uint16_t* __restrict__ g_packed, const uint16_t* __restrict__ g_starts,
	const uint8_t* __restrict__ g_gains, size_t tile0 ) {
	constexpr int packed_size = engine::modules::trellis_tile_u16( K );

	int t = threadIdx.x;
	int lane_id = t % 32;
	int warp_id = t / 32;

	__shared__ uint32_t s_packed[8][packed_size / 2];
	const size_t tile_idx = tile0 + warp_id;
	g_packed += tile0 * packed_size;
	if ( t < packed_size ) {
		( (int4*) s_packed )[t] = ( (const int4*) g_packed )[t];
	}
	__syncthreads();

	FragB frag[2];
	dq_dispatch_ext<K, cb>( s_packed[warp_id], lane_id * 8, frag[0], frag[1], g_starts ? g_starts[tile_idx] : 0u, g_starts != nullptr, g_gains != nullptr,
		g_gains ? tile_gain2( g_gains[tile_idx] ) : __half2half2( __ushort_as_half( 0x3c00 ) ) );

#if EXL3_VOLTA_LAYOUT
	// Volta tile order: lane (q, t_, hh) holds value 4 * s + i = element (k = 4 * (2 * (q >> 1) + s) + i, column 8 * (q & 1) + t_ + 4 * hh)
	{
		const int q = ( lane_id >> 2 ) & 3, col = 8 * ( q & 1 ) + ( lane_id & 3 ) + 4 * ( lane_id >> 4 );
		half* th = reinterpret_cast<half*>( &tile[0][0][0] );
		#pragma unroll
		for ( int s = 0; s < 2; ++s )
			#pragma unroll
			for ( int i = 0; i < 4; ++i ) {
				const half v = ( i & 1 ) ? __high2half( frag[s][i >> 1] ) : __low2half( frag[s][i >> 1] );
				th[( ( 4 * ( 2 * ( q >> 1 ) + s ) + i ) * 8 + warp_id ) * 16 + col] = v;
			}
	}
#else
	half2 n0 = __shfl_down_sync( 0xFFFFFFFF, frag[0][0], 4, 32 );
	half2 n1 = __shfl_down_sync( 0xFFFFFFFF, frag[0][1], 4, 32 );
	half2 n2 = __shfl_down_sync( 0xFFFFFFFF, frag[1][0], 4, 32 );
	half2 n3 = __shfl_down_sync( 0xFFFFFFFF, frag[1][1], 4, 32 );

	if ( !( lane_id & 4 ) ) {
		half2 m0 = __halves2half2( __low2half( frag[0][0] ), __low2half( n0 ) );
		half2 m1 = __halves2half2( __high2half( frag[0][0] ), __high2half( n0 ) );
		half2 m2 = __halves2half2( __low2half( frag[0][1] ), __low2half( n1 ) );
		half2 m3 = __halves2half2( __high2half( frag[0][1] ), __high2half( n1 ) );
		half2 m4 = __halves2half2( __low2half( frag[1][0] ), __low2half( n2 ) );
		half2 m5 = __halves2half2( __high2half( frag[1][0] ), __high2half( n2 ) );
		half2 m6 = __halves2half2( __low2half( frag[1][1] ), __low2half( n3 ) );
		half2 m7 = __halves2half2( __high2half( frag[1][1] ), __high2half( n3 ) );
		int r0 = ( lane_id % 4 ) * 2;
		int r1 = r0 + 1;
		int r2 = r0 + 8;
		int r3 = r0 + 9;
		int c0 = lane_id / 8;
		int c1 = c0 + 4;
		tile[r0][warp_id][c0] = m0;
		tile[r1][warp_id][c0] = m1;
		tile[r2][warp_id][c0] = m2;
		tile[r3][warp_id][c0] = m3;
		tile[r0][warp_id][c1] = m4;
		tile[r1][warp_id][c1] = m5;
		tile[r2][warp_id][c1] = m6;
		tile[r3][warp_id][c1] = m7;
	}
#endif
	__syncthreads();
}

template <int K, int cb>
__global__ __launch_bounds__( 256 )
void reconstruct_kernel(
	half* __restrict__ g_unpacked,
	const uint16_t* __restrict__ g_packed,
	const uint16_t* __restrict__ g_starts,
	const uint8_t* __restrict__ g_gains,
	int packed_blocks_n,
	int packed_n_offset
) {
	int t = threadIdx.x;
	int k = blockIdx.y;
	int n = blockIdx.x * 8;
	int tiles_n = gridDim.x;
	int out_blocks_n = tiles_n * 8;

	__shared__ half2 tile[16][8][8];
	decode_strip<K, cb>( tile, g_packed, g_starts, g_gains, (size_t) k * packed_blocks_n + packed_n_offset + n );

	int r = t / 16;
	int c = t % 16;
	int4* tile_int4 = (reinterpret_cast<int4*>( tile ));
	int4* out_int4 = ( (int4*) g_unpacked ) + ( k * 16 + r ) * 2 * out_blocks_n + n * 2 + c;
	*out_int4 = tile_int4[t];
}

// Per tile of a strip of 8 (block (x, y): the view's tile row y, its tiles 8 x .. 8 x + 7): dg[tile] += sum over the tile of dT o R / g, R the
// decoded tile (its gain g applied) and dT [rows, ld] the view's gradient of the cores before the rotations. tile_row0: the view's first tile
// row of the tensor; table (batch z): 5 * count pointers, the codes, starts, gains, dT and dg of every entry (whole tensors)
template <int K, int cb>
__global__ __launch_bounds__( 256 )
void tile_gain_grad_kernel( const void* const* __restrict__ table, int count, int packed_blocks_n, int packed_n_offset, int tile_row0, int ld,
	const uint16_t* packed, const uint16_t* starts, const uint8_t* gains, const float* dT, float* dg ) {
	if ( table ) {
		const int z = blockIdx.z;
		packed = static_cast<const uint16_t*>( table[z] );
		starts = static_cast<const uint16_t*>( table[count + z] );
		gains = static_cast<const uint8_t*>( table[2 * count + z] );
		dT = static_cast<const float*>( table[3 * count + z] );
		dg = static_cast<float*>( const_cast<void*>( table[4 * count + z] ) );
	}
	const int t = threadIdx.x, lane = t % 32;
	const int k = blockIdx.y, n = blockIdx.x * 8;
	const size_t tile0 = (size_t) ( tile_row0 + k ) * packed_blocks_n + packed_n_offset + n;
	__shared__ half2 tile[16][8][8];
	__shared__ float acc[8];
	if ( t < 8 ) acc[t] = 0.0f;
	decode_strip<K, cb>( tile, packed, starts, gains, tile0 );
	// thread t: row t / 16 of the strip, its columns 8 ( t % 16 ) .. + 7 (tile ( t % 16 ) / 2)
	const int r = t / 16, c = t % 16;
	const half* v = reinterpret_cast<const half*>( &tile[0][0][0] ) + r * 128 + c * 8;
	const float* d = dT + (size_t) ( k * 16 + r ) * ld + n * 16 + c * 8;
	const float4 d0 = *reinterpret_cast<const float4*>( d ), d1 = *reinterpret_cast<const float4*>( d + 4 );
	float s = __half2float( v[0] ) * d0.x + __half2float( v[1] ) * d0.y + __half2float( v[2] ) * d0.z + __half2float( v[3] ) * d0.w +
		__half2float( v[4] ) * d1.x + __half2float( v[5] ) * d1.y + __half2float( v[6] ) * d1.z + __half2float( v[7] ) * d1.w;
	// lanes 2 q, 2 q + 1, 2 q + 16, 2 q + 17 hold tile q of the warp's two rows
	s += __shfl_xor_sync( 0xffffffff, s, 1 );
	s += __shfl_xor_sync( 0xffffffff, s, 16 );
	if ( lane < 16 && !( lane & 1 ) ) atomicAdd( &acc[lane / 2], s );
	__syncthreads();
	if ( t < 8 ) {
		const float g = gains ? ( 896.0f + gains[tile0 + t] ) * ( 1.0f / 1024.0f ) : 1.0f;
		dg[tile0 + t] += acc[t] / g;
	}
}

#define __(i, cb) reconstruct_kernel<i, cb>
constexpr auto reconstruct_kernel_instances = std::array{
	__(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
	__(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
	__(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2),
	__(9, 2), __(10, 2), __(11, 2)   // the half-integer rates 1.5, 2.5, 3.5 (mul1 only)
};
#undef __

#define RH_THREADS 256

// The body of reconstruct_had_kernel. OUT 0: FP16 times the scales; with unit scales, 1: FP32 out (g_unpacked then a float*), 2: FP16 out
template <int K, int cb, int OUT>
__device__ __forceinline__ void reconstruct_had_body(
	void* __restrict__ g_unpacked,
	const uint16_t* __restrict__ g_packed,
	const uint16_t* __restrict__ g_starts,
	const uint8_t* __restrict__ g_gains,
	const half* __restrict__ suh,
	const half* __restrict__ svh,
	int packed_blocks_n,
	int packed_n_offset,
	int k_block_offset   // the first 128-row block of the codes (the output starts at row 0)
) {
	constexpr int packed_size = engine::modules::trellis_tile_u16( K );
	constexpr float r_scale = 0.08838834764831845f;

	int t = threadIdx.x;
	int lane_id = t % 32;
	int warp_id = t / 32;
	int kb = blockIdx.y;
	const int kbp = kb + k_block_offset;
	int nb = blockIdx.x;
	int n = nb * 8;
	int row_len = gridDim.x * 128;

	__shared__ uint32_t s_packed[8][8][packed_size / 2];
	__shared__ half2 stile[128 * 64];

	auto tix = [&]( int R, int q, int p ) {
		return R * 64 + ( q ^ ( ( R >> 2 ) & 31 ) ) * 2 + p;
	};

	constexpr int j_int4 = packed_size / 8;
	for ( int u = t; u < 8 * 8 * j_int4; u += RH_THREADS ) {
		int j = u / ( 8 * j_int4 );
		int r = u % ( 8 * j_int4 );
		const uint16_t* gp = g_packed +
			( (size_t) ( ( kbp * 8 + j ) * packed_blocks_n + packed_n_offset + n ) ) * packed_size;
		( (int4*) s_packed[j] )[r] = ( (const int4*) gp )[r];
	}
	__syncthreads();

	for ( int jj = 0; jj < 8 * 8 / ( RH_THREADS / 32 ); ++jj ) {
		int j = ( warp_id / 8 ) * ( 8 / ( RH_THREADS / 256 ) ) + jj;
		int wn = warp_id % 8;
		FragB frag[2];
		const size_t tile_idx = (size_t) ( kbp * 8 + j ) * packed_blocks_n + packed_n_offset + n + wn;
		dq_dispatch_ext<K, cb>( s_packed[j][wn], lane_id * 8, frag[0], frag[1], g_starts ? g_starts[tile_idx] : 0u, g_starts != nullptr,
			g_gains != nullptr, g_gains ? tile_gain2( g_gains[tile_idx] ) : __half2half2( __ushort_as_half( 0x3c00 ) ) );

#if EXL3_VOLTA_LAYOUT
		{
			const int q = ( lane_id >> 2 ) & 3, col = 8 * ( q & 1 ) + ( lane_id & 3 ) + 4 * ( lane_id >> 4 );
			const int cc = wn * 8 + ( col >> 1 );
			#pragma unroll
			for ( int s = 0; s < 2; ++s )
				#pragma unroll
				for ( int i = 0; i < 4; ++i ) {
					const half v = ( i & 1 ) ? __high2half( frag[s][i >> 1] ) : __low2half( frag[s][i >> 1] );
					const int R = j * 16 + 4 * ( 2 * ( q >> 1 ) + s ) + i;
					reinterpret_cast<half*>( &stile[tix( R, cc >> 1, cc & 1 )] )[col & 1] = v;
				}
		}
#else
		half2 n0 = __shfl_down_sync( 0xFFFFFFFF, frag[0][0], 4, 32 );
		half2 n1 = __shfl_down_sync( 0xFFFFFFFF, frag[0][1], 4, 32 );
		half2 n2 = __shfl_down_sync( 0xFFFFFFFF, frag[1][0], 4, 32 );
		half2 n3 = __shfl_down_sync( 0xFFFFFFFF, frag[1][1], 4, 32 );

		if ( !( lane_id & 4 ) ) {
			half2 m0 = __halves2half2( __low2half( frag[0][0] ), __low2half( n0 ) );
			half2 m1 = __halves2half2( __high2half( frag[0][0] ), __high2half( n0 ) );
			half2 m2 = __halves2half2( __low2half( frag[0][1] ), __low2half( n1 ) );
			half2 m3 = __halves2half2( __high2half( frag[0][1] ), __high2half( n1 ) );
			half2 m4 = __halves2half2( __low2half( frag[1][0] ), __low2half( n2 ) );
			half2 m5 = __halves2half2( __high2half( frag[1][0] ), __high2half( n2 ) );
			half2 m6 = __halves2half2( __low2half( frag[1][1] ), __low2half( n3 ) );
			half2 m7 = __halves2half2( __high2half( frag[1][1] ), __high2half( n3 ) );
			int r0 = j * 16 + ( lane_id % 4 ) * 2;
			int r1 = r0 + 1;
			int r2 = r0 + 8;
			int r3 = r0 + 9;
			int c0 = lane_id / 8;
			int q0 = ( wn * 8 + c0 ) >> 1, p0 = c0 & 1;
			int q1 = ( wn * 8 + c0 + 4 ) >> 1, p1 = c0 & 1;
			stile[tix( r0, q0, p0 )] = m0;
			stile[tix( r1, q0, p0 )] = m1;
			stile[tix( r2, q0, p0 )] = m2;
			stile[tix( r3, q0, p0 )] = m3;
			stile[tix( r0, q1, p1 )] = m4;
			stile[tix( r1, q1, p1 )] = m5;
			stile[tix( r2, q1, p1 )] = m6;
			stile[tix( r3, q1, p1 )] = m7;
		}
#endif
	}
	__syncthreads();

	const half2 rs2 = __float2half2_rn( r_scale );
	constexpr int CHUNKS_PW = 32 / ( RH_THREADS / 32 );
	#pragma unroll
	for ( int qq = 0; qq < CHUNKS_PW; ++qq ) {
		int q = warp_id * CHUNKS_PW + qq;
		int qs = q ^ lane_id;
		half2 a[4], b[4];
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			half4 v = *( (const half4*) ( stile + ( lane_id * 4 + i ) * 64 + qs * 2 ) );
			a[i] = v.x;
			b[i] = v.y;
		}
		#pragma unroll
		for ( int x = 0; x < 2; ++x ) {
			half2* v = x == 0 ? a : b;
			half2 s0 = __hadd2( v[0], v[1] ), d0 = __hsub2( v[0], v[1] );
			half2 s1 = __hadd2( v[2], v[3] ), d1 = __hsub2( v[2], v[3] );
			v[0] = __hmul2( __hadd2( s0, s1 ), rs2 );
			v[1] = __hmul2( __hadd2( d0, d1 ), rs2 );
			v[2] = __hmul2( __hsub2( s0, s1 ), rs2 );
			v[3] = __hmul2( __hsub2( d0, d1 ), rs2 );
			#pragma unroll
			for ( int i = 0; i < 4; ++i ) {
				v[i] = shuffle_had_h2x32( v[i], lane_id );
			}
		}
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			half4 v;
			v.x = a[i];
			v.y = b[i];
			*( (half4*) ( stile + ( lane_id * 4 + i ) * 64 + qs * 2 ) ) = v;
		}
	}
	__syncthreads();

	constexpr int ROWS_PW = 128 / ( RH_THREADS / 32 );
	const half4 sv4 = OUT != 0 ? half4{} : ( (const half4*) svh )[nb * 32 + lane_id];
	#pragma unroll
	for ( int rr = 0; rr < ROWS_PW; ++rr ) {
		int R = warp_id * ROWS_PW + rr;
		int base = R * 64 + ( lane_id ^ ( ( R >> 2 ) & 31 ) ) * 2;
		half2 v01 = stile[base];
		half2 v23 = stile[base + 1];
		float v0 = __low2float( v01 ), v1 = __high2float( v01 );
		float v2 = __low2float( v23 ), v3 = __high2float( v23 );
		float s0 = v0 + v1, d0 = v0 - v1;
		float s1 = v2 + v3, d1 = v2 - v3;
		half2 h01 = __hmul2( __floats2half2_rn( s0 + s1, d0 + d1 ), rs2 );
		half2 h23 = __hmul2( __floats2half2_rn( s0 - s1, d0 - d1 ), rs2 );
		h01 = shuffle_had_h2x32( h01, lane_id );
		h23 = shuffle_had_h2x32( h23, lane_id );
		const size_t at = (size_t) ( kb * 128 + R ) * row_len + nb * 128 + lane_id * 4;
		if constexpr ( OUT == 1 ) {
			*( (float4*) ( static_cast<float*>( g_unpacked ) + at ) ) = make_float4( __low2float( h01 ), __high2float( h01 ), __low2float( h23 ), __high2float( h23 ) );
		}
		else if constexpr ( OUT == 2 ) {
			half4 v;
			v.x = h01;
			v.y = h23;
			*( (half4*) ( static_cast<half*>( g_unpacked ) + at ) ) = v;
		}
		else {
			half2 su2 = __half2half2( suh[kb * 128 + R] );
			half4 v;
			v.x = __hmul2( __hmul2( h01, su2 ), sv4.x );
			v.y = __hmul2( __hmul2( h23, su2 ), sv4.y );
			*( (half4*) ( static_cast<half*>( g_unpacked ) + at ) ) = v;
		}
	}
}

template <int K, int cb>
__global__ __launch_bounds__( RH_THREADS )
void reconstruct_had_kernel(
	half* __restrict__ g_unpacked,
	const uint16_t* __restrict__ g_packed,
	const uint16_t* __restrict__ g_starts,
	const uint8_t* __restrict__ g_gains,
	const half* __restrict__ suh,
	const half* __restrict__ svh,
	int packed_blocks_n,
	int packed_n_offset
) {
	reconstruct_had_body<K, cb, 0>( g_unpacked, g_packed, g_starts, g_gains, suh, svh, packed_blocks_n, packed_n_offset, 0 );
}

// Unit scales, FP32 (OUT 1) or FP16 (OUT 2): batch z of a table of 4 * count pointers (codes, starts, gains, out per entry; whole tensors), or
// without a table one block of rows and columns of the tensor given directly
template <int K, int cb, int OUT>
__global__ __launch_bounds__( RH_THREADS )
void reconstruct_had_batch_kernel( const void* const* __restrict__ table, int count, int packed_blocks_n, int packed_n_offset, int k_block_offset, void* out,
	const uint16_t* packed, const uint16_t* starts, const uint8_t* gains ) {
	const int z = blockIdx.z;
	if ( table ) {
		out = const_cast<void*>( table[3 * count + z] );
		packed = static_cast<const uint16_t*>( table[z] );
		starts = static_cast<const uint16_t*>( table[count + z] );
		gains = static_cast<const uint8_t*>( table[2 * count + z] );
	}
	reconstruct_had_body<K, cb, OUT>( out, packed, starts, gains, nullptr, nullptr, packed_blocks_n, packed_n_offset, k_block_offset );
}

#define __(i, cb) reconstruct_had_kernel<i, cb>
constexpr auto reconstruct_had_kernel_instances = std::array{
	__(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
	__(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
	__(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2),
	__(9, 2), __(10, 2), __(11, 2)   // the half-integer rates 1.5, 2.5, 3.5 (mul1 only)
};
#undef __
#define __(i, cb) reconstruct_had_batch_kernel<i, cb, 1>
constexpr auto reconstruct_had_batch_f32_instances = std::array{
	__(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
	__(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
	__(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2),
	__(9, 2), __(10, 2), __(11, 2)
};
#undef __
#define __(i, cb) reconstruct_had_batch_kernel<i, cb, 2>
constexpr auto reconstruct_had_batch_f16_instances = std::array{
	__(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
	__(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
	__(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2),
	__(9, 2), __(10, 2), __(11, 2)
};
#undef __
#define __(i, cb) tile_gain_grad_kernel<i, cb>
constexpr auto tile_gain_grad_instances = std::array{
	__(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
	__(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
	__(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2),
	__(9, 2), __(10, 2), __(11, 2)
};
#undef __

namespace {

void reconstruct_cols_impl( half* unpacked, const uint16_t* packed, const uint16_t* starts, const uint8_t* gains, int K, bool mcg, bool mul1, int rows,
	int cols, int64_t n_offset, int stride, cudaStream_t stream ) {
	if ( K < 1 || K > 11 || ( K > 8 && !mul1 ) ) return;
	int packed_cols = stride / 16;
	int packed_n_offset = n_offset / 16;

	dim3 blockDim( 256 );
	dim3 gridDim( cols / 128, rows );

	int cbi = K - 1;
	if ( K > 8 ) cbi = 24 + K - 9;
	else if ( mcg ) cbi += 8;
	else if ( mul1 ) cbi += 16;

	reconstruct_kernel_instances[cbi]<<<gridDim, blockDim, 0, stream>>>(
		unpacked,
		packed,
		starts,
		gains,
		packed_cols,
		packed_n_offset
	);
}

} // namespace

void reconstruct_cols(
	half* unpacked,
	const uint16_t* packed,
	int K,
	bool mcg,
	bool mul1,
	int rows,
	int cols,
	int64_t n_offset,
	int stride,
	cudaStream_t stream
) {
	reconstruct_cols_impl( unpacked, packed, nullptr, nullptr, K, mcg, mul1, rows, cols, n_offset, stride, stream );
}

void reconstruct_slice_ext( half* unpacked, const uint16_t* packed, const uint16_t* starts, const uint8_t* gains, int K, bool mcg, bool mul1, int rows,
	int cols, int64_t n_offset, cudaStream_t stream ) {
	reconstruct_cols_impl( unpacked, packed, starts, gains, K, mcg, mul1, rows, cols, n_offset, cols, stream );
}

// defined after reconstruct_cols: the Volta-order copy (namespace exl3_vl) must call its own
void reconstruct_slice(
	half* unpacked,
	const uint16_t* packed,
	int K,
	bool mcg,
	bool mul1,
	int rows,
	int cols,
	int64_t n_offset,
	cudaStream_t stream
) {
	reconstruct_cols( unpacked, packed, K, mcg, mul1, rows, cols, n_offset, cols, stream );
}

void reconstruct_had_slice_ext( half* unpacked, const uint16_t* packed, const uint16_t* starts, const uint8_t* gains, const half* suh, const half* svh,
	int K, bool mcg, bool mul1, int rows, int cols, int64_t n_offset, cudaStream_t stream ) {
	if ( K < 1 || K > 11 || ( K > 8 && !mul1 ) ) return;
	int packed_cols = cols / 16;
	int packed_n_offset = n_offset / 16;

	dim3 blockDim( RH_THREADS );
	dim3 gridDim( cols / 128, rows / 128 );

	int cbi = K - 1;
	if ( K > 8 ) cbi = 24 + K - 9;
	else if ( mcg ) cbi += 8;
	else if ( mul1 ) cbi += 16;

	reconstruct_had_kernel_instances[cbi]<<<gridDim, blockDim, 0, stream>>>(
		unpacked,
		packed,
		starts,
		gains,
		suh,
		svh,
		packed_cols,
		packed_n_offset
	);
}

void reconstruct_had_batch( const void* const* table, int count, int K, bool mcg, bool mul1, int rows, int cols, bool half_out, cudaStream_t stream ) {
	if ( count <= 0 || K < 1 || K > 11 || ( K > 8 && !mul1 ) ) return;
	int cbi = K - 1;
	if ( K > 8 ) cbi = 24 + K - 9;
	else if ( mcg ) cbi += 8;
	else if ( mul1 ) cbi += 16;
	const auto fn = half_out ? reconstruct_had_batch_f16_instances[cbi] : reconstruct_had_batch_f32_instances[cbi];
	fn<<<dim3( cols / 128, rows / 128, count ), RH_THREADS, 0, stream>>>( table, count, cols / 16, 0, 0, nullptr, nullptr, nullptr, nullptr );
}

void reconstruct_had_block( void* out, const uint16_t* packed, const uint16_t* starts, const uint8_t* gains, int K, bool mcg, bool mul1, int k0, int rows,
	int n0, int cols, int stride, bool half_out, cudaStream_t stream ) {
	if ( K < 1 || K > 11 || ( K > 8 && !mul1 ) ) return;
	int cbi = K - 1;
	if ( K > 8 ) cbi = 24 + K - 9;
	else if ( mcg ) cbi += 8;
	else if ( mul1 ) cbi += 16;
	const auto fn = half_out ? reconstruct_had_batch_f16_instances[cbi] : reconstruct_had_batch_f32_instances[cbi];
	fn<<<dim3( cols / 128, rows / 128, 1 ), RH_THREADS, 0, stream>>>( nullptr, 1, stride / 16, n0 / 16, k0 / 128, out, packed, starts, gains );
}

void tile_gain_grad( const float* dT, float* dg, const uint16_t* packed, const uint16_t* starts, const uint8_t* gains, int K, bool mcg, bool mul1, int k0,
	int rows, int n0, int cols, int stride, cudaStream_t stream ) {
	if ( K < 1 || K > 11 || ( K > 8 && !mul1 ) || rows <= 0 ) return;
	int cbi = K - 1;
	if ( K > 8 ) cbi = 24 + K - 9;
	else if ( mcg ) cbi += 8;
	else if ( mul1 ) cbi += 16;
	tile_gain_grad_instances[cbi]<<<dim3( cols / 128, rows / 16, 1 ), 256, 0, stream>>>( nullptr, 1, stride / 16, n0 / 16, k0 / 16, cols, packed, starts,
		gains, dT, dg );
}

void tile_gain_grad_batch( const void* const* table, int count, int K, bool mcg, bool mul1, int rows, int cols, cudaStream_t stream ) {
	if ( count <= 0 || K < 1 || K > 11 || ( K > 8 && !mul1 ) ) return;
	int cbi = K - 1;
	if ( K > 8 ) cbi = 24 + K - 9;
	else if ( mcg ) cbi += 8;
	else if ( mul1 ) cbi += 16;
	tile_gain_grad_instances[cbi]<<<dim3( cols / 128, rows / 16, count ), 256, 0, stream>>>( table, count, cols / 16, 0, 0, cols, nullptr, nullptr, nullptr,
		nullptr, nullptr );
}

void reconstruct_had_slice(
	half* unpacked,
	const uint16_t* packed,
	const half* suh,
	const half* svh,
	int K,
	bool mcg,
	bool mul1,
	int rows,
	int cols,
	int64_t n_offset,
	cudaStream_t stream
) {
	reconstruct_had_slice_ext( unpacked, packed, nullptr, nullptr, suh, svh, K, mcg, mul1, rows, cols, n_offset, stream );
}
