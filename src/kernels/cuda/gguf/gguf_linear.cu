// GGUF linear on CUDA. Up to kGemvRows rows (decode, verify blocks): the activations quantized to q8_1, then one GEMV kernel per block
// type built on llama.cpp's vec_dot functions (ggml/vecdotq.cuh). Up to kTileRows rows: every 256-value span of a tile of weight rows is
// decoded once into shared memory (llama.cpp's own block decoders, ggml/dequantize.cuh, or a vectorized one for Q4_K) and multiplied
// with all rows on the tensor cores. More rows (prefill chunks): the weight dequantized to FP16 by llama.cpp's converters (ggml/convert.cu) and one cuBLAS
// GEMM. The ggml files are vendored unchanged.
#include "engine/modules/gguf.h"
#include "engine/core/batch.h"
#include "engine/backend/runtime.h"
#include <algorithm>
#include <cstdlib>
#include "gguf_common.cuh"
#include "ggml/convert.cuh"
#include "ggml/dequantize.cuh"
#include "engine/modules/gemm.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <mutex>
#include <type_traits>

namespace engine::modules {

namespace {

constexpr int kWarps = 4;
constexpr int kGemvRows = 8;    // up to this many rows the GEMV (each row its own vec_dot pass: below the tile kernel's fixed cost)
constexpr int kTileRows = 16;   // up to this many rows the tile kernel; above, dequantize the whole weight and run a GEMM


// x [m, k] in the weight's column order (the prefill GEMM's operand)
__global__ void tile_columns_kernel( const half* __restrict__ x, half* __restrict__ y, int k, GgufColTiling t ) {
	const int row = blockIdx.y;
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < k ) y[static_cast<size_t>( row ) * k + i] = x[static_cast<size_t>( row ) * k + tiled_source( i, t, k )];
}

// One block of kWarps warps per output row; every row of the batch in the same pass over the weight row
template <GgmlType T, int M>
__global__ __launch_bounds__( 32 * kWarps )
void gemv_kernel( const void* __restrict__ w, const block_q8_1* __restrict__ y, half* __restrict__ out, int n, int k ) {
	using Tr = Traits<T>;
	const int row = blockIdx.x;
	const int tid = threadIdx.y * 32 + threadIdx.x;
	const int blocks_per_row = k / Tr::qk, y_blocks = k / QK8_1;
	constexpr int per_iter = Tr::vdr * kWarps * 32 / Tr::qi;
	float acc[M] = {};
	for ( int kbx = tid / ( Tr::qi / Tr::vdr ); kbx < blocks_per_row; kbx += per_iter ) {
		const int kby = kbx * ( Tr::qk / QK8_1 );
		const int kqs = Tr::vdr * ( tid % ( Tr::qi / Tr::vdr ) );
		#pragma unroll
		for ( int j = 0; j < M; ++j ) acc[j] += Tr::dot( w, y + static_cast<size_t>( j ) * y_blocks + kby, row * blocks_per_row + kbx, kqs );
	}
	__shared__ float part[kWarps][M];
	#pragma unroll
	for ( int j = 0; j < M; ++j ) {
		float v = acc[j];
		#pragma unroll
		for ( int o = 16; o > 0; o >>= 1 ) v += __shfl_xor_sync( 0xffffffff, v, o );
		if ( threadIdx.x == 0 ) part[threadIdx.y][j] = v;
	}
	__syncthreads();
	if ( threadIdx.y == 0 && threadIdx.x < M ) {
		float v = 0.0f;
		#pragma unroll
		for ( int wi = 0; wi < kWarps; ++wi ) v += part[wi][threadIdx.x];
		out[static_cast<size_t>( threadIdx.x ) * n + row] = __float2half_rn( v );
	}
}

template <GgmlType T>
void launch_rows( const void* w, const block_q8_1* y, half* out, int n, int k, int m, cudaStream_t stream ) {
	const dim3 grid( n ), block( 32, kWarps );
	switch ( m ) {
		case 1: gemv_kernel<T, 1><<<grid, block, 0, stream>>>( w, y, out, n, k ); break;
		case 2: gemv_kernel<T, 2><<<grid, block, 0, stream>>>( w, y, out, n, k ); break;
		case 3: gemv_kernel<T, 3><<<grid, block, 0, stream>>>( w, y, out, n, k ); break;
		case 4: gemv_kernel<T, 4><<<grid, block, 0, stream>>>( w, y, out, n, k ); break;
		case 5: gemv_kernel<T, 5><<<grid, block, 0, stream>>>( w, y, out, n, k ); break;
		case 6: gemv_kernel<T, 6><<<grid, block, 0, stream>>>( w, y, out, n, k ); break;
		case 7: gemv_kernel<T, 7><<<grid, block, 0, stream>>>( w, y, out, n, k ); break;
		default: gemv_kernel<T, 8><<<grid, block, 0, stream>>>( w, y, out, n, k ); break;
	}
}

template <GgmlType T>
void gemv_tiles( const void* w, const block_q8_1* y, half* out, int n, int k, int m, cudaStream_t stream ) {
	for ( int r = 0; r < m; r += 8 ) launch_rows<T>( w, y + static_cast<size_t>( r ) * ( k / QK8_1 ), out + static_cast<size_t>( r ) * n, n, k, std::min( 8, m - r ), stream );
}

// Tile kernel: kTileN weight rows per block, the reduction dimension in 256-value spans. Per span: the tile's raw blocks are copied to
// shared memory (coalesced, the next span's copy in flight in registers while this one is decoded and multiplied), every row's span is
// decoded from there by a team of threads with llama.cpp's decoder of its type (the team size its converters launch with), and four
// warps multiply the decoded span with up to kTileRows activation rows (wmma 16x16x16, FP32 accumulation), one 64-wide quarter each.
constexpr int kTileN = 32, kSpan = 256, kTilePad = 8, kTileThreads = 128;

template <GgmlType T> struct Tile;
using TileRows = half ( * )[kSpan + kTilePad];
// Every row's span decoded by teams of TEAM threads with a per-block decoder
template <int TEAM, int SPAN_BYTES, typename F>
__device__ __forceinline__ void decode_rows( const uint8_t* raw, TileRows ws, F decode ) {
	const int team = threadIdx.x / TEAM, tid = threadIdx.x % TEAM;
	#pragma unroll
	for ( int r = team; r < kTileN; r += kTileThreads / TEAM ) decode( raw + r * SPAN_BYTES, ws[r], tid );
}
// Super-block types: one call decodes one 256-value block
#define GGUF_TILE_SUPER( T, BLOCK, TEAM, FN ) \
	template <> struct Tile<GgmlType::T> { \
		static constexpr int span_bytes = sizeof( BLOCK ); \
		static __device__ __forceinline__ void decode_tile( const uint8_t* raw, TileRows ws ) { \
			decode_rows<TEAM, span_bytes>( raw, ws, []( const uint8_t* b, half* d, int tid ) { FN<half>( b, 0, d, tid ); } ); \
		} \
	};
// Small-block types: value pairs, as llama.cpp's generic dequantize_block kernel takes them
template <int QK, int QR, dequantize_kernel_t F>
__device__ __forceinline__ void decode_pairs( const void* blocks, half* dst, int tid ) {
	for ( int p = tid; p < kSpan / 2; p += 32 ) {
		const int i = 2 * p, iqs = ( i % QK ) / QR, base = i - i % QK;
		float2 v;
		F( blocks, i / QK, iqs, v );
		dst[base + iqs] = __float2half( v.x );
		dst[base + iqs + ( QR == 1 ? 1 : QK / 2 )] = __float2half( v.y );
	}
}
#define GGUF_TILE_PAIRS( T, BLOCK, QK, QR, FN ) \
	template <> struct Tile<GgmlType::T> { \
		static constexpr int span_bytes = kSpan / QK * sizeof( BLOCK ); \
		static __device__ __forceinline__ void decode_tile( const uint8_t* raw, TileRows ws ) { \
			decode_rows<32, span_bytes>( raw, ws, []( const uint8_t* b, half* d, int tid ) { decode_pairs<QK, QR, FN>( b, d, tid ); } ); \
		} \
	};
GGUF_TILE_PAIRS( Q4_0, block_q4_0, QK4_0, QR4_0, dequantize_q4_0 )
GGUF_TILE_PAIRS( Q4_1, block_q4_1, QK4_1, QR4_1, dequantize_q4_1 )
GGUF_TILE_PAIRS( Q5_0, block_q5_0, QK5_0, QR5_0, dequantize_q5_0 )
GGUF_TILE_PAIRS( Q5_1, block_q5_1, QK5_1, QR5_1, dequantize_q5_1 )
GGUF_TILE_PAIRS( Q8_0, block_q8_0, QK8_0, QR8_0, dequantize_q8_0 )
GGUF_TILE_PAIRS( Q2_0, block_q2_0, QK2_0, QR2_0, dequantize_q2_0 )
GGUF_TILE_SUPER( Q2_K, block_q2_K, 64, dequantize_q2_K )
GGUF_TILE_SUPER( Q3_K, block_q3_K, 64, dequantize_q3_K )
GGUF_TILE_SUPER( Q5_K, block_q5_K, 64, dequantize_q5_K )
GGUF_TILE_SUPER( Q6_K, block_q6_K, 64, dequantize_q6_K )
GGUF_TILE_SUPER( IQ2_XXS, block_iq2_xxs, 32, dequantize_iq2_xxs )
GGUF_TILE_SUPER( IQ2_XS, block_iq2_xs, 32, dequantize_iq2_xs )
GGUF_TILE_SUPER( IQ2_S, block_iq2_s, 32, dequantize_iq2_s )
GGUF_TILE_SUPER( IQ3_XXS, block_iq3_xxs, 32, dequantize_iq3_xxs )
GGUF_TILE_SUPER( IQ3_S, block_iq3_s, 32, dequantize_iq3_s )
GGUF_TILE_SUPER( IQ1_S, block_iq1_s, 32, dequantize_iq1_s )
GGUF_TILE_SUPER( IQ1_M, block_iq1_m, 32, dequantize_iq1_m )
GGUF_TILE_SUPER( IQ4_XS, block_iq4_xs, 32, dequantize_iq4_xs )
// IQ4_NL: llama.cpp's decoder takes the eight 32-value blocks of a 256-value span at once
template <> struct Tile<GgmlType::IQ4_NL> {
	static constexpr int span_bytes = kSpan / QK4_NL * sizeof( block_iq4_nl );
	static __device__ __forceinline__ void decode_tile( const uint8_t* raw, TileRows ws ) {
		decode_rows<32, span_bytes>( raw, ws, []( const uint8_t* b, half* d, int tid ) { dequantize_iq4_nl<half>( b, 0, d, tid ); } );
	}
};

// Four 4-bit values (the low nibbles of the bytes of q) as two FP16 pairs: 1024 + q as a half, minus 1024, is exact
__device__ __forceinline__ void nibbles_to_half2( uint32_t q, half2& a, half2& b ) {
	const half2 bias = __half2half2( __float2half( 1024.0f ) );
	const uint32_t lo = __byte_perm( q, 0x64646464u, 0x4140 ), hi = __byte_perm( q, 0x64646464u, 0x4342 );
	a = __hsub2( *reinterpret_cast<const half2*>( &lo ), bias );
	b = __hsub2( *reinterpret_cast<const half2*>( &hi ), bias );
}

// Q4_K, decoded here instead of by llama.cpp's per-value decoder: one thread per 32-bit word of quants (four low and four high nibbles,
// values 64 j + 4 t .. + 3 of sub-block 2 j and the same 32 further on of sub-block 2 j + 1, as dequantize_q4_K lays them out), FP16
// scale arithmetic, 8-byte stores
template <> struct Tile<GgmlType::Q4_K> {
	static constexpr int span_bytes = sizeof( block_q4_K );
	static __device__ __forceinline__ void decode_tile( const uint8_t* raw, TileRows ws ) {
		#pragma unroll
		for ( int it = 0; it < kTileN * 32 / kTileThreads; ++it ) {
			const int i = threadIdx.x + it * kTileThreads, r = i / 32, wd = i % 32, j = wd / 8, t = wd % 8;
			const block_q4_K* b = reinterpret_cast<const block_q4_K*>( raw + r * span_bytes );
			const float2 dm = __half22float2( b->dm );
			uint8_t sc, mn;
			get_scale_min_k4( 2 * j, b->scales, sc, mn );
			const half2 d1 = __float2half2_rn( dm.x * sc ), m1 = __float2half2_rn( dm.y * mn );
			get_scale_min_k4( 2 * j + 1, b->scales, sc, mn );
			const half2 d2 = __float2half2_rn( dm.x * sc ), m2 = __float2half2_rn( dm.y * mn );
			const uint32_t q = reinterpret_cast<const uint32_t*>( b->qs )[wd];
			half2 v0, v1, v2, v3;
			nibbles_to_half2( q & 0x0F0F0F0Fu, v0, v1 );
			nibbles_to_half2( ( q >> 4 ) & 0x0F0F0F0Fu, v2, v3 );
			half2 lo[2] = { __hfma2( v0, d1, __hneg2( m1 ) ), __hfma2( v1, d1, __hneg2( m1 ) ) };
			half2 hi[2] = { __hfma2( v2, d2, __hneg2( m2 ) ), __hfma2( v3, d2, __hneg2( m2 ) ) };
			*reinterpret_cast<uint2*>( &ws[r][64 * j + 4 * t] ) = *reinterpret_cast<const uint2*>( lo );
			*reinterpret_cast<uint2*>( &ws[r][64 * j + 32 + 4 * t] ) = *reinterpret_cast<const uint2*>( hi );
		}
	}
};
#undef GGUF_TILE_SUPER
#undef GGUF_TILE_PAIRS

// out [m, n] (or, split along k, FP32 partial sums part [split][kTileRows][n]) for blocks of kTileN rows; spans [s0, s1) of this split
template <GgmlType T>
__global__ __launch_bounds__( kTileThreads )
void tile_kernel( const void* __restrict__ w, size_t row_bytes, const half* __restrict__ x, half* __restrict__ out, float* __restrict__ part,
	int n, int k, int m, GgufColTiling tiling, int spans_per_split ) {
	using namespace nvcuda;
	using Tl = Tile<T>;
	// the raw span of every row in units of the widest load its alignment allows
	using Unit = typename std::conditional<Tl::span_bytes % 4 == 0, uint32_t, uint16_t>::type;
	constexpr int units = Tl::span_bytes / sizeof( Unit ), per_thread = ( kTileN * units + kTileThreads - 1 ) / kTileThreads;
	constexpr int x_vecs = kSpan / 8;   // x as uint4 (8 halves)
	__shared__ __align__( 32 ) half ws[kTileN][kSpan + kTilePad];
	__shared__ __align__( 32 ) half xs[kTileRows][kSpan + kTilePad];
	__shared__ __align__( 16 ) Unit raw[kTileN * units];
	const int row0 = blockIdx.x * kTileN, warp = threadIdx.x / 32;
	const int s0 = blockIdx.y * spans_per_split, s1 = min( k / kSpan, s0 + spans_per_split );
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[kTileN / 16];
	#pragma unroll
	for ( int t = 0; t < kTileN / 16; ++t ) wmma::fill_fragment( acc[t], 0.0f );
	for ( int i = threadIdx.x; i < ( kTileRows - m ) * kSpan; i += kTileThreads ) xs[m + i / kSpan][i % kSpan] = __float2half( 0.0f );
	Unit rw[per_thread];
	uint4 rx[kTileRows * x_vecs / kTileThreads];
	auto fetch = [&]( int s ) {
		#pragma unroll
		for ( int j = 0; j < per_thread; ++j ) {
			const int u = threadIdx.x + j * kTileThreads, r = u / units;
			rw[j] = u < kTileN * units && row0 + r < n
				? reinterpret_cast<const Unit*>( static_cast<const uint8_t*>( w ) + static_cast<size_t>( row0 + r ) * row_bytes + static_cast<size_t>( s ) * Tl::span_bytes )[u % units]
				: Unit( 0 );
		}
		#pragma unroll
		for ( int j = 0; j < kTileRows * x_vecs / kTileThreads; ++j ) {
			const int v = threadIdx.x + j * kTileThreads, r = v / x_vecs;
			if ( r < m ) rx[j] = *reinterpret_cast<const uint4*>( x + static_cast<size_t>( r ) * k + tiled_source( s * kSpan + ( v % x_vecs ) * 8, tiling, k ) );
		}
	};
	if ( s0 < s1 ) fetch( s0 );
	for ( int s = s0; s < s1; ++s ) {
		#pragma unroll
		for ( int j = 0; j < per_thread; ++j ) {
			const int u = threadIdx.x + j * kTileThreads;
			if ( u < kTileN * units ) raw[u] = rw[j];
		}
		#pragma unroll
		for ( int j = 0; j < kTileRows * x_vecs / kTileThreads; ++j ) {
			const int v = threadIdx.x + j * kTileThreads, r = v / x_vecs;
			if ( r < m ) *reinterpret_cast<uint4*>( &xs[r][( v % x_vecs ) * 8] ) = rx[j];
		}
		__syncthreads();
		if ( s + 1 < s1 ) fetch( s + 1 );   // in flight while this span is decoded and multiplied
		Tl::decode_tile( reinterpret_cast<const uint8_t*>( raw ), ws );
		__syncthreads();
		#pragma unroll
		for ( int kk = warp * ( kSpan / 4 ); kk < ( warp + 1 ) * ( kSpan / 4 ); kk += 16 ) {
			wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
			wmma::load_matrix_sync( a, &xs[0][kk], kSpan + kTilePad );
			#pragma unroll
			for ( int t = 0; t < kTileN / 16; ++t ) {
				// the weight rows are the columns of B: B[kk + i][j] = ws[16 t + j][kk + i]
				wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
				wmma::load_matrix_sync( b, &ws[16 * t][kk], kSpan + kTilePad );
				wmma::mma_sync( acc[t], a, b, acc[t] );
			}
		}
		__syncthreads();
	}
	// the four warps' quarters summed through shared memory (the weight tile's space, free after the last span)
	float* red = reinterpret_cast<float*>( &ws[0][0] );   // [warp][kTileRows][kTileN]
	static_assert( sizeof( ws ) >= 4 * kTileRows * kTileN * sizeof( float ), "reduction space" );
	#pragma unroll
	for ( int t = 0; t < kTileN / 16; ++t ) wmma::store_matrix_sync( red + warp * kTileRows * kTileN + 16 * t, acc[t], kTileN, wmma::mem_row_major );
	__syncthreads();
	for ( int i = threadIdx.x; i < m * kTileN; i += kTileThreads ) {
		const int r = i / kTileN, c = i % kTileN;
		if ( row0 + c >= n ) continue;
		float v = 0.0f;
		#pragma unroll
		for ( int wi = 0; wi < 4; ++wi ) v += red[( wi * kTileRows + r ) * kTileN + c];
		if ( gridDim.y == 1 ) out[static_cast<size_t>( r ) * n + row0 + c] = __float2half_rn( v );
		else part[( static_cast<size_t>( blockIdx.y ) * kTileRows + r ) * n + row0 + c] = v;
	}
}

__global__ void tile_reduce_kernel( const float* __restrict__ part, half* __restrict__ out, int n, int m, int splits ) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= m * n ) return;
	const int r = i / n, c = i % n;
	float v = 0.0f;
	for ( int s = 0; s < splits; ++s ) v += part[( static_cast<size_t>( s ) * kTileRows + r ) * n + c];
	out[i] = __float2half_rn( v );
}

// Volta tensor-core kernel (2 to 16 rows, the types with an MmaDec): mma.m8n8k4, whose 8-lane quad pairs each multiply an 8 x 4 block of
// activations (row p of a lane: p = lane % 4 + 4 * (lane >= 16)) with a 4 x 8 block of the weight (column p of a lane: weight row p of the
// pair). A lane decodes 4 consecutive values of its own weight row per step straight into that B fragment, so every weight value is decoded
// once for all rows and multiplied on the tensor cores; a warp's four quad pairs hold 32 weight rows, the warps of a block and, when the
// grid is small, the blocks split k. Fragment layouts measured on sm_70 (C: rows (lane & 1) + 4 (lane >= 16) and + 2, columns (lane & 2)
// + {0, 1, 4, 5}, in register order (r, c) (r, c + 1) (r + 2, c) (r + 2, c + 1), then the same 4 columns on).
constexpr int kMmaWarps = 4, kMmaRows = 32;

__device__ __forceinline__ void mma884( float d[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1 ) {
	asm( "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
		: "+f"( d[0] ), "+f"( d[1] ), "+f"( d[2] ), "+f"( d[3] ), "+f"( d[4] ), "+f"( d[5] ), "+f"( d[6] ), "+f"( d[7] ) : "r"( a0 ), "r"( a1 ), "r"( b0 ), "r"( b1 ) );
}

__device__ __forceinline__ uint32_t h2u( half2 h ) { return *reinterpret_cast<const uint32_t*>( &h ); }

// A type's decode for the kernel: its group (values staged per row at a time), the bytes of a group, the staging stride, and pair(): the
// products of one 64-value stretch of a lane's row with its activation rows
template <GgmlType T> struct MmaDec { static constexpr bool on = false; };

// Q4_K: a group is one 256-value super-block, staged per warp (32 rows, stride bytes each: 16-byte aligned rows) and decoded by a lane for its
// own row: sub-block pair j (values 64 j .. 64 j + 63) is the low and the high nibbles of quant bytes 32 j .. 32 j + 31
template <> struct MmaDec<GgmlType::Q4_K> {
	static constexpr bool on = true;
	// align: the staging unit (rows and blocks are multiples of it); stage: bytes staged per row and group; stride: their spacing in shared memory
	// min_rows: from this many rows on the kernel beats the GEMV
	static constexpr int group = 256, bytes = sizeof( block_q4_K ), align = 16, stage = bytes, stride = bytes + 16, min_rows = 4;
	// the 64 values of pair j times the activations, for MT tiles of 8 rows; xr[t]: this lane's activation row of tile t at the pair's start
	template <int MT>
	static __device__ __forceinline__ void pair( const uint8_t* blk, int j, const half* const xr[MT], const bool live[MT], float d[MT][8] ) {
		const block_q4_K* b = reinterpret_cast<const block_q4_K*>( blk );
		const float2 dm = __half22float2( b->dm );
		uint8_t sc, mn;
		get_scale_min_k4( 2 * j, b->scales, sc, mn );
		const half2 dlo = __float2half2_rn( dm.x * sc ), mlo = __float2half2_rn( -dm.y * mn );
		get_scale_min_k4( 2 * j + 1, b->scales, sc, mn );
		const half2 dhi = __float2half2_rn( dm.x * sc ), mhi = __float2half2_rn( -dm.y * mn );
		const uint4 q0 = *reinterpret_cast<const uint4*>( b->qs + 32 * j ), q1 = *reinterpret_cast<const uint4*>( b->qs + 32 * j + 16 );
		const uint32_t words[8] = { q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w };
		#pragma unroll
		for ( int h = 0; h < 2; ++h ) {
			#pragma unroll
			for ( int c = 0; c < 4; ++c ) {
				// steps 2 c and 2 c + 1 of this half: words 2 c and 2 c + 1, activations 8 c .. 8 c + 7 of the half (one 16-byte load per tile)
				uint4 xa[MT];
				#pragma unroll
				for ( int t = 0; t < MT; ++t ) xa[t] = live[t] ? __ldg( reinterpret_cast<const uint4*>( xr[t] + 32 * h + 8 * c ) ) : make_uint4( 0, 0, 0, 0 );
				#pragma unroll
				for ( int e = 0; e < 2; ++e ) {
					const uint32_t word = words[2 * c + e];
					half2 v0, v1;
					nibbles_to_half2( h ? ( word >> 4 ) & 0x0F0F0F0Fu : word & 0x0F0F0F0Fu, v0, v1 );
					const uint32_t b0 = h2u( __hfma2( v0, h ? dhi : dlo, h ? mhi : mlo ) ), b1 = h2u( __hfma2( v1, h ? dhi : dlo, h ? mhi : mlo ) );
					#pragma unroll
					for ( int t = 0; t < MT; ++t ) mma884( d[t], e ? xa[t].z : xa[t].x, e ? xa[t].w : xa[t].y, b0, b1 );
				}
			}
		}
	}
};

// Q5_K: Q4_K with a fifth bit per value: bit 2 j (low nibbles) or 2 j + 1 (high nibbles) of qh[l] for value l of the pair's half
template <> struct MmaDec<GgmlType::Q5_K> {
	static constexpr bool on = true;
	static constexpr int group = 256, bytes = sizeof( block_q5_K ), align = 16, stage = bytes, stride = bytes + 16, min_rows = 6;
	template <int MT>
	static __device__ __forceinline__ void pair( const uint8_t* blk, int j, const half* const xr[MT], const bool live[MT], float d[MT][8] ) {
		const block_q5_K* b = reinterpret_cast<const block_q5_K*>( blk );
		const float2 dm = __half22float2( b->dm );
		uint8_t sc, mn;
		get_scale_min_k4( 2 * j, b->scales, sc, mn );
		const half2 dlo = __float2half2_rn( dm.x * sc ), mlo = __float2half2_rn( -dm.y * mn );
		get_scale_min_k4( 2 * j + 1, b->scales, sc, mn );
		const half2 dhi = __float2half2_rn( dm.x * sc ), mhi = __float2half2_rn( -dm.y * mn );
		const uint4 q0 = *reinterpret_cast<const uint4*>( b->qs + 32 * j ), q1 = *reinterpret_cast<const uint4*>( b->qs + 32 * j + 16 );
		const uint4 h0 = *reinterpret_cast<const uint4*>( b->qh ), h1 = *reinterpret_cast<const uint4*>( b->qh + 16 );
		const uint32_t words[8] = { q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w };
		const uint32_t highs[8] = { h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w };
		#pragma unroll
		for ( int h = 0; h < 2; ++h ) {
			#pragma unroll
			for ( int c = 0; c < 4; ++c ) {
				uint4 xa[MT];
				#pragma unroll
				for ( int t = 0; t < MT; ++t ) xa[t] = live[t] ? __ldg( reinterpret_cast<const uint4*>( xr[t] + 32 * h + 8 * c ) ) : make_uint4( 0, 0, 0, 0 );
				#pragma unroll
				for ( int e = 0; e < 2; ++e ) {
					const uint32_t word = words[2 * c + e];
					const uint32_t q = ( h ? ( word >> 4 ) & 0x0F0F0F0Fu : word & 0x0F0F0F0Fu ) | ( ( ( highs[2 * c + e] >> ( 2 * j + h ) ) & 0x01010101u ) << 4 );
					const uint32_t w0 = __byte_perm( q, 0x64646464u, 0x4140 ), w1 = __byte_perm( q, 0x64646464u, 0x4342 );
					const half2 bias = __half2half2( __float2half( 1024.0f ) );
					const uint32_t b0 = h2u( __hfma2( __hsub2( *reinterpret_cast<const half2*>( &w0 ), bias ), h ? dhi : dlo, h ? mhi : mlo ) );
					const uint32_t b1 = h2u( __hfma2( __hsub2( *reinterpret_cast<const half2*>( &w1 ), bias ), h ? dhi : dlo, h ? mhi : mlo ) );
					#pragma unroll
					for ( int t = 0; t < MT; ++t ) mma884( d[t], e ? xa[t].z : xa[t].x, e ? xa[t].w : xa[t].y, b0, b1 );
				}
			}
		}
	}
};

// IQ4_XS: 32-value sub-blocks ib with a 6-bit scale; values 0..15 of a sub-block are the low nibbles of qs[16 ib .. 16 ib + 15], 16..31 the
// high ones, each an index into the non-linear grid kvalues_iq4nl (dequantize_iq4_xs); the grid lookup is llama.cpp's byte-permute one
template <> struct MmaDec<GgmlType::IQ4_XS> {
	static constexpr bool on = true;
	static constexpr int group = 256, bytes = sizeof( block_iq4_xs ), align = 8, stage = bytes, stride = bytes + 8, min_rows = 6;
	template <int MT>
	static __device__ __forceinline__ void pair( const uint8_t* blk, int j, const half* const xr[MT], const bool live[MT], float d[MT][8] ) {
		const block_iq4_xs* b = reinterpret_cast<const block_iq4_xs*>( blk );
		half dh;
		memcpy( &dh, &b->d, sizeof( dh ) );
		const float df = __half2float( dh );
		const half2 bias = __half2half2( __float2half( 1152.0f ) );   // 1024 + 128: the grid value + 128 as a half above 1024, recentred
		#pragma unroll
		for ( int sb = 0; sb < 2; ++sb ) {
			const int ib = 2 * j + sb;
			const int scale = ( ( ( b->scales_l[ib / 2] >> ( 4 * ( ib % 2 ) ) ) & 0xF ) | ( ( ( b->scales_h >> ( 2 * ib ) ) & 3 ) << 4 ) ) - 32;
			const half2 sc = __float2half2_rn( df * scale );
			const uint2 q0 = *reinterpret_cast<const uint2*>( b->qs + 16 * ib ), q1 = *reinterpret_cast<const uint2*>( b->qs + 16 * ib + 8 );
			const uint32_t words[4] = { q0.x, q0.y, q1.x, q1.y };
			#pragma unroll
			for ( int c = 0; c < 4; ++c ) {
				// values 8 c .. 8 c + 7 of the sub-block: c < 2 the low nibbles of words 2 c, 2 c + 1, else the high nibbles of words 2 c - 4, 2 c - 3
				uint4 xa[MT];
				#pragma unroll
				for ( int t = 0; t < MT; ++t ) xa[t] = live[t] ? __ldg( reinterpret_cast<const uint4*>( xr[t] + 32 * sb + 8 * c ) ) : make_uint4( 0, 0, 0, 0 );
				#pragma unroll
				for ( int e = 0; e < 2; ++e ) {
					const int2 v = get_int_from_table_16( static_cast<int>( words[2 * ( c % 2 ) + e] ), kvalues_iq4nl );
					const uint32_t g = static_cast<uint32_t>( c < 2 ? v.x : v.y ) ^ 0x80808080u;   // grid value + 128, as unsigned bytes
					const uint32_t w0 = __byte_perm( g, 0x64646464u, 0x4140 ), w1 = __byte_perm( g, 0x64646464u, 0x4342 );
					const uint32_t b0 = h2u( __hmul2( __hsub2( *reinterpret_cast<const half2*>( &w0 ), bias ), sc ) );
					const uint32_t b1 = h2u( __hmul2( __hsub2( *reinterpret_cast<const half2*>( &w1 ), bias ), sc ) );
					#pragma unroll
					for ( int t = 0; t < MT; ++t ) mma884( d[t], e ? xa[t].z : xa[t].x, e ? xa[t].w : xa[t].y, b0, b1 );
				}
			}
		}
	}
};

// Q6_K: a group is one 256-value super-block, staged block-aligned (the warp shifts each row's 8-byte aligned stretch by the block's offset
// in it while staging). Half h (128 values) quarter Q (32 values): value l is the low (Q < 2) or high nibble of ql[64 h + 32 (Q & 1) + l]
// with bits 2 Q, 2 Q + 1 of qh[32 h + l] on top, minus 32, times scales[8 h + 2 Q + l / 16] and d (dequantize_q6_K)
template <> struct MmaDec<GgmlType::Q6_K> {
	static constexpr bool on = true;
	static constexpr int group = 256, bytes = sizeof( block_q6_K ), align = 8, stage = ( bytes + align - 2 + align - 1 ) / align * align, stride = stage + 8, min_rows = 6;
	template <int MT>
	static __device__ __forceinline__ void pair( const uint8_t* blk, int j, const half* const xr[MT], const bool live[MT], float d[MT][8] ) {
		const block_q6_K* b = reinterpret_cast<const block_q6_K*>( blk );
		const int h = j / 2;
		half dd;
		memcpy( &dd, &b->d, sizeof( dd ) );
		const float df = __half2float( dd );
		const half2 bias = __half2half2( __float2half( 1056.0f ) );   // 1024 + 32: the 6-bit code as a half above 1024, recentred
		const uint4 h0 = *reinterpret_cast<const uint4*>( b->qh + 32 * h ), h1 = *reinterpret_cast<const uint4*>( b->qh + 32 * h + 16 );
		const uint32_t hw[8] = { h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w };
		#pragma unroll
		for ( int qq = 0; qq < 2; ++qq ) {
			const int Q = 2 * ( j % 2 ) + qq;
			const uint4 l0 = *reinterpret_cast<const uint4*>( b->ql + 64 * h + 32 * ( Q & 1 ) ), l1 = *reinterpret_cast<const uint4*>( b->ql + 64 * h + 32 * ( Q & 1 ) + 16 );
			const uint32_t lw[8] = { l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w };
			const half2 sc[2] = { __float2half2_rn( df * b->scales[8 * h + 2 * Q] ), __float2half2_rn( df * b->scales[8 * h + 2 * Q + 1] ) };
			#pragma unroll
			for ( int c = 0; c < 4; ++c ) {
				// values 8 c .. 8 c + 7 of the quarter: two steps, one 16-byte activation load per tile
				uint4 xa[MT];
				#pragma unroll
				for ( int t = 0; t < MT; ++t ) xa[t] = live[t] ? __ldg( reinterpret_cast<const uint4*>( xr[t] + 32 * qq + 8 * c ) ) : make_uint4( 0, 0, 0, 0 );
				#pragma unroll
				for ( int e = 0; e < 2; ++e ) {
					const int wi = 2 * c + e;
					const uint32_t q = ( ( Q < 2 ? lw[wi] : lw[wi] >> 4 ) & 0x0F0F0F0Fu ) | ( ( ( hw[wi] >> ( 2 * Q ) ) & 0x03030303u ) << 4 );
					const uint32_t w0 = __byte_perm( q, 0x64646464u, 0x4140 ), w1 = __byte_perm( q, 0x64646464u, 0x4342 );
					const uint32_t b0 = h2u( __hmul2( __hsub2( *reinterpret_cast<const half2*>( &w0 ), bias ), sc[c / 2] ) );
					const uint32_t b1 = h2u( __hmul2( __hsub2( *reinterpret_cast<const half2*>( &w1 ), bias ), sc[c / 2] ) );
					#pragma unroll
					for ( int t = 0; t < MT; ++t ) mma884( d[t], e ? xa[t].z : xa[t].x, e ? xa[t].w : xa[t].y, b0, b1 );
				}
			}
		}
	}
};

// out [m, n] (or, split along k, FP32 partials part [split][kTileRows][n]) for blocks of kMmaRows weight rows; MT tiles of 8 activation rows.
// Shared memory: per warp its 32 rows' current group (D::stride each), loaded with coalesced 16-byte loads, the next group into registers
// while the current one is multiplied; the activations are read from global memory (L1)
template <GgmlType T, int MT>
__global__ __launch_bounds__( 32 * kMmaWarps )
void mma_rows_kernel( const void* __restrict__ w, size_t row_bytes, const half* __restrict__ x, half* __restrict__ out, float* __restrict__ part, int n, int k, int m,
	GgufColTiling tiling, int groups_per_split ) {
	using D = MmaDec<T>;
	using Unit = typename std::conditional<D::align == 16, uint4, uint2>::type;
	constexpr int per_row = D::stage / D::align, chunks = ( kMmaRows * per_row + 31 ) / 32;   // staging loads per lane for a warp's group
	static_assert( D::stage % D::align == 0 && D::stride % D::align == 0, "aligned staging" );
	extern __shared__ __align__( 16 ) uint8_t smem[];
	const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
	const int q = ( lane >> 2 ) & 3, p = ( lane & 3 ) + 4 * ( lane >> 4 );
	const int g0 = blockIdx.y * groups_per_split, g1 = min( k / D::group, g0 + groups_per_split );
	uint8_t* ws = smem + static_cast<size_t>( warp ) * kMmaRows * D::stride;
	const int row_base = blockIdx.x * kMmaRows;
	// a lane's chunks: chunk c is unit o of row r's stretch, the aligned run of units holding the group's bytes (rows of a type whose blocks are
	// not whole units start anywhere: the stretch is per row)
	uintptr_t rowp[chunks];
	int unit[chunks];
	Unit* dst[chunks];
	#pragma unroll
	for ( int c = 0; c < chunks; ++c ) {
		const int i = c * 32 + lane, r = min( i / per_row, kMmaRows - 1 ), o = i % per_row;
		rowp[c] = reinterpret_cast<uintptr_t>( static_cast<const uint8_t*>( w ) + static_cast<size_t>( min( row_base + r, n - 1 ) ) * row_bytes );
		unit[c] = o;
		dst[c] = reinterpret_cast<Unit*>( ws + r * D::stride + D::align * o );
	}
	constexpr bool tail = kMmaRows * per_row % 32 != 0;   // the last chunk round is partial
	const bool last_ok = !tail || ( ( chunks - 1 ) * 32 + lane ) < kMmaRows * per_row;
	// a group's stretch may end past its row's bytes (the next row's, or the end of the weight + the allocation's slack)
	Unit pre[chunks];
	int shift[chunks];   // bytes from the stretch's start to the group's
	auto fetch = [&]( int g ) {
		#pragma unroll
		for ( int c = 0; c < chunks; ++c ) {
			const uintptr_t a = rowp[c] + static_cast<size_t>( g ) * D::bytes;
			shift[c] = static_cast<int>( a % D::align );
			if ( c < chunks - 1 || last_ok ) pre[c] = __ldg( reinterpret_cast<const Unit*>( a - shift[c] ) + unit[c] );
		}
	};
	// staged block-aligned: the group's bytes start shift bytes into the row's stretch; a chunk takes the bytes it lacks from the next one (the
	// next lane's, or for lane 31 lane 0's of the next round)
	auto stash = [&]() {
		if constexpr ( D::bytes % D::align == 0 ) {
			#pragma unroll
			for ( int c = 0; c < chunks; ++c ) {
				if ( c < chunks - 1 || last_ok ) *dst[c] = pre[c];
			}
		}
		else {
			static_assert( D::align == 8, "a shifted staging unit is 8 bytes" );
			#pragma unroll
			for ( int c = 0; c < chunks; ++c ) {
				// lane 31 needs lane 0's chunk of round c + 1: lane 0 sends that one, every other lane its round-c chunk
				const uint32_t s0 = lane == 0 && c + 1 < chunks ? pre[c + 1].x : pre[c].x, s1 = lane == 0 && c + 1 < chunks ? pre[c + 1].y : pre[c].y;
				uint2 nx;
				nx.x = __shfl_sync( 0xffffffff, s0, ( lane + 1 ) & 31 );
				nx.y = __shfl_sync( 0xffffffff, s1, ( lane + 1 ) & 31 );
				// the next chunk is the same row's (past a row's last chunk the bytes are past its block and never read)
				uint2 v = pre[c];
				if ( shift[c] ) {
					const uint64_t lo = ( static_cast<uint64_t>( v.y ) << 32 ) | v.x, hi = ( static_cast<uint64_t>( nx.y ) << 32 ) | nx.x;
					const uint64_t r = ( lo >> ( 8 * shift[c] ) ) | ( hi << ( 64 - 8 * shift[c] ) );
					v = make_uint2( static_cast<uint32_t>( r ), static_cast<uint32_t>( r >> 32 ) );
				}
				if ( c < chunks - 1 || last_ok ) *dst[c] = v;
			}
		}
	};
	// this lane's activation rows (rows past m read nothing); the tiled columns move whole 64-value pairs (the tiled head dim is a multiple of 64)
	bool live[MT];
	#pragma unroll
	for ( int t = 0; t < MT; ++t ) live[t] = 8 * t + p < m;
	float d[MT][8] = {};
	int g = g0 + warp;
	if ( g < g1 ) fetch( g );
	for ( ; g < g1; g += kMmaWarps ) {
		stash();
		__syncwarp();
		if ( g + kMmaWarps < g1 ) fetch( g + kMmaWarps );
		const uint8_t* b = ws + ( 8 * q + p ) * D::stride;
		#pragma unroll
		for ( int j = 0; j < 4; ++j ) {
			const int src = tiled_source( g * D::group + 64 * j, tiling, k );
			const half* xr[MT];
			#pragma unroll
			for ( int t = 0; t < MT; ++t ) xr[t] = x + static_cast<size_t>( live[t] ? 8 * t + p : 0 ) * k + src;
			D::template pair<MT>( b, j, xr, live, d );
		}
		__syncwarp();
	}
	__syncthreads();
	// the warps' partial sums through the staging space
	static_assert( kMmaRows * D::stride >= 2 * 8 * 32 * sizeof( float ), "reduction space" );
	float* red = reinterpret_cast<float*>( smem );   // [warp][t][i][lane]
	#pragma unroll
	for ( int t = 0; t < MT; ++t ) {
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) red[( ( warp * MT + t ) * 8 + i ) * 32 + lane] = d[t][i];
	}
	__syncthreads();
	if ( warp != 0 ) return;
	const int r0 = ( lane & 1 ) + 4 * ( lane >> 4 ), c0 = lane & 2;
	#pragma unroll
	for ( int t = 0; t < MT; ++t ) {
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) {
			float v = 0.0f;
			#pragma unroll
			for ( int wi = 0; wi < kMmaWarps; ++wi ) v += red[( ( wi * MT + t ) * 8 + i ) * 32 + lane];
			const int mr = 8 * t + r0 + 2 * ( ( i >> 1 ) & 1 ), wrow = row_base + 8 * q + c0 + ( i & 1 ) + 4 * ( i >> 2 );
			if ( mr >= m || wrow >= n ) continue;
			if ( gridDim.y == 1 ) out[static_cast<size_t>( mr ) * n + wrow] = __float2half_rn( v );
			else part[( static_cast<size_t>( blockIdx.y ) * kTileRows + mr ) * n + wrow] = v;
		}
	}
}

// Per-device state, each behind its own lock: a cudaMalloc waits for its device, whose queue may hold a collective the other ranks
// have to join, so one rank must never hold a lock another rank's device is waiting behind
std::mutex& device_mutex( int dev ) {
	static std::mutex mu[64];
	return mu[dev];
}

// The q8_1 activations of one call, per device, kept between calls (decode graphs capture the pointer): allocated at load
// (LinearGguf::prepare) to cover a prefill chunk of 32K-wide rows, so it does not grow inside a forward
block_q8_1* q8_workspace( size_t blocks ) {
	static block_q8_1* buf[64] = {};
	static size_t cap[64] = {};
	int dev = 0;
	cudaGetDevice( &dev );
	std::lock_guard<std::mutex> lk( device_mutex( dev ) );
	if ( blocks > cap[dev] ) {
		if ( buf[dev] ) cudaFree( buf[dev] );
		const size_t want = std::max( blocks, static_cast<size_t>( pod::MAX_ROWS ) * 1024 );   // a prefill chunk of 32K-wide rows without regrowing
		ENGINE_CHECK( cudaMalloc( &buf[dev], want * sizeof( block_q8_1 ) ) );
		cap[dev] = want;
	}
	return buf[dev];
}

// The FP16 copy of one weight for the prefill GEMM, per device; grows to the largest weight (prefill runs eagerly, never inside a graph)
half* recon_workspace( size_t elems, cudaStream_t stream ) {
	static half* buf[64] = {};
	static size_t cap[64] = {};
	int dev = 0;
	cudaGetDevice( &dev );
	std::lock_guard<std::mutex> lk( device_mutex( dev ) );
	if ( elems > cap[dev] ) {
		ENGINE_CHECK( cudaStreamSynchronize( stream ) );   // the previous copy may still be read by queued work
		if ( buf[dev] ) cudaFree( buf[dev] );
		ENGINE_CHECK( cudaMalloc( &buf[dev], elems * sizeof( half ) ) );
		cap[dev] = elems;
	}
	return buf[dev];
}

// The tile kernel's split-k partial sums, per device, made at load (a fixed size: splits only fill the SMs of narrow weights) and the
// device's SM count
constexpr size_t kPartFloats = static_cast<size_t>( 1 ) << 20;
struct TileState {
	float* part = nullptr;
	int sms = 0;
};
TileState& tile_state() {
	static TileState st[64];
	int dev = 0;
	cudaGetDevice( &dev );
	std::lock_guard<std::mutex> lk( device_mutex( dev ) );
	if ( st[dev].part == nullptr ) {
		ENGINE_CHECK( cudaMalloc( &st[dev].part, kPartFloats * sizeof( float ) ) );
		ENGINE_CHECK( cudaDeviceGetAttribute( &st[dev].sms, cudaDevAttrMultiProcessorCount, dev ) );
	}
	return st[dev];
}

template <GgmlType T>
void launch_tile( const void* w, const half* x, half* out, int n, int k, int m, GgufColTiling tiling, cudaStream_t stream ) {
	const TileState& st = tile_state();
	static int resident[64] = {};   // blocks of this type's kernel per SM, per device
	int dev = 0;
	cudaGetDevice( &dev );
	if ( resident[dev] == 0 ) ENGINE_CHECK( cudaOccupancyMaxActiveBlocksPerMultiprocessor( &resident[dev], tile_kernel<T>, kTileThreads, 0 ) );
	const int tiles = ( n + kTileN - 1 ) / kTileN, spans = k / kSpan, slots = resident[dev] * st.sms;
	// split along k into close to three full waves of blocks (a partly filled last wave idles most SMs)
	const int splits = std::max( 1, std::min( spans, 3 * slots / tiles ) ), per_split = ( spans + splits - 1 ) / splits;
	ENGINE_ASSERT( splits == 1 || static_cast<size_t>( splits ) * kTileRows * n <= kPartFloats, "LinearGguf: split-k workspace too small" );
	const size_t row_bytes = static_cast<size_t>( k / ggml_block( T ).elems ) * ggml_block( T ).bytes;
	const int used = ( spans + per_split - 1 ) / per_split;
	tile_kernel<T><<<dim3( tiles, used ), kTileThreads, 0, stream>>>( w, row_bytes, x, out, st.part, n, k, m, tiling, per_split );
	if ( used > 1 ) tile_reduce_kernel<<<( m * n + 255 ) / 256, 256, 0, stream>>>( st.part, out, n, m, used );
}

template <GgmlType T>
void launch_mma( const void* w, const half* x, half* out, int n, int k, int m, GgufColTiling tiling, cudaStream_t stream ) {
	using D = MmaDec<T>;
	const TileState& st = tile_state();
	const int blocks = ( n + kMmaRows - 1 ) / kMmaRows, groups = k / D::group;
	// split along k into close to whole waves of resident blocks (three if they leave each warp a group), at least one group per warp
	const int mt = m <= 8 ? 1 : 2;
	static int resident[2][64] = {};
	int dev = 0;
	cudaGetDevice( &dev );
	if ( resident[mt - 1][dev] == 0 ) {
		const size_t sm = static_cast<size_t>( kMmaWarps ) * kMmaRows * D::stride;
		ENGINE_CHECK( mt == 1 ? cudaOccupancyMaxActiveBlocksPerMultiprocessor( &resident[0][dev], mma_rows_kernel<T, 1>, 32 * kMmaWarps, sm )
		                      : cudaOccupancyMaxActiveBlocksPerMultiprocessor( &resident[1][dev], mma_rows_kernel<T, 2>, 32 * kMmaWarps, sm ) );
	}
	const int slots = std::max( 1, resident[mt - 1][dev] ) * st.sms, max_splits = std::max( 1, groups / kMmaWarps );
	int splits = 1;
	for ( int waves = 3; waves >= 1 && splits == 1; --waves ) splits = std::max( 1, std::min( max_splits, waves * slots / blocks ) );
	const int per_split = ( groups + splits - 1 ) / splits, used = ( groups + per_split - 1 ) / per_split;
	ENGINE_ASSERT( used == 1 || static_cast<size_t>( used ) * kTileRows * n <= kPartFloats, "LinearGguf: split-k workspace too small" );
	const size_t row_bytes = static_cast<size_t>( k / ggml_block( T ).elems ) * ggml_block( T ).bytes;
	const size_t smem = static_cast<size_t>( kMmaWarps ) * kMmaRows * D::stride;
	const dim3 grid( blocks, used );
	if ( mt == 1 ) {
		static bool attr = false;
		if ( !attr ) { ENGINE_CHECK( cudaFuncSetAttribute( mma_rows_kernel<T, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024 ) ); attr = true; }
		mma_rows_kernel<T, 1><<<grid, 32 * kMmaWarps, smem, stream>>>( w, row_bytes, x, out, st.part, n, k, m, tiling, per_split );
	}
	else {
		static bool attr = false;
		if ( !attr ) { ENGINE_CHECK( cudaFuncSetAttribute( mma_rows_kernel<T, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024 ) ); attr = true; }
		mma_rows_kernel<T, 2><<<grid, 32 * kMmaWarps, smem, stream>>>( w, row_bytes, x, out, st.part, n, k, m, tiling, per_split );
	}
	if ( used > 1 ) tile_reduce_kernel<<<( m * n + 255 ) / 256, 256, 0, stream>>>( st.part, out, n, m, used );
}

// ENGINE_DEBUG_GGUF_MMA=0 takes the tensor-core rows kernel out (the GEMV / tile paths), for A/B timing
bool mma_off() {
	static const bool off = [] { const char* e = std::getenv( "ENGINE_DEBUG_GGUF_MMA" ); return e != nullptr && e[0] == '0'; }();
	return off;
}

} // namespace

void LinearGguf::prepare() {
	q8_workspace( 1 );
	tile_state();
}

void LinearGguf::dequant( const void* w, GgmlType type, int64_t elems, half* out, cudaStream_t stream ) {
	const to_fp16_cuda_t fn = ggml_get_to_fp16_cuda( static_cast<ggml_type>( type ) );
	ENGINE_ASSERT( fn != nullptr, "LinearGguf::dequant: no converter for this block type" );
	fn( w, out, elems, stream );
}

bool gguf_linear_supported( GgmlType type ) {
	switch ( type ) {
		case GgmlType::Q4_0: case GgmlType::Q4_1: case GgmlType::Q5_0: case GgmlType::Q5_1: case GgmlType::Q8_0: case GgmlType::Q2_0:
		case GgmlType::Q2_K: case GgmlType::Q3_K: case GgmlType::Q4_K: case GgmlType::Q5_K: case GgmlType::Q6_K:
		case GgmlType::IQ2_XXS: case GgmlType::IQ2_XS: case GgmlType::IQ2_S: case GgmlType::IQ3_XXS: case GgmlType::IQ3_S:
		case GgmlType::IQ1_S: case GgmlType::IQ1_M: case GgmlType::IQ4_NL: case GgmlType::IQ4_XS: return true;
		default: return false;
	}
}

void LinearGguf::forward( const pod::TensorView& x, const void* w, GgmlType type, int n, int k, pod::TensorView& out, cudaStream_t stream,
	GgufColTiling tiling ) {
	const int m = x.shape[0];
	ENGINE_ASSERT( x.dtype == pod::DType::FP16 && out.dtype == pod::DType::FP16, "LinearGguf: FP16 activations and output" );
	ENGINE_ASSERT( x.shape[1] == k && k % ggml_block( type ).elems == 0 && k % QK8_1 == 0, "LinearGguf: k does not match the weight's blocks" );
	ENGINE_ASSERT( tiling.heads == 0 || ( tiling.dim > 0 && k % ( tiling.heads * tiling.dim ) == 0 ), "LinearGguf: tiling does not divide k" );
	if ( m > kTileRows ) {
		half* wf = recon_workspace( static_cast<size_t>( n ) * k + ( tiling.heads ? static_cast<size_t>( m ) * k : 0 ), stream );
		dequant( w, type, static_cast<int64_t>( n ) * k, wf, stream );
		const half* xs = x.ptr<const half>();
		if ( tiling.heads ) {
			half* xt = wf + static_cast<size_t>( n ) * k;
			tile_columns_kernel<<<dim3( ( k + 255 ) / 256, m ), 256, 0, stream>>>( xs, xt, k, tiling );
			xs = xt;
		}
		Gemm::tn( xs, k, wf, k, out.data, n, false, m, n, k, 0.0f, stream );
		return;
	}
	// up to 16 rows of a type with a tensor-core decoder (from its min_rows on: fewer go faster through the GEMV); a type whose blocks are whole
	// staging units needs rows that are too, the others realign every row
	if ( m >= 2 && !mma_off() && k % 256 == 0 ) {
		const size_t row_bytes = static_cast<size_t>( k / ggml_block( type ).elems ) * ggml_block( type ).bytes;
		switch ( type ) {
#define CASE( T ) case GgmlType::T: if ( m >= MmaDec<GgmlType::T>::min_rows && ( row_bytes % MmaDec<GgmlType::T>::align == 0 || MmaDec<GgmlType::T>::bytes % MmaDec<GgmlType::T>::align != 0 ) ) { launch_mma<GgmlType::T>( w, x.ptr<const half>(), out.ptr<half>(), n, k, m, tiling, stream ); return; } break;
			CASE( Q4_K ) CASE( Q5_K ) CASE( Q6_K ) CASE( IQ4_XS )
#undef CASE
			default: break;
		}
	}
	if ( m > kGemvRows && k % kSpan == 0 ) {
		const half* xs = x.ptr<const half>();
		half* o = out.ptr<half>();
		switch ( type ) {
#define CASE( T ) case GgmlType::T: launch_tile<GgmlType::T>( w, xs, o, n, k, m, tiling, stream ); return;
			CASE( Q4_0 ) CASE( Q4_1 ) CASE( Q5_0 ) CASE( Q5_1 ) CASE( Q8_0 ) CASE( Q2_0 ) CASE( Q2_K ) CASE( Q3_K ) CASE( Q4_K ) CASE( Q5_K ) CASE( Q6_K )
			CASE( IQ2_XXS ) CASE( IQ2_XS ) CASE( IQ2_S ) CASE( IQ3_XXS ) CASE( IQ3_S ) CASE( IQ1_S ) CASE( IQ1_M ) CASE( IQ4_NL ) CASE( IQ4_XS )
#undef CASE
			default: break;
		}
	}
	block_q8_1* y = q8_workspace( static_cast<size_t>( m ) * ( k / QK8_1 ) );
	quantize_q8_1_kernel<<<dim3( ( k + 255 ) / 256, m ), 256, 0, stream>>>( x.ptr<const half>(), y, k, tiling );
	half* o = out.ptr<half>();
	switch ( type ) {
#define CASE( T ) case GgmlType::T: gemv_tiles<GgmlType::T>( w, y, o, n, k, m, stream ); break;
		CASE( Q4_0 ) CASE( Q4_1 ) CASE( Q5_0 ) CASE( Q5_1 ) CASE( Q8_0 ) CASE( Q2_0 ) CASE( Q2_K ) CASE( Q3_K ) CASE( Q4_K ) CASE( Q5_K ) CASE( Q6_K )
		CASE( IQ2_XXS ) CASE( IQ2_XS ) CASE( IQ2_S ) CASE( IQ3_XXS ) CASE( IQ3_S ) CASE( IQ1_S ) CASE( IQ1_M ) CASE( IQ4_NL ) CASE( IQ4_XS )
#undef CASE
		default: ENGINE_ASSERT( false, "LinearGguf: unsupported block type" );
	}
}

} // namespace engine::modules
