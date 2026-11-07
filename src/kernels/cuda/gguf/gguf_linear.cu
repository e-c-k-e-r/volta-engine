// GGUF linear on CUDA. Up to kGemvRows rows (decode, verify blocks): the activations quantized to q8_1, then one GEMV kernel per block
// type built on llama.cpp's vec_dot functions (ggml/vecdotq.cuh). Up to kTileRows rows: every 256-value span of a tile of weight rows is
// decoded once into shared memory (llama.cpp's own block decoders, ggml/dequantize.cuh, or a vectorized one for Q4_K) and multiplied
// with all rows on the tensor cores. More rows (prefill chunks): the weight dequantized to FP16 by llama.cpp's converters (ggml/convert.cu) and one cuBLAS
// GEMM. The ggml files are vendored unchanged.
#include "engine/modules/gguf.h"
#include "engine/core/batch.h"
#include "engine/backend/runtime.h"
#include <algorithm>
#include "gguf_common.cuh"
#include "ggml/convert.cuh"
#include "ggml/dequantize.cuh"
#include <cublas_v2.h>
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

cublasHandle_t cublas_handle() {
	static cublasHandle_t h[64] = {};
	int dev = 0;
	cudaGetDevice( &dev );
	std::lock_guard<std::mutex> lk( device_mutex( dev ) );
	if ( !h[dev] ) ENGINE_ASSERT( cublasCreate( &h[dev] ) == CUBLAS_STATUS_SUCCESS, "LinearGguf: cublasCreate failed" );
	return h[dev];
}

} // namespace

void LinearGguf::prepare() {
	q8_workspace( 1 );
	tile_state();
	cublas_handle();
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
		// W [n][k] row-major is W^T in column-major terms: C (column-major [n][m]) = op(W)^T [n, k] times x^T [k, m]
		half* wf = recon_workspace( static_cast<size_t>( n ) * k + ( tiling.heads ? static_cast<size_t>( m ) * k : 0 ), stream );
		dequant( w, type, static_cast<int64_t>( n ) * k, wf, stream );
		const half* xs = x.ptr<const half>();
		if ( tiling.heads ) {
			half* xt = wf + static_cast<size_t>( n ) * k;
			tile_columns_kernel<<<dim3( ( k + 255 ) / 256, m ), 256, 0, stream>>>( xs, xt, k, tiling );
			xs = xt;
		}
		cublasHandle_t h = cublas_handle();
		cublasSetStream( h, stream );
		const float alpha = 1.0f, beta = 0.0f;
		const cublasStatus_t st = cublasGemmEx( h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, wf, CUDA_R_16F, k, xs, CUDA_R_16F, k,
			&beta, out.data, CUDA_R_16F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP );
		ENGINE_ASSERT( st == CUBLAS_STATUS_SUCCESS, "LinearGguf: cublasGemmEx failed" );
		return;
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
