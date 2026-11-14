// GGUF routed experts on CUDA (MoEGguf in engine/modules/gguf.h). The routes of a pass are grouped by expert into work items of up to
// kItemRoutes routes; each projection is a GEMV of one output row per block and work item, the expert's weight row read once for all the
// item's routes through llama.cpp's vec_dot of the projection's block type (LinearGguf's multi-row GEMV structure). Routes of experts on
// other ranks are left out, and the combine skips them. Prefill chunks (more rows than one pass of that path) take the tile path instead:
// passes of kTileRowsPerPass rows, items of up to 16 or 32 routes, and per item a tile of output rows whose weight blocks are decoded
// to FP16 in shared memory once and multiplied with all the item's routes on the tensor cores (the vec_dot path decodes them per route).
#include "engine/modules/gguf.h"
#include "engine/backend/runtime.h"
#include "engine/runtime/options.h"
#include "gguf_common.cuh"
#include "ggml/dequantize.cuh"
#include <mma.h>
#include <algorithm>

namespace engine::modules {

namespace {

constexpr int kMoEWarps = 4;

constexpr int kItemRoutes = 8;

// A work item: one local expert and up to kItemRoutes of its routes (route ids in `sorted`, from `begin`)
struct Item {
	int expert, begin, count, pad;
};

// Groups the pass's routes by local expert (counting sort in one block): items[0 .. *n_items) and the route ids in expert order; counts the
// experts' routes into heat
__global__ void moe_dispatch_kernel( const int* __restrict__ topk, int routes, int first, int num_local, Item* __restrict__ items, int* __restrict__ n_items,
	int* __restrict__ sorted, uint32_t* __restrict__ heat, int item_routes ) {
	extern __shared__ int sh[];
	int* count = sh;                  // [num_local]
	int* start = sh + num_local;      // [num_local]
	for ( int e = threadIdx.x; e < num_local; e += blockDim.x ) count[e] = 0;
	__syncthreads();
	for ( int r = threadIdx.x; r < routes; r += blockDim.x ) {
		const int e = topk[r] - first;
		if ( e >= 0 && e < num_local ) atomicAdd( count + e, 1 );
	}
	__syncthreads();
	if ( threadIdx.x == 0 ) {
		int at = 0, n = 0;
		for ( int e = 0; e < num_local; ++e ) {
			start[e] = at;
			for ( int b = 0; b < count[e]; b += item_routes ) items[n++] = Item{ e, at + b, min( item_routes, count[e] - b ), 0 };
			if ( heat != nullptr && count[e] > 0 ) heat[e] += static_cast<uint32_t>( count[e] );
			at += count[e];
		}
		*n_items = n;
	}
	__syncthreads();
	// route ids in route order within an expert (deterministic): one thread per expert walks the routes
	for ( int e = threadIdx.x; e < num_local; e += blockDim.x ) {
		if ( count[e] == 0 ) continue;
		int at = start[e];
		for ( int r = 0; r < routes; ++r ) if ( topk[r] - first == e ) sorted[at++] = r;
	}
}

// out[route][row] = expert row `row` . input of each route of the work item (the route's token's q8_1 row, or the route's own), scaled by
// the route's weight when `weighted`. One warp per output row, kMoEWarps rows per block: expert rows are short (640 or 2560 values), too
// short to keep a whole block of threads on one
template <GgmlType T>
__global__ __launch_bounds__( 32 * kMoEWarps )
void moe_proj_kernel( const void* const* __restrict__ ptrs, const Item* __restrict__ items, const int* __restrict__ n_items, const int* __restrict__ sorted,
	const block_q8_1* __restrict__ in, bool per_token, int k, int n, const half* __restrict__ topw, int top_k, bool weighted, float* __restrict__ out ) {
	using Tr = Traits<T>;
	if ( static_cast<int>( blockIdx.y ) >= *n_items ) return;
	const Item it = items[blockIdx.y];
	const int row = blockIdx.x * kMoEWarps + threadIdx.y;
	if ( row >= n ) return;
	const int lane = threadIdx.x;
	const void* w = ptrs[it.expert];
	const int blocks_per_row = k / Tr::qk, y_blocks = k / QK8_1;
	const block_q8_1* y[kItemRoutes];
	#pragma unroll
	for ( int j = 0; j < kItemRoutes; ++j ) {
		const int r = sorted[it.begin + min( j, it.count - 1 )];
		y[j] = in + static_cast<size_t>( per_token ? r / top_k : r ) * y_blocks;
	}
	constexpr int per_iter = Tr::vdr * 32 / Tr::qi;   // weight blocks a warp covers per iteration
	float acc[kItemRoutes] = {};
	for ( int kbx = lane / ( Tr::qi / Tr::vdr ); kbx < blocks_per_row; kbx += per_iter ) {
		const int kby = kbx * ( Tr::qk / QK8_1 );
		const int kqs = Tr::vdr * ( lane % ( Tr::qi / Tr::vdr ) );
		#pragma unroll
		for ( int j = 0; j < kItemRoutes; ++j ) {
			if ( j < it.count ) acc[j] += Tr::dot( w, y[j] + kby, row * blocks_per_row + kbx, kqs );
		}
	}
	#pragma unroll
	for ( int j = 0; j < kItemRoutes; ++j ) {
		#pragma unroll
		for ( int off = 16; off > 0; off >>= 1 ) acc[j] += __shfl_xor_sync( 0xffffffff, acc[j], off );
	}
	if ( lane < it.count ) {
		float v = 0.0f;
		#pragma unroll
		for ( int j = 0; j < kItemRoutes; ++j ) v = lane == j ? acc[j] : v;
		const int r = sorted[it.begin + lane];
		out[static_cast<size_t>( r ) * n + row] = weighted ? v * __half2float( topw[r] ) : v;
	}
}

// act = SiLU( gate ) * up of every route, quantized to q8_1 ([route][inter / 32]) as quantize_q8_1_kernel does; limit > 0 clamps the
// gate from above and up to [-limit, limit] first (SwiGLU::forward's activation limit)
__global__ void moe_act_q8_kernel( const float* __restrict__ g, const float* __restrict__ u, block_q8_1* __restrict__ q, int inter, float limit ) {
	const int r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= inter ) return;
	float gv = g[static_cast<size_t>( r ) * inter + i], uv = u[static_cast<size_t>( r ) * inter + i];
	if ( limit > 0.0f ) {
		gv = fminf( gv, limit );
		uv = fmaxf( -limit, fminf( uv, limit ) );
	}
	const float a = gv / ( 1.0f + __expf( -gv ) ) * uv;
	float amax = fabsf( a ), sum = a;
	#pragma unroll
	for ( int o = 16; o > 0; o >>= 1 ) {
		amax = fmaxf( amax, __shfl_xor_sync( 0xffffffff, amax, o ) );
		sum += __shfl_xor_sync( 0xffffffff, sum, o );
	}
	const float d = amax / 127.0f;
	block_q8_1& b = q[static_cast<size_t>( r ) * ( inter / QK8_1 ) + i / QK8_1];
	b.qs[i % QK8_1] = amax == 0.0f ? 0 : static_cast<int8_t>( roundf( a / d ) );
	if ( i % QK8_1 == 0 ) b.ds = make_half2( d, sum );
}

// out[t][j] = sum over the token's routes to this rank's experts, in route order
__global__ void moe_combine_kernel( const float* __restrict__ part, const int* __restrict__ topk, int first, int num_local, half* __restrict__ out, int rows, int top_k,
	int hidden ) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= rows * hidden ) return;
	const int t = i / hidden, j = i % hidden;
	float v = 0.0f;
	for ( int s = 0; s < top_k; ++s ) {
		const int e = topk[t * top_k + s] - first;
		if ( e >= 0 && e < num_local ) v += part[( static_cast<size_t>( t ) * top_k + s ) * hidden + j];
	}
	out[i] = __float2half_rn( v );
}

// At most this many work items in a pass of `routes` routes: one per active expert, plus one per kItemRoutes more of its routes
int max_items( int routes, int num_local ) { return std::min( routes, num_local + routes / kItemRoutes ); }

template <GgmlType T>
void launch_proj( const void* const* ptrs, const Item* items, const int* n_items, const int* sorted, int max_n, const block_q8_1* in, bool per_token, int k, int n,
	const half* topw, int top_k, bool weighted, float* out, cudaStream_t stream ) {
	moe_proj_kernel<T><<<dim3( ( n + kMoEWarps - 1 ) / kMoEWarps, max_n ), dim3( 32, kMoEWarps ), 0, stream>>>( ptrs, items, n_items, sorted, in, per_token, k, n, topw, top_k, weighted, out );
}

void proj( GgmlType type, const void* const* ptrs, const Item* items, const int* n_items, const int* sorted, int max_n, const block_q8_1* in, bool per_token, int k,
	int n, const half* topw, int top_k, bool weighted, float* out, cudaStream_t stream ) {
	switch ( type ) {
#define CASE( T ) case GgmlType::T: launch_proj<GgmlType::T>( ptrs, items, n_items, sorted, max_n, in, per_token, k, n, topw, top_k, weighted, out, stream ); break;
		CASE( Q4_0 ) CASE( Q4_1 ) CASE( Q5_0 ) CASE( Q5_1 ) CASE( Q8_0 ) CASE( Q2_0 ) CASE( Q2_K ) CASE( Q3_K ) CASE( Q4_K ) CASE( Q5_K ) CASE( Q6_K )
		CASE( IQ2_XXS ) CASE( IQ2_XS ) CASE( IQ2_S ) CASE( IQ3_XXS ) CASE( IQ3_S ) CASE( IQ1_S ) CASE( IQ1_M ) CASE( IQ4_NL ) CASE( IQ4_XS )
#undef CASE
		default: ENGINE_ASSERT( false, "MoEGguf: unsupported expert block type" );
	}
}

// The scratch of one pass: the rows' q8_1, gate and up outputs, the activation's q8_1, the down partials, the work items and the routes in
// expert order
struct Layout {
	size_t xq, g, u, aq, part, items, n_items, sorted, total;
};
Layout layout( int top_k, int hidden, int inter ) {
	const size_t R = MoEGguf::kMoERowsPerPass, routes = R * top_k;
	auto up = []( size_t b ) { return ( b + 255 ) / 256 * 256; };
	Layout l{};
	l.xq = 0;
	l.g = l.xq + up( R * ( hidden / QK8_1 ) * sizeof( block_q8_1 ) );
	l.u = l.g + up( routes * inter * sizeof( float ) );
	l.aq = l.u + up( routes * inter * sizeof( float ) );
	l.part = l.aq + up( routes * ( inter / QK8_1 ) * sizeof( block_q8_1 ) );
	l.items = l.part + up( routes * hidden * sizeof( float ) );
	l.n_items = l.items + up( routes * sizeof( Item ) );
	l.sorted = l.n_items + 256;
	l.total = l.sorted + up( routes * sizeof( int ) );
	return l;
}


// ---- tile path (prefill) ----
constexpr int kTileRowsPerPass = 1024;   // rows of a pass: about 16 routes of each local expert at top 6 over 384 experts on 4 ranks
constexpr int kTileRoutesMax = 32;      // routes of an item: 16, or 32 when a pass gives the local experts more than about 12 routes each
constexpr int kTileN = 32;              // output rows of a block: 26 KB of shared memory, three blocks an SM
constexpr int kTileThreads = 256;       // 8 warps: the decode needs them to hide its loads
constexpr int kTileLd = 256 + 16;       // halves per shared-memory row: one block of 256 inputs, padded to keep the WMMA rows 32-byte aligned

// The block struct of a tile type (its bytes are staged in shared memory before the decode)
template <GgmlType T> struct TileBlk;
#define TILE_BLK( T, B ) template <> struct TileBlk<GgmlType::T> { using type = B; };
TILE_BLK( IQ1_S, block_iq1_s ) TILE_BLK( IQ1_M, block_iq1_m ) TILE_BLK( IQ2_XXS, block_iq2_xxs ) TILE_BLK( IQ2_XS, block_iq2_xs ) TILE_BLK( IQ2_S, block_iq2_s )
TILE_BLK( IQ3_XXS, block_iq3_xxs ) TILE_BLK( IQ3_S, block_iq3_s ) TILE_BLK( IQ4_XS, block_iq4_xs ) TILE_BLK( Q2_K, block_q2_K ) TILE_BLK( Q3_K, block_q3_K )
TILE_BLK( Q4_K, block_q4_K ) TILE_BLK( Q5_K, block_q5_K ) TILE_BLK( Q6_K, block_q6_K )
#undef TILE_BLK
// bytes of a staged block in shared memory (its size rounded to 16), and the kernel's shared memory: the decoded tile, the routes' inputs and two steps of raw blocks
template <GgmlType T> constexpr int tile_raw_ld() { return ( static_cast<int>( sizeof( typename TileBlk<T>::type ) ) + 15 ) / 16 * 16; }
template <GgmlType T, int RT> constexpr size_t tile_smem() { return static_cast<size_t>( kTileN + RT ) * kTileLd * sizeof( half ) + 2 * kTileN * tile_raw_ld<T>(); }

// Eight IQ1 values d * ( q + delta ), q = 0..2 per nibble of a grid word, as FP16 without a conversion per value: the three values a q can
// give are rounded once (exactly as d * ( q + delta ) is), and each pair of nibbles picks its two halves out of them with one byte permute
// (selector nibbles 2 qa, 2 qa + 1, 2 qb, 2 qb + 1 = the pair's bytes times 34 plus 0x1010)
__device__ __forceinline__ uint4 iq1_eight_h( uint32_t g, float d, float delta ) {
	const half v0 = __float2half_rn( d * ( 0.0f + delta ) ), v1 = __float2half_rn( d * ( 1.0f + delta ) ), v2 = __float2half_rn( d * ( 2.0f + delta ) );
	const uint32_t t0 = static_cast<uint32_t>( __half_as_ushort( v0 ) ) | static_cast<uint32_t>( __half_as_ushort( v1 ) ) << 16, t1 = __half_as_ushort( v2 );
	const uint32_t lo = g & 0x0f0f0f0f, hi = ( g >> 4 ) & 0x0f0f0f0f;
	uint4 out;
	out.x = __byte_perm( t0, t1, ( lo & 0xffff ) * 34 + 0x1010 );
	out.y = __byte_perm( t0, t1, ( lo >> 16 ) * 34 + 0x1010 );
	out.z = __byte_perm( t0, t1, ( hi & 0xffff ) * 34 + 0x1010 );
	out.w = __byte_perm( t0, t1, ( hi >> 16 ) * 34 + 0x1010 );
	return out;
}

// Eight IQ2 values d * grid * sign (grid bytes 8, 25 or 43: byte >> 4 is 0, 1, 2) the same way, the signs set by flipping the halves' sign bits
__device__ __forceinline__ uint4 iq2_eight_h( uint64_t grid, float d, uint32_t signs ) {
	const half v0 = __float2half_rn( d * 8.0f ), v1 = __float2half_rn( d * 25.0f ), v2 = __float2half_rn( d * 43.0f );
	const uint32_t t0 = static_cast<uint32_t>( __half_as_ushort( v0 ) ) | static_cast<uint32_t>( __half_as_ushort( v1 ) ) << 16, t1 = __half_as_ushort( v2 );
	const uint32_t lo = ( static_cast<uint32_t>( grid ) >> 4 ) & 0x0f0f0f0f, hi = ( static_cast<uint32_t>( grid >> 32 ) >> 4 ) & 0x0f0f0f0f;
	auto flip = [&]( int p ) { const uint32_t b = signs >> 2 * p; return ( ( b & 1 ) | ( b & 2 ) << 15 ) << 15; };   // value 2 p's and 2 p + 1's sign bits
	uint4 out;
	out.x = __byte_perm( t0, t1, ( lo & 0xffff ) * 34 + 0x1010 ) ^ flip( 0 );
	out.y = __byte_perm( t0, t1, ( lo >> 16 ) * 34 + 0x1010 ) ^ flip( 1 );
	out.z = __byte_perm( t0, t1, ( hi & 0xffff ) * 34 + 0x1010 ) ^ flip( 2 );
	out.w = __byte_perm( t0, t1, ( hi >> 16 ) * 34 + 0x1010 ) ^ flip( 3 );
	return out;
}

// The FP16 values of block ib of a 256-wide block type, decoded by one warp. The IQ1 and IQ2 types put lane l's eight values at 8 l (sub-block
// l / 4, eighth l % 4) with one 16-byte store, so a warp's stores are consecutive, bit for bit llama.cpp's dequantizers' values; the others take
// those dequantizers (32 or 64 threads a block)
template <GgmlType T>
__device__ __forceinline__ void tile_dequant( const void* w, int64_t ib, half* y, int lane ) {
	if constexpr ( T == GgmlType::IQ1_S ) {
		const block_iq1_s* x = static_cast<const block_iq1_s*>( w ) + ib;
		const int sb = lane >> 2, il = lane & 3;
		const uint16_t qh = x->qh[sb];
		const float delta = qh & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
		const float d = static_cast<float>( x->d ) * ( 2 * ( ( qh >> 12 ) & 7 ) + 1 );
		const uint32_t g = iq1s_grid_gpu[x->qs[4 * sb + il] | ( ( ( qh >> 3 * il ) & 7 ) << 8 )];
		*reinterpret_cast<uint4*>( y + 8 * lane ) = iq1_eight_h( g, d, delta );
	}
	else if constexpr ( T == GgmlType::IQ1_M ) {
		const block_iq1_m* x = static_cast<const block_iq1_m*>( w ) + ib;
		const int sb = lane >> 2, il = lane & 3;
		const uint16_t* sc = reinterpret_cast<const uint16_t*>( x->scales );
		iq1m_scale_t scale;
		scale.u16 = ( sc[0] >> 12 ) | ( ( sc[1] >> 8 ) & 0x00f0 ) | ( ( sc[2] >> 4 ) & 0x0f00 ) | ( sc[3] & 0xf000 );
		const int ib16 = 2 * sb + il / 2;
		const float d = static_cast<float>( scale.f16 ) * ( 2 * ( ( sc[ib16 / 4] >> 3 * ( ib16 % 4 ) ) & 0x7 ) + 1 );
		const uint8_t qh = x->qh[2 * sb + il / 2];
		const float delta = qh & ( 0x08 << 4 * ( il % 2 ) ) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
		const uint32_t g = iq1s_grid_gpu[x->qs[4 * sb + il] | ( ( ( qh >> 4 * ( il % 2 ) ) & 7 ) << 8 )];
		*reinterpret_cast<uint4*>( y + 8 * lane ) = iq1_eight_h( g, d, delta );
	}
	else if constexpr ( T == GgmlType::IQ2_XXS ) {
		const block_iq2_xxs* x = static_cast<const block_iq2_xxs*>( w ) + ib;
		const int sb = lane >> 2, il = lane & 3;
		const uint16_t* q2 = x->qs + 4 * sb;
		const uint32_t aux32 = q2[2] | ( q2[3] << 16 );
		const float d = static_cast<float>( x->d ) * ( 0.5f + ( aux32 >> 28 ) ) * 0.25f;
		*reinterpret_cast<uint4*>( y + 8 * lane ) = iq2_eight_h( iq2xxs_grid[reinterpret_cast<const uint8_t*>( q2 )[il]], d, ksigns_iq2xs[( aux32 >> 7 * il ) & 127] );
	}
	else if constexpr ( T == GgmlType::IQ2_XS ) {
		const block_iq2_xs* x = static_cast<const block_iq2_xs*>( w ) + ib;
		const int sb = lane >> 2, il = lane & 3;
		const uint16_t q2 = x->qs[4 * sb + il];
		const float d = static_cast<float>( x->d ) * ( 0.5f + ( ( x->scales[sb] >> 4 * ( il / 2 ) ) & 0xf ) ) * 0.25f;
		*reinterpret_cast<uint4*>( y + 8 * lane ) = iq2_eight_h( iq2xs_grid[q2 & 511], d, ksigns_iq2xs[q2 >> 9] );
	}
	else if constexpr ( T == GgmlType::IQ2_S ) {
		const block_iq2_s* x = static_cast<const block_iq2_s*>( w ) + ib;
		const int sb = lane >> 2, il = lane & 3;
		const float d = static_cast<float>( x->d ) * ( 0.5f + ( ( x->scales[sb] >> 4 * ( il / 2 ) ) & 0xf ) ) * 0.25f;
		*reinterpret_cast<uint4*>( y + 8 * lane ) = iq2_eight_h( iq2s_grid[x->qs[4 * sb + il] | ( ( x->qh[sb] << ( 8 - 2 * il ) ) & 0x300 )], d, x->qs[QK_K / 8 + 4 * sb + il] );
	}
	else if constexpr ( T == GgmlType::IQ3_XXS ) dequantize_iq3_xxs<half>( w, ib, y, lane );
	else if constexpr ( T == GgmlType::IQ3_S ) dequantize_iq3_s<half>( w, ib, y, lane );
	else if constexpr ( T == GgmlType::IQ4_XS ) dequantize_iq4_xs<half>( w, ib, y, lane );
	else if constexpr ( T == GgmlType::Q4_K ) dequantize_q4_K<half>( w, ib, y, lane );
	else if constexpr ( T == GgmlType::Q2_K ) { dequantize_q2_K<half>( w, ib, y, lane ); dequantize_q2_K<half>( w, ib, y, lane + 32 ); }
	else if constexpr ( T == GgmlType::Q3_K ) { dequantize_q3_K<half>( w, ib, y, lane ); dequantize_q3_K<half>( w, ib, y, lane + 32 ); }
	else if constexpr ( T == GgmlType::Q5_K ) { dequantize_q5_K<half>( w, ib, y, lane ); dequantize_q5_K<half>( w, ib, y, lane + 32 ); }
	else if constexpr ( T == GgmlType::Q6_K ) { dequantize_q6_K<half>( w, ib, y, lane ); dequantize_q6_K<half>( w, ib, y, lane + 32 ); }
}

bool tile_type( GgmlType t ) {
	switch ( t ) {
		case GgmlType::IQ1_S: case GgmlType::IQ1_M: case GgmlType::IQ2_XXS: case GgmlType::IQ2_XS: case GgmlType::IQ2_S: case GgmlType::IQ3_XXS:
		case GgmlType::IQ3_S: case GgmlType::IQ4_XS: case GgmlType::Q2_K: case GgmlType::Q3_K: case GgmlType::Q4_K: case GgmlType::Q5_K: case GgmlType::Q6_K:
			return true;
		default: return false;
	}
}

// out[route][row] = expert row `row` . the route's FP16 input row (the token's row of x, or the route's own row of the activation) for the item's
// routes and kTileN rows: per block of 256 inputs, the warps decode the tile's weight rows into shared memory (4 rows each), the routes'
// inputs are copied beside them, and each warp multiplies a quarter of the block's inputs for one of the two 16 x 16 pieces (WMMA, FP32
// accumulation); the quarters are summed at the end. Down projections are scaled by the route's weight into FP32 partials, gate and up FP16.
// GU: gate and up of the same block type in one pass, the tile's first 16 rows from the gate and the last 16 the same rows of the up projection
// (the routes' inputs copied once for both), and the epilogue writes SiLU( gate ) * up (moe_act16_kernel's activation on the FP16-rounded
// gate and up, so the result is that of the two passes) to out16 [route][n]
template <GgmlType T, bool GU, int RT>
__global__ __launch_bounds__( kTileThreads, 3 )
void moe_tile_kernel( const void* const* __restrict__ ptrs, const void* const* __restrict__ ptrs_up, const Item* __restrict__ items, const int* __restrict__ n_items,
	const int* __restrict__ sorted, const half* __restrict__ in, bool per_token, int k, int n, const half* __restrict__ topw, int top_k, float* __restrict__ part32,
	half* __restrict__ out16, float act_limit ) {
	using namespace nvcuda;
	using Blk = typename TileBlk<T>::type;
	constexpr int kTileRawLd = tile_raw_ld<T>();
	static_assert( sizeof( Blk ) % 2 == 0, "blocks are whole 16-bit words" );
	constexpr int RU = sizeof( Blk ) / 2, UNITS = kTileN * RU, PER = ( UNITS + kTileThreads - 1 ) / kTileThreads;   // 16-bit words of the tile's blocks
	if ( static_cast<int>( blockIdx.y ) >= *n_items ) return;
	extern __shared__ __align__( 32 ) unsigned char tile_smem[];
	half* sW = reinterpret_cast<half*>( tile_smem );                       // [kTileN][kTileLd]
	half* sX = sW + kTileN * kTileLd;                                      // [RT][kTileLd]
	unsigned char* sR = reinterpret_cast<unsigned char*>( sX + RT * kTileLd );   // [2][kTileN][kTileRawLd]: the raw blocks of this and the next step
	__shared__ int s_route[RT];
	const Item it = items[blockIdx.y];
	constexpr int kRows = GU ? kTileN / 2 : kTileN;   // output rows of the block (of each of gate and up)
	const int row0 = blockIdx.x * kRows, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	if ( threadIdx.x < RT ) s_route[threadIdx.x] = threadIdx.x < it.count ? sorted[it.begin + threadIdx.x] : -1;
	const uint16_t* w = static_cast<const uint16_t*>( ptrs[it.expert] );
	const uint16_t* wu = GU ? static_cast<const uint16_t*>( ptrs_up[it.expert] ) : w;
	const int bpr = k / 256;
	// the raw words of step kb's blocks (rows past n read as zero blocks: they decode to zeros), fetched a step ahead into registers; each
	// thread's words and their shared-memory places are fixed for the whole K loop (bit p of `upper`: the word is the up projection's)
	int src[PER], dst[PER];
	unsigned upper = 0;
	#pragma unroll
	for ( int p = 0; p < PER; ++p ) {
		const int u = threadIdx.x + p * kTileThreads, row = u / RU, j = u % RU, gr = row0 + ( GU ? row % kRows : row );
		src[p] = u < UNITS && gr < n ? gr * bpr * RU + j : -1;
		dst[p] = u < UNITS ? row * ( kTileRawLd / 2 ) + j : -1;
		if ( GU && row >= kRows ) upper |= 1u << p;
	}
	uint16_t rr[PER];
	auto fetch = [&]( int kb ) {
		#pragma unroll
		for ( int p = 0; p < PER; ++p ) rr[p] = src[p] >= 0 ? ( ( upper >> p ) & 1 ? wu : w )[src[p] + kb * RU] : 0;
	};
	auto stash = [&]( int buf ) {
		uint16_t* d = reinterpret_cast<uint16_t*>( sR + buf * kTileN * kTileRawLd );
		#pragma unroll
		for ( int p = 0; p < PER; ++p ) if ( dst[p] >= 0 ) d[dst[p]] = rr[p];
	};
	// the routes' input rows this thread copies (16 routes x 32 pieces of 8 halves over the threads)
	constexpr int XPER = RT * 32 / kTileThreads;
	const half* xsrc[XPER];
	#pragma unroll
	for ( int i = 0; i < XPER; ++i ) {
		const int v = threadIdx.x + i * kTileThreads, j = v >> 5, c = v & 31, r = j < it.count ? sorted[it.begin + j] : -1;
		xsrc[i] = r >= 0 ? in + static_cast<size_t>( per_token ? r / top_k : r ) * k + c * 8 : nullptr;
	}
	fetch( 0 );
	stash( 0 );
	__syncthreads();
	const int wn = ( warp & 1 ) * 16, kh = ( warp >> 1 ) * 64;   // the warp's rows and quarter of the inputs
	constexpr int RF = RT / 16;   // 16-route fragments
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[RF];
	#pragma unroll
	for ( int f = 0; f < RF; ++f ) wmma::fill_fragment( acc[f], 0.0f );
	for ( int kb = 0; kb < bpr; ++kb ) {
		if ( kb + 1 < bpr ) fetch( kb + 1 );   // in flight while this step decodes and multiplies
		const unsigned char* raw = sR + ( kb & 1 ) * kTileN * kTileRawLd;
		#pragma unroll
		for ( int i = 0; i < kTileN / 8; ++i ) {
			const int r = warp * ( kTileN / 8 ) + i;
			tile_dequant<T>( raw + r * kTileRawLd, 0, sW + r * kTileLd, lane );
		}
		#pragma unroll
		for ( int i = 0; i < XPER; ++i ) {
			const int v = threadIdx.x + i * kTileThreads;
			const uint4 val = xsrc[i] != nullptr ? *reinterpret_cast<const uint4*>( xsrc[i] + kb * 256 ) : make_uint4( 0, 0, 0, 0 );
			*reinterpret_cast<uint4*>( sX + ( v >> 5 ) * kTileLd + ( v & 31 ) * 8 ) = val;
		}
		__syncthreads();
		#pragma unroll
		for ( int kk = 0; kk < 64; kk += 16 ) {
			wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
			wmma::load_matrix_sync( b, sW + wn * kTileLd + kh + kk, kTileLd );
			#pragma unroll
			for ( int f = 0; f < RF; ++f ) {
				if ( f * 16 >= it.count ) break;
				wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
				wmma::load_matrix_sync( a, sX + f * 16 * kTileLd + kh + kk, kTileLd );
				wmma::mma_sync( acc[f], a, b, acc[f] );
			}
		}
		if ( kb + 1 < bpr ) stash( ( kb + 1 ) & 1 );
		__syncthreads();
	}
	// the four input quarters' pieces through shared memory, summed in order, to the routes' rows
	float* sC = reinterpret_cast<float*>( tile_smem );   // [4][RT][kTileN + 4]
	constexpr int ldc = kTileN + 4;
	#pragma unroll
	for ( int f = 0; f < RF; ++f ) wmma::store_matrix_sync( sC + ( ( warp >> 1 ) * RT + f * 16 ) * ldc + wn, acc[f], ldc, wmma::mem_row_major );
	__syncthreads();
	for ( int v = threadIdx.x; v < RT * kRows; v += kTileThreads ) {
		const int j = v / kRows, c = v % kRows, r = s_route[j], gr = row0 + c;
		if ( r < 0 || gr >= n ) continue;
		const float* q = sC + j * ldc + c;
		const float val = ( q[0] + q[RT * ldc] ) + ( q[2 * RT * ldc] + q[3 * RT * ldc] );
		if constexpr ( GU ) {
			const float* qu = q + kRows;
			float gv = __half2float( __float2half_rn( val ) );
			float uv = __half2float( __float2half_rn( ( qu[0] + qu[RT * ldc] ) + ( qu[2 * RT * ldc] + qu[3 * RT * ldc] ) ) );
			if ( act_limit > 0.0f ) {
				gv = fminf( gv, act_limit );
				uv = fmaxf( -act_limit, fminf( uv, act_limit ) );
			}
			out16[static_cast<size_t>( r ) * n + gr] = __float2half_rn( gv / ( 1.0f + __expf( -gv ) ) * uv );
		}
		else if ( part32 != nullptr ) part32[static_cast<size_t>( r ) * n + gr] = val * __half2float( topw[r] );
		else out16[static_cast<size_t>( r ) * n + gr] = __float2half_rn( val );
	}
}

template <GgmlType T, bool GU, int RT>
void launch_tile( const void* const* ptrs, const void* const* ptrs_up, const Item* items, const int* n_items, const int* sorted, int max_n, const half* in, bool per_token,
	int k, int n, const half* topw, int top_k, float* part32, half* out16, float act_limit, cudaStream_t stream ) {
	static bool attr[64] = {};
	int dev = 0;
	cudaGetDevice( &dev );
	if ( !attr[dev] ) {
		CUDA_CHECK( cudaFuncSetAttribute( moe_tile_kernel<T, GU, RT>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>( tile_smem<T, RT>() ) ) );
		attr[dev] = true;
	}
	constexpr int rows = GU ? kTileN / 2 : kTileN;
	moe_tile_kernel<T, GU, RT><<<dim3( ( n + rows - 1 ) / rows, max_n ), kTileThreads, tile_smem<T, RT>(), stream>>>( ptrs, ptrs_up, items, n_items, sorted, in, per_token, k,
		n, topw, top_k, part32, out16, act_limit );
}

// ptrs_up != null: gate (ptrs) and up of the same type in one pass, the activation to out16; rt: the items' routes (16 or 32)
void tile_proj( GgmlType type, int rt, const void* const* ptrs, const void* const* ptrs_up, const Item* items, const int* n_items, const int* sorted, int max_n,
	const half* in, bool per_token, int k, int n, const half* topw, int top_k, float* part32, half* out16, float act_limit, cudaStream_t stream ) {
	switch ( type ) {
#define LAUNCH( T, RT ) \
		if ( ptrs_up != nullptr ) launch_tile<GgmlType::T, true, RT>( ptrs, ptrs_up, items, n_items, sorted, max_n, in, per_token, k, n, topw, top_k, part32, out16, act_limit, stream ); \
		else launch_tile<GgmlType::T, false, RT>( ptrs, nullptr, items, n_items, sorted, max_n, in, per_token, k, n, topw, top_k, part32, out16, act_limit, stream );
#define CASE( T ) case GgmlType::T: if ( rt > 16 ) { LAUNCH( T, 32 ) } else { LAUNCH( T, 16 ) } break;
		CASE( IQ1_S ) CASE( IQ1_M ) CASE( IQ2_XXS ) CASE( IQ2_XS ) CASE( IQ2_S ) CASE( IQ3_XXS ) CASE( IQ3_S ) CASE( IQ4_XS ) CASE( Q2_K ) CASE( Q3_K ) CASE( Q4_K )
		CASE( Q5_K ) CASE( Q6_K )
#undef CASE
#undef LAUNCH
		default: ENGINE_ASSERT( false, "MoEGguf: no tile path for the expert block type" );
	}
}

// act = SiLU( gate ) * up of every route in FP16 ([route][inter]), the gate and up from the tile path; limit > 0 clamps as moe_act_q8_kernel
__global__ void moe_act16_kernel( const half* __restrict__ g, const half* __restrict__ u, half* __restrict__ a, size_t count, float limit ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i >= count ) return;
	float gv = __half2float( g[i] ), uv = __half2float( u[i] );
	if ( limit > 0.0f ) {
		gv = fminf( gv, limit );
		uv = fmaxf( -limit, fminf( uv, limit ) );
	}
	a[i] = __float2half_rn( gv / ( 1.0f + __expf( -gv ) ) * uv );
}

// The scratch of one tile pass: the activation in FP16, the down partials in FP32 (FP16 ones moved the first token's distribution by a KL of
// up to 0.08 on 20K-token prompts; the gate and up of two block types, FP16, sit in their place: the activation reads them before the down
// projection writes there), the work items and the routes in expert order
struct TileLayout {
	size_t a, part, g, u, items, n_items, sorted, total;
};
TileLayout tile_layout( int top_k, int hidden, int inter ) {
	const size_t routes = static_cast<size_t>( kTileRowsPerPass ) * top_k;
	auto up = []( size_t b ) { return ( b + 255 ) / 256 * 256; };
	TileLayout l{};
	l.a = 0;
	l.part = l.a + up( routes * inter * sizeof( half ) );
	l.g = l.part;
	l.u = l.g + up( routes * inter * sizeof( half ) );
	l.items = l.part + std::max( up( routes * hidden * sizeof( float ) ), 2 * up( routes * inter * sizeof( half ) ) );
	l.n_items = l.items + up( routes * sizeof( Item ) );
	l.sorted = l.n_items + 256;
	l.total = l.sorted + up( routes * sizeof( int ) );
	return l;
}

} // namespace

size_t MoEGguf::scratch_bytes( int top_k, int hidden, int inter ) {
	const bool tile = hidden % 256 == 0 && inter % 256 == 0;   // the widths the tile path takes
	return std::max( layout( top_k, hidden, inter ).total, tile ? tile_layout( top_k, hidden, inter ).total : 0 );
}

void MoEGguf::forward( const Table& t, const half* x, int m, const int* topk_idx, const half* topk_w, int top_k, half* out, void* scratch, uint32_t* heat,
	cudaStream_t stream ) {
	ENGINE_ASSERT( t.hidden % QK8_1 == 0 && t.inter % QK8_1 == 0, "MoEGguf: widths must be whole q8_1 blocks" );
	char* ws = static_cast<char*>( scratch );
	const void* const* gate = t.ptrs, * const* up = t.ptrs + t.num_local, * const* down = t.ptrs + 2 * t.num_local;
	static const bool tile_on = engine::runtime::options().gguf_moe_tile;
	if ( tile_on && m > kMoERowsPerPass && t.hidden % 256 == 0 && t.inter % 256 == 0 && tile_type( t.type[0] ) && tile_type( t.type[1] ) && tile_type( t.type[2] ) ) {
		const TileLayout l = tile_layout( top_k, t.hidden, t.inter );
		auto* a16 = reinterpret_cast<half*>( ws + l.a );
		auto* part = reinterpret_cast<float*>( ws + l.part );
		auto* g16 = reinterpret_cast<half*>( ws + l.g );
		auto* u16 = reinterpret_cast<half*>( ws + l.u );
		auto* items = reinterpret_cast<Item*>( ws + l.items );
		auto* n_items = reinterpret_cast<int*>( ws + l.n_items );
		auto* sorted = reinterpret_cast<int*>( ws + l.sorted );
		for ( int r0 = 0; r0 < m; r0 += kTileRowsPerPass ) {
			const int rn = std::min( kTileRowsPerPass, m - r0 ), routes = rn * top_k;
			// 32-route items once the local experts average more than about 12 routes in the pass (fewer decodes, at a lower occupancy)
			const int rt = routes > 12 * 4 * t.num_local ? kTileRoutesMax : 16, mi = std::min( routes, t.num_local + routes / rt );
			const int* ti = topk_idx + static_cast<size_t>( r0 ) * top_k;
			const half* tw = topk_w + static_cast<size_t>( r0 ) * top_k;
			const half* xr = x + static_cast<size_t>( r0 ) * t.hidden;
			moe_dispatch_kernel<<<1, 256, 2 * t.num_local * sizeof( int ), stream>>>( ti, routes, t.first_expert, t.num_local, items, n_items, sorted, heat, rt );
			if ( t.type[0] == t.type[1] ) tile_proj( t.type[0], rt, gate, up, items, n_items, sorted, mi, xr, true, t.hidden, t.inter, tw, top_k, nullptr, a16, t.act_limit, stream );
			else {
				tile_proj( t.type[0], rt, gate, nullptr, items, n_items, sorted, mi, xr, true, t.hidden, t.inter, tw, top_k, nullptr, g16, 0.0f, stream );
				tile_proj( t.type[1], rt, up, nullptr, items, n_items, sorted, mi, xr, true, t.hidden, t.inter, tw, top_k, nullptr, u16, 0.0f, stream );
				const size_t act = static_cast<size_t>( routes ) * t.inter;
				moe_act16_kernel<<<static_cast<unsigned>( ( act + 255 ) / 256 ), 256, 0, stream>>>( g16, u16, a16, act, t.act_limit );
			}
			tile_proj( t.type[2], rt, down, nullptr, items, n_items, sorted, mi, a16, false, t.inter, t.hidden, tw, top_k, part, nullptr, 0.0f, stream );
			moe_combine_kernel<<<( rn * t.hidden + 255 ) / 256, 256, 0, stream>>>( part, ti, t.first_expert, t.num_local, out + static_cast<size_t>( r0 ) * t.hidden, rn, top_k, t.hidden );
		}
		return;
	}
	const Layout l = layout( top_k, t.hidden, t.inter );
	auto* xq = reinterpret_cast<block_q8_1*>( ws + l.xq );
	auto* g = reinterpret_cast<float*>( ws + l.g );
	auto* u = reinterpret_cast<float*>( ws + l.u );
	auto* aq = reinterpret_cast<block_q8_1*>( ws + l.aq );
	auto* part = reinterpret_cast<float*>( ws + l.part );
	auto* items = reinterpret_cast<Item*>( ws + l.items );
	auto* n_items = reinterpret_cast<int*>( ws + l.n_items );
	auto* sorted = reinterpret_cast<int*>( ws + l.sorted );
	for ( int r0 = 0; r0 < m; r0 += kMoERowsPerPass ) {
		const int rn = std::min( kMoERowsPerPass, m - r0 ), routes = rn * top_k, mi = max_items( routes, t.num_local );
		const int* ti = topk_idx + static_cast<size_t>( r0 ) * top_k;
		const half* tw = topk_w + static_cast<size_t>( r0 ) * top_k;
		moe_dispatch_kernel<<<1, 256, 2 * t.num_local * sizeof( int ), stream>>>( ti, routes, t.first_expert, t.num_local, items, n_items, sorted, heat, kItemRoutes );
		quantize_q8_1_kernel<<<dim3( ( t.hidden + 255 ) / 256, rn ), 256, 0, stream>>>( x + static_cast<size_t>( r0 ) * t.hidden, xq, t.hidden, GgufColTiling{} );
		proj( t.type[0], gate, items, n_items, sorted, mi, xq, true, t.hidden, t.inter, tw, top_k, false, g, stream );
		proj( t.type[1], up, items, n_items, sorted, mi, xq, true, t.hidden, t.inter, tw, top_k, false, u, stream );
		moe_act_q8_kernel<<<dim3( ( t.inter + 127 ) / 128, routes ), 128, 0, stream>>>( g, u, aq, t.inter, t.act_limit );
		proj( t.type[2], down, items, n_items, sorted, mi, aq, false, t.inter, t.hidden, tw, top_k, true, part, stream );
		moe_combine_kernel<<<( rn * t.hidden + 255 ) / 256, 256, 0, stream>>>( part, ti, t.first_expert, t.num_local, out + static_cast<size_t>( r0 ) * t.hidden, rn, top_k, t.hidden );
	}
}

} // namespace engine::modules
