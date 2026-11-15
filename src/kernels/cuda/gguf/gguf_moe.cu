// GGUF routed experts on CUDA (MoEGguf in engine/modules/gguf.h). The routes of a pass are grouped by expert into work items of up to
// kItemRoutes routes; the gate and up projections are a GEMV of one output row per warp and work item, the expert's weight row read once for
// all the item's routes through llama.cpp's vec_dot of the projection's block type (LinearGguf's multi-row GEMV structure); the down projection
// takes one output row of one token per warp over the token's routes and sums them as it goes. Routes of experts on other ranks are left out. Prefill chunks (more rows than one pass of that path) take the tile path instead:
// passes of kTileRowsPerPass rows, items of up to 16 or 32 routes, and per item a tile of output rows whose weight blocks are decoded
// to FP16 in shared memory once and multiplied with all the item's routes on the tensor cores (the vec_dot path decodes them per route).
#include "engine/modules/gguf.h"
#include "engine/backend/runtime.h"
#include "engine/runtime/options.h"
#include "gguf_common.cuh"
#include "ggml/dequantize.cuh"
#include "../trellis/ptx.cuh"
#include <mma.h>
#include <algorithm>

namespace engine::modules {

namespace {

constexpr int kMoEWarps = 4;

// Routes of a work item at most: an expert with more takes several items, each reading its weights (mostly from the L2 then); items of more
// routes held accumulators for all of them in every warp, which halved the occupancy, and verify passes (4 to 16 rows) ran at half the speed
constexpr int kItemRoutes = 2;
constexpr int kMoEWaveBlocks = 2560;   // four waves of moe_proj_kernel blocks on 80 SMs

constexpr int kDispatchWarps = 8;   // moe_dispatch_kernel's block of 256 threads
size_t dispatch_smem( int num_local ) { return static_cast<size_t>( 2 + kDispatchWarps ) * num_local * sizeof( int ); }

// A work item: one local expert and up to kItemRoutes of its routes (route ids in `sorted`, from `begin`)
struct Item {
	int expert, begin, count, pad;
};

// Groups the pass's routes by local expert (counting sort in block 0): items[0 .. *n_items) and the route ids in expert order; counts the
// experts' routes into heat. Blocks 1 .. (xq != nullptr) quantize the pass's rows of x to q8_1, qx blocks of 256 values a row
__global__ void moe_dispatch_kernel( const int* __restrict__ topk, int routes, int first, int num_local, Item* __restrict__ items, int* __restrict__ n_items,
	int* __restrict__ sorted, uint32_t* __restrict__ heat, int item_routes, const half* __restrict__ x = nullptr, block_q8_1* __restrict__ xq = nullptr, int k = 0,
	int qx = 1 ) {
	if ( blockIdx.x > 0 ) {
		const int b = blockIdx.x - 1;
		quantize_q8_1_at( x, xq, k, GgufColTiling{}, b / qx, b % qx * blockDim.x + threadIdx.x );
		return;
	}
	if ( routes <= 32 ) {
		// a decode pass, on one warp without barriers: lane r ranks its route among the pass's local ones (expert order, then route order), and the
		// first route of each expert writes the expert's items after those of the experts before it: the same lists as the block below
		if ( threadIdx.x >= 32 ) return;
		const int lane = threadIdx.x;
		const int e = lane < routes ? topk[lane] - first : -1;
		const bool ok = e >= 0 && e < num_local;
		const int key = ok ? e : 0x7fffffff;
		int rank = 0, cnt = 0;
		bool lead = ok;
		#pragma unroll
		for ( int j = 0; j < 32; ++j ) {
			const int kj = __shfl_sync( 0xffffffff, key, j );
			rank += kj < key || ( kj == key && j < lane );
			cnt += kj == key;
			lead = lead && !( kj == key && j < lane );
		}
		if ( ok ) sorted[rank] = lane;
		const int ni = lead ? ( cnt + item_routes - 1 ) / item_routes : 0;
		int before = 0, total = 0;
		#pragma unroll
		for ( int j = 0; j < 32; ++j ) {
			const int nj = __shfl_sync( 0xffffffff, ni, j ), kj = __shfl_sync( 0xffffffff, key, j );
			before += kj < key ? nj : 0;
			total += nj;
		}
		if ( lead ) {
			for ( int b = 0, i = before; b < cnt; b += item_routes, ++i ) items[i] = Item{ e, rank + b, min( item_routes, cnt - b ), 0 };
			if ( heat != nullptr ) heat[e] += static_cast<uint32_t>( cnt );
		}
		if ( lane == 0 ) *n_items = total;
		return;
	}
	// counting sort, stable: warp w takes routes [w seg, (w + 1) seg) in chunks of 32 and counts its routes per expert, a scan gives each expert
	// its first route and items, then each warp's routes of an expert go after the earlier warps' (route order within an expert)
	extern __shared__ int sh[];
	int* count = sh;                    // [num_local]
	int* start = sh + num_local;        // [num_local]
	int* wcnt = sh + 2 * num_local;     // [kDispatchWarps][num_local]: a warp's routes per expert, then where its next one goes
	const int warp = threadIdx.x >> 5, wl = threadIdx.x & 31;
	const unsigned below = ( 1u << wl ) - 1;
	for ( int i = threadIdx.x; i < ( 2 + kDispatchWarps ) * num_local; i += blockDim.x ) sh[i] = 0;
	__syncthreads();
	const int seg = ( ( routes + kDispatchWarps - 1 ) / kDispatchWarps + 31 ) / 32 * 32;
	const int s0 = warp * seg, s1 = min( routes, s0 + seg );
	int* mine = wcnt + warp * num_local;
	for ( int c = s0; c < s1; c += 32 ) {
		const int e = c + wl < s1 ? topk[c + wl] - first : -1;
		const bool ok = e >= 0 && e < num_local;
		const unsigned peers = __match_any_sync( 0xffffffff, ok ? e : -1 );
		if ( ok && ( peers & below ) == 0 ) mine[e] += __popc( peers );
		__syncwarp();
	}
	__syncthreads();
	for ( int e = threadIdx.x; e < num_local; e += blockDim.x ) {
		int c = 0;
		for ( int w = 0; w < kDispatchWarps; ++w ) c += wcnt[w * num_local + e];
		count[e] = c;
	}
	__syncthreads();
	if ( threadIdx.x < 32 ) {
		// warp 0, 32 experts at a time: scans of the routes and the items give each expert its first route and first item (expert order)
		const int lane = threadIdx.x;
		int at = 0, n = 0;
		for ( int e0 = 0; e0 < num_local; e0 += 32 ) {
			const int e = e0 + lane;
			const int c = e < num_local ? count[e] : 0, ni = ( c + item_routes - 1 ) / item_routes;
			int sc = c, sn = ni;
			#pragma unroll
			for ( int o = 1; o < 32; o <<= 1 ) {
				const int tc = __shfl_up_sync( 0xffffffff, sc, o ), tn = __shfl_up_sync( 0xffffffff, sn, o );
				if ( lane >= o ) { sc += tc; sn += tn; }
			}
			if ( e < num_local ) {
				const int a = at + sc - c;
				start[e] = a;
				for ( int b = 0, j = n + sn - ni; b < c; b += item_routes, ++j ) items[j] = Item{ e, a + b, min( item_routes, c - b ), 0 };
				if ( heat != nullptr && c > 0 ) heat[e] += static_cast<uint32_t>( c );
			}
			at += __shfl_sync( 0xffffffff, sc, 31 );
			n += __shfl_sync( 0xffffffff, sn, 31 );
		}
		if ( lane == 0 ) *n_items = n;
	}
	__syncthreads();
	for ( int e = threadIdx.x; e < num_local; e += blockDim.x ) {
		int run = start[e];
		for ( int w = 0; w < kDispatchWarps; ++w ) {
			const int t = wcnt[w * num_local + e];
			wcnt[w * num_local + e] = run;
			run += t;
		}
	}
	__syncthreads();
	for ( int c = s0; c < s1; c += 32 ) {
		const int e = c + wl < s1 ? topk[c + wl] - first : -1;
		const bool ok = e >= 0 && e < num_local;
		const unsigned peers = __match_any_sync( 0xffffffff, ok ? e : -1 );
		if ( ok ) sorted[mine[e] + __popc( peers & below )] = c + wl;
		__syncwarp();
		if ( ok && ( peers & below ) == 0 ) mine[e] += __popc( peers );
		__syncwarp();
	}
}

// out[route][row] = expert row `row` . input of each route of the work item (the route's token's q8_1 row, or the route's own), scaled by
// the route's weight when `weighted`. One warp per output row, kMoEWarps rows per block: expert rows are short (640 or 2560 values), too
// short to keep a whole block of threads on one. IR: the items' most routes (1 for one-row passes, else kItemRoutes); GU: the same row of the
// up projection (ptrs_up) too, into out_up, sharing the input loads.
// Block row y takes items y, y + gridDim.y, ...
template <GgmlType T, int IR, bool GU>
__global__ __launch_bounds__( 32 * kMoEWarps, 8 )
void moe_proj_kernel( const void* const* __restrict__ ptrs, const void* const* __restrict__ ptrs_up, const Item* __restrict__ items, const int* __restrict__ n_items,
	const int* __restrict__ sorted, const block_q8_1* __restrict__ in, bool per_token, int k, int n, const half* __restrict__ topw, int top_k, bool weighted,
	float* __restrict__ out, float* __restrict__ out_up ) {
	using Tr = Traits<T>;
	const int row = blockIdx.x * kMoEWarps + threadIdx.y;
	if ( row >= n ) return;
	const int lane = threadIdx.x;
	const int items_n = *n_items;
	for ( int ii = blockIdx.y; ii < items_n; ii += gridDim.y ) {   // the pass's items, gridDim.y rows of blocks taking every gridDim.y-th
	const Item it = items[ii];
	const void* w = ptrs[it.expert];
	const void* wu = GU ? ptrs_up[it.expert] : nullptr;
	const int blocks_per_row = k / Tr::qk, y_blocks = k / QK8_1;
	const block_q8_1* y[IR];
	#pragma unroll
	for ( int j = 0; j < IR; ++j ) {
		const int r = sorted[it.begin + min( j, it.count - 1 )];
		y[j] = in + static_cast<size_t>( per_token ? r / top_k : r ) * y_blocks;
	}
	constexpr int per_iter = Tr::vdr * 32 / Tr::qi;   // weight blocks a warp covers per iteration
	float acc[IR] = {}, acc_u[IR] = {};
	for ( int kbx = lane / ( Tr::qi / Tr::vdr ); kbx < blocks_per_row; kbx += per_iter ) {
		const int kby = kbx * ( Tr::qk / QK8_1 );
		const int kqs = Tr::vdr * ( lane % ( Tr::qi / Tr::vdr ) );
		#pragma unroll
		for ( int j = 0; j < IR; ++j ) {
			if ( j < it.count ) {
				acc[j] += Tr::dot( w, y[j] + kby, row * blocks_per_row + kbx, kqs );
				if constexpr ( GU ) acc_u[j] += Tr::dot( wu, y[j] + kby, row * blocks_per_row + kbx, kqs );
			}
		}
	}
	#pragma unroll
	for ( int j = 0; j < IR; ++j ) {
		#pragma unroll
		for ( int off = 16; off > 0; off >>= 1 ) {
			acc[j] += __shfl_xor_sync( 0xffffffff, acc[j], off );
			if constexpr ( GU ) acc_u[j] += __shfl_xor_sync( 0xffffffff, acc_u[j], off );
		}
	}
	if ( lane < it.count ) {
		float v = 0.0f, vu = 0.0f;
		#pragma unroll
		for ( int j = 0; j < IR; ++j ) {
			v = lane == j ? acc[j] : v;
			vu = lane == j ? acc_u[j] : vu;
		}
		const int r = sorted[it.begin + lane];
		out[static_cast<size_t>( r ) * n + row] = weighted ? v * __half2float( topw[r] ) : v;
		if constexpr ( GU ) out_up[static_cast<size_t>( r ) * n + row] = vu;
	}
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

// A one-row pass's down projection and combine in one launch: warp `row` takes every item (one route each: a token routes to an expert once), two
// at a time, each dot as moe_proj_kernel's, and the weighted rows are summed in route order as moe_combine_kernel sums them: out is bit-identical
// to the pair
template <GgmlType T>
__global__ __launch_bounds__( 32 * kMoEWarps, 8 )
void moe_down_combine_kernel( const void* const* __restrict__ ptrs, const Item* __restrict__ items, const int* __restrict__ n_items, const int* __restrict__ sorted,
	const block_q8_1* __restrict__ in, int k, int n, const half* __restrict__ topw, int top_k, half* __restrict__ out ) {
	using Tr = Traits<T>;
	const int row = blockIdx.x * kMoEWarps + threadIdx.y;
	if ( row >= n ) return;
	const int lane = threadIdx.x;
	const int items_n = *n_items;
	const int blocks_per_row = k / Tr::qk, y_blocks = k / QK8_1;
	constexpr int per_iter = Tr::vdr * 32 / Tr::qi;
	const int kqs = Tr::vdr * ( lane % ( Tr::qi / Tr::vdr ) );
	float v_mine = 0.0f;   // lane i: item i's weighted value
	int r_mine = -1;       // and its route
	for ( int i = 0; i < items_n; i += 2 ) {
		const bool two = i + 1 < items_n;
		const Item a = items[i], b = items[two ? i + 1 : i];
		const int ra = sorted[a.begin], rb = sorted[b.begin];
		const void* wa = ptrs[a.expert];
		const void* wb = ptrs[b.expert];
		const block_q8_1* ya = in + static_cast<size_t>( ra ) * y_blocks;
		const block_q8_1* yb = in + static_cast<size_t>( rb ) * y_blocks;
		float acc_a = 0.0f, acc_b = 0.0f;
		for ( int kbx = lane / ( Tr::qi / Tr::vdr ); kbx < blocks_per_row; kbx += per_iter ) {
			const int kby = kbx * ( Tr::qk / QK8_1 );
			acc_a += Tr::dot( wa, ya + kby, row * blocks_per_row + kbx, kqs );
			if ( two ) acc_b += Tr::dot( wb, yb + kby, row * blocks_per_row + kbx, kqs );
		}
		#pragma unroll
		for ( int off = 16; off > 0; off >>= 1 ) {
			acc_a += __shfl_xor_sync( 0xffffffff, acc_a, off );
			acc_b += __shfl_xor_sync( 0xffffffff, acc_b, off );
		}
		if ( lane == i ) { v_mine = acc_a * __half2float( topw[ra] ); r_mine = ra; }
		if ( two && lane == i + 1 ) { v_mine = acc_b * __half2float( topw[rb] ); r_mine = rb; }
	}
	float sum = 0.0f;
	for ( int s = 0; s < top_k; ++s ) {
		const unsigned m = __ballot_sync( 0xffffffff, r_mine == s );
		if ( m ) sum += __shfl_sync( 0xffffffff, v_mine, __ffs( m ) - 1 );
	}
	if ( lane == 0 ) out[row] = __float2half_rn( sum );
}

// A multi-row pass's down projection and combine by token: warp (row, token) runs over the token's routes to this rank's experts, the routes' loads
// in flight together, and sums each route's weighted dot in its lanes, reduced once. For short rows (an expert's width sliced over the ranks: a few
// blocks), where a (row, work item) warp pays its pointer chain and reductions per couple of blocks; an expert routed by several tokens is read
// once per route instead of once per item (from the L2 after the first)
template <GgmlType T>
__global__ __launch_bounds__( 32 * kMoEWarps, 8 )
void moe_down_tok_kernel( const void* const* __restrict__ ptrs, const int* __restrict__ topk, int first, int num_local, const block_q8_1* __restrict__ in, int k, int n,
	const half* __restrict__ topw, int top_k, half* __restrict__ out ) {
	using Tr = Traits<T>;
	const int row = blockIdx.x * kMoEWarps + threadIdx.y;
	if ( row >= n ) return;
	const int t = blockIdx.y, lane = threadIdx.x;
	const int blocks_per_row = k / Tr::qk, y_blocks = k / QK8_1;
	constexpr int per_iter = Tr::vdr * 32 / Tr::qi;
	const int kqs = Tr::vdr * ( lane % ( Tr::qi / Tr::vdr ) );
	float acc = 0.0f;
	#pragma unroll 4
	for ( int s = 0; s < top_k; ++s ) {
		const int r = t * top_k + s;
		const int e = topk[r] - first;
		if ( e < 0 || e >= num_local ) continue;   // the same for the whole warp
		const void* w = ptrs[e];
		const block_q8_1* y = in + static_cast<size_t>( r ) * y_blocks;
		float a = 0.0f;
		for ( int kbx = lane / ( Tr::qi / Tr::vdr ); kbx < blocks_per_row; kbx += per_iter ) a += Tr::dot( w, y + kbx * ( Tr::qk / QK8_1 ), row * blocks_per_row + kbx, kqs );
		acc += a * __half2float( topw[r] );
	}
	#pragma unroll
	for ( int off = 16; off > 0; off >>= 1 ) acc += __shfl_xor_sync( 0xffffffff, acc, off );
	if ( lane == 0 ) out[static_cast<size_t>( t ) * n + row] = __float2half_rn( acc );
}

void down_tok( GgmlType type, const void* const* ptrs, const int* topk, int first, int num_local, const block_q8_1* in, int k, int n, const half* topw, int top_k, int rows,
	half* out, cudaStream_t stream ) {
	const dim3 grid( ( n + kMoEWarps - 1 ) / kMoEWarps, rows ), block( 32, kMoEWarps );
	switch ( type ) {
#define CASE( T ) case GgmlType::T: moe_down_tok_kernel<GgmlType::T><<<grid, block, 0, stream>>>( ptrs, topk, first, num_local, in, k, n, topw, top_k, out ); break;
		CASE( Q4_0 ) CASE( Q4_1 ) CASE( Q5_0 ) CASE( Q5_1 ) CASE( Q8_0 ) CASE( Q2_0 ) CASE( Q2_K ) CASE( Q3_K ) CASE( Q4_K ) CASE( Q5_K ) CASE( Q6_K )
		CASE( IQ2_XXS ) CASE( IQ2_XS ) CASE( IQ2_S ) CASE( IQ3_XXS ) CASE( IQ3_S ) CASE( IQ1_S ) CASE( IQ1_M ) CASE( IQ4_NL ) CASE( IQ4_XS )
#undef CASE
		default: ENGINE_ASSERT( false, "MoEGguf: unsupported expert block type" );
	}
}

void down_combine( GgmlType type, const void* const* ptrs, const Item* items, const int* n_items, const int* sorted, const block_q8_1* in, int k, int n, const half* topw,
	int top_k, half* out, cudaStream_t stream ) {
	const dim3 grid( ( n + kMoEWarps - 1 ) / kMoEWarps ), block( 32, kMoEWarps );
	switch ( type ) {
#define CASE( T ) case GgmlType::T: moe_down_combine_kernel<GgmlType::T><<<grid, block, 0, stream>>>( ptrs, items, n_items, sorted, in, k, n, topw, top_k, out ); break;
		CASE( Q4_0 ) CASE( Q4_1 ) CASE( Q5_0 ) CASE( Q5_1 ) CASE( Q8_0 ) CASE( Q2_0 ) CASE( Q2_K ) CASE( Q3_K ) CASE( Q4_K ) CASE( Q5_K ) CASE( Q6_K )
		CASE( IQ2_XXS ) CASE( IQ2_XS ) CASE( IQ2_S ) CASE( IQ3_XXS ) CASE( IQ3_S ) CASE( IQ1_S ) CASE( IQ1_M ) CASE( IQ4_NL ) CASE( IQ4_XS )
#undef CASE
		default: ENGINE_ASSERT( false, "MoEGguf: unsupported expert block type" );
	}
}

template <GgmlType T, int IR>
void launch_proj_ir( const void* const* ptrs, const void* const* ptrs_up, const Item* items, const int* n_items, const int* sorted, int max_n, const block_q8_1* in,
	bool per_token, int k, int n, const half* topw, int top_k, bool weighted, float* out, float* out_up, cudaStream_t stream ) {
	// passes route to a few of max_n possible items: fewer rows of blocks, each looping over its share, leave no idle blocks; a short output
	// (an expert's width sliced over the ranks) takes more rows of blocks, about four waves in all, or its last wave runs a few blocks long
	const int gx = ( n + kMoEWarps - 1 ) / kMoEWarps;
	const dim3 grid( gx, std::min( max_n, std::max( 2 + 2 * IR, kMoEWaveBlocks / gx ) ) ), block( 32, kMoEWarps );
	if ( ptrs_up != nullptr ) moe_proj_kernel<T, IR, true><<<grid, block, 0, stream>>>( ptrs, ptrs_up, items, n_items, sorted, in, per_token, k, n, topw, top_k, weighted, out, out_up );
	else moe_proj_kernel<T, IR, false><<<grid, block, 0, stream>>>( ptrs, nullptr, items, n_items, sorted, in, per_token, k, n, topw, top_k, weighted, out, nullptr );
}

template <GgmlType T>
void launch_proj( int ir, const void* const* ptrs, const void* const* ptrs_up, const Item* items, const int* n_items, const int* sorted, int max_n, const block_q8_1* in,
	bool per_token, int k, int n, const half* topw, int top_k, bool weighted, float* out, float* out_up, cudaStream_t stream ) {
	if ( ir == 1 ) launch_proj_ir<T, 1>( ptrs, ptrs_up, items, n_items, sorted, max_n, in, per_token, k, n, topw, top_k, weighted, out, out_up, stream );
	else launch_proj_ir<T, kItemRoutes>( ptrs, ptrs_up, items, n_items, sorted, max_n, in, per_token, k, n, topw, top_k, weighted, out, out_up, stream );
}

// ir: the items' most routes (1 or kItemRoutes); ptrs_up != null: gate (ptrs) and up of the same type in one launch, up into out_up
void proj( GgmlType type, int ir, const void* const* ptrs, const void* const* ptrs_up, const Item* items, const int* n_items, const int* sorted, int max_n,
	const block_q8_1* in, bool per_token, int k, int n, const half* topw, int top_k, bool weighted, float* out, float* out_up, cudaStream_t stream ) {
	switch ( type ) {
#define CASE( T ) case GgmlType::T: launch_proj<GgmlType::T>( ir, ptrs, ptrs_up, items, n_items, sorted, max_n, in, per_token, k, n, topw, top_k, weighted, out, out_up, stream ); break;
		CASE( Q4_0 ) CASE( Q4_1 ) CASE( Q5_0 ) CASE( Q5_1 ) CASE( Q8_0 ) CASE( Q2_0 ) CASE( Q2_K ) CASE( Q3_K ) CASE( Q4_K ) CASE( Q5_K ) CASE( Q6_K )
		CASE( IQ2_XXS ) CASE( IQ2_XS ) CASE( IQ2_S ) CASE( IQ3_XXS ) CASE( IQ3_S ) CASE( IQ1_S ) CASE( IQ1_M ) CASE( IQ4_NL ) CASE( IQ4_XS )
#undef CASE
		default: ENGINE_ASSERT( false, "MoEGguf: unsupported expert block type" );
	}
}

// The scratch of one pass: the rows' q8_1, gate and up outputs, the activation's q8_1, the work items and the routes in expert order
struct Layout {
	size_t xq, g, u, aq, items, n_items, sorted, total;
};
Layout layout( int top_k, int hidden, int inter ) {
	const size_t R = MoEGguf::kMoERowsPerPass, routes = R * top_k;
	auto up = []( size_t b ) { return ( b + 255 ) / 256 * 256; };
	Layout l{};
	l.xq = 0;
	l.g = l.xq + up( R * ( hidden / QK8_1 ) * sizeof( block_q8_1 ) );
	l.u = l.g + up( routes * inter * sizeof( float ) );
	l.aq = l.u + up( routes * inter * sizeof( float ) );
	l.items = l.aq + up( routes * ( inter / QK8_1 ) * sizeof( block_q8_1 ) );
	l.n_items = l.items + up( routes * sizeof( Item ) );
	l.sorted = l.n_items + 256;
	l.total = l.sorted + up( routes * sizeof( int ) );
	return l;
}


// ---- tile path (prefill) ----
constexpr int kTileRowsPerPass = 1024;   // rows of a pass: about 16 routes of each local expert at top 6 over 384 experts on 4 ranks
constexpr int kTileRoutesMax = 32;      // routes of an item: 16, or 32 when a pass gives the local experts more than about 12 routes each
constexpr int kTileN = 32;              // output rows of a block: 26 KB of shared memory
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

// Eight Q2_K values dall * s * q - dmin * m (scale byte sc: s = sc & 0xf, m = sc >> 4; q the 2-bit fields at `shift` of the bytes of w0, w1)
// the same way out of the four values a q can give: the products are exact in FP32, so each value is rounded once, as the dequantizer's
__device__ __forceinline__ uint4 q2k_eight_h( uint32_t w0, uint32_t w1, int shift, float dall, float dmin, uint32_t sc ) {
	const float sd = dall * ( sc & 0xf ), md = dmin * ( sc >> 4 );
	const half v0 = __float2half_rn( sd * 0.0f - md ), v1 = __float2half_rn( sd * 1.0f - md ), v2 = __float2half_rn( sd * 2.0f - md ), v3 = __float2half_rn( sd * 3.0f - md );
	const uint32_t t0 = static_cast<uint32_t>( __half_as_ushort( v0 ) ) | static_cast<uint32_t>( __half_as_ushort( v1 ) ) << 16;
	const uint32_t t1 = static_cast<uint32_t>( __half_as_ushort( v2 ) ) | static_cast<uint32_t>( __half_as_ushort( v3 ) ) << 16;
	const uint32_t s0 = ( ( w0 >> shift ) & 0x03030303 ) * 0x22 + 0x10101010, s1 = ( ( w1 >> shift ) & 0x03030303 ) * 0x22 + 0x10101010;   // selectors of four pairs
	uint4 out;
	out.x = __byte_perm( t0, t1, s0 );
	out.y = __byte_perm( t0, t1, s0 >> 16 );
	out.z = __byte_perm( t0, t1, s1 );
	out.w = __byte_perm( t0, t1, s1 >> 16 );
	return out;
}

// Eight FP32 values as FP16, each rounded once
__device__ __forceinline__ uint4 eight_h( const float* v ) {
	uint32_t o[4];
	#pragma unroll
	for ( int i = 0; i < 4; ++i ) {
		const __half2 h = __floats2half2_rn( v[2 * i], v[2 * i + 1] );
		o[i] = *reinterpret_cast<const uint32_t*>( &h );
	}
	return make_uint4( o[0], o[1], o[2], o[3] );
}

// The FP16 values of two IQ3 grid words times d, the signs of `signs` (bit j: value j) set by flipping the halves' sign bits
__device__ __forceinline__ uint4 iq3_eight_h( uint32_t g1, uint32_t g2, float d, uint32_t signs ) {
	float v[8];
	#pragma unroll
	for ( int j = 0; j < 4; ++j ) {
		v[j] = d * static_cast<float>( ( g1 >> 8 * j ) & 0xff );
		v[j + 4] = d * static_cast<float>( ( g2 >> 8 * j ) & 0xff );
	}
	uint4 out = eight_h( v );
	auto flip = [&]( int p ) { const uint32_t b = signs >> 2 * p; return ( ( b & 1 ) | ( b & 2 ) << 15 ) << 15; };
	out.x ^= flip( 0 );
	out.y ^= flip( 1 );
	out.z ^= flip( 2 );
	out.w ^= flip( 3 );
	return out;
}

// The tile types: every 256-wide block type, whose eight values of a chunk decode into registers (chunk_h)
template <GgmlType T> constexpr bool chunk_type() {
	return T == GgmlType::IQ1_S || T == GgmlType::IQ1_M || T == GgmlType::IQ2_XXS || T == GgmlType::IQ2_XS || T == GgmlType::IQ2_S || T == GgmlType::IQ3_XXS ||
		T == GgmlType::IQ3_S || T == GgmlType::IQ4_XS || T == GgmlType::Q2_K || T == GgmlType::Q3_K || T == GgmlType::Q4_K || T == GgmlType::Q5_K || T == GgmlType::Q6_K;
}

// The FP16 values 8 c .. 8 c + 7 of block blk (for the IQ types sub-block c / 4, eighth c % 4), bit for bit llama.cpp's dequantizers' values:
// every product the dequantizers form is exact in FP32 (FP16 scales times integers of at most 13 bits together), so each value is rounded
// once whatever the order of the operations
template <GgmlType T>
__device__ __forceinline__ uint4 chunk_h( const void* blk, int c ) {
	const int sb = c >> 2, il = c & 3;
	if constexpr ( T == GgmlType::IQ1_S ) {
		const block_iq1_s* x = static_cast<const block_iq1_s*>( blk );
		const uint16_t qh = x->qh[sb];
		const float delta = qh & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
		const float d = static_cast<float>( x->d ) * ( 2 * ( ( qh >> 12 ) & 7 ) + 1 );
		return iq1_eight_h( iq1s_grid_gpu[x->qs[4 * sb + il] | ( ( ( qh >> 3 * il ) & 7 ) << 8 )], d, delta );
	}
	else if constexpr ( T == GgmlType::IQ1_M ) {
		const block_iq1_m* x = static_cast<const block_iq1_m*>( blk );
		const uint16_t* sc = reinterpret_cast<const uint16_t*>( x->scales );
		iq1m_scale_t scale;
		scale.u16 = ( sc[0] >> 12 ) | ( ( sc[1] >> 8 ) & 0x00f0 ) | ( ( sc[2] >> 4 ) & 0x0f00 ) | ( sc[3] & 0xf000 );
		const int ib16 = 2 * sb + il / 2;
		const float d = static_cast<float>( scale.f16 ) * ( 2 * ( ( sc[ib16 / 4] >> 3 * ( ib16 % 4 ) ) & 0x7 ) + 1 );
		const uint8_t qh = x->qh[2 * sb + il / 2];
		const float delta = qh & ( 0x08 << 4 * ( il % 2 ) ) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
		return iq1_eight_h( iq1s_grid_gpu[x->qs[4 * sb + il] | ( ( ( qh >> 4 * ( il % 2 ) ) & 7 ) << 8 )], d, delta );
	}
	else if constexpr ( T == GgmlType::IQ2_XXS ) {
		const block_iq2_xxs* x = static_cast<const block_iq2_xxs*>( blk );
		const uint16_t* q2 = x->qs + 4 * sb;
		const uint32_t aux32 = q2[2] | ( q2[3] << 16 );
		const float d = static_cast<float>( x->d ) * ( 0.5f + ( aux32 >> 28 ) ) * 0.25f;
		return iq2_eight_h( iq2xxs_grid[reinterpret_cast<const uint8_t*>( q2 )[il]], d, ksigns_iq2xs[( aux32 >> 7 * il ) & 127] );
	}
	else if constexpr ( T == GgmlType::IQ2_XS ) {
		const block_iq2_xs* x = static_cast<const block_iq2_xs*>( blk );
		const uint16_t q2 = x->qs[4 * sb + il];
		const float d = static_cast<float>( x->d ) * ( 0.5f + ( ( x->scales[sb] >> 4 * ( il / 2 ) ) & 0xf ) ) * 0.25f;
		return iq2_eight_h( iq2xs_grid[q2 & 511], d, ksigns_iq2xs[q2 >> 9] );
	}
	else if constexpr ( T == GgmlType::Q2_K ) {
		// values 8 c .. 8 c + 7 = 128 n + 32 j + l0 .. + 7: the 2-bit fields 2 j of bytes l0 .. l0 + 7 of the qs half n, one scale byte
		const block_q2_K* x = static_cast<const block_q2_K*>( blk );
		const int n = c >> 4, j = ( c >> 2 ) & 3, l0 = 8 * ( c & 3 );
		const uint32_t* qw = reinterpret_cast<const uint32_t*>( x->qs + 32 * n + l0 );
		return q2k_eight_h( qw[0], qw[1], 2 * j, __low2float( x->dm ), __high2float( x->dm ), x->scales[8 * n + 2 * j + ( c & 3 ) / 2] );
	}
	else if constexpr ( T == GgmlType::IQ2_S ) {
		const block_iq2_s* x = static_cast<const block_iq2_s*>( blk );
		const float d = static_cast<float>( x->d ) * ( 0.5f + ( ( x->scales[sb] >> 4 * ( il / 2 ) ) & 0xf ) ) * 0.25f;
		return iq2_eight_h( iq2s_grid[x->qs[4 * sb + il] | ( ( x->qh[sb] << ( 8 - 2 * il ) ) & 0x300 )], d, x->qs[QK_K / 8 + 4 * sb + il] );
	}
	else if constexpr ( T == GgmlType::IQ3_XXS ) {
		const block_iq3_xxs* x = static_cast<const block_iq3_xxs*>( blk );
		const uint8_t* q3 = x->qs + 8 * sb;
		const uint16_t* gas = reinterpret_cast<const uint16_t*>( x->qs + QK_K / 4 ) + 2 * sb;
		const uint32_t aux32 = gas[0] | ( gas[1] << 16 );
		const float d = static_cast<float>( x->d ) * ( 0.5f + ( aux32 >> 28 ) ) * 0.5f;
		return iq3_eight_h( iq3xxs_grid[q3[2 * il]], iq3xxs_grid[q3[2 * il + 1]], d, ksigns_iq2xs[( aux32 >> 7 * il ) & 127] );
	}
	else if constexpr ( T == GgmlType::IQ3_S ) {
		const block_iq3_s* x = static_cast<const block_iq3_s*>( blk );
		const uint8_t* qs = x->qs + 8 * sb;
		const float d = static_cast<float>( x->d ) * ( 1 + 2 * ( ( x->scales[sb / 2] >> 4 * ( sb % 2 ) ) & 0xf ) );
		return iq3_eight_h( iq3s_grid[qs[2 * il] | ( ( x->qh[sb] << ( 8 - 2 * il ) ) & 256 )], iq3s_grid[qs[2 * il + 1] | ( ( x->qh[sb] << ( 7 - 2 * il ) ) & 256 )], d,
			x->signs[4 * sb + il] );
	}
	else if constexpr ( T == GgmlType::IQ4_XS ) {
		// sub-block sb: values 0..15 the low nibbles of its 16 bytes, 16..31 the high nibbles
		const block_iq4_xs* x = static_cast<const block_iq4_xs*>( blk );
		const float d = static_cast<float>( x->d ) * ( ( ( ( x->scales_l[sb / 2] >> 4 * ( sb % 2 ) ) & 0xf ) | ( ( ( x->scales_h >> 2 * sb ) & 3 ) << 4 ) ) - 32 );
		const uint32_t* q = reinterpret_cast<const uint32_t*>( x->qs + 16 * sb + 8 * ( il & 1 ) );
		const int shift = 4 * ( il >> 1 );
		float v[8];
		#pragma unroll
		for ( int k = 0; k < 8; ++k ) v[k] = d * kvalues_iq4nl[( q[k >> 2] >> ( 8 * ( k & 3 ) + shift ) ) & 0xf];
		return eight_h( v );
	}
	else if constexpr ( T == GgmlType::Q2_K ) {
		// values 8 c .. 8 c + 7 = 128 n + 32 j + l0 .. + 7: the 2-bit fields 2 j of bytes l0 .. l0 + 7 of the qs half n, one scale byte
		const block_q2_K* x = static_cast<const block_q2_K*>( blk );
		const int n = c >> 4, j = ( c >> 2 ) & 3, l0 = 8 * ( c & 3 );
		const uint32_t* qw = reinterpret_cast<const uint32_t*>( x->qs + 32 * n + l0 );
		return q2k_eight_h( qw[0], qw[1], 2 * j, __low2float( x->dm ), __high2float( x->dm ), x->scales[8 * n + 2 * j + ( c & 3 ) / 2] );
	}
	else if constexpr ( T == GgmlType::Q3_K ) {
		// the same positions as Q2_K; a value's high bit is bit 4 n + j of its hmask byte (clear: minus 4), its 6-bit scale split over the scale bytes
		const block_q3_K* x = static_cast<const block_q3_K*>( blk );
		const int n = c >> 4, j = ( c >> 2 ) & 3, l0 = 8 * ( c & 3 ), is = 8 * n + 2 * j + ( c & 3 ) / 2;
		const uint8_t* sc = x->scales;
		const int us = is < 4 ? ( sc[is] & 0xF ) | ( ( ( sc[is + 8] >> 0 ) & 3 ) << 4 ) : is < 8 ? ( sc[is] & 0xF ) | ( ( ( sc[is + 4] >> 2 ) & 3 ) << 4 ) :
			is < 12 ? ( sc[is - 8] >> 4 ) | ( ( ( sc[is] >> 4 ) & 3 ) << 4 ) : ( sc[is - 8] >> 4 ) | ( ( ( sc[is - 4] >> 6 ) & 3 ) << 4 );
		const float dl = static_cast<float>( x->d ) * ( us - 32 );
		const uint32_t* q = reinterpret_cast<const uint32_t*>( x->qs + 32 * n + l0 );
		const uint32_t* hm = reinterpret_cast<const uint32_t*>( x->hmask + l0 );
		float v[8];
		#pragma unroll
		for ( int k = 0; k < 8; ++k ) {
			const int b = 8 * ( k & 3 );
			v[k] = dl * ( static_cast<int>( ( q[k >> 2] >> ( b + 2 * j ) ) & 3 ) - ( ( hm[k >> 2] >> ( b + 4 * n + j ) ) & 1 ? 0 : 4 ) );
		}
		return eight_h( v );
	}
	else if constexpr ( T == GgmlType::Q4_K || T == GgmlType::Q5_K ) {
		// values 8 c .. = 64 il + 32 h + l0 ..: nibble h of bytes l0 .. l0 + 7 of the qs quarter il, scale and min 2 il + h; Q5_K's fifth bit is
		// bit 2 il + h of qh byte l0 ..
		const typename TileBlk<T>::type* x = static_cast<const typename TileBlk<T>::type*>( blk );
		const int i4 = c >> 3, h = ( c >> 2 ) & 1, l0 = 8 * ( c & 3 );
		uint8_t sc, m;
		get_scale_min_k4( 2 * i4 + h, x->scales, sc, m );
		const float d1 = __low2float( x->dm ) * sc, m1 = __high2float( x->dm ) * m;
		const uint32_t* q = reinterpret_cast<const uint32_t*>( x->qs + 32 * i4 + l0 );
		float v[8];
		#pragma unroll
		for ( int k = 0; k < 8; ++k ) {
			int qv = ( q[k >> 2] >> ( 8 * ( k & 3 ) + 4 * h ) ) & 0xF;
			if constexpr ( T == GgmlType::Q5_K ) qv += ( reinterpret_cast<const uint32_t*>( x->qh + l0 )[k >> 2] >> ( 8 * ( k & 3 ) + 2 * i4 + h ) ) & 1 ? 16 : 0;
			v[k] = d1 * qv - m1;
		}
		return eight_h( v );
	}
	else {
		// values 8 c .. = 128 ip + 32 g + l0 ..: nibble g / 2 of ql bytes 64 ip + 32 ( g % 2 ) + l0 .., bits 2 g of qh bytes 32 ip + l0 .., minus 32
		static_assert( T == GgmlType::Q6_K, "chunk_h: a tile type" );
		const block_q6_K* x = static_cast<const block_q6_K*>( blk );
		const int ip = c >> 4, g = ( c >> 2 ) & 3, l0 = 8 * ( c & 3 );
		const uint32_t* ql = reinterpret_cast<const uint32_t*>( x->ql + 64 * ip + 32 * ( g & 1 ) + l0 );
		const uint32_t* qh = reinterpret_cast<const uint32_t*>( x->qh + 32 * ip + l0 );
		const float d = x->d;
		const int sc = x->scales[8 * ip + l0 / 16 + 2 * g];
		float v[8];
		#pragma unroll
		for ( int k = 0; k < 8; ++k ) {
			const int b = 8 * ( k & 3 );
			v[k] = d * sc * ( static_cast<int>( ( ( ql[k >> 2] >> ( b + 4 * ( g >> 1 ) ) ) & 0xF ) | ( ( ( qh[k >> 2] >> ( b + 2 * g ) ) & 3 ) << 4 ) ) - 32 );
		}
		return eight_h( v );
	}
}

// The FP16 values of block ib of a tile type, decoded by one warp: lane l's eight values at 8 l with one 16-byte store, so a warp's stores
// are consecutive
template <GgmlType T>
__device__ __forceinline__ void tile_dequant( const void* w, int64_t ib, half* y, int lane ) {
	*reinterpret_cast<uint4*>( y + 8 * lane ) = chunk_h<T>( static_cast<const typename TileBlk<T>::type*>( w ) + ib, lane );
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
// resident blocks an SM: three, two for the K-quants above Q2_K, the IQ3 types and IQ4_XS (their dequantizers would spill registers at three)
template <GgmlType T> constexpr int tile_min_blocks() {
	return T == GgmlType::Q3_K || T == GgmlType::Q4_K || T == GgmlType::Q5_K || T == GgmlType::Q6_K || T == GgmlType::IQ3_XXS || T == GgmlType::IQ3_S || T == GgmlType::IQ4_XS ? 2 : 3;
}

template <GgmlType T, bool GU, int RT>
__global__ __launch_bounds__( kTileThreads, tile_min_blocks<T>() )
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

// ---- SM_70: the decoded values straight into the m8n8k4 B operands (no decoded tile in shared memory) ----
// Built for SM_70 only (HMMA.884; later architectures run moe_tile_kernel's WMMA): elsewhere the kernel body is empty, and tile_proj launches
// it only on a compute capability 7.0 device.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
#define GGUF_TILE_884 1
#else
#define GGUF_TILE_884 0
#endif

// bytes of a staged block: an odd number of 32-bit words, so the eight rows a quarter decodes read eight banks
template <GgmlType T> constexpr int tile884_raw_ld() { return ( ( static_cast<int>( sizeof( typename TileBlk<T>::type ) ) + 3 ) / 4 | 1 ) * 4; }
// the kernel's shared memory: the routes' inputs and two steps of raw blocks, or the input pieces' FP32 sums at the end
template <GgmlType T, int RT, int R> constexpr size_t tile884_smem() {
	return std::max( static_cast<size_t>( RT ) * kTileLd * sizeof( half ) + 2 * kTileN * tile884_raw_ld<T>(), static_cast<size_t>( 2 * R ) * RT * ( kTileN + 4 ) * sizeof( float ) );
}

// moe_tile_kernel for the chunk types (IQ1, IQ2, Q2_K) on SM_70. In m8n8k4 a warp is four independent quarters; warp w takes R groups of 8 of the block's
// 32 rows and 16 / R of each step's 32 chunks of 8 inputs, its quarter q the chunks q, q + 4, ... of those. A lane decodes the eight values of
// its chunk for each of its rows (B column (l & 3) + 4 (l >= 16) of a group): they are its B operands of two k4 steps, multiplied with each
// 8 routes' A operands (route (l & 3) + 4 (l >= 16) of the 8, the chunk's 16 bytes from shared memory). The quarters' sums meet over shuffles
// (lanes l, l ^ 4, l ^ 8, l ^ 12 hold the same output slots), the warps' input pieces in shared memory, summed in order.
template <GgmlType T, bool GU, int RT, int R>
__global__ __launch_bounds__( kTileThreads, 2 )
void moe_tile884_kernel( const void* const* __restrict__ ptrs, const void* const* __restrict__ ptrs_up, const Item* __restrict__ items, const int* __restrict__ n_items,
	const int* __restrict__ sorted, const half* __restrict__ in, bool per_token, int k, int n, const half* __restrict__ topw, int top_k, float* __restrict__ part32,
	half* __restrict__ out16, float act_limit ) {
#if GGUF_TILE_884
	using Blk = typename TileBlk<T>::type;
	constexpr int kRawLd = tile884_raw_ld<T>();
	constexpr int RU = sizeof( Blk ) / 2, UNITS = kTileN * RU, PER = ( UNITS + kTileThreads - 1 ) / kTileThreads;
	constexpr int SETS = 4 / R, PIECES = 8 / SETS, CQ = 4 / R;   // row sets; input pieces (warps per row set); chunks of a quarter a step
	constexpr int G = RT / 8;                                     // groups of 8 routes
	if ( static_cast<int>( blockIdx.y ) >= *n_items ) return;
	extern __shared__ __align__( 32 ) unsigned char tile_smem[];
	half* sX = reinterpret_cast<half*>( tile_smem );                                  // [RT][kTileLd]
	unsigned char* sR = reinterpret_cast<unsigned char*>( sX + RT * kTileLd );       // [2][kTileN][kRawLd]
	__shared__ int s_route[RT];
	const Item it = items[blockIdx.y];
	constexpr int kRows = GU ? kTileN / 2 : kTileN;
	const int row0 = blockIdx.x * kRows, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	if ( threadIdx.x < RT ) s_route[threadIdx.x] = threadIdx.x < it.count ? sorted[it.begin + threadIdx.x] : -1;
	const uint16_t* w = static_cast<const uint16_t*>( ptrs[it.expert] );
	const uint16_t* wu = GU ? static_cast<const uint16_t*>( ptrs_up[it.expert] ) : w;
	const int bpr = k / 256;
	int src[PER], dst[PER];
	unsigned upper = 0;
	#pragma unroll
	for ( int p = 0; p < PER; ++p ) {
		const int u = threadIdx.x + p * kTileThreads, row = u / RU, j = u % RU, gr = row0 + ( GU ? row % kRows : row );
		src[p] = u < UNITS && gr < n ? gr * bpr * RU + j : -1;
		dst[p] = u < UNITS ? row * ( kRawLd / 2 ) + j : -1;
		if ( GU && row >= kRows ) upper |= 1u << p;
	}
	uint16_t rr[PER];
	auto fetch = [&]( int kb ) {
		#pragma unroll
		for ( int p = 0; p < PER; ++p ) rr[p] = src[p] >= 0 ? ( ( upper >> p ) & 1 ? wu : w )[src[p] + kb * RU] : 0;
	};
	auto stash = [&]( int buf ) {
		uint16_t* d = reinterpret_cast<uint16_t*>( sR + buf * kTileN * kRawLd );
		#pragma unroll
		for ( int p = 0; p < PER; ++p ) if ( dst[p] >= 0 ) d[dst[p]] = rr[p];
	};
	constexpr int XPER = RT * 32 / kTileThreads;
	const half* xsrc[XPER];
	#pragma unroll
	for ( int i = 0; i < XPER; ++i ) {
		const int v = threadIdx.x + i * kTileThreads, j = v >> 5, c = v & 31, r = j < it.count ? sorted[it.begin + j] : -1;
		xsrc[i] = r >= 0 ? in + static_cast<size_t>( per_token ? r / top_k : r ) * k + c * 8 : nullptr;
	}
	fetch( 0 );
	stash( 0 );
	const int rs = warp % SETS, piece = warp / SETS;                             // the warp's rows (groups rs R ..) and piece of the inputs
	const int q = ( lane >> 2 ) & 3, rl = ( lane & 3 ) + 4 * ( lane >> 4 );     // the lane's quarter; its B column and A row within 8
	float acc[R][G][8];
	#pragma unroll
	for ( int h = 0; h < R; ++h )
		#pragma unroll
		for ( int g = 0; g < G; ++g )
			#pragma unroll
			for ( int r = 0; r < 8; ++r ) acc[h][g][r] = 0.0f;
	for ( int kb = 0; kb < bpr; ++kb ) {
		if ( kb + 1 < bpr ) fetch( kb + 1 );
		#pragma unroll
		for ( int i = 0; i < XPER; ++i ) {
			const int v = threadIdx.x + i * kTileThreads;
			const uint4 val = xsrc[i] != nullptr ? *reinterpret_cast<const uint4*>( xsrc[i] + kb * 256 ) : make_uint4( 0, 0, 0, 0 );
			*reinterpret_cast<uint4*>( sX + ( v >> 5 ) * kTileLd + ( v & 31 ) * 8 ) = val;
		}
		__syncthreads();
		const unsigned char* raw = sR + ( ( kb & 1 ) * kTileN + rs * R * 8 + rl ) * kRawLd;
		#pragma unroll
		for ( int j = 0; j < CQ; ++j ) {
			const int c = piece * 4 * CQ + q + 4 * j;   // with rows kTileLd apart (8 banks), a phase's eight 16-byte A reads take all 32 banks
			uint4 b[R];
			#pragma unroll
			for ( int h = 0; h < R; ++h ) b[h] = chunk_h<T>( raw + h * 8 * kRawLd, c );
			#pragma unroll
			for ( int g = 0; g < G; ++g ) {
				if ( g * 8 >= it.count ) break;   // uniform over the block
				const uint4 a = *reinterpret_cast<const uint4*>( sX + ( g * 8 + rl ) * kTileLd + c * 8 );
				#pragma unroll
				for ( int h = 0; h < R; ++h ) {
					volta_mma_m8n8k4( a.x, a.y, b[h].x, b[h].y, acc[h][g] );
					volta_mma_m8n8k4( a.z, a.w, b[h].z, b[h].w, acc[h][g] );
				}
			}
		}
		if ( kb + 1 < bpr ) stash( ( kb + 1 ) & 1 );
		__syncthreads();
	}
	float* sC = reinterpret_cast<float*>( tile_smem );   // [PIECES][RT][kTileN + 4]
	constexpr int ldc = kTileN + 4;
	#pragma unroll
	for ( int h = 0; h < R; ++h )
		#pragma unroll
		for ( int g = 0; g < G; ++g )
			#pragma unroll
			for ( int r = 0; r < 8; ++r ) {
				float v = acc[h][g][r];
				v += __shfl_xor_sync( 0xffffffffu, v, 4 );
				v += __shfl_xor_sync( 0xffffffffu, v, 8 );
				int m, nn;
				volta_d_slot( lane, r, &m, &nn );
				if ( q == 0 ) sC[( piece * RT + g * 8 + m ) * ldc + ( rs * R + h ) * 8 + nn] = v;
			}
	__syncthreads();
	for ( int v = threadIdx.x; v < RT * kRows; v += kTileThreads ) {
		const int j = v / kRows, c = v % kRows, r = s_route[j], gr = row0 + c;
		if ( r < 0 || gr >= n ) continue;
		auto sum = [&]( int col ) {
			float t = 0.0f;
			#pragma unroll
			for ( int p = 0; p < PIECES; ++p ) t += sC[( p * RT + j ) * ldc + col];
			return t;
		};
		const float val = sum( c );
		if constexpr ( GU ) {
			float gv = __half2float( __float2half_rn( val ) ), uv = __half2float( __float2half_rn( sum( c + kRows ) ) );
			if ( act_limit > 0.0f ) {
				gv = fminf( gv, act_limit );
				uv = fmaxf( -act_limit, fminf( uv, act_limit ) );
			}
			out16[static_cast<size_t>( r ) * n + gr] = __float2half_rn( gv / ( 1.0f + __expf( -gv ) ) * uv );
		}
		else if ( part32 != nullptr ) part32[static_cast<size_t>( r ) * n + gr] = val * __half2float( topw[r] );
		else out16[static_cast<size_t>( r ) * n + gr] = __float2half_rn( val );
	}
#endif   // elsewhere empty: tile_proj takes this kernel on compute capability 7.0 only
}

template <GgmlType T, bool GU, int RT>
void launch_tile884( const void* const* ptrs, const void* const* ptrs_up, const Item* items, const int* n_items, const int* sorted, int max_n, const half* in,
	bool per_token, int k, int n, const half* topw, int top_k, float* part32, half* out16, float act_limit, cudaStream_t stream ) {
	if constexpr ( chunk_type<T>() ) {
		constexpr int R = RT > 16 ? 1 : 2;   // row groups of a warp: each A read feeds R rows' operands, 8 R G accumulators a lane
		static bool attr[64] = {};
		int dev = 0;
		cudaGetDevice( &dev );
		if ( !attr[dev] ) {
			CUDA_CHECK( cudaFuncSetAttribute( moe_tile884_kernel<T, GU, RT, R>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>( tile884_smem<T, RT, R>() ) ) );
			attr[dev] = true;
		}
		constexpr int rows = GU ? kTileN / 2 : kTileN;
		moe_tile884_kernel<T, GU, RT, R><<<dim3( ( n + rows - 1 ) / rows, max_n ), kTileThreads, tile884_smem<T, RT, R>(), stream>>>( ptrs, ptrs_up, items, n_items,
			sorted, in, per_token, k, n, topw, top_k, part32, out16, act_limit );
	}
}

// The device runs the SM_70 kernel: compute capability 7.0 exactly
bool tile884_device() {
	static int cc[64] = {};
	int dev = 0;
	cudaGetDevice( &dev );
	if ( cc[dev] == 0 ) {
		int major = 0, minor = 0;
		CUDA_CHECK( cudaDeviceGetAttribute( &major, cudaDevAttrComputeCapabilityMajor, dev ) );
		CUDA_CHECK( cudaDeviceGetAttribute( &minor, cudaDevAttrComputeCapabilityMinor, dev ) );
		cc[dev] = 10 * major + minor;
	}
	return cc[dev] == 70;
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
	static const bool m884_on = engine::runtime::options().gguf_moe_tile884;
	const bool m884 = m884_on && tile884_device();
	switch ( type ) {
#define LAUNCH( T, RT ) \
		if ( m884 && chunk_type<GgmlType::T>() ) { \
			if ( ptrs_up != nullptr ) launch_tile884<GgmlType::T, true, RT>( ptrs, ptrs_up, items, n_items, sorted, max_n, in, per_token, k, n, topw, top_k, part32, out16, act_limit, stream ); \
			else launch_tile884<GgmlType::T, false, RT>( ptrs, nullptr, items, n_items, sorted, max_n, in, per_token, k, n, topw, top_k, part32, out16, act_limit, stream ); \
		} \
		else if ( ptrs_up != nullptr ) launch_tile<GgmlType::T, true, RT>( ptrs, ptrs_up, items, n_items, sorted, max_n, in, per_token, k, n, topw, top_k, part32, out16, act_limit, stream ); \
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
			// 32-route items once the local experts average more than about 12 routes in the pass (fewer decodes, at a lower occupancy): a pass's
			// routes over every rank's experts (experts sharded over the ranks: all of them local)
			const int global = t.num_global > 0 ? t.num_global : 4 * t.num_local;
			const int rt = routes > 12 * global ? kTileRoutesMax : 16, mi = std::min( routes, t.num_local + routes / rt );
			const int* ti = topk_idx + static_cast<size_t>( r0 ) * top_k;
			const half* tw = topk_w + static_cast<size_t>( r0 ) * top_k;
			const half* xr = x + static_cast<size_t>( r0 ) * t.hidden;
			moe_dispatch_kernel<<<1, 256, dispatch_smem( t.num_local ), stream>>>( ti, routes, t.first_expert, t.num_local, items, n_items, sorted, heat, rt );
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
	auto* items = reinterpret_cast<Item*>( ws + l.items );
	auto* n_items = reinterpret_cast<int*>( ws + l.n_items );
	auto* sorted = reinterpret_cast<int*>( ws + l.sorted );
	for ( int r0 = 0; r0 < m; r0 += kMoERowsPerPass ) {
		const int rn = std::min( kMoERowsPerPass, m - r0 ), routes = rn * top_k;
		// a token routes to an expert once, so a one-row pass's items hold one route each
		const int ir = rn <= 1 ? 1 : kItemRoutes, mi = std::min( routes, t.num_local + routes / ir );
		const int* ti = topk_idx + static_cast<size_t>( r0 ) * top_k;
		const half* tw = topk_w + static_cast<size_t>( r0 ) * top_k;
		const int qx = ( t.hidden + 255 ) / 256;   // the dispatch block, then the blocks that quantize the pass's rows
		moe_dispatch_kernel<<<1 + qx * rn, 256, dispatch_smem( t.num_local ), stream>>>( ti, routes, t.first_expert, t.num_local, items, n_items, sorted, heat, ir,
			x + static_cast<size_t>( r0 ) * t.hidden, xq, t.hidden, qx );
		if ( t.type[0] == t.type[1] ) proj( t.type[0], ir, gate, up, items, n_items, sorted, mi, xq, true, t.hidden, t.inter, tw, top_k, false, g, u, stream );
		else {
			proj( t.type[0], ir, gate, nullptr, items, n_items, sorted, mi, xq, true, t.hidden, t.inter, tw, top_k, false, g, nullptr, stream );
			proj( t.type[1], ir, up, nullptr, items, n_items, sorted, mi, xq, true, t.hidden, t.inter, tw, top_k, false, u, nullptr, stream );
		}
		moe_act_q8_kernel<<<dim3( ( t.inter + 127 ) / 128, routes ), 128, 0, stream>>>( g, u, aq, t.inter, t.act_limit );
		if ( rn == 1 && top_k <= 32 ) {   // one token: its routes summed by the down projection's warps
			down_combine( t.type[2], down, items, n_items, sorted, aq, t.inter, t.hidden, tw, top_k, out + static_cast<size_t>( r0 ) * t.hidden, stream );
			continue;
		}
		down_tok( t.type[2], down, ti, t.first_expert, t.num_local, aq, t.inter, t.hidden, tw, top_k, rn, out + static_cast<size_t>( r0 ) * t.hidden, stream );
	}
}

} // namespace engine::modules
