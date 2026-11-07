// GGUF routed experts on CUDA (MoEGguf in engine/modules/gguf.h). The routes of a pass are grouped by expert into work items of up to
// kItemRoutes routes; each projection is a GEMV of one output row per block and work item, the expert's weight row read once for all the
// item's routes through llama.cpp's vec_dot of the projection's block type (LinearGguf's multi-row GEMV structure). Routes of experts on
// other ranks are left out, and the combine skips them.
#include "engine/modules/gguf.h"
#include "engine/backend/runtime.h"
#include "gguf_common.cuh"
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
	int* __restrict__ sorted, uint32_t* __restrict__ heat ) {
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
			for ( int b = 0; b < count[e]; b += kItemRoutes ) items[n++] = Item{ e, at + b, min( kItemRoutes, count[e] - b ), 0 };
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

} // namespace

size_t MoEGguf::scratch_bytes( int top_k, int hidden, int inter ) { return layout( top_k, hidden, inter ).total; }

void MoEGguf::forward( const Table& t, const half* x, int m, const int* topk_idx, const half* topk_w, int top_k, half* out, void* scratch, uint32_t* heat,
	cudaStream_t stream ) {
	ENGINE_ASSERT( t.hidden % QK8_1 == 0 && t.inter % QK8_1 == 0, "MoEGguf: widths must be whole q8_1 blocks" );
	const Layout l = layout( top_k, t.hidden, t.inter );
	char* ws = static_cast<char*>( scratch );
	auto* xq = reinterpret_cast<block_q8_1*>( ws + l.xq );
	auto* g = reinterpret_cast<float*>( ws + l.g );
	auto* u = reinterpret_cast<float*>( ws + l.u );
	auto* aq = reinterpret_cast<block_q8_1*>( ws + l.aq );
	auto* part = reinterpret_cast<float*>( ws + l.part );
	auto* items = reinterpret_cast<Item*>( ws + l.items );
	auto* n_items = reinterpret_cast<int*>( ws + l.n_items );
	auto* sorted = reinterpret_cast<int*>( ws + l.sorted );
	const void* const* gate = t.ptrs, * const* up = t.ptrs + t.num_local, * const* down = t.ptrs + 2 * t.num_local;
	for ( int r0 = 0; r0 < m; r0 += kMoERowsPerPass ) {
		const int rn = std::min( kMoERowsPerPass, m - r0 ), routes = rn * top_k, mi = max_items( routes, t.num_local );
		const int* ti = topk_idx + static_cast<size_t>( r0 ) * top_k;
		const half* tw = topk_w + static_cast<size_t>( r0 ) * top_k;
		moe_dispatch_kernel<<<1, 256, 2 * t.num_local * sizeof( int ), stream>>>( ti, routes, t.first_expert, t.num_local, items, n_items, sorted, heat );
		quantize_q8_1_kernel<<<dim3( ( t.hidden + 255 ) / 256, rn ), 256, 0, stream>>>( x + static_cast<size_t>( r0 ) * t.hidden, xq, t.hidden, GgufColTiling{} );
		proj( t.type[0], gate, items, n_items, sorted, mi, xq, true, t.hidden, t.inter, tw, top_k, false, g, stream );
		proj( t.type[1], up, items, n_items, sorted, mi, xq, true, t.hidden, t.inter, tw, top_k, false, u, stream );
		moe_act_q8_kernel<<<dim3( ( t.inter + 127 ) / 128, routes ), 128, 0, stream>>>( g, u, aq, t.inter, t.act_limit );
		proj( t.type[2], down, items, n_items, sorted, mi, aq, false, t.inter, t.hidden, tw, top_k, true, part, stream );
		moe_combine_kernel<<<( rn * t.hidden + 255 ) / 256, 256, 0, stream>>>( part, ti, t.first_expert, t.num_local, out + static_cast<size_t>( r0 ) * t.hidden, rn, top_k, t.hidden );
	}
}

} // namespace engine::modules
