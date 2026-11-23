#pragma once
// Route grouping shared by the expert kernels that work per expert over a pass's routes (GGUF, FP8): work items of one local expert and
// up to item_routes of its routes, the route ids in expert order, and the combine of the routes' FP32 partials per token
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>

namespace engine::modules {

namespace {

constexpr int kDispatchWarps = 8;   // the grouping's block of 256 threads
inline size_t dispatch_smem( int num_local ) { return static_cast<size_t>( 2 + kDispatchWarps ) * num_local * sizeof( int ); }

// A work item: one local expert and up to item_routes of its routes (route ids in `sorted`, from `begin`)
struct Item {
	int expert, begin, count, pad;
};

// Groups the pass's routes by local expert (counting sort, one block of 256 threads with dispatch_smem bytes): items[0 .. *n_items) and the
// route ids in expert order; counts the experts' routes into heat
__device__ __forceinline__ void moe_group_routes( const int* __restrict__ topk, int routes, int first, int num_local, Item* __restrict__ items, int* __restrict__ n_items,
	int* __restrict__ sorted, uint32_t* __restrict__ heat, int item_routes ) {
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

} // namespace

} // namespace engine::modules
