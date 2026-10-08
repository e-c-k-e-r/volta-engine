#include "engine/modules/moe.h"
#include "exl3_moe_kernel.cuh"
#include "exl3_moe_coop.cuh"
#include "engine/core/batch.h"
#include <cuda_fp16.h>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>

namespace engine::modules {

namespace {

// One thread per expert bin (plus the overflow bin): up to 511 experts per rank
constexpr int MOE_HIST_THREADS = 512;

__global__ void moe_dispatch_hist_kernel(
	const int* __restrict__ selected_experts,
	const int n,
	const int num_experts,
	const int routing_first,
	const int routing_last,
	int64_t* __restrict__ expert_count,
	int* __restrict__ cursors,
	int4* __restrict__ active_experts,
	int* __restrict__ sched,
	int num_groups,
	int chunk_rows,
	uint32_t* __restrict__ heat
) {
	__shared__ int hist[MOE_HIST_THREADS];
	const int threads = blockDim.x;
	for ( int i = threadIdx.x; i <= num_experts; i += threads ) hist[i] = 0;
	__syncthreads();

	for ( int i = threadIdx.x; i < n; i += threads ) {
		int e = selected_experts[i];
		int el = ( e >= routing_first && e < routing_last ) ? ( e - routing_first ) : num_experts;
		atomicAdd( &hist[el], 1 );
	}
	__syncthreads();

	if ( heat != nullptr && threadIdx.x < num_experts && hist[threadIdx.x] > 0 ) atomicAdd( &heat[threadIdx.x], static_cast<uint32_t>( hist[threadIdx.x] ) );
	__shared__ int scan[MOE_HIST_THREADS];
	scan[threadIdx.x] = ( threadIdx.x <= num_experts ) ? hist[threadIdx.x] : 0;
	__syncthreads();
	for ( int d = 1; d < MOE_HIST_THREADS; d <<= 1 ) {
		int v = ( threadIdx.x >= d ) ? scan[threadIdx.x - d] : 0;
		__syncthreads();
		scan[threadIdx.x] += v;
		__syncthreads();
	}

	if ( threadIdx.x <= num_experts ) {
		int my_start = scan[threadIdx.x] - hist[threadIdx.x];
		cursors[threadIdx.x] = my_start;
		expert_count[threadIdx.x] = static_cast<int64_t>( hist[threadIdx.x] );
	}

	if ( active_experts ) {
		// Work items are (expert, first row, rows <= chunk_rows): a hot expert's chunks spread over all expert groups
		// instead of serializing in one. Item 0 is the header (item count); items follow, ordered by expert then row.
		__syncthreads();
		const int nchunks = ( threadIdx.x < num_experts ) ? ( hist[threadIdx.x] + chunk_rows - 1 ) / chunk_rows : 0;
		scan[threadIdx.x] = nchunks;
		__syncthreads();
		for ( int d = 1; d < MOE_HIST_THREADS; d <<= 1 ) {
			int v = ( threadIdx.x >= d ) ? scan[threadIdx.x - d] : 0;
			__syncthreads();
			scan[threadIdx.x] += v;
			__syncthreads();
		}

		if ( nchunks > 0 ) {
			const int first_item = scan[threadIdx.x] - nchunks;
			const int my_start = cursors[threadIdx.x];
			for ( int c = 0; c < nchunks; ++c ) {
				active_experts[1 + first_item + c] = make_int4( threadIdx.x, my_start + c * chunk_rows, MIN( chunk_rows, hist[threadIdx.x] - c * chunk_rows ), 0 );
			}
		}

		if ( threadIdx.x == 0 ) {
			int total_items = ( num_experts > 0 ) ? scan[num_experts - 1] : 0;
			active_experts[0] = make_int4( total_items, 0, 0, 0 );
			// exl3_moe_kernel hands out tickets as num_groups + atomicAdd( &sched[0], 1 ) (tickets
			// 0..num_groups-1 are implicit) and resets sched[0] to 0 on exit: seed must be 0.
			// Seeding num_groups here skipped active experts num_groups..2*num_groups-1.
			if ( sched ) {
				sched[0] = 0;
				sched[1] = 0;
			}
		}
	}
}

__global__ void moe_dispatch_scatter_kernel(
	const int* __restrict__ selected_experts,
	const half* __restrict__ routing_weights,
	const int n,
	const int top_k,
	const int num_experts,
	const int routing_first,
	const int routing_last,
	int64_t* __restrict__ token_sorted,
	half* __restrict__ weight_sorted,
	int* __restrict__ cursors
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= n ) return;
	int e = selected_experts[i];
	if ( e >= routing_first && e < routing_last ) {
		// Position = the expert's start + the number of earlier routes to it, so every expert's rows are in route order
		// and a chunk always holds the same tokens (an atomic cursor made the order, and with it the chunk-to-group
		// assignment and the plane rounding, vary from run to run).
		int rank = 0;
		for ( int j = 0; j < i; ++j ) rank += ( selected_experts[j] == e ) ? 1 : 0;
		const int pos = cursors[e - routing_first] + rank;
		token_sorted[pos] = static_cast<int64_t>( i / top_k );
		weight_sorted[pos] = routing_weights[i];
	}
}

// Router scratch: split-K partial logits [row][k slice of 128][expert] handed from the dot-product kernel to the
// selection kernel, which sums the slices in slice order (deterministic).
#define ROUTER_MAX_KSLICES 32
__device__ float g_router_partials[pod::MAX_ROWS * ROUTER_MAX_KSLICES * 288];

__global__ __launch_bounds__( 32 )
void router_logits_kernel(
	const half* __restrict__ x,
	const half* __restrict__ router_weights,
	int hidden_size,
	int num_experts
) {
	// Block (expert tile, row, k slice): 128 input dims for 32 experts. Many small blocks spread the 288 x 4096 weight
	// read over the GPU; one warp per tile left it latency bound at ~90 us.
	__shared__ float s_x[128];
	const int m = blockIdx.y;
	const int ks = blockIdx.z;
	const int d0 = ks * 128;
	x += static_cast<size_t>( m ) * hidden_size + d0;
	for ( int d = threadIdx.x; d < 128; d += 32 ) s_x[d] = __half2float( x[d] );
	__syncwarp();

	const int expert = blockIdx.x * 32 + threadIdx.x;
	if ( expert >= num_experts ) return;
	const half* wcol = router_weights + static_cast<size_t>( d0 ) * num_experts + expert;
	half w[128];
	#pragma unroll
	for ( int j = 0; j < 128; ++j ) {
		w[j] = __ldg( wcol + static_cast<size_t>( j ) * num_experts );
	}
	float logit = 0.0f;
	#pragma unroll
	for ( int j = 0; j < 128; ++j ) {
		logit = fmaf( s_x[j], __half2float( w[j] ), logit );
	}
	g_router_partials[( static_cast<size_t>( m ) * ROUTER_MAX_KSLICES + ks ) * 288 + expert] = logit;
}

__global__ __launch_bounds__( 32 )
void router_select_kernel(
	const half* __restrict__ e_score_bias,
	int* __restrict__ topk_indices,
	half* __restrict__ topk_weights,
	int num_experts,
	int top_k,
	float routed_scaling_factor,
	int k_slices
) {
	__shared__ float s_raw[288];
	__shared__ float s_select[288];
	const int m = blockIdx.x;
	topk_indices += m * top_k;
	topk_weights += m * top_k;
	for ( int expert = threadIdx.x; expert < num_experts; expert += 32 ) {
		float logit = 0.0f;
		for ( int ks = 0; ks < k_slices; ++ks ) {
			logit += g_router_partials[( static_cast<size_t>( m ) * ROUTER_MAX_KSLICES + ks ) * 288 + expert];
		}
		const float raw_score = 1.0f / ( 1.0f + __expf( -logit ) );
		float select_score = raw_score;
		if ( e_score_bias != nullptr ) {
			select_score += __half2float( e_score_bias[expert] );
		}
		s_raw[expert] = raw_score;
		s_select[expert] = select_score;
	}
	__syncwarp();

	// Top-k by warp 0: each lane owns experts lane, lane + 32, ...; k rounds of a warp argmax (ties go to the
	// lowest expert index), removing the winner each round. Same descending order as the old selection sort.
	if ( threadIdx.x < 32 ) {
		const int lane = threadIdx.x;
		bool taken[9];
		#pragma unroll
		for ( int q = 0; q < 9; ++q ) taken[q] = false;
		int sel[16];
		for ( int i = 0; i < top_k; ++i ) {
			float best = -INFINITY;
			int best_idx = 0x7fffffff;
			#pragma unroll
			for ( int q = 0; q < 9; ++q ) {
				const int e = lane + q * 32;
				if ( e < num_experts && !taken[q] ) {
					const float v = s_select[e];
					if ( v > best || ( v == best && e < best_idx ) ) {
						best = v;
						best_idx = e;
					}
				}
			}
			#pragma unroll
			for ( int off = 16; off > 0; off >>= 1 ) {
				const float ov = __shfl_xor_sync( 0xffffffff, best, off );
				const int oi = __shfl_xor_sync( 0xffffffff, best_idx, off );
				if ( ov > best || ( ov == best && oi < best_idx ) ) {
					best = ov;
					best_idx = oi;
				}
			}
			sel[i] = best_idx;
			if ( ( best_idx & 31 ) == lane ) taken[best_idx >> 5] = true;
		}
		if ( lane == 0 ) {
			float sum_scores = 0.0f;
			for ( int i = 0; i < top_k; ++i ) {
				sum_scores += s_raw[sel[i]];
			}
			const float inv_sum = routed_scaling_factor / ( sum_scores + 1e-9f );
			for ( int i = 0; i < top_k; ++i ) {
				topk_indices[i] = sel[i];
				topk_weights[i] = __float2half_rn( s_raw[sel[i]] * inv_sum );
			}
		}
	}
}

// out = sum over group planes in group order (fixed order, so the result is independent of group timing)
__global__ void moe_sum_planes_kernel(
	const half* __restrict__ planes,
	half* __restrict__ out,
	int n,
	int num_groups
) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= n ) return;
	float acc = __half2float( planes[i] );
	for ( int g = 1; g < num_groups; ++g ) {
		acc += __half2float( planes[static_cast<size_t>( g ) * n + i] );
	}
	out[i] = __float2half_rn( acc );
}

// Same sum, eight halves per thread (128-bit loads and store); same FP32 order per element
__global__ void moe_sum_planes8_kernel(
	const half* __restrict__ planes,
	half* __restrict__ out,
	int n8,
	int num_groups
) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= n8 ) return;
	const uint4* p = reinterpret_cast<const uint4*>( planes );
	float acc[8];
	#pragma unroll
	for ( int g = 0; g < MOE_MAX_GROUPS; ++g ) {
		if ( g >= num_groups ) break;
		const uint4 raw = p[static_cast<size_t>( g ) * n8 + i];
		const half2* h2 = reinterpret_cast<const half2*>( &raw );
		#pragma unroll
		for ( int j = 0; j < 4; ++j ) {
			const float2 f = __half22float2( h2[j] );
			if ( g == 0 ) {
				acc[2 * j] = f.x;
				acc[2 * j + 1] = f.y;
			}
			else {
				acc[2 * j] += f.x;
				acc[2 * j + 1] += f.y;
			}
		}
	}
	uint4 res;
	half2* r2 = reinterpret_cast<half2*>( &res );
	#pragma unroll
	for ( int j = 0; j < 4; ++j ) r2[j] = __floats2half2_rn( acc[2 * j], acc[2 * j + 1] );
	reinterpret_cast<uint4*>( out )[i] = res;
}

__global__ void moe_combine_kernel(
	const half* __restrict__ expert_parts,
	const half* __restrict__ topk_weights,
	const half* __restrict__ shared_out,
	half* __restrict__ output,
	int hidden_size,
	int top_k
) {
	int d = blockIdx.x * blockDim.x + threadIdx.x;
	if ( d >= hidden_size ) return;

	float sum = ( shared_out != nullptr ) ? __half2float( shared_out[d] ) : 0.0f;
	#pragma unroll
	for ( int k = 0; k < top_k; ++k ) {
		float w = __half2float( topk_weights[k] );
		float part = __half2float( expert_parts[k * hidden_size + d] );
		sum = fmaf( part, w, sum );
	}
	output[d] = __float2half_rn( sum );
}

__global__ void moe_accumulate_kernel(
	half* __restrict__ accum_out,
	const half* __restrict__ expert_out,
	half weight,
	int hidden_size
) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ( tid < hidden_size ) {
		float cur = __half2float( accum_out[tid] );
		float val = __half2float( expert_out[tid] );
		float w = __half2float( weight );
		accum_out[tid] = __float2half_rn( fmaf( val, w, cur ) );
	}
}

} // namespace

void MoERouter::route_topk(
	const pod::TensorView& x,
	const pod::TensorView& router_weights,
	const pod::TensorView& e_score_bias,
	int* d_topk_indices,
	half* d_topk_weights,
	int hidden_size,
	int num_experts,
	int top_k,
	float routed_scaling_factor,
	cudaStream_t stream,
	int rows
) {
	ENGINE_ASSERT( rows >= 1 && rows <= pod::MAX_ROWS, "MoERouter::route_topk: rows out of range" );
	ENGINE_ASSERT( top_k <= 16 && num_experts <= 288 && hidden_size % 128 == 0, "MoERouter::route_topk: top_k / num_experts / hidden_size out of range" );
	ENGINE_ASSERT( hidden_size / 128 <= ROUTER_MAX_KSLICES, "MoERouter::route_topk: hidden_size exceeds the router k slices" );
	router_logits_kernel<<<dim3( ( num_experts + 31 ) / 32, rows, hidden_size / 128 ), dim3( 32 ), 0, stream>>>(
		x.ptr<const half>(),
		router_weights.ptr<const half>(),
		hidden_size,
		num_experts
	);
	router_select_kernel<<<dim3( rows ), dim3( 32 ), 0, stream>>>(
		e_score_bias.ptr<const half>(),
		d_topk_indices,
		d_topk_weights,
		num_experts,
		top_k,
		routed_scaling_factor,
		hidden_size / 128
	);
}

void MoERouter::combine(
	const half* expert_parts,
	const half* topk_weights,
	const half* shared_out,
	half* output,
	int hidden_size,
	int top_k,
	cudaStream_t stream
) {
	int threads = 256;
	int blocks = ( hidden_size + threads - 1 ) / threads;
	moe_combine_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
		expert_parts,
		topk_weights,
		shared_out,
		output,
		hidden_size,
		top_k
	);
}

void MoERouter::accumulate(
	half* accum_out,
	const half* expert_out,
	half weight,
	int hidden_size,
	cudaStream_t stream
) {
	int threads = 256;
	int blocks = ( hidden_size + threads - 1 ) / threads;
	moe_accumulate_kernel<<<dim3( blocks ), dim3( threads ), 0, stream>>>(
		accum_out, expert_out, weight, hidden_size
	);
}

void MoERouter::accumulate(
	pod::TensorView& accum_out,
	const pod::TensorView& expert_out,
	half weight,
	int hidden_size,
	cudaStream_t stream
) {
	accumulate( accum_out.ptr<half>(), expert_out.ptr<const half>(), weight, hidden_size, stream );
}

void MoERouter::exl3_moe_dispatch(
	const int* selected_experts,
	const half* routing_weights,
	int64_t* expert_count,
	int64_t* token_sorted,
	half* weight_sorted,
	int* cursors,
	int num_experts,
	int routing_first,
	int routing_last,
	int4* active_experts,
	cudaStream_t stream,
	int rows,
	int chunk_rows,
	uint32_t* heat
) {
	const int n = 8 * rows;
	int device = 0;
	cudaGetDevice( &device );
	int* locks = engine::modules::trellis_get_locks( device );
	int* sched = locks + MOE_SCHED_OFFSET;
	int num_groups = 1;

	moe_dispatch_hist_kernel<<<dim3( 1 ), dim3( MOE_HIST_THREADS ), 0, stream>>>(
		selected_experts,
		n,
		num_experts,
		routing_first,
		routing_last,
		expert_count,
		cursors,
		active_experts,
		sched,
		num_groups,
		chunk_rows,
		heat
	);

	moe_dispatch_scatter_kernel<<<dim3( ( n + 31 ) / 32 ), dim3( 32 ), 0, stream>>>(
		selected_experts,
		routing_weights,
		n,
		8,
		num_experts,
		routing_first,
		routing_last,
		token_sorted,
		weight_sorted,
		cursors
	);
}

void MoE::init_device_ptrs(
	pod::MoELayerWeights& weights,
	pod::MemoryArena& arena,
	cudaStream_t stream
) {
	if ( weights.type == pod::QuantType::TRELLIS ) {
		const int E = weights.num_local_experts;
		std::vector<const void*> h_ptrs( 9 * E );
		for ( int e = 0; e < E; ++e ) {
			h_ptrs[0 * E + e] = weights.gate[e].w0.data;
			h_ptrs[1 * E + e] = weights.gate[e].w1.data;
			h_ptrs[2 * E + e] = weights.gate[e].w2.data;
			h_ptrs[3 * E + e] = weights.up[e].w0.data;
			h_ptrs[4 * E + e] = weights.up[e].w1.data;
			h_ptrs[5 * E + e] = weights.up[e].w2.data;
			h_ptrs[6 * E + e] = weights.down[e].w0.data;
			h_ptrs[7 * E + e] = weights.down[e].w1.data;
			h_ptrs[8 * E + e] = weights.down[e].w2.data;
		}
		void* d_ptrs = engine::core::arena_alloc( arena, 9 * E * sizeof( const void* ) );
		CUDA_CHECK( cudaMemcpyAsync( d_ptrs, h_ptrs.data(), 9 * E * sizeof( const void* ), cudaMemcpyHostToDevice, stream ) );
		if ( stream ) {
			CUDA_CHECK( cudaStreamSynchronize( stream ) );
		}
		weights.device_ptrs = d_ptrs;
	}
}

pod::MoEScratch MoE::alloc_scratch(
	pod::MemoryArena& arena,
	int num_local_experts,
	int top_k,
	int hidden_size,
	int intermediate_size,
	int max_rows
) {
	const int groups = MOE_MAX_GROUPS;
	const size_t E1 = static_cast<size_t>( num_local_experts ) + 1;
	pod::MoEScratch s;
	s.expert_count = static_cast<int64_t*>( engine::core::arena_alloc( arena, E1 * sizeof( int64_t ) ) );
	s.token_sorted = static_cast<int64_t*>( engine::core::arena_alloc( arena, static_cast<size_t>( top_k ) * max_rows * sizeof( int64_t ) ) );
	s.weight_sorted = static_cast<half*>( engine::core::arena_alloc( arena, static_cast<size_t>( top_k ) * max_rows * sizeof( half ) ) );
	s.cursors = static_cast<int*>( engine::core::arena_alloc( arena, E1 * sizeof( int ) ) );
	// header + one item per (expert, chunk of >= 4 rows)
	s.active_experts = static_cast<int4*>( engine::core::arena_alloc( arena, ( 1 + E1 + static_cast<size_t>( top_k ) * max_rows / 4 ) * sizeof( int4 ) ) );
	const size_t state_elems = static_cast<size_t>( groups ) * 64 * hidden_size;
	const size_t interm_elems = static_cast<size_t>( groups ) * 64 * intermediate_size;
	s.state_g = static_cast<half*>( engine::core::arena_alloc( arena, state_elems * sizeof( half ) ) );
	s.state_u = static_cast<half*>( engine::core::arena_alloc( arena, state_elems * sizeof( half ) ) );
	s.interm_g = static_cast<half*>( engine::core::arena_alloc( arena, interm_elems * sizeof( half ) ) );
	s.interm_u = static_cast<half*>( engine::core::arena_alloc( arena, interm_elems * sizeof( half ) ) );
	s.planes = static_cast<half*>( engine::core::arena_alloc( arena, static_cast<size_t>( groups ) * max_rows * hidden_size * sizeof( half ) ) );
	// Fused decode path: the rotated inputs and the gate/up partials alias the persistent kernel's staging buffers (a call runs
	// one path or the other, in stream order); the activation and the down partials are its own
	if ( hidden_size % 128 == 0 && intermediate_size % 128 == 0 ) {
		const size_t slots = MOE_COOP_MAX_SLOTS;
		s.coop_had_g = s.state_g;
		s.coop_had_u = s.state_u;
		s.coop_gu_g = reinterpret_cast<float*>( s.interm_g );
		s.coop_gu_u = reinterpret_cast<float*>( s.interm_u );
		s.coop_act = static_cast<half*>( engine::core::arena_alloc( arena, slots * intermediate_size * sizeof( half ) ) );
		s.coop_d_out = static_cast<float*>( engine::core::arena_alloc( arena, slots * hidden_size * sizeof( float ) ) );
		const size_t ctr_bytes = static_cast<size_t>( moe_coop_ctr_len( hidden_size, intermediate_size ) ) * sizeof( int );
		s.coop_ctr = static_cast<int*>( engine::core::arena_alloc( arena, ctr_bytes ) );
		CUDA_CHECK( cudaMemset( s.coop_ctr, 0, ctr_bytes ) );
	}
	return s;
}

void MoE::forward(
	const pod::TensorView& x,
	const pod::MoELayerWeights& weights,
	const int* d_topk_indices,
	const half* d_topk_weights,
	pod::TensorView& out,
	int start_expert,
	int end_expert,
	const pod::MoEScratch& scratch,
	float act_limit,
	cudaStream_t stream
) {
	const int rows = ( out.ndim >= 2 && out.shape[0] > 0 ) ? out.shape[0] : 1;
	// Concurrent expert groups: a rank has ~2 active experts at one row, ~13 at eight, all 72 in a prefill chunk, and
	// each group has fewer blocks, so wide grouping only pays once there are experts to run side by side.
	static const int env_groups = std::getenv( "EXL3_MOE_GROUPS" ) ? std::atoi( std::getenv( "EXL3_MOE_GROUPS" ) ) : 0;
	// Sharded experts (every rank runs every active expert, a quarter wide) want all groups at any row count: the active experts
	// are few per token but each pass is short, so the fixed phase costs hide behind the other groups
	const bool sharded = weights.num_local_experts > 72;
	const int num_groups = env_groups > 0 ? MIN( env_groups, MOE_MAX_GROUPS ) : ( sharded ? MOE_MAX_GROUPS : rows == 1 ? 1 : rows <= 16 ? 4 : MOE_MAX_GROUPS );
	// Fused decode path: gate/up and down GEMV kernels over runs of rows that share an expert (exl3_moe_coop.cuh)
	static const bool coop_on = std::getenv( "EXL3_MOE_COOP" ) ? std::atoi( std::getenv( "EXL3_MOE_COOP" ) ) != 0 : true;
	if ( coop_on && weights.type == pod::QuantType::TRELLIS && rows <= MOE_COOP_MAX_ROWS && scratch.coop_ctr != nullptr
		&& weights.gate[0].bits == 3 && weights.up[0].bits == 3 && weights.down[0].bits == 3 && weights.up[0].mul1 && weights.gate[0].mul1 && weights.down[0].mul1 ) {
		const int E = weights.num_local_experts;
		const void** d_ptrs = static_cast<const void**>( weights.device_ptrs );
		MoeCoopParams p = {};
		p.x = x.ptr<const half>();
		p.sel = d_topk_indices;
		p.rw = d_topk_weights;
		p.out = out.ptr<half>();
		p.heat = weights.heat;
		p.bsz = rows;
		p.min_expert = start_expert;
		p.max_expert = end_expert;
		p.H = weights.up[0].w1.shape[0];
		p.I = weights.up[0].w2.shape[0];
		p.topk = 8;
		p.act_limit = act_limit;
		p.g_trellis = reinterpret_cast<const uint16_t* const*>( d_ptrs + 0 * E );
		p.g_suh = reinterpret_cast<const half* const*>( d_ptrs + 1 * E );
		p.g_svh = reinterpret_cast<const half* const*>( d_ptrs + 2 * E );
		p.u_trellis = reinterpret_cast<const uint16_t* const*>( d_ptrs + 3 * E );
		p.u_suh = reinterpret_cast<const half* const*>( d_ptrs + 4 * E );
		p.u_svh = reinterpret_cast<const half* const*>( d_ptrs + 5 * E );
		p.d_trellis = reinterpret_cast<const uint16_t* const*>( d_ptrs + 6 * E );
		p.d_suh = reinterpret_cast<const half* const*>( d_ptrs + 7 * E );
		p.d_svh = reinterpret_cast<const half* const*>( d_ptrs + 8 * E );
		p.had_g = scratch.coop_had_g;
		p.had_u = scratch.coop_had_u;
		p.gu_g = scratch.coop_gu_g;
		p.gu_u = scratch.coop_gu_u;
		p.act_out = scratch.coop_act;
		p.d_out = scratch.coop_d_out;
		p.ctr_a = scratch.coop_ctr;
		p.ctr_b = p.ctr_a + MOE_COOP_MAX_SLOTS * ( p.I / 128 );
		p.runs = p.ctr_b + MOE_COOP_MAX_ROWS * ( p.H / 128 );
		int device = 0;
		CUDA_CHECK( cudaGetDevice( &device ) );
		CUDA_CHECK( cudaMemsetAsync( p.out, 0, static_cast<size_t>( rows ) * p.H * sizeof( half ), stream ) );
		exl3_moe_coop_launch( p, device, stream );
		return;
	}
	const size_t plane_elems = static_cast<size_t>( rows ) * out.shape[out.ndim - 1];
	half* target = ( num_groups == 1 ) ? out.ptr<half>() : scratch.planes;
	CUDA_CHECK( cudaMemsetAsync( target, 0, plane_elems * num_groups * sizeof( half ), stream ) );

	switch ( weights.type ) {
		case pod::QuantType::TRELLIS: {
			const int E = weights.num_local_experts;
			const int K_gate = weights.gate[0].bits;
			const int K_up = weights.up[0].bits;
			const int K_down = weights.down[0].bits;
			ENGINE_ASSERT( weights.device_ptrs != nullptr, "MoE::forward: init_device_ptrs not called" );
			ENGINE_ASSERT( K_gate == 3 && K_up == 3 && K_down == 3, "MoE::forward: persistent trellis MoE kernel is instantiated for 3-bit experts only" );

			const int wide = static_cast<int>( std::getenv( "EXL3_MOE_WIDE" ) ? std::atoi( std::getenv( "EXL3_MOE_WIDE" ) ) : ( rows > 8 ? 2 : 0 ) );
			MoERouter::exl3_moe_dispatch(
				d_topk_indices,
				d_topk_weights,
				scratch.expert_count,
				scratch.token_sorted,
				scratch.weight_sorted,
				scratch.cursors,
				E,
				start_expert,
				end_expert,
				scratch.active_experts,
				stream,
				rows,
				wide == 2 ? 16 : 4,
				weights.heat
			);

			const void** d_ptrs = static_cast<const void**>( weights.device_ptrs );
			const uint16_t** gate_trellis = reinterpret_cast<const uint16_t**>( d_ptrs + 0 * E );
			const half** gate_suh = reinterpret_cast<const half**>( d_ptrs + 1 * E );
			const half** gate_svh = reinterpret_cast<const half**>( d_ptrs + 2 * E );
			const uint16_t** up_trellis = reinterpret_cast<const uint16_t**>( d_ptrs + 3 * E );
			const half** up_suh = reinterpret_cast<const half**>( d_ptrs + 4 * E );
			const half** up_svh = reinterpret_cast<const half**>( d_ptrs + 5 * E );
			const uint16_t** down_trellis = reinterpret_cast<const uint16_t**>( d_ptrs + 6 * E );
			const half** down_suh = reinterpret_cast<const half**>( d_ptrs + 7 * E );
			const half** down_svh = reinterpret_cast<const half**>( d_ptrs + 8 * E );

			ext::exl3_moe(
				x.data,
				target,
				scratch.expert_count,
				scratch.token_sorted,
				scratch.weight_sorted,
				scratch.state_g,
				scratch.state_u,
				scratch.interm_g,
				scratch.interm_u,
				0,
				K_gate, K_up, K_down,
				gate_trellis, gate_suh, gate_svh,
				up_trellis, up_suh, up_svh,
				down_trellis, down_suh, down_svh,
				false, weights.gate[0].mul1, false, weights.up[0].mul1, false, weights.down[0].mul1,
				act_limit,
				-1,
				scratch.active_experts,
				num_groups,
				plane_elems,
				wide,
				weights.up[0].w1.shape[0],      // hidden: the length of suh
				weights.up[0].w2.shape[0],      // this rank's intermediate width: the length of svh (2048, or 512 when sharded)
				E,
				stream
			);
			if ( num_groups > 1 ) {
				if ( plane_elems % 8 == 0 && num_groups <= MOE_MAX_GROUPS ) {
					moe_sum_planes8_kernel<<<dim3( ( plane_elems / 8 + 255 ) / 256 ), dim3( 256 ), 0, stream>>>(
						scratch.planes, out.ptr<half>(), static_cast<int>( plane_elems / 8 ), num_groups );
				}
				else {
					moe_sum_planes_kernel<<<dim3( ( plane_elems + 255 ) / 256 ), dim3( 256 ), 0, stream>>>(
						scratch.planes, out.ptr<half>(), static_cast<int>( plane_elems ), num_groups );
				}
			}
		} break;
		default: {
			ENGINE_ASSERT( false, "MoE::forward: no routed-expert driver for this weight format (only TRELLIS is implemented)" );
		} break;
	}
}

} // namespace engine::modules

namespace ext {

static bool attr_set[ENGINE_NUM_GPUS] = { false };

void exl3_moe_reset_caches() {
	for ( auto& b : attr_set ) b = false;
}

void exl3_moe(
	const void* hidden_state,
	void* output_state,
	const int64_t* expert_count,
	const int64_t* token_sorted,
	const half* weight_sorted,
	half* temp_state_g,
	half* temp_state_u,
	half* temp_intermediate_g,
	half* temp_intermediate_u,
	int act_function,
	int K_gate, int K_up, int K_down,
	const uint16_t** gate_trellis, const half** gate_suh, const half** gate_svh,
	const uint16_t** up_trellis, const half** up_suh, const half** up_svh,
	const uint16_t** down_trellis, const half** down_suh, const half** down_svh,
	bool gate_mcg, bool gate_mul1,
	bool up_mcg, bool up_mul1,
	bool down_mcg, bool down_mul1,
	float act_limit,
	int num_active,
	const int4* active_experts,
	int num_groups,
	size_t out_plane_stride,
	int wide,
	int hidden_dim,
	int intermediate_dim,
	int num_experts,
	cudaStream_t stream
) {
	(void) gate_mcg; (void) gate_mul1; (void) up_mcg; (void) up_mul1; (void) down_mcg; (void) down_mul1;
	(void) num_active;
	int device = 0;
	cudaGetDevice( &device );
	int num_sms = engine::modules::trellis_get_num_sms( device );
	int* locks = engine::modules::trellis_get_locks( device );

	// num_groups experts run concurrently on num_sms / num_groups blocks each, all co-resident
	ENGINE_ASSERT( num_groups >= 1 && num_groups <= MOE_MAX_GROUPS, "exl3_moe: bad group count" );
	int group_size = num_sms / num_groups;
	dim3 grid_dim( group_size, 1, num_groups );
	int block_dim = 256 * MOE_TILESIZE_K / 16;

	const half* _hidden_state = static_cast<const half*>( hidden_state );
	half* _output_state = static_cast<half*>( output_state );

	int num_experts_per_tok = 8;
	int max_tokens_per_expert = wide == 2 ? 16 : 4; // rows per expert pass (the GEMM tile); larger batches run in chunks
	int concurrency = num_groups;
	int chunk_all = 1; // experts above max_tokens_per_expert tokens are evaluated in chunks of max_tokens_per_expert

	void* kernel_args[] = {
		(void*) &_hidden_state,
		(void*) &temp_state_g,
		(void*) &temp_state_u,
		(void*) &temp_intermediate_g,
		(void*) &temp_intermediate_u,
		(void*) &_output_state,
		(void*) &gate_trellis,
		(void*) &gate_suh,
		(void*) &gate_svh,
		(void*) &up_trellis,
		(void*) &up_suh,
		(void*) &up_svh,
		(void*) &down_trellis,
		(void*) &down_suh,
		(void*) &down_svh,
		(void*) &expert_count,
		(void*) &token_sorted,
		(void*) &weight_sorted,
		(void*) &hidden_dim,
		(void*) &intermediate_dim,
		(void*) &num_experts,
		(void*) &num_experts_per_tok,
		(void*) &max_tokens_per_expert,
		(void*) &concurrency,
		(void*) &chunk_all,
		(void*) &act_limit,
		(void*) &act_function,
		(void*) &K_gate,
		(void*) &K_up,
		(void*) &K_down,
		(void*) &locks,
		(void*) &active_experts,
		(void*) &out_plane_stride
	};

	if ( device >= 0 && device < ENGINE_NUM_GPUS && !attr_set[device] ) {
		CUDA_CHECK( cudaFuncSetAttribute( (void*) exl3_moe_kernel<3, 128, 2, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX ) );
		CUDA_CHECK( cudaFuncSetAttribute( (void*) exl3_moe_kernel<3, 128, 2, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX ) );
		CUDA_CHECK( cudaFuncSetAttribute( (void*) exl3_moe_kernel<3, 256, 2, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX ) );
		CUDA_CHECK( cudaFuncSetAttribute( (void*) exl3_moe_kernel<3, 256, 2, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX ) );
		CUDA_CHECK( cudaFuncSetAttribute( (void*) exl3_moe_kernel<3, 128, 2, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX ) );
		attr_set[device] = true;
	}

	// The wide kernel carries the 16-row GEMM tile as well; keeping it out of the decode kernel preserves its registers
	// Output-tile width of the expert GEMMs: 256 is 18-29% faster than 128 below 64 rows on both the sharded and the whole-expert
	// layouts (fewer K-split reductions per weight decoded) and level at 512; EXL3_MOE_TILE_N=128 selects the old tile
	static const int tile_n = std::getenv( "EXL3_MOE_TILE_N" ) ? std::atoi( std::getenv( "EXL3_MOE_TILE_N" ) ) : 256;
	void* moe_fn = tile_n == 128
		? ( wide == 2 ? (void*) exl3_moe_kernel<3, 128, 2, 2> : (void*) exl3_moe_kernel<3, 128, 2, 0> )
		: ( wide == 2 ? (void*) exl3_moe_kernel<3, 256, 2, 2> : (void*) exl3_moe_kernel<3, 256, 2, 0> );
	CUDA_CHECK( cudaLaunchKernel(
		moe_fn,
		grid_dim,
		dim3( block_dim ),
		kernel_args,
		SMEM_MAX,
		stream
	) );
}

} // namespace ext
