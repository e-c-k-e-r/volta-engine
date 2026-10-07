#include "engine/modules/moe.h"
#include "exl3_moe_kernel.cuh"
#include <cuda_fp16.h>
#include <cmath>
#include <vector>
#include <cstdio>

namespace engine::modules {

namespace {

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
	int num_groups
) {
	__shared__ int hist[128];
	const int threads = blockDim.x;
	for ( int i = threadIdx.x; i <= num_experts; i += threads ) hist[i] = 0;
	__syncthreads();

	for ( int i = threadIdx.x; i < n; i += threads ) {
		int e = selected_experts[i];
		int el = ( e >= routing_first && e < routing_last ) ? ( e - routing_first ) : num_experts;
		atomicAdd( &hist[el], 1 );
	}
	__syncthreads();

	__shared__ int scan[128];
	scan[threadIdx.x] = ( threadIdx.x <= num_experts ) ? hist[threadIdx.x] : 0;
	__syncthreads();
	for ( int d = 1; d < 128; d <<= 1 ) {
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
		__syncthreads();
		scan[threadIdx.x] = ( threadIdx.x < num_experts && hist[threadIdx.x] > 0 ) ? 1 : 0;
		__syncthreads();
		for ( int d = 1; d < 128; d <<= 1 ) {
			int v = ( threadIdx.x >= d ) ? scan[threadIdx.x - d] : 0;
			__syncthreads();
			scan[threadIdx.x] += v;
			__syncthreads();
		}

		if ( threadIdx.x < num_experts && hist[threadIdx.x] > 0 ) {
			int act_idx = scan[threadIdx.x] - 1;
			int my_start = cursors[threadIdx.x];
			active_experts[act_idx] = make_int4( threadIdx.x, my_start, hist[threadIdx.x], 0 );
		}

		if ( threadIdx.x == 0 ) {
			int total_active = ( num_experts > 0 ) ? scan[num_experts - 1] : 0;
			active_experts[num_experts] = make_int4( total_active, 0, 0, 0 );
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
		int el = e - routing_first;
		int pos = atomicAdd( &cursors[el], 1 );
		token_sorted[pos] = static_cast<int64_t>( i / top_k );
		weight_sorted[pos] = routing_weights[i];
	}
}

// Router scratch: the 288 logits handed from the dot-product kernel to the selection kernel.
__device__ float g_router_logits[288];

// One warp per 32 experts, one thread per expert, summed in the same sequential order as the original
// single-block kernel (bit-identical logits). The single block was bound by memory-level parallelism on
// one SM (0.43 ms per layer for 288 x 4096 weights); spreading the experts over nine blocks gives nine
// SMs' worth of outstanding loads, with 128 independent weight loads in flight per thread.
__global__ __launch_bounds__( 32 )
void router_logits_kernel(
	const half* __restrict__ x,
	const half* __restrict__ router_weights,
	int hidden_size,
	int num_experts
) {
	extern __shared__ float s_x[];
	for ( int d = threadIdx.x; d < hidden_size; d += blockDim.x ) {
		s_x[d] = __half2float( x[d] );
	}
	__syncthreads();

	const int expert = blockIdx.x * 32 + threadIdx.x;
	if ( expert >= num_experts ) return;
	float logit = 0.0f;
	const half* wcol = router_weights + expert;
	for ( int d0 = 0; d0 < hidden_size; d0 += 128 ) {
		// Unconditional loads (hidden_size % 128 == 0, asserted on the host) so all 128 issue back to back.
		half w[128];
		#pragma unroll
		for ( int j = 0; j < 128; ++j ) {
			w[j] = __ldg( wcol + static_cast<size_t>( d0 + j ) * num_experts );
		}
		#pragma unroll
		for ( int j = 0; j < 128; ++j ) {
			logit = fmaf( s_x[d0 + j], __half2float( w[j] ), logit );
		}
	}
	g_router_logits[expert] = logit;
}

__global__ __launch_bounds__( 32 )
void router_select_kernel(
	const half* __restrict__ e_score_bias,
	int* __restrict__ topk_indices,
	half* __restrict__ topk_weights,
	int num_experts,
	int top_k,
	float routed_scaling_factor
) {
	__shared__ float s_raw[288];
	__shared__ float s_select[288];
	for ( int expert = threadIdx.x; expert < num_experts; expert += 32 ) {
		const float raw_score = 1.0f / ( 1.0f + __expf( -g_router_logits[expert] ) );
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
	cudaStream_t stream
) {
	ENGINE_ASSERT( top_k <= 16 && num_experts <= 288 && hidden_size % 128 == 0, "MoERouter::route_topk: top_k / num_experts / hidden_size out of range" );
	const size_t smem = static_cast<size_t>( hidden_size ) * sizeof( float );
	ENGINE_ASSERT( smem <= 48 * 1024, "MoERouter::route_topk: hidden_size exceeds the staged activation" );
	router_logits_kernel<<<dim3( ( num_experts + 31 ) / 32 ), dim3( 32 ), smem, stream>>>(
		x.ptr<const half>(),
		router_weights.ptr<const half>(),
		hidden_size,
		num_experts
	);
	router_select_kernel<<<dim3( 1 ), dim3( 32 ), 0, stream>>>(
		e_score_bias.ptr<const half>(),
		d_topk_indices,
		d_topk_weights,
		num_experts,
		top_k,
		routed_scaling_factor
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
	cudaStream_t stream
) {
	int n = 8;
	int device = 0;
	cudaGetDevice( &device );
	int* locks = engine::modules::trellis_get_locks( device );
	int* sched = locks + MOE_SCHED_OFFSET;
	int num_groups = 1;

	moe_dispatch_hist_kernel<<<dim3( 1 ), dim3( 128 ), 0, stream>>>(
		selected_experts,
		n,
		num_experts,
		routing_first,
		routing_last,
		expert_count,
		cursors,
		active_experts,
		sched,
		num_groups
	);

	moe_dispatch_scatter_kernel<<<dim3( 1 ), dim3( 32 ), 0, stream>>>(
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
	int intermediate_size
) {
	const int groups = 10;
	const size_t E1 = static_cast<size_t>( num_local_experts ) + 1;
	pod::MoEScratch s;
	s.expert_count = static_cast<int64_t*>( engine::core::arena_alloc( arena, E1 * sizeof( int64_t ) ) );
	s.token_sorted = static_cast<int64_t*>( engine::core::arena_alloc( arena, top_k * sizeof( int64_t ) ) );
	s.weight_sorted = static_cast<half*>( engine::core::arena_alloc( arena, top_k * sizeof( half ) ) );
	s.cursors = static_cast<int*>( engine::core::arena_alloc( arena, E1 * sizeof( int ) ) );
	s.active_experts = static_cast<int4*>( engine::core::arena_alloc( arena, E1 * sizeof( int4 ) ) );
	const size_t state_elems = static_cast<size_t>( groups ) * top_k * hidden_size;
	const size_t interm_elems = static_cast<size_t>( groups ) * top_k * intermediate_size;
	s.state_g = static_cast<half*>( engine::core::arena_alloc( arena, state_elems * sizeof( half ) ) );
	s.state_u = static_cast<half*>( engine::core::arena_alloc( arena, state_elems * sizeof( half ) ) );
	s.interm_g = static_cast<half*>( engine::core::arena_alloc( arena, interm_elems * sizeof( half ) ) );
	s.interm_u = static_cast<half*>( engine::core::arena_alloc( arena, interm_elems * sizeof( half ) ) );
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
	CUDA_CHECK( cudaMemsetAsync( out.data, 0, out.size_bytes(), stream ) );

	switch ( weights.type ) {
		case pod::QuantType::TRELLIS: {
			const int E = weights.num_local_experts;
			const int K_gate = weights.gate[0].bits;
			const int K_up = weights.up[0].bits;
			const int K_down = weights.down[0].bits;
			ENGINE_ASSERT( weights.device_ptrs != nullptr, "MoE::forward: init_device_ptrs not called" );
			ENGINE_ASSERT( K_gate == 3 && K_up == 3 && K_down == 3, "MoE::forward: persistent trellis MoE kernel is instantiated for 3-bit experts only" );

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
				stream
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
				out.data,
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
				stream
			);
		} break;
		default: {
			ENGINE_ASSERT( false, "MoE::forward: no routed-expert driver for this weight format (only TRELLIS is implemented)" );
		} break;
	}
}

} // namespace engine::modules

namespace ext {

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
	cudaStream_t stream
) {
	(void) gate_mcg; (void) gate_mul1; (void) up_mcg; (void) up_mul1; (void) down_mcg; (void) down_mul1;
	(void) num_active;
	int device = 0;
	cudaGetDevice( &device );
	int num_sms = engine::modules::trellis_get_num_sms( device );
	int* locks = engine::modules::trellis_get_locks( device );

	int group_size = 80;
	int num_groups = num_sms / group_size;
	dim3 grid_dim( group_size, 1, num_groups );
	int block_dim = 256 * MOE_TILESIZE_K / 16;

	const half* _hidden_state = static_cast<const half*>( hidden_state );
	half* _output_state = static_cast<half*>( output_state );

	int hidden_dim = 4096;
	int intermediate_dim = 2048;
	int num_experts = 72;
	int num_experts_per_tok = 8;
	int max_tokens_per_expert = 8;
	int concurrency = num_groups;
	int chunk_all = 0;

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
		(void*) &active_experts
	};

	static bool attr_set[ENGINE_NUM_GPUS] = { false };
	if ( device >= 0 && device < ENGINE_NUM_GPUS && !attr_set[device] ) {
		CUDA_CHECK( cudaFuncSetAttribute( (void*) exl3_moe_kernel<3, 128, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX ) );
		attr_set[device] = true;
	}

	CUDA_CHECK( cudaLaunchKernel(
		(void*) exl3_moe_kernel<3, 128, 2>,
		grid_dim,
		dim3( block_dim ),
		kernel_args,
		SMEM_MAX,
		stream
	) );
}

} // namespace ext
