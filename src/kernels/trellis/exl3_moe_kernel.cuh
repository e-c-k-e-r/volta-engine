#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/atomic>
#include <cstdio>
#include <cstdlib>

#include "exl3_moe_common.cuh"
#include "util.cuh"
#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"
#include "exl3_gemm_inner.cuh"
#include "ptx.cuh"
#include "trellis_context.h"

template<int t_bits, int MOE_TILESIZE_N, int cb>
__device__ inline void moe_gemm_m4(
	const half* in_addr,
	const uint16_t* trellis,
	half* out_addr,
	int rows,
	int dim_k,
	int dim_n,
	int* locks
) {
	exl3_gemm_kernel_inner<t_bits, false, cb, 4, MOE_TILESIZE_K, MOE_TILESIZE_N, MOE_SH_STAGES, MOE_FRAG_STAGES, false>(
		in_addr, trellis, out_addr, rows, dim_k, dim_n, locks, nullptr
	);
}

template<bool gated>
__device__ inline void moe_gather_gu_in(
	const half* hidden_state,
	half* temp_state_g,
	half* temp_state_u,
	const half* exp_gate_suh,
	const half* exp_up_suh,
	int rows,
	const int64_t* top_x,
	int hidden_dim,
	int warp_idx0,
	int warps_per_group,
	int group_idx,
	int group_size,
	int* barrier_counters_sense
) {
	const int warps_per_token = hidden_dim / 128;
	const int total_warps = rows * warps_per_token;
	for ( int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group ) {
		int token_idx = top_x[warp_idx / warps_per_token];
		int token_off = warp_idx % warps_per_token;
		const half* in_ptr = hidden_state + token_idx * hidden_dim + token_off * 128;
		if ( gated ) {
			had_hf_r_128_inner<true, false>(
				in_ptr,
				temp_state_g + 128 * warp_idx,
				exp_gate_suh + 128 * token_off,
				0.088388347648f
			);
		}
		had_hf_r_128_inner<true, false>(
			in_ptr,
			temp_state_u + 128 * warp_idx,
			exp_up_suh + 128 * token_off,
			0.088388347648f
		);
	}
	group_barrier( group_idx, group_size, barrier_counters_sense );
}

__device__ inline void moe_had_guad(
	half* temp_intermediate_g,
	half* temp_intermediate_u,
	const half* exp_gate_svh,
	const half* exp_up_svh,
	const half* exp_down_suh,
	int rows,
	int intermediate_dim,
	float act_limit,
	int act_function,
	int warp_idx0,
	int warps_per_group,
	int group_idx,
	int group_size,
	int* barrier_counters_sense
) {
	const int warps_per_token = intermediate_dim / 128;
	const int total_warps = rows * warps_per_token;
	for ( int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group ) {
		int token_off = warp_idx % warps_per_token;
		had_hf_r_128_guad_inner(
			temp_intermediate_g + 128 * warp_idx,
			temp_intermediate_u + 128 * warp_idx,
			temp_intermediate_g + 128 * warp_idx,
			exp_gate_svh + 128 * token_off,
			exp_up_svh + 128 * token_off,
			exp_down_suh + 128 * token_off,
			0.088388347648f,
			act_limit,
			act_function
		);
	}
	group_barrier( group_idx, group_size, barrier_counters_sense );
}

__device__ inline void moe_had_d_out(
	const half* temp_state_g,
	half* output_state,
	const half* exp_down_svh,
	int rows,
	const int64_t* top_x,
	const half* weights,
	int hidden_dim,
	int warp_idx0,
	int warps_per_group
) {
	const int warps_per_token = hidden_dim / 128;
	const int total_warps = rows * warps_per_token;
	for ( int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group ) {
		int token_idx = top_x[warp_idx / warps_per_token];
		half weight = weights[warp_idx / warps_per_token];
		int token_off = warp_idx % warps_per_token;
		half* out_ptr = output_state + token_idx * hidden_dim + token_off * 128;
		had_hf_r_128_d_inner(
			temp_state_g + 128 * warp_idx,
			out_ptr,
			exp_down_svh + 128 * token_off,
			0.088388347648f * __half2float( weight )
		);
	}
}

template<int t_bits, int MOE_TILESIZE_N, int cb>
__global__ __launch_bounds__( EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16 )
void exl3_moe_kernel( EXL3_MOE_KERNEL_ARGS ) {
	const int group_idx = blockIdx.z;
	const int block_idx = blockIdx.x;
	const int group_size = gridDim.x;
	const int num_groups = gridDim.z;
	const int block_threads = EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16;
	const int group_threads = group_size * block_threads;
	const int warp_id = threadIdx.x / 32;
	const int warps_per_group = group_threads / 32;
	const int warps_per_block = block_threads / 32;
	const int warp_idx0 = block_idx * warps_per_block + warp_id;

	temp_state_g += group_idx * max_tokens_per_expert * hidden_dim;
	temp_state_u += group_idx * max_tokens_per_expert * hidden_dim;
	temp_intermediate_g += group_idx * max_tokens_per_expert * intermediate_dim;
	temp_intermediate_u += group_idx * max_tokens_per_expert * intermediate_dim;

	int* barrier_counters_sense = locks + BARRIER_LOCKS_OFFSET;
	int* sched = locks + MOE_SCHED_OFFSET;
	locks += group_idx * MAX( hidden_dim, intermediate_dim ) / 128;

	int ticket = group_idx;
	const bool gated = ( act_function != MOE_ACT_RELU2_NOGATE );

	auto eval_one_expert = [&]( int expert_idx, int start, int token_count ) {
		const uint16_t* exp_gate_trellis = gate_trellis[expert_idx];
		const half* exp_gate_suh = gate_suh[expert_idx];
		const half* exp_gate_svh = gate_svh[expert_idx];
		const uint16_t* exp_up_trellis = up_trellis[expert_idx];
		const half* exp_up_suh = up_suh[expert_idx];
		const half* exp_up_svh = up_svh[expert_idx];
		const uint16_t* exp_down_trellis = down_trellis[expert_idx];
		const half* exp_down_suh = down_suh[expert_idx];
		const half* exp_down_svh = down_svh[expert_idx];

		for ( int row0 = 0; row0 < token_count; row0 += max_tokens_per_expert ) {
			int rows = MIN( max_tokens_per_expert, token_count - row0 );
			const int64_t* top_x = token_sorted + start + row0;
			const half* weights = weight_sorted + start + row0;

			if ( gated ) {
				moe_gather_gu_in<true>(
					hidden_state, temp_state_g, temp_state_u,
					exp_gate_suh, exp_up_suh, rows, top_x,
					hidden_dim, warp_idx0, warps_per_group, group_idx, group_size, barrier_counters_sense
				);

				moe_gemm_m4<t_bits, MOE_TILESIZE_N, cb>( temp_state_g, exp_gate_trellis, temp_intermediate_g, rows, hidden_dim, intermediate_dim, locks );
				moe_gemm_m4<t_bits, MOE_TILESIZE_N, cb>( temp_state_u, exp_up_trellis, temp_intermediate_u, rows, hidden_dim, intermediate_dim, locks );
			}
			else {
				moe_gather_gu_in<false>(
					hidden_state, temp_state_g, temp_state_u,
					exp_gate_suh, exp_up_suh, rows, top_x,
					hidden_dim, warp_idx0, warps_per_group, group_idx, group_size, barrier_counters_sense
				);

				moe_gemm_m4<t_bits, MOE_TILESIZE_N, cb>( temp_state_u, exp_up_trellis, temp_intermediate_u, rows, hidden_dim, intermediate_dim, locks );
			}
			group_barrier( group_idx, group_size, barrier_counters_sense );

			moe_had_guad(
				temp_intermediate_g, temp_intermediate_u,
				exp_gate_svh, exp_up_svh, exp_down_suh,
				rows, intermediate_dim, act_limit, act_function,
				warp_idx0, warps_per_group, group_idx, group_size, barrier_counters_sense
			);

			moe_gemm_m4<t_bits, MOE_TILESIZE_N, cb>( temp_intermediate_g, exp_down_trellis, temp_state_g, rows, intermediate_dim, hidden_dim, locks );
			group_barrier( group_idx, group_size, barrier_counters_sense );

			moe_had_d_out( temp_state_g, output_state, exp_down_svh, rows, top_x, weights, hidden_dim, warp_idx0, warps_per_group );
			group_barrier( group_idx, group_size, barrier_counters_sense );
		}
	};

	if ( active_experts ) {
		const int num_active = active_experts[num_experts].x;

		while ( ticket < num_active ) {
			int4 act = active_experts[ticket];
			int expert_idx = act.x;
			int start = act.y;
			int token_count = act.z;

			if ( token_count <= max_tokens_per_expert || chunk_all ) {
				eval_one_expert( expert_idx, start, token_count );
			}

			if ( block_idx == 0 && threadIdx.x == 0 ) {
				sched[2 + group_idx] = num_groups + atomicAdd( &sched[0], 1 );
			}
			group_barrier( group_idx, group_size, barrier_counters_sense );
			ticket = sched[2 + group_idx];
		}
	}
	else {
		int start = 0;
		int end = 0;
		int expert_idx = 0;
		int expert_idx_assign = 0;
		for ( ; expert_idx < num_experts; ++expert_idx ) {
			start = end;
			end += expert_count[expert_idx];
			int token_count = end - start;
			if ( token_count == 0 ) continue;
			if ( token_count > max_tokens_per_expert && !chunk_all ) continue;

			if ( expert_idx_assign++ != ticket ) continue;

			eval_one_expert( expert_idx, start, token_count );

			if ( block_idx == 0 && threadIdx.x == 0 ) {
				sched[2 + group_idx] = num_groups + atomicAdd( &sched[0], 1 );
			}
			group_barrier( group_idx, group_size, barrier_counters_sense );
			ticket = sched[2 + group_idx];
		}
	}

	if ( block_idx == 0 && threadIdx.x == 0 ) {
		cuda::atomic_ref<int, cuda::thread_scope_device> next_ticket( sched[0] );
		cuda::atomic_ref<int, cuda::thread_scope_device> retired_groups( sched[1] );
		int retired = retired_groups.fetch_add( 1, cuda::memory_order_acq_rel );
		if ( retired == num_groups - 1 ) {
			next_ticket.store( 0, cuda::memory_order_relaxed );
			retired_groups.store( 0, cuda::memory_order_relaxed );
		}
	}
}
