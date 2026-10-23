#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

#define MOE_ACT_SILU 0
#define MOE_ACT_GELU 1
#define MOE_ACT_RELU2_NOGATE 2

#define MOE_SMS_PER_EXPERT 8
#define MOE_MAX_SMS_PER_EXPERT 32
#define MOE_TILESIZE_K 16
#define MOE_TILESIZE_M 16
#define MOE_SH_STAGES 3
#define MOE_FRAG_STAGES 3
// The prefill variant (wide == 3, chunks of more than 128 rows) trades pipeline depth for registers: 2 fragment stages fit 128 registers, so two
// blocks share an SM and the warps of one hide the barrier and shuffle stalls of the other (1 block of 230 registers left 8 warps per SM)
#define MOE_PREFILL_FRAG_STAGES 2
#ifndef MOE_PREFILL_BLOCKS_PER_SM
#define MOE_PREFILL_BLOCKS_PER_SM 2
#endif

#ifndef EXL3_GEMM_BASE_THREADS
#define EXL3_GEMM_BASE_THREADS 256
#endif

#define EXL3_MOE_KERNEL_ARGS                    \
	const half* __restrict__ hidden_state,      \
	half* __restrict__ temp_state_g,            \
	half* __restrict__ temp_state_u,            \
	half* __restrict__ temp_intermediate_g,     \
	half* __restrict__ temp_intermediate_u,     \
	half* __restrict__ output_state,            \
	                                            \
	const uint16_t** __restrict__ gate_trellis, \
	const half** __restrict__ gate_suh,         \
	const half** __restrict__ gate_svh,         \
	const uint16_t** __restrict__ up_trellis,   \
	const half** __restrict__ up_suh,           \
	const half** __restrict__ up_svh,           \
	const uint16_t** __restrict__ down_trellis, \
	const half** __restrict__ down_suh,         \
	const half** __restrict__ down_svh,         \
	                                            \
	const int64_t* __restrict__ expert_count,   \
	const int64_t* __restrict__ token_sorted,   \
	const half* __restrict__ weight_sorted,     \
	                                            \
	const int hidden_dim,                       \
	const int intermediate_dim,                 \
	const int num_experts,                      \
	const int num_experts_per_tok,              \
	const int max_tokens_per_expert,            \
	const int concurrency,                      \
	const int chunk_all,                        \
	const float act_limit,                      \
	const int act_function,                     \
	const int K_gate,                           \
	const int K_up,                             \
	const int K_down,                           \
	                                            \
	int* __restrict__ locks,                    \
	const int4* __restrict__ active_experts,    \
	const size_t out_plane_stride,              \
	const uint8_t* __restrict__ bits_tab
