// The trellis MoE expert kernel's host launcher (ext::exl3_moe): one persistent kernel over every active expert of a layer, the gate / up
// GEMMs, the activation and the down GEMM per expert. Built twice on SM_70 builds, once per tile order (see exl3_vl.cuh).
#include "engine/config.h"
#include "engine/modules/moe.h"
#include "engine/modules/trellis_bits.h"
#include "exl3_moe_kernel.cuh"
#include "trellis_context.h"
#include <cuda_fp16.h>

namespace ext {

// The persistent kernel of a (bits, output tile) variant and row mode (0: 4-row tile, 2: 16 rows, 3: the wide prefill tile)
static void* moe_kernel_fn( int variant, int mode ) {
	switch ( variant ) {
		case 0: return mode == 3 ? (void*) exl3_moe_kernel<3, 256, 2, 3> : mode == 2 ? (void*) exl3_moe_kernel<3, 256, 2, 2> : (void*) exl3_moe_kernel<3, 256, 2, 0>;
		case 1: return mode == 3 ? (void*) exl3_moe_kernel<3, 128, 2, 3> : mode == 2 ? (void*) exl3_moe_kernel<3, 128, 2, 2> : (void*) exl3_moe_kernel<3, 128, 2, 0>;
		case 3: return mode == 3 ? (void*) exl3_moe_kernel<2, 128, 2, 3> : mode == 2 ? (void*) exl3_moe_kernel<2, 128, 2, 2> : (void*) exl3_moe_kernel<2, 128, 2, 0>;
		case 4: return mode == 3 ? (void*) exl3_moe_kernel<5, 128, 2, 3> : mode == 2 ? (void*) exl3_moe_kernel<5, 128, 2, 2> : (void*) exl3_moe_kernel<5, 128, 2, 0>;
		case 5: return mode == 3 ? (void*) exl3_moe_kernel<1, 128, 2, 3> : mode == 2 ? (void*) exl3_moe_kernel<1, 128, 2, 2> : (void*) exl3_moe_kernel<1, 128, 2, 0>;
		case 8: return mode == 3 ? (void*) exl3_moe_kernel<10, 128, 2, 3> : mode == 2 ? (void*) exl3_moe_kernel<10, 128, 2, 2> : (void*) exl3_moe_kernel<10, 128, 2, 0>;
		case 9: return mode == 3 ? (void*) exl3_moe_kernel<11, 128, 2, 3> : mode == 2 ? (void*) exl3_moe_kernel<11, 128, 2, 2> : (void*) exl3_moe_kernel<11, 128, 2, 0>;
		case 10: return mode == 3 ? (void*) exl3_moe_kernel<9, 128, 2, 3> : mode == 2 ? (void*) exl3_moe_kernel<9, 128, 2, 2> : (void*) exl3_moe_kernel<9, 128, 2, 0>;
		case 7: return mode == 3 ? (void*) exl3_moe_kernel<MOE_BITS_HALF_DYN, 128, 2, 3> : mode == 2 ? (void*) exl3_moe_kernel<MOE_BITS_HALF_DYN, 128, 2, 2> : (void*) exl3_moe_kernel<MOE_BITS_HALF_DYN, 128, 2, 0>;
		case 6: return mode == 3 ? (void*) exl3_moe_kernel<0, 128, 2, 3> : mode == 2 ? (void*) exl3_moe_kernel<0, 128, 2, 2> : (void*) exl3_moe_kernel<0, 128, 2, 0>;
		default: return mode == 3 ? (void*) exl3_moe_kernel<4, 128, 2, 3> : mode == 2 ? (void*) exl3_moe_kernel<4, 128, 2, 2> : (void*) exl3_moe_kernel<4, 128, 2, 0>;
	}
}

static bool attr_done[ENGINE_NUM_GPUS][11] = {};

void exl3_moe_reset_caches() {
	for ( auto& d : attr_done ) for ( bool& b : d ) b = false;
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
	cudaStream_t stream,
	int top_k,
	const uint8_t* bits_tab, bool has_half
) {
	(void) gate_mcg; (void) gate_mul1; (void) up_mcg; (void) up_mul1; (void) down_mcg; (void) down_mul1;
	(void) num_active;
	int device = 0;
	(void) cudaGetDevice( &device );
	int num_sms = engine::modules::trellis_get_num_sms( device );
	int* locks = engine::modules::trellis_get_locks( device );

	// num_groups experts run concurrently on num_sms / num_groups blocks each, all co-resident
	ENGINE_ASSERT( num_groups >= 1 && num_groups <= MOE_MAX_GROUPS, "exl3_moe: bad group count" );
	const int blocks_per_sm = wide == 3 ? MOE_PREFILL_BLOCKS_PER_SM : 1;
	int group_size = num_sms * blocks_per_sm / num_groups;
	dim3 grid_dim( group_size, 1, num_groups );
	int block_dim = 256 * MOE_TILESIZE_K / 16;

	const half* _hidden_state = static_cast<const half*>( hidden_state );
	half* _output_state = static_cast<half*>( output_state );

	int num_experts_per_tok = top_k;
	int max_tokens_per_expert = wide == 3 ? 32 : wide == 2 ? 16 : 4; // rows per expert pass (the GEMM tile); larger batches run in chunks
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
		(void*) &out_plane_stride,
		(void*) &bits_tab
	};

	// 3-bit experts with a 256-divisible intermediate width (GLM) run 256-wide output tiles; every other shape (4-bit experts, and
	// 3-bit ones with Flash-Next's intermediate 640) takes the 128-wide tiles
	auto kok = []( int b ) { return ( b >= 1 && b <= 5 ) || ( b >= 9 && b <= 11 ); };
	ENGINE_ASSERT( kok( K_gate ) && kok( K_up ) && kok( K_down ), "exl3_moe: expert bits" );
	const bool mixed = bits_tab != nullptr || K_gate != K_up || K_up != K_down;   // the projections of the experts differ in width: one kernel that picks each GEMM's instantiation at run time
	// the widest tile of the three (width codes do not order by size: 11 is 3.5 bpw)
	int K_max = K_gate;
	for ( int kq : { K_up, K_down } ) if ( engine::modules::trellis_tile_u16( kq ) > engine::modules::trellis_tile_u16( K_max ) ) K_max = kq;
	const bool tile128 = mixed || K_gate != 3 || intermediate_dim % 256 != 0;
	ENGINE_ASSERT( intermediate_dim % ( tile128 ? 128 : 256 ) == 0, "exl3_moe: intermediate width vs the output tile" );
	const int variant = mixed ? ( has_half ? 7 : 6 ) : K_gate == 9 ? 10 : K_gate == 10 ? 8 : K_gate == 11 ? 9 : K_gate == 5 ? 4 : K_gate == 4 ? 2 : K_gate == 2 ? 3 : K_gate == 1 ? 5 : ( tile128 ? 1 : 0 );   // 0: <3,256>, 1: <3,128>, 2: <4,128>, 3: <2,128>, 4: <5,128> (DeepSeek-V4.1's few 5-bit experts), 5: <1,128>
	if ( device >= 0 && device < ENGINE_NUM_GPUS && !attr_done[device][variant] ) {
		void* fns[3] = { moe_kernel_fn( variant, 0 ), moe_kernel_fn( variant, 2 ), moe_kernel_fn( variant, 3 ) };
		for ( void* fn : fns ) CUDA_CHECK( cudaFuncSetAttribute( fn, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX ) );
		attr_done[device][variant] = true;
	}

	// The wide kernel carries the 16-row GEMM tile as well; keeping it out of the decode kernel preserves its registers
	// Output-tile width of the expert GEMMs: 256 is 18-29% faster than 128 below 64 rows on both the sharded and the whole-expert
	// layouts (fewer K-split reductions per weight decoded) and level at 512
	void* moe_fn = moe_kernel_fn( variant, wide == 3 ? 3 : wide == 2 ? 2 : 0 );
#if defined(USE_ROCM)
	// RDNA grants a workgroup 64 KB, below the 90 KB cap the CUDA launch reserves: request what the tile uses (the 16-row tile's
	// footprint, ~22 KB at most; the decode variant's 4-row tile needs less)
	const size_t moe_smem = static_cast<size_t>( exl3_gemm_smem_bytes( 16, MOE_TILESIZE_K, tile128 ? 128 : 256, MOE_SH_STAGES, MOE_FRAG_STAGES, K_max, false ) );
	ENGINE_ASSERT( moe_smem <= static_cast<size_t>( rocm_smem_optin_max() ), "exl3_moe: tile shared memory above the device limit" );
#else
	const size_t moe_smem = blocks_per_sm > 1 ? static_cast<size_t>( exl3_gemm_smem_bytes( 32, MOE_TILESIZE_K, tile128 ? 128 : 256, MOE_SH_STAGES, MOE_PREFILL_FRAG_STAGES, K_max, false ) ) : SMEM_MAX;
#endif
	CUDA_CHECK( cudaLaunchKernel(
		moe_fn,
		grid_dim,
		dim3( block_dim ),
		kernel_args,
		moe_smem,
		stream
	) );
}

} // namespace ext
