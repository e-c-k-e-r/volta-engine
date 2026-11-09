#pragma once
// The Volta tile order copies of the trellis entry points. Every translation unit whose kernels decode tiles is compiled twice on SM_70 CUDA
// builds (ENGINE_EXL3_VL): as is, for the stock order, and through exl3_vl_unit.cu.in with EXL3_VOLTA_LAYOUT=1 inside namespace exl3_vl.
// A weight's LinearWeight::volta_tiles picks the copy at the call. The signatures repeat the stock headers' (a mismatch fails to link).
#include "exl3_moe_coop.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

#if ENGINE_EXL3_VL
namespace exl3_vl {

void exl3_gemm( const half* A, const uint16_t* B, void* C, const half* suh, half* A_had, const half* svh, int size_m, int size_k, int size_n, int K,
	bool mcg, bool mul1, bool c_fp32, cudaStream_t stream, int force_shape );
void exl3_gemv( const half* A, const uint16_t* B, void* C, const half* suh, half* A_had, const half* svh, int size_m, int size_k, int size_n, int K,
	bool mcg, bool mul1, bool c_fp32, cudaStream_t stream );
void reconstruct_slice( half* unpacked, const uint16_t* packed, int K, bool mcg, bool mul1, int rows, int cols, int64_t n_offset, cudaStream_t stream );
void reconstruct_had_slice( half* unpacked, const uint16_t* packed, const half* suh, const half* svh, int K, bool mcg, bool mul1, int rows, int cols,
	int64_t n_offset, cudaStream_t stream );
void exl3_moe_coop_launch( const MoeCoopParams& p, int device, cudaStream_t stream );

void exl3_gemm_init_all_attrs( int device );
void exl3_gemv_int8_init( int device );
void exl3_gemm_reset_caches();
void exl3_gemv_reset_caches();
void exl3_gemv_int8_reset_caches();

namespace ext {
void exl3_moe( const void* hidden_state, void* output_state, const int64_t* expert_count, const int64_t* token_sorted, const half* weight_sorted,
	half* temp_state_g, half* temp_state_u, half* temp_intermediate_g, half* temp_intermediate_u, int act_function, int K_gate, int K_up, int K_down,
	const uint16_t** gate_trellis, const half** gate_suh, const half** gate_svh, const uint16_t** up_trellis, const half** up_suh, const half** up_svh,
	const uint16_t** down_trellis, const half** down_suh, const half** down_svh, bool gate_mcg, bool gate_mul1, bool up_mcg, bool up_mul1,
	bool down_mcg, bool down_mul1, float act_limit, int num_active, const int4* active_experts, int num_groups, size_t out_plane_stride, int wide,
	int hidden_dim, int intermediate_dim, int num_experts, cudaStream_t stream, int top_k, const uint8_t* bits_tab, bool bits_half );
void exl3_moe_reset_caches();
} // namespace ext

} // namespace exl3_vl

// fn( args ) in the copy for the weight's tile order (every argument spelled out: the Volta declarations above carry no defaults)
#define EXL3_TILES_CALL( volta, fn, ... ) ( ( volta ) ? exl3_vl::fn( __VA_ARGS__ ) : ::fn( __VA_ARGS__ ) )
#else
#define EXL3_TILES_CALL( volta, fn, ... ) ( (void) ( volta ), ::fn( __VA_ARGS__ ) )
#endif
