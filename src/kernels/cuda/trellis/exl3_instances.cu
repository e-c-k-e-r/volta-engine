#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "util.cuh"
#include "ptx.cuh"
#include "exl3_gemm_kernel.cuh"
#include "exl3_kernel_map.cuh"
#include "exl3_mgemm_gv_kernel.cuh"

// Stub out unused bitwidths 1, 7, 8 to avoid hundreds of unused template instantiations
#define STUB_K(K) \
	fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_b##K##_cb0[5] = {}; \
	fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_b##K##_cb0[5] = {}; \
	fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp32_b##K##_cb0[5] = {}; \
	fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp16_b##K##_cb0[5] = {}; \
	fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_b##K##_cb1[5] = {}; \
	fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_b##K##_cb1[5] = {}; \
	fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp32_b##K##_cb1[5] = {}; \
	fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp16_b##K##_cb1[5] = {}; \
	fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_b##K##_cb2[5] = {}; \
	fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_b##K##_cb2[5] = {}; \
	fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp32_b##K##_cb2[5] = {}; \
	fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp16_b##K##_cb2[5] = {};

STUB_K(1)
STUB_K(7)
STUB_K(8)
#undef STUB_K

// Real instantiations for Trellis active bitwidths: 2, 3, 4, 5, 6
#define INSTANTIATE_K(K) \
	EXL3_KERNEL_INSTANCES_CB(K, 0) \
	EXL3_KERNEL_INSTANCES_CB(K, 1) \
	EXL3_KERNEL_INSTANCES_CB(K, 2)

INSTANTIATE_K(2)
INSTANTIATE_K(3)
INSTANTIATE_K(4)
INSTANTIATE_K(5)
INSTANTIATE_K(6)
#undef INSTANTIATE_K

// Instantiations for SM70 m=1 GEMV-class mgemm restructure (exl3_mgemm_gv_kernel)
template __global__ void exl3_mgemm_gv_kernel<2, false, 0, false>( EXL3_MGEMM_ARGS );
template __global__ void exl3_mgemm_gv_kernel<2, false, 2, false>( EXL3_MGEMM_ARGS );
template __global__ void exl3_mgemm_gv_kernel<3, false, 0, false>( EXL3_MGEMM_ARGS );
template __global__ void exl3_mgemm_gv_kernel<3, false, 2, false>( EXL3_MGEMM_ARGS );
template __global__ void exl3_mgemm_gv_kernel<4, false, 0, false>( EXL3_MGEMM_ARGS );
template __global__ void exl3_mgemm_gv_kernel<4, false, 2, false>( EXL3_MGEMM_ARGS );
template __global__ void exl3_mgemm_gv_kernel<5, false, 2, false>( EXL3_MGEMM_ARGS );
template __global__ void exl3_mgemm_gv_kernel<2, true, 0, false>( EXL3_MGEMM_ARGS );
template __global__ void exl3_mgemm_gv_kernel<2, true, 2, false>( EXL3_MGEMM_ARGS );
template __global__ void exl3_mgemm_gv_kernel<3, true, 0, false>( EXL3_MGEMM_ARGS );
template __global__ void exl3_mgemm_gv_kernel<3, true, 2, false>( EXL3_MGEMM_ARGS );
template __global__ void exl3_mgemm_gv_kernel<4, true, 2, false>( EXL3_MGEMM_ARGS );
template __global__ void exl3_mgemm_gv_kernel<5, true, 2, false>( EXL3_MGEMM_ARGS );

void* get_mgemm_gv_kernel( int K, bool c_fp32, int cb ) {
	if ( cb == 0 ) {
		if ( K == 2 ) return c_fp32 ? (void*) exl3_mgemm_gv_kernel<2, true, 0, false> : (void*) exl3_mgemm_gv_kernel<2, false, 0, false>;
		if ( K == 3 ) return c_fp32 ? (void*) exl3_mgemm_gv_kernel<3, true, 0, false> : (void*) exl3_mgemm_gv_kernel<3, false, 0, false>;
		if ( K == 4 ) return c_fp32 ? (void*) exl3_mgemm_gv_kernel<4, true, 0, false> : (void*) exl3_mgemm_gv_kernel<4, false, 0, false>;
	}
	if ( cb == 2 ) {
		if ( K == 2 ) return c_fp32 ? (void*) exl3_mgemm_gv_kernel<2, true, 2, false> : (void*) exl3_mgemm_gv_kernel<2, false, 2, false>;
		if ( K == 3 ) return c_fp32 ? (void*) exl3_mgemm_gv_kernel<3, true, 2, false> : (void*) exl3_mgemm_gv_kernel<3, false, 2, false>;
		if ( K == 4 ) return c_fp32 ? (void*) exl3_mgemm_gv_kernel<4, true, 2, false> : (void*) exl3_mgemm_gv_kernel<4, false, 2, false>;
		if ( K == 5 ) return c_fp32 ? (void*) exl3_mgemm_gv_kernel<5, true, 2, false> : (void*) exl3_mgemm_gv_kernel<5, false, 2, false>;
	}
	return nullptr;
}

// v2 (EXL3_SM70_CB0V2): K2/cb0 batch-pipelined inner, A staged as float
template __global__ void exl3_mgemm_gv_kernel<2, false, 0, true>( EXL3_MGEMM_ARGS );
template __global__ void exl3_mgemm_gv_kernel<2, true, 0, true>( EXL3_MGEMM_ARGS );

void* get_mgemm_gv2_kernel( int K, bool c_fp32, int cb ) {
	if ( K == 2 && cb == 0 ) {
		return c_fp32 ? (void*) exl3_mgemm_gv_kernel<2, true, 0, true> : (void*) exl3_mgemm_gv_kernel<2, false, 0, true>;
	}
	return nullptr;
}

// Instantiations for SM70 m=1 GEMV-class single gemm restructure (exl3_gemm_gv_kernel)
template __global__ void exl3_gemm_gv_kernel<2, false, 0, false>( EXL3_GEMM_ARGS );
template __global__ void exl3_gemm_gv_kernel<2, false, 2, false>( EXL3_GEMM_ARGS );
template __global__ void exl3_gemm_gv_kernel<3, false, 0, false>( EXL3_GEMM_ARGS );
template __global__ void exl3_gemm_gv_kernel<3, false, 2, false>( EXL3_GEMM_ARGS );
template __global__ void exl3_gemm_gv_kernel<4, false, 0, false>( EXL3_GEMM_ARGS );
template __global__ void exl3_gemm_gv_kernel<4, false, 2, false>( EXL3_GEMM_ARGS );
template __global__ void exl3_gemm_gv_kernel<5, false, 2, false>( EXL3_GEMM_ARGS );
template __global__ void exl3_gemm_gv_kernel<6, false, 2, false>( EXL3_GEMM_ARGS );
template __global__ void exl3_gemm_gv_kernel<7, false, 2, false>( EXL3_GEMM_ARGS );
template __global__ void exl3_gemm_gv_kernel<2, true, 0, false>( EXL3_GEMM_ARGS );
template __global__ void exl3_gemm_gv_kernel<2, true, 2, false>( EXL3_GEMM_ARGS );
template __global__ void exl3_gemm_gv_kernel<3, true, 0, false>( EXL3_GEMM_ARGS );
template __global__ void exl3_gemm_gv_kernel<3, true, 2, false>( EXL3_GEMM_ARGS );
template __global__ void exl3_gemm_gv_kernel<4, true, 2, false>( EXL3_GEMM_ARGS );
template __global__ void exl3_gemm_gv_kernel<5, true, 2, false>( EXL3_GEMM_ARGS );
template __global__ void exl3_gemm_gv_kernel<6, true, 2, false>( EXL3_GEMM_ARGS );
template __global__ void exl3_gemm_gv_kernel<7, true, 2, false>( EXL3_GEMM_ARGS );

void* get_gemm_gv_kernel( int K, bool c_fp32, int cb ) {
	if ( cb == 0 ) {
		if ( K == 2 ) return c_fp32 ? (void*) exl3_gemm_gv_kernel<2, true, 0, false> : (void*) exl3_gemm_gv_kernel<2, false, 0, false>;
		if ( K == 3 ) return c_fp32 ? (void*) exl3_gemm_gv_kernel<3, true, 0, false> : (void*) exl3_gemm_gv_kernel<3, false, 0, false>;
		if ( K == 4 ) return c_fp32 ? (void*) exl3_gemm_gv_kernel<4, true, 0, false> : (void*) exl3_gemm_gv_kernel<4, false, 0, false>;
	}
	if ( cb == 2 ) {
		if ( K == 2 ) return c_fp32 ? (void*) exl3_gemm_gv_kernel<2, true, 2, false> : (void*) exl3_gemm_gv_kernel<2, false, 2, false>;
		if ( K == 3 ) return c_fp32 ? (void*) exl3_gemm_gv_kernel<3, true, 2, false> : (void*) exl3_gemm_gv_kernel<3, false, 2, false>;
		if ( K == 4 ) return c_fp32 ? (void*) exl3_gemm_gv_kernel<4, true, 2, false> : (void*) exl3_gemm_gv_kernel<4, false, 2, false>;
		if ( K == 5 ) return c_fp32 ? (void*) exl3_gemm_gv_kernel<5, true, 2, false> : (void*) exl3_gemm_gv_kernel<5, false, 2, false>;
		if ( K == 6 ) return c_fp32 ? (void*) exl3_gemm_gv_kernel<6, true, 2, false> : (void*) exl3_gemm_gv_kernel<6, false, 2, false>;
		if ( K == 7 ) return c_fp32 ? (void*) exl3_gemm_gv_kernel<7, true, 2, false> : (void*) exl3_gemm_gv_kernel<7, false, 2, false>;
	}
	return nullptr;
}

// v2 (EXL3_SM70_CB0V2): K2/cb0 batch-pipelined inner, A staged as float
template __global__ void exl3_gemm_gv_kernel<2, false, 0, true>( EXL3_GEMM_ARGS );
template __global__ void exl3_gemm_gv_kernel<2, true, 0, true>( EXL3_GEMM_ARGS );

void* get_gemm_gv2_kernel( int K, bool c_fp32, int cb ) {
	if ( K == 2 && cb == 0 ) {
		return c_fp32 ? (void*) exl3_gemm_gv_kernel<2, true, 0, true> : (void*) exl3_gemm_gv_kernel<2, false, 0, true>;
	}
	return nullptr;
}
