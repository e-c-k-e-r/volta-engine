// ROCm EXL3 linears (src/kernels/rocm/trellis), declared through src/kernels/cuda/trellis/exl3_gemv.cuh on ROCm
#pragma once

// ROCm: plain-launch v_dot2 GEMV (exl3_gemv.cu in src/kernels/rocm/trellis) for 1..EXL3_GEMV_ROCM_MAX_M rows, bits 1-8, every codebook, FP16 output with
// the Hadamard pair. False when the call is outside that (the caller takes another path)
#define EXL3_GEMV_ROCM_MAX_M 8
bool exl3_gemv_rocm(
	const half* A,
	const uint16_t* B,
	void* C,
	const half* suh,
	half* A_had,
	const half* svh,
	int size_m,
	int size_k,
	int size_n,
	int K,
	bool mcg,
	bool mul1,
	bool c_fp32,
	cudaStream_t stream
);

// ROCm: the WMMA multi-row GEMM (exl3_gemm.cu in src/kernels/rocm/trellis) for EXL3_GEMV_ROCM_MAX_M < rows <= EXL3_GEMM_ROCM_MAX_M: the packed weight is read
// once (the dequantize + hipBLAS path writes and re-reads it as FP16). False when the call is outside that
#define EXL3_GEMM_ROCM_MAX_M 64
bool exl3_gemm_rocm(
	const half* A,
	const uint16_t* B,
	void* C,
	const half* suh,
	half* A_had,
	const half* svh,
	int size_m,
	int size_k,
	int size_n,
	int K,
	bool mcg,
	bool mul1,
	bool c_fp32,
	cudaStream_t stream
);
