#pragma once

#include <cuda_fp16.h>
#include <cstdint>

bool exl3_gemv_try_launch(
	void** kernel_args,
	int size_m,
	int size_k,
	int size_n,
	int K,
	int cb,
	bool c_fp32,
	bool has_su_sv,
	int device,
	cudaStream_t stream,
	void** launched_kernel = nullptr,
	bool force = false
);

void exl3_gemv(
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
	bool c_fp32 = false,
	cudaStream_t stream = nullptr
);

#if defined(USE_ROCM)
// ROCm: plain-launch v_dot2 GEMV (exl3_gemv_rocm.cu) for 1..EXL3_GEMV_ROCM_MAX_M rows, bits 1-8, every codebook, FP16 output with
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
#endif
