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

// count (2 to 4) projections in one launch, tensor t of input A[t] [size_m, size_k], where exl3_gemv would take the int8 GEMV for each:
// every tensor of width K and the codebook flags, size_n[t] outputs, multiples of 256. False when the call does not qualify (nothing enqueued)
bool exl3_gemv_multi(
	const half* const* A,
	int count,
	const uint16_t* const* B,
	void* const* C,
	const half* const* suh,
	const half* const* svh,
	const int* size_n,
	int size_m,
	int size_k,
	int K,
	bool mcg,
	bool mul1,
	bool c_fp32,
	cudaStream_t stream
);

#if defined(USE_ROCM)
#include "../../rocm/trellis/exl3_rocm.cuh"
#endif
