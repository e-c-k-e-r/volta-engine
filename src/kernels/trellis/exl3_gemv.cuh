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
