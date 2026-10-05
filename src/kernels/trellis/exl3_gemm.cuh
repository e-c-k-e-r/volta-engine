#pragma once

#include <cuda_fp16.h>
#include <cstdint>

void exl3_gemm(
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

void exl3_gemm_init_all_attrs( int device );
