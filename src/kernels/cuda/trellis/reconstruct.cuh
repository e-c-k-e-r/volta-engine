#pragma once

#include <cuda_fp16.h>
#include <cstdint>

void reconstruct_slice(
	half* unpacked,
	const uint16_t* packed,
	int K,
	bool mcg,
	bool mul1,
	int rows,
	int cols,
	int64_t n_offset,
	cudaStream_t stream = nullptr
);

// cols columns from n_offset of a tensor stride columns wide, into an unpacked [rows * 16, cols] block
void reconstruct_cols(
	half* unpacked,
	const uint16_t* packed,
	int K,
	bool mcg,
	bool mul1,
	int rows,
	int cols,
	int64_t n_offset,
	int stride,
	cudaStream_t stream = nullptr
);

void reconstruct_had_slice(
	half* unpacked,
	const uint16_t* packed,
	const half* suh,
	const half* svh,
	int K,
	bool mcg,
	bool mul1,
	int rows,
	int cols,
	int64_t n_offset,
	cudaStream_t stream = nullptr
);
