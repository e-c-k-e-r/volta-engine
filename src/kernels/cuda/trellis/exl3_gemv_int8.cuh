#pragma once

// Host dispatch for the fused int8-activation GEMV (mul1 cb 2 tensors), ported from exllamav3's
// exl3_gemv_int8.cu / exl3_gemv_int8.cuh and adapted to our CUDA-graph decode path.
//
// Mode 1 (fixed): int8 + error-feedback residual pass (~15-16 bit effective activation precision, size_m <= 2).
//
// Only the per-slice-scale kernel is ported (plain launch, fixed workspace, no grid.sync), so it
// captures into static graphs. The kernel argument list matches exl3_gemv's, with `locks` replaced by
// the int8 workspace pointer.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Highest K the int8 path accepts. Above it the regular fp16 tensor-core kernel wins.
int exl3_gemv_int8_max_k( int device );

// True if the mode enables the path at all.
bool exl3_gemv_int8_enabled();

// Set shared-memory attributes and warm the occupancy caches. Must run before any capture.
void exl3_gemv_int8_init( int device );

// Returns true if the operation was handled (false -> caller falls through to the regular kernel).
bool exl3_gemv_int8(
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
	int device,
	cudaStream_t stream
);
