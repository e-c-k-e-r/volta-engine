#pragma once

#include <cuda_fp16.h>

void had_r_128(
	const half* input,
	half* output,
	const half* pre_scale,
	const half* post_scale,
	float scale,
	int rows,
	int cols,
	cudaStream_t stream = nullptr
);
