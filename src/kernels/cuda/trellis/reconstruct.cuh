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

// With the tile companions of an extended trellis: starts (uint16 per tile, the bits before its stream) and gains (uint8 per tile), either null
void reconstruct_slice_ext( half* unpacked, const uint16_t* packed, const uint16_t* starts, const uint8_t* gains, int K, bool mcg, bool mul1, int rows,
	int cols, int64_t n_offset, cudaStream_t stream );
void reconstruct_had_slice_ext( half* unpacked, const uint16_t* packed, const uint16_t* starts, const uint8_t* gains, const half* suh, const half* svh,
	int K, bool mcg, bool mul1, int rows, int cols, int64_t n_offset, cudaStream_t stream );
// count whole [rows, cols] tensors of one format with unit scales into FP32 (half_out: FP16): table (device) holds 4 * count pointers, the
// codes of every entry, then their starts, gains (either null) and outputs
void reconstruct_had_batch( const void* const* table, int count, int K, bool mcg, bool mul1, int rows, int cols, bool half_out, cudaStream_t stream );
// Rows k0 .. k0 + rows and columns n0 .. n0 + cols (multiples of 128) of a tensor stride columns wide with unit scales, into [rows, cols] FP32
// (half_out: FP16)
void reconstruct_had_block( void* out, const uint16_t* packed, const uint16_t* starts, const uint8_t* gains, int K, bool mcg, bool mul1, int k0, int rows,
	int n0, int cols, int stride, bool half_out, cudaStream_t stream );
// The gradient of the tile gains over rows k0 .. k0 + rows and columns n0 .. n0 + cols (k0, rows multiples of 16; n0, cols of 128) of a tensor
// stride columns wide: per tile dg[tile] += sum over it of dT o R / g, R the tile's decoded values (its gain g applied, unit scales, before the
// Hadamard rotations: the core is H R H in 128-blocks), dT [rows, cols] FP32 the gradient with respect to R; dg per tile of the whole tensor
// (k / 16 x stride / 16, the gains' order)
void tile_gain_grad( const float* dT, float* dg, const uint16_t* packed, const uint16_t* starts, const uint8_t* gains, int K, bool mcg, bool mul1, int k0,
	int rows, int n0, int cols, int stride, cudaStream_t stream );
// count whole [rows, cols] tensors of one format: table (device) holds 5 * count pointers, the codes of every entry, then their starts, gains,
// dT and dg
void tile_gain_grad_batch( const void* const* table, int count, int K, bool mcg, bool mul1, int rows, int cols, cudaStream_t stream );
