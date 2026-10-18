#pragma once

#include "engine/kernels/cpu/linear.h"
#include <cstddef>
#include <vector>

namespace engine::kernels::cpu {

// EXL3 trellis linear (MUL1 codebook, whole bits per weight, 1 to 8). The checkpoint's [kt, nt, 16 * bits] codes are repacked once into a
// layout the AVX2 kernel reads without gathers: the eight lanes of a vector are the eight output columns of a 16 x 16 tile, so every lane
// shares one shift and one input row. The kernel decodes a window, forms the codebook value from the byte sum of window * 0x83DCD12D and
// multiplies it into FP32 accumulators. Input: x * suh, Hadamard, 1/sqrt(128); output: Hadamard, 1/sqrt(128), svh
class TrellisLinear : public Linear {
public:
	static bool supported( int bits );
	static size_t repacked_bytes( int kt, int nt, int bits );
	// trellis: the codes [kt][nt][16 * bits] u16, read here only; suh [kt * 16], svh [nt * 16]: FP16 bit patterns, copied. storage: where the
	// repacked codes go (repacked_bytes bytes, caller-owned, e.g. NUMA-bound memory); null: owned by this object
	TrellisLinear( const uint16_t* trellis, int kt, int nt, int bits, const uint16_t* suh, const uint16_t* svh, void* storage = nullptr );
	int in() const override { return kt_ * 16; }
	int out() const override { return nt_ * 16; }
	int bits() const { return bits_; }
	bool needs_row_sum() const override { return true; }
	void prepare_block( const float* x, int b, float* xin ) const override;
	void run( int rows, const float* xin, const float* row_sum, int u0, int u1, float* raw ) const override;
	void finish_block( float* v, int b ) const override;

private:
	int kt_, nt_, bits_;
	const uint32_t* data_ = nullptr;
	std::vector<uint32_t> owned_;
	std::vector<float> suh_, svh_;
};

}   // namespace engine::kernels::cpu
