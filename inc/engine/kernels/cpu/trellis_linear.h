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
	// Only the rotations (prepare_block / finish_block) of a matrix whose codes were decoded with dequantize_panels; run() must not be called
	TrellisLinear( int kt, int nt, int bits, const uint16_t* suh, const uint16_t* svh );
	// The matrix as dense FP16 for a product by a block of rows (a vision tower's thousands of patches, where the windows would be decoded once per
	// few rows): panels[nt][kt * 16][16], the codebook value of every weight of the tiles in the basis run() multiplies in, so that
	// prepare_block, a dense product with the panels and finish_block give the same linear as run(). trellis: the checkpoint's [kt][nt][16 * bits] codes,
	// in the Volta tile order when `volta` (tile_layout volta of the checkpoint; run() reads only the stock one)
	static void dequantize_panels( const uint16_t* trellis, int kt, int nt, int bits, bool volta, uint16_t* panels );
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
