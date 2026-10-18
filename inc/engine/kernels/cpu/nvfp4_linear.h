#pragma once

#include "engine/kernels/cpu/linear.h"
#include <cstddef>
#include <vector>

namespace engine::kernels::cpu {

// NVFP4 W4A16 linear: E2M1 codes with an E4M3 scale per 16 weights of a column and one FP32 global scale. The prepacked codes of src/loader/nvfp4.cpp
// ([tile][group][lane], 32 columns per tile, 16-wide K groups) are read in place, so a tier's pinned copy serves the GPU and the CPU alike; only the
// scales are copied (1/8 of the codes), in the order the kernel wants them. The kernel shuffles the lanes of a tile into one dword per column, looks the
// nibbles up one by one (a permute gives the magnitude, a shifted bit the sign) and accumulates in FP32; each group's partial is scaled once.
// No input or output transform.
class Nvfp4Linear : public Linear {
public:
	// codes: [N / 32][K / 16][32][8] bytes, not copied and not owned (must outlive this object); scales: [N / 32][K / 16][32] E4M3 bytes, copied.
	// K a multiple of 64, N of 128
	Nvfp4Linear( const uint8_t* codes, const uint8_t* scales, float global_scale, int k, int n );
	// The codes moved (the tier swapped the expert's buffer)
	void rebind( const uint8_t* codes ) { codes_ = codes; }
	int in() const override { return k_; }
	int out() const override { return n_; }
	void prepare_block( const float* x, int b, float* xin ) const override;
	void run( int rows, const float* xin, const float* row_sum, int u0, int u1, float* raw ) const override;
	void finish_block( float* v, int b ) const override;

private:
	int k_, n_;
	const uint8_t* codes_;
	std::vector<uint8_t> scales_;    // [unit][group][16] in the kernel's column order
	float lut_[256];                 // E4M3 byte -> scale * global / 2
};

}   // namespace engine::kernels::cpu
