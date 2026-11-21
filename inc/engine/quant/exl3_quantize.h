#pragma once

// EXL3 quantization of linear weights on the GPU (src/kernels/cuda/trellis/quant/exl3_quantize.cu), the algorithm of exllamav3's quantize_exl3:
// the input Hessian damped, sign-flipped and rotated by the 128-point Hadamard on both sides and factored into a block LDL (16-row blocks);
// each weight scaled per output and input channel, rotated, and scaled by a searched global factor; the 16-row blocks quantized from the last to
// the first with the LDL error feedback, through the trellis encoder (engine/quant/exl3_encode.h); the per-channel scales refitted to the
// quantized weight in the Hessian metric. Projections that share an input share the damped Hessian and its sign flips; each factors it with its
// own input scales folded in (scaled_hessian, exllamav3 1.6.7's default: the feedback then weights a row by what its error costs at the output),
// or all share one factor of the sign-flipped Hessian. Since the feedback runs along the input dimension only, the columns of all of them go
// through the encoder together, block row by block row.
#include "engine/backend/types.h"
#include <cstdint>
#include <vector>

namespace engine::quant {

struct Exl3QuantOptions {
	int code = 4;             // width code (engine/modules/trellis_bits.h)
	int cb = 2;               // codebook: 0 3INST, 1 mcg, 2 mul1
	bool volta = true;        // tile order of the packed trellis
	float sigma_reg = 0.025f; // diagonal damping, a share of the mean Hessian diagonal
	int out_scales = -1;      // per-output-channel scales: -1 auto (on), 0 off, 1 on
	bool refit = true;        // refit the channel scales after quantization (needs a Hessian)
	bool scaled_hessian = true; // one factor per projection with its input scales folded in; false: the input's shared factor (needs one)
	bool ldl_metric = false;  // the Viterbi weights each row's squared error by its block's LDL diagonal (CurveTQ's metric; needs scaled_hessian)
	bool open_trellis = false; // no tail-biting ring: a free start state per tile (the bits before its stream), Exl3Tensor::starts
	bool tile_gain = false;   // a gain per tile, the least-squares one (in the LDL metric with ldl_metric), 8 bits: g = (896 + c) / 1024, applied
	                          // before the error feedback sees the tile; Exl3Tensor::gains
	float scale_hint = 0.0f;  // where the global scale search starts (a like tensor's g_scale): the fine grid around it, the coarse grid only when its
	                          // minimum is at an edge; 0: the coarse grid first. A group starts each projection at the one before's
	int scale_search = 2;     // from a hint: 2 the fine grid of five on the whole sample, 1 three scales on a one-wide diagonal (five when the
	                          // minimum is at an edge), 0 the hint as it is; without a hint always the coarse grid, then the fine one
};

// The input side shared by the projections of one input. Without a usable Hessian (no rows, or a diagonal that is zero or not finite) the
// projections are quantized without error feedback
struct Exl3InputFactor {
	int k = 0;
	bool fallback = true;
	float* H = nullptr;       // damped Hessian, original basis, k x k (device)
	float* L = nullptr;       // shared block-LDL factor of P S H S P with zero diagonal blocks, k x k (device); only with shared_ldl
	float* su_signs = nullptr;// k (device), +1 / -1
};

// h_sum: the k x k sum of x^T x over `rows` input rows (device, as the engine's capture writes it), or nullptr. shared_ldl: also factor the
// sign-flipped Hessian once for every projection of the input (Exl3QuantOptions::scaled_hessian = false). False when the device memory or the
// factorization fails after the damping retries
bool exl3_input_factor( Exl3InputFactor& f, const float* h_sum, int k, int64_t rows, float sigma_reg, uint64_t seed, backend::stream_t stream,
	bool shared_ldl = false );
void exl3_input_factor_free( Exl3InputFactor& f );

// One quantized projection
struct Exl3Tensor {
	int k = 0, n = 0;
	std::vector<uint16_t> trellis;  // k / 16 x n / 16 x trellis_tile_u16( code )
	std::vector<uint16_t> suh;      // k fp16 bits (input scales and signs)
	std::vector<uint16_t> svh;      // n fp16 bits (output scales and signs)
	double proxy_err = 0.0;         // tr(E^T H E) / tr(W^T H W) of what is stored, the damping taken out of H; fallback: |E|^2 / |W|^2
	float g_scale = 1.0f;
	std::vector<uint16_t> starts;   // open_trellis: k / 16 x n / 16 start states (the 16 - K bits before a tile's stream, in its low bits)
	std::vector<uint8_t> gains;     // tile_gain: k / 16 x n / 16 gain codes
};

// Seconds per phase of exl3_quantize_group, added to (each phase boundary then synchronizes the stream)
struct Exl3QuantTimes {
	double regularize = 0.0;   // channel scales, rotations and the global scale search (its trial encodes)
	double factor = 0.0;       // each projection's scaled Hessian factor (scaled_hessian)
	double ldlq = 0.0;         // the blocks through the encoder with the error feedback
	double finish = 0.0;       // the back rotation, the scale refit and proxy error, the packing and the copies out
};

// weights[i]: device fp32 k x ns[i], input-major (the transpose of a [out, in] checkpoint tensor); ns[i] % 128 == 0, k % 128 == 0. seeds[i]
// draws the output sign flips. The weights are not modified. times: the phases' seconds added there (null: not timed)
bool exl3_quantize_group( const Exl3InputFactor& f, const std::vector<const float*>& weights, const std::vector<int>& ns,
	const std::vector<uint64_t>& seeds, const Exl3QuantOptions& o, std::vector<Exl3Tensor>& out, backend::stream_t stream,
	Exl3QuantTimes* times = nullptr );

// tr(E^T H E) / tr(W^T H W) with E = W - Q, from device fp32 W, Q (k x n) and H (k x k, e.g. a mean Hessian)
double exl3_proxy_error( const float* W, const float* Q, const float* H, int k, int n, backend::stream_t stream );

} // namespace engine::quant
