#pragma once
// NVFP4 quantization of linear weights on the GPU (src/kernels/cuda/nvfp4/quant/nvfp4_quantize.cu), the modelopt layout: every 16 input values
// of an output column share an E4M3 scale, the values are E2M1 (0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives), one FP32 global scale per
// matrix (amax / (6 * 448)). With the input Hessian H = X^T X of the projection, block LDLQ as exllamav3's quantizer runs it without the trellis:
// H averaged over its rows and damped, H = L L^T, L scaled by the inverses of its 16 x 16 diagonal blocks (which are then dropped); the 16-row
// blocks of the input-major weight are quantized from the last to the first, each with the error the blocks after it left fed back through L.
// A block's quantizer searches each column's scale over the two E4M3 codes either side of the nearest one to amax / 6 for the least squared
// error; the values round to the nearest E2M1 magnitude (ties to the lower one).
#include "engine/backend/types.h"
#include <cstdint>
#include <vector>

namespace engine::quant {

struct Nvfp4QuantOptions {
	float sigma_reg = 0.025f;  // diagonal damping, a share of the mean Hessian diagonal (doubled on every failed factorization, up to 10 tries)
	bool rtn_proxy = true;     // also quantize without the feedback for its proxy error (Nvfp4Tensor::proxy_rtn)
};

// One quantized projection in the modelopt tensors' layout
struct Nvfp4Tensor {
	int k = 0, n = 0;
	std::vector<uint8_t> codes;     // .weight [n, k / 2]: two E2M1 codes a byte, the lower input index in the low nibble
	std::vector<uint8_t> scales;    // .weight_scale [n, k / 16] E4M3
	float global_scale = 0.0f;      // .weight_scale_2
	double proxy_err = 0.0;         // tr(E^T H E) / tr(W^T H W) with the undamped mean Hessian; without one |E|^2 / |W|^2
	double proxy_rtn = -1.0;        // the same for round to nearest (rtn_proxy), -1 otherwise
	bool fallback = false;          // round to nearest: no Hessian, or it did not factor
};

// The global scale amax( |w| ) / ( 6 * 448 ) of count device fp32 values
float nvfp4_global_scale( const float* w, size_t count, backend::stream_t stream );

// w: device fp32 k x n, input-major (the transpose of a [out, in] checkpoint tensor), not modified; k % 16 == 0. h_sum: the k x k sum of x^T x
// over `rows` input rows (device), or nullptr (round to nearest). global_scale <= 0: nvfp4_global_scale of w (several tensors that the engine
// stacks into one weight pass the largest of theirs). False when the device memory runs out
bool nvfp4_quantize( const float* w, int k, int n, const float* h_sum, int64_t rows, float global_scale, const Nvfp4QuantOptions& o, Nvfp4Tensor& out,
	backend::stream_t stream );

} // namespace engine::quant
