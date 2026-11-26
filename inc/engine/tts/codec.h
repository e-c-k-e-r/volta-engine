// The NeMo 44.1 kHz audio codec's decoder (nvidia/audio-codec-44khz: a HiFi-GAN generator), FSQ digits to audio, from a
// vall-e export (python -m vall_e.export codec: FP32 weights with the weight norm folded in, the layout in the metadata).
// The decoder: the dequantized frames [32, T] through an input conv, then per up-sample stage a HalfSnake (Snake on the
// first half of the channels, leaky ReLU on the rest), a transposed conv (stride s, kernel 2s) and a residual layer (the
// mean of one branch per kernel size, each a chain of residual blocks over the dilations), then a HalfSnake, a
// one-channel output conv and a clamp to [-1, 1]. Convolutions pad by reflection, transposed ones with zeros.
// Everything runs in FP32, as vall_e decodes; each convolution is GEMMs over chunks of time.
#pragma once

#include "engine/core/arena.h"
#include "engine/loader/safetensors.h"
#include <cstdint>
#include <string>
#include <vector>

namespace engine::tts {

struct CodecConfig {
	int sample_rate = 44100, input_dim = 32, base_channels = 768, in_kernel = 7, out_kernel = 3;
	std::vector<int> up_rates, res_kernels, res_dilations, levels;  // levels: an FSQ group's levels per dimension
	int groups = 8;
	int samples_per_frame() const;
};

class CodecDecoder {
public:
	bool load( const std::string& path, int device = 0 );
	// digits [n_frames * 32] (frame-major FSQ level indices, as TtsModel::generate returns) -> [n_frames * samples_per_frame]
	std::vector<float> decode( const std::vector<uint8_t>& digits, int n_frames );
	const CodecConfig& config() const { return cfg_; }
	~CodecDecoder();

private:
	struct Conv {  // Conv1d [co, ci, k] (reflect padding (k - 1) / 2 * dil)
		const float *w = nullptr, *b = nullptr;
		int co = 0, ci = 0, k = 0, dil = 1;
	};
	struct UpConv {  // ConvTranspose1d [ci, co, 2s] regrouped by output phase: [s * co, 3 * ci] (input shifts -1, 0, +1)
		const float *w = nullptr, *b = nullptr;
		int co = 0, ci = 0, s = 1;
	};
	struct ResBlock { Conv in, skip; const float *alpha_in = nullptr, *alpha_skip = nullptr; };
	struct Stage {
		const float* alpha = nullptr;  // the HalfSnake before the up-sample
		UpConv up;
		std::vector<std::vector<ResBlock>> branches;  // [kernel size][dilation]
	};

	void conv( const Conv& c, const float* x, float* y, int T );
	void up_conv( const UpConv& u, const float* x, float* y, int T_in );
	void half_snake( float* x, const float* alpha, int C, int T );
	void reserve( int n_frames );

	CodecConfig cfg_;
	Conv pre_, post_;
	std::vector<Stage> stages_;
	const float* post_alpha_ = nullptr;
	void* blas_ = nullptr;  // cublasHandle_t
	pod::SafeTensorsFile st_;
	pod::MemoryArena arena_ {};
	// activations (each the largest stage's [C, T]) and the GEMM chunks' columns / outputs
	float *x_ = nullptr, *cur_ = nullptr, *a_ = nullptr, *b_ = nullptr, *acc_ = nullptr, *cols_ = nullptr, *ph_ = nullptr;
	uint8_t* digits_ = nullptr;
	size_t act_elems_ = 0;
	int frames_ = 0;
};

} // namespace engine::tts
