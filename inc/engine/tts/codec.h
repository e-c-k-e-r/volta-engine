// The NeMo 44.1 kHz audio codec (nvidia/audio-codec-44khz): a HiFi-GAN encoder, FSQ (8 groups of 4 dimensions) and a
// HiFi-GAN decoder, from a vall-e export (python -m vall_e.export codec: FP32 weights with the weight norm folded in, the
// layout in the metadata).
// The decoder: the dequantized frames [32, T] through an input conv, then per up-sample stage a HalfSnake (Snake on the
// first half of the channels, leaky ReLU on the rest), a transposed conv (stride s, kernel 2s) and a residual layer (the
// mean of one branch per kernel size, each a chain of residual blocks over the dilations), then a HalfSnake, a
// one-channel output conv and a clamp to [-1, 1]. The encoder mirrors it with leaky ReLUs: an input conv, then per stage
// a residual layer, the activation and a strided conv (stride s, kernel 2s), then the activation, an output conv to 32
// channels and FSQ. Convolutions pad by reflection, transposed ones with zeros. Everything runs in FP32, as vall_e does;
// each convolution is GEMMs over chunks of time.
#pragma once

#include "engine/core/arena.h"
#include "engine/loader/safetensors.h"
#include <cstdint>
#include <string>
#include <vector>

namespace engine::tts {

struct CodecConfig {
	int sample_rate = 44100, samples_per_frame = 512, groups = 8;
	float fsq_eps = 1e-3f;
	std::vector<int> levels;  // an FSQ group's levels per dimension
	int enc_channels = 48, enc_dim = 32, enc_in_kernel = 7, enc_out_kernel = 7;
	int dec_dim = 32, dec_channels = 768, dec_in_kernel = 7, dec_out_kernel = 3;
	std::vector<int> down_rates, up_rates, enc_res_kernels, enc_res_dilations, dec_res_kernels, dec_res_dilations;
};

class NemoCodec {
public:
	bool load( const std::string& path, int device = 0 );
	// digits [n_frames * 32] (frame-major FSQ level indices, as TtsModel::generate returns) -> [n_frames * samples_per_frame]
	std::vector<float> decode( const std::vector<uint8_t>& digits, int n_frames );
	// audio (44.1 kHz mono, [-1, 1]) -> digits [n_frames * 32], zero-padded to whole frames first (as AudioCodecModel.encode)
	std::vector<uint8_t> encode( const std::vector<float>& audio );
	// The activations for n_frames frames either way (each the largest [C, T]: ~0.5 MB a frame in all, 42 MB a second); kept and grown on demand by
	// decode / encode, which return nothing when the GPU has no room for them. False if it hasn't
	bool reserve( int n_frames );
	const CodecConfig& config() const { return cfg_; }
	~NemoCodec();

private:
	struct Conv {  // Conv1d [co, ci, k], reflect padding: (k - 1) / 2 * dil, or (k - stride + 1) / 2 for a strided one
		const float *w = nullptr, *b = nullptr;
		int co = 0, ci = 0, k = 0, dil = 1, stride = 1, pad = 0;
	};
	struct UpConv {  // ConvTranspose1d [ci, co, 2s] regrouped by output phase: [s * co, 3 * ci] (input shifts -1, 0, +1)
		const float *w = nullptr, *b = nullptr;
		int co = 0, ci = 0, s = 1;
	};
	struct ResBlock { Conv in, skip; const float *alpha_in = nullptr, *alpha_skip = nullptr; };  // alphas null: leaky ReLU
	using ResLayer = std::vector<std::vector<ResBlock>>;  // [kernel size][dilation]
	struct DecStage { const float* alpha = nullptr; UpConv up; ResLayer res; };
	struct EncStage { ResLayer res; Conv down; };

	int conv( const Conv& c, const float* x, float* y, int T );  // returns the output length
	void up_conv( const UpConv& u, const float* x, float* y, int T_in );
	void act( const float* x, float* y, const float* alpha, int C, int T );
	void res_layer( const ResLayer& r, int C, int T );  // x_ in place (cur_, a_, b_, acc_ as scratch)

	CodecConfig cfg_;
	Conv enc_pre_, enc_post_, dec_pre_, dec_post_;
	std::vector<EncStage> enc_;
	std::vector<DecStage> dec_;
	const float* dec_post_alpha_ = nullptr;
	pod::SafeTensorsFile st_;
	pod::MemoryArena arena_ {};
	// activations (each the largest [C, T] of either half) and the GEMM chunks' columns / outputs
	float *x_ = nullptr, *cur_ = nullptr, *a_ = nullptr, *b_ = nullptr, *acc_ = nullptr, *cols_ = nullptr, *ph_ = nullptr;
	uint8_t* digits_ = nullptr;
	int frames_ = 0;
};

} // namespace engine::tts
