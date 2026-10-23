#pragma once

#include "engine/config.h"
#include <cstddef>
#include <vector>

// MiniMax-H3 audio VAE: reference-audio encoder (DAC conv stack + causal attention projection) and decoder (BigVGAN with anti-aliased SnakeBeta activations), FP32, CUDA only.
namespace engine::video {

class H3AudioVAE {
public:
	H3AudioVAE();
	~H3AudioVAE();
	H3AudioVAE( const H3AudioVAE& ) = delete;
	H3AudioVAE& operator=( const H3AudioVAE& ) = delete;

	// dir: the audio_vae component directory (config.json with the latent statistics, model.safetensors)
	bool load( const char* dir, int device, int max_frames = 1024 );

	// rows: the denoised audio rows [2 * frames, 32] (device, fp32, normalized) in the DiT's packed order (channel-major).
	// pcm: stereo samples [2][frames * 800] at 32 kHz, clamped to [-1, 1] (host).
	bool decode( const float* d_rows, int frames, std::vector<float>& pcm );

	// Reference-audio encoder: wave = stereo samples [2][samples] at 32 kHz (host). rows: normalized latent rows [2 * frames, 32] in the DiT's packed order
	// (channel-major); frames = ceil(samples / 800). Loads the encoder weights on first use.
	bool encode( const std::vector<float>& wave, int samples, std::vector<float>* rows, int* frames );

private:
	struct Impl;
	Impl* p_;
};

} // namespace engine::video
