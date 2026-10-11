#pragma once

#include "engine/config.h"
#include <cstddef>
#include <vector>

// MiniMax-H3 audio VAE decoder (BigVGAN with anti-aliased SnakeBeta activations), FP32, CUDA only.
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

	int sample_rate() const { return 32000; }

private:
	struct Impl;
	Impl* p_;
};

} // namespace engine::video
