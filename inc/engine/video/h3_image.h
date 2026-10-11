#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Reference-image helpers of the H3 pipeline (host side).
namespace engine::video {

// PIL-compatible Lanczos resize (support 3, antialiased on downscale, 8-bit intermediate between the horizontal and vertical pass); rgb [h, w, 3]
std::vector<uint8_t> h3_resize_lanczos( const uint8_t* rgb, int in_h, int in_w, int out_h, int out_w );

// ImageNet-normalized CHW float input of the VAE encoder from an RGB8 image
std::vector<float> h3_vae_input( const uint8_t* rgb, int h, int w );

// Image encoder output (moments [48, h, w]: 24 mean + 24 log-variance channels, host) -> condition rows [h/2 * w/2, 96]:
// sample = mean + exp(0.5 logvar) * noise (rounded to FP16 like the reference), normalized with the latent statistics, patchified (1, 2, 2),
// then mixed with fresh noise at timestep noise_aug (rows = noise_aug * clean + (1 - noise_aug) * noise).
std::vector<float> h3_keyframe_rows( const std::vector<float>& moments, int h, int w, const std::vector<float>& latent_mean, const std::vector<float>& latent_std,
	uint64_t seed, float noise_aug );

// The same for a clip: moments [T][48][h][w] (host) -> rows [T * h/2 * w/2, 96] in (t, hh, ww) order
std::vector<float> h3_condition_rows( const std::vector<float>& moments, int T, int h, int w, const std::vector<float>& latent_mean, const std::vector<float>& latent_std,
	uint64_t seed, float noise_aug );

// Qwen3-VL video input of a reference video: the frames sampled for the language model (RGB8 [h, w, 3] each) -> per temporal block (two frames) the vision-tower
// patches [grid_h * grid_w, 1536] (merge-block order), after the processor's smart resize (24M pixel budget over the clip)
struct H3VideoBlocks {
	int grid_h = 0, grid_w = 0;
	std::vector<std::vector<float>> patches;
};
H3VideoBlocks h3_video_blocks( const std::vector<const uint8_t*>& frames, int h, int w );

} // namespace engine::video
