#pragma once

#include "engine/config.h"
#include <cstddef>
#include <cstdint>
#include <vector>

// MiniMax-H3 visual VAE decoder: 24-channel latents -> RGB frames (ViT3D decoder, 16x spatial / 4x temporal), CUDA only.
// Port of the checkpoint's AutoencoderKLLegacy.decode_base for a 3D-conv/ViT-decoder config: spatial tiling with linear blending,
// temporal chunks of 5 latent frames with 2 tokens of overlap, ImageNet pixel de-normalization.
namespace engine::video {

class H3StageGroup;

struct H3VaeParams {
	int vae_ratio = 16;
	int vae_ratio_t = 4;
	int clip_length = 17;
	int token_drop = 3;
	int tile_size = 256;
	int tile_overlap_min = 64;
	std::vector<float> mean, stdev; // latents_mean / latents_std (24)
};

// Spatial tiling of the checkpoint (shared by the decoder and the image encoder): tile starts / lengths / overlaps along one axis, in pixels
struct H3Tiles {
	std::vector<int> start, len, overlap;
};
H3Tiles h3_split_tiles( int input_len, int tile_size, int overlap_min, int ratio );

// Reads <dir>/config.json (vae_* fields, latent statistics)
bool h3_read_vae_params( const char* dir, H3VaeParams& out );

class H3VideoVAE {
public:
	H3VideoVAE();
	~H3VideoVAE();
	H3VideoVAE( const H3VideoVAE& ) = delete;
	H3VideoVAE& operator=( const H3VideoVAE& ) = delete;

	// dir: the video_vae component directory (config.json, source/model.safetensors)
	// `group` (optional): the decoders of several GPUs loading in parallel share one read of the weight file (see h3_stager.h); every member must be loaded
	// by its own thread with its own rank, and all must succeed or fail together
	bool load( const char* dir, int device, int max_tokens = 4096, H3StageGroup* group = nullptr, int rank = 0 );

	// Raw ViT decoder on one tile of denormalized latents d_z [24, T, H, W] (device, fp32); pixels [3, 4T, 16H, 16W] (device, fp32) in the
	// network's normalized space (before the ImageNet de-normalization).
	void decode_tile( const float* d_z, int T, int H, int W, float* d_pix );

	// Full decode of a DiT latent z [24, T, H, W] (device, fp32, as produced by unpatchifying the denoised rows): returns RGB uint8 frames
	// [F, 16H, 16W, 3] on the host.
	bool decode_video( const float* d_z, int T, int H, int W, std::vector<uint8_t>& rgb, int& frames );
	// The same with the tiles of every temporal clip decoded by several decoders (vaes[0] on the GPU that holds d_z and assembles the frames, the others on other GPUs).
	static bool decode_video( const std::vector<H3VideoVAE*>& vaes, const float* d_z, int T, int H, int W, std::vector<uint8_t>& rgb, int& frames );

	const H3VaeParams& params() const;
	size_t weight_bytes() const;

private:
	struct Impl;
	Impl* p_;
};

} // namespace engine::video
