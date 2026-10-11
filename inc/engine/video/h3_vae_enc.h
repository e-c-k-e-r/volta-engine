#pragma once

#include "engine/config.h"
#include <cstddef>
#include <vector>

// MiniMax-H3 video VAE encoder for single images (keyframe conditioning), FP32, CUDA only.
// The checkpoint's causal 3-D encoder sees one frame, so every 3-D conv reduces to its last temporal tap (a 2-D conv with reflect padding).
namespace engine::video {

class H3ImageEncoder {
public:
	H3ImageEncoder();
	~H3ImageEncoder();
	H3ImageEncoder( const H3ImageEncoder& ) = delete;
	H3ImageEncoder& operator=( const H3ImageEncoder& ) = delete;

	// dir: the video_vae component directory (source/model.safetensors)
	bool load( const char* dir, int device );

	// rgb: [3, H, W] fp32 (device), already normalized with the ImageNet mean / std; H and W multiples of 16.
	// moments: [48, H/16, W/16] (device, fp32): 24 mean channels followed by 24 log-variance channels.
	// Images larger than 256 px are encoded in overlapping 256 px tiles with linearly blended moments, as the checkpoint does.
	bool encode( const float* d_rgb, int H, int W, float* d_moments );

private:
	bool encode_tile( const float* d_rgb, int H, int W, float* d_moments );
	struct Impl;
	Impl* p_;
};

} // namespace engine::video
