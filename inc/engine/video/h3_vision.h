#pragma once

#include "engine/config.h"
#include <cstddef>
#include <vector>

// Vision tower of the H3 text encoder (Qwen3-VL ViT: 27 blocks, 2-D rotary, patch merger, DeepStack taps after blocks 8 / 16 / 24). Dense BF16 weights
// converted to FP16 on load; built on the engine's Vision operators. CUDA only.
namespace engine::video {

class H3VisionTower {
public:
	H3VisionTower();
	~H3VisionTower();
	H3VisionTower( const H3VisionTower& ) = delete;
	H3VisionTower& operator=( const H3VisionTower& ) = delete;

	// dir: the text_encoder component directory (model.visual.* tensors)
	bool load( const char* dir, int device, int max_patches = 16384 );

	// patches: [grid_h * grid_w, 1536] fp32 in merge-block order (engine::loader::PreparedImage).
	// merged: [tokens, 5120]; deepstack[i]: [tokens, 5120] (tokens = patches / 4); fp32 on the host.
	bool encode( const std::vector<float>& patches, int grid_h, int grid_w, std::vector<float>* merged, std::vector<float>* deepstack );

	int max_patches() const;

private:
	struct Impl;
	Impl* p_;
};

} // namespace engine::video
