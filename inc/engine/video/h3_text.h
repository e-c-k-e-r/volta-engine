#pragma once

#include "engine/config.h"
#include "engine/video/h3_tp.h"
#include <cstddef>
#include <string>
#include <functional>
#include <vector>

// MiniMax-H3 text conditioning: the Qwen3-VL text tower's first 50 decoder layers (no final norm) over plain prompt tokens.
// BF16 weights stay on disk (mmap) and are streamed to the GPU one layer at a time (the layer is converted to FP16 on the device),
// residual stream FP32, FP16 tensor-core GEMMs. CUDA only.
namespace engine::video {

// Vision features of one reference image (from H3VisionTower) as they enter the language model
struct H3ImageFeatures {
	int grid_h = 0, grid_w = 0;      // patch grid (multiples of 2)
	std::vector<float> merged;       // [tokens, 5120]
	std::vector<float> deepstack[3]; // [tokens, 5120], added to the stream after decoder layers 0 / 1 / 2
};

class H3TextEncoder {
public:
	H3TextEncoder();
	~H3TextEncoder();
	H3TextEncoder( const H3TextEncoder& ) = delete;
	H3TextEncoder& operator=( const H3TextEncoder& ) = delete;

	// dir: the text_encoder component directory (config.json, model.safetensors.index.json, shards)
	// With an EXL3 checkpoint (trellis projections) the layers stay resident; tp_ranks > 1 loads this rank's shard (16 of 64 query heads, 2 of 8 KV heads and a quarter
	// of the MLP width at four ranks) and needs set_tp() before encode (see H3TextGroup).
	bool load( const char* dir, int device, int max_tokens = 2048, int layers = 50, int tp_rank = 0, int tp_ranks = 1 );
	void set_tp( H3TpComm comm );

	// Hidden states after `layers` decoder layers for the token ids: d_out [ids.size(), 5120] (device, fp32).
	bool encode( const std::vector<int>& ids, float* d_out );
	// With image placeholders (image_token_id runs in `ids`, one run per entry of `images`, in order): image rows replace the embeddings, 3-D (interleaved
	// mRoPE) positions follow Qwen3-VL's get_rope_index, DeepStack features are added after layers 0..2.
	bool encode_mm( const std::vector<int>& ids, int image_token_id, const std::vector<H3ImageFeatures>& images, float* d_out );

	int hidden() const { return 5120; }

	// Debug hook: called after every decoder layer with the residual stream [S, 5120] (device, fp32)
	std::function<void( int layer, const float* d_x, int S )> probe;

private:
	struct Impl;
	Impl* p_;
};

} // namespace engine::video
