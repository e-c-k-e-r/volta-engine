#pragma once

#include "engine/config.h"
#include <cstddef>
#include <string>
#include <vector>

// MiniMax-H3 text conditioning: the Qwen3-VL text tower's first 50 decoder layers (no final norm) over plain prompt tokens.
// BF16 weights stay on disk (mmap) and are streamed to the GPU one layer at a time (the layer is converted to FP16 on the device),
// residual stream FP32, FP16 tensor-core GEMMs. CUDA only.
namespace engine::video {

class H3TextEncoder {
public:
	H3TextEncoder();
	~H3TextEncoder();
	H3TextEncoder( const H3TextEncoder& ) = delete;
	H3TextEncoder& operator=( const H3TextEncoder& ) = delete;

	// dir: the text_encoder component directory (config.json, model.safetensors.index.json, shards)
	bool load( const char* dir, int device, int max_tokens = 2048, int layers = 50 );

	// Hidden states after `layers` decoder layers for the token ids: d_out [ids.size(), 5120] (device, fp32).
	bool encode( const std::vector<int>& ids, float* d_out );

	int hidden() const { return 5120; }

private:
	struct Impl;
	Impl* p_;
};

} // namespace engine::video
