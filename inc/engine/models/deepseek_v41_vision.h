#pragma once

#include "engine/core/arena.h"
#include "engine/core/tensor.h"
#include "engine/models/vision_tower.h"
#include <vector>

namespace engine::models {

struct DeepseekV41VisionConfig {
	int depth = 32;
	int hidden = 1024;
	int heads = 16;
	int intermediate = 2816;
	int patch_size = 14;
	int downsample = 3;
	int out_hidden = 5120;     // the language model's hidden size (the aligner's output)
	float rope_theta = 10000.0f;
	float eps = 1e-6f;

	int head_dim() const { return hidden / heads; }
	int patch_dim() const { return 3 * patch_size * patch_size; }
};

// The DeepSeek-V4.1 image encoder (inference/vision.py): a patch projection, RMSNorm + SwiGLU ViT blocks over the whole image with a fused
// qkv projection and 2-D rotary (half the rotated pairs by the row, half by the column), a final norm, then the aligner: downsample x downsample
// patches (raster order, zero padded at the edges) are concatenated channel-major and projected through two linears with GELU between. The
// weights are the checkpoint's BF16 vision.* / aligner.* tensors, kept as FP16. encode() returns the image's whole token span as the prompt
// carries it: [start] ([image] * llm_w + [newline]) * llm_h [end], the three delimiters being learned embeddings.
class DeepseekV41VisionTower : public VisionTower {
public:
	// False without a vision_config or without the vision tensors, or when the arena has no room
	bool load( const char* model_dir, pod::MemoryArena& arena, int max_image_tokens, backend::stream_t stream );
	bool loaded() const override { return loaded_; }
	const engine::loader::VisionParams& params() const override { return params_; }
	bool encode( const engine::loader::PreparedImage& img, std::vector<backend::half_t>* out, backend::stream_t stream ) override;

private:
	struct Block {
		pod::TensorView norm1_w, norm2_w;
		pod::TensorView wqkv, qkv_b, wo, wo_b, w1, w2;
	};

	bool loaded_ = false;
	DeepseekV41VisionConfig cfg_;
	engine::loader::VisionParams params_;
	int max_patches_ = 0, max_tokens_ = 0;
	std::vector<Block> blocks_;
	pod::TensorView patch_w_, patch_b_, norm_w_, zero_bias_;
	pod::TensorView al_w1_, al_b1_, al_w2_, al_b2_;
	std::vector<backend::half_t> start_, end_, newline_;   // host copies of the delimiter embeddings

	backend::half_t* d_patches_ = nullptr;
	float* d_cos_ = nullptr;
	float* d_sin_ = nullptr;
	float* d_hidden_ = nullptr;
	backend::half_t* d_ln_ = nullptr;
	backend::half_t* d_q_ = nullptr;
	backend::half_t* d_k_ = nullptr;
	backend::half_t* d_v_ = nullptr;
	backend::half_t* d_attn_ = nullptr;
	backend::half_t* d_tmp_ = nullptr;
	backend::half_t* d_gate_ = nullptr;
	backend::half_t* d_up_ = nullptr;
	backend::half_t* d_mlp_ = nullptr;
	backend::half_t* d_gathered_ = nullptr;
	backend::half_t* d_al1_ = nullptr;
	backend::half_t* d_out_ = nullptr;
};

} // namespace engine::models
