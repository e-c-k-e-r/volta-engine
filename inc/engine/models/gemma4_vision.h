#pragma once

#include "engine/core/arena.h"
#include "engine/core/tensor.h"
#include "engine/modules/linear.h"
#include "engine/models/vision_tower.h"
#include <vector>

namespace engine::models {

struct Gemma4VisionConfig {
	int depth = 27;
	int hidden = 1152;
	int heads = 16;
	int head_dim = 72;
	int intermediate = 4304;
	int out_hidden = 5376;      // the language model's hidden size
	int patch_size = 16;
	int pool = 3;               // a soft token pools pool x pool patches
	int max_soft_tokens = 280;
	int pos_table = 10240;      // rows of each of the x / y position tables
	float rope_theta = 100.0f;
	float eps = 1e-6f;
	float text_eps = 1e-6f;     // the embedding norm before the projection

	int patch_dim() const { return 3 * patch_size * patch_size; }
};

// Gemma 4's image encoder: a patch projection plus learned x / y position embeddings, sandwich-normed ViT blocks (per-head q / k norms, an
// unweighted v norm, a 2-D rotary, softmax scale 1, a GELU-tanh gated MLP), a pool x pool average pooler with the checkpoint's
// standardization, an unweighted RMSNorm and the projection into the language model. Weights are the checkpoint's model.vision_tower.* and
// model.embed_vision.* tensors (BF16, loaded as FP16).
class Gemma4VisionTower : public VisionTower {
public:
	// False without a vision_config or without the vision tensors
	bool load( const char* model_dir, pod::MemoryArena& arena, int max_soft_tokens, backend::stream_t stream );
	bool loaded() const override { return loaded_; }
	const engine::loader::VisionParams& params() const override { return params_; }
	bool encode( const engine::loader::PreparedImage& img, std::vector<backend::half_t>* out, backend::stream_t stream ) override;

private:
	struct Block {
		pod::TensorView input_norm, post_attn_norm, pre_ffn_norm, post_ffn_norm, q_norm, k_norm;
		pod::LinearWeight q, k, v, o, gate, up, down;
	};

	bool loaded_ = false;
	Gemma4VisionConfig cfg_;
	engine::loader::VisionParams params_;
	int max_patches_ = 0;
	std::vector<Block> blocks_;
	pod::LinearWeight patch_proj_, out_proj_;
	pod::TensorView std_bias_, std_scale_;
	std::vector<uint16_t> pos_table_;   // [2, pos_table, hidden] BF16 on the host: the rows of an image are gathered per encode

	backend::half_t* d_patches_ = nullptr;
	backend::half_t* d_zero_bias_ = nullptr;
	backend::half_t* d_pos_ = nullptr;
	int* d_pos_xy_ = nullptr;
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
	backend::half_t* d_pooled_ = nullptr;
	backend::half_t* d_normed_ = nullptr;
	backend::half_t* d_out_ = nullptr;
};

} // namespace engine::models
