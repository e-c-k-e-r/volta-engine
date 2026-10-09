#pragma once

#include "engine/core/arena.h"
#include "engine/core/tensor.h"
#include "engine/modules/linear.h"
#include "engine/loader/image.h"
#include "engine/backend/types.h"
#include <cstdint>
#include <string>
#include <vector>

namespace engine::models {

struct Qwen3VisionConfig {
	int depth = 27;
	int hidden = 1152;
	int heads = 16;
	int intermediate = 4304;
	int out_hidden = 5120;
	int patch_size = 16;
	int merge_size = 2;
	int temporal_patch_size = 2;
	int num_position_embeddings = 2304;
	float eps = 1e-6f;

	int head_dim() const { return hidden / heads; }
	int patch_dim() const { return 3 * temporal_patch_size * patch_size * patch_size; }
};

// The Qwen3-VL image encoder (patch embedding, 2-D rotary ViT blocks, patch merger) on one device. Weights come from the
// checkpoint's model.visual.* tensors (trellis-quantized projections, FP16 norms / biases / embeddings).
class Qwen3VisionTower {
public:
	// False without a vision_config or without the quantized visual tensors (the checkpoint then serves text only).
	bool load( const char* model_dir, pod::MemoryArena& arena, int max_patches, backend::stream_t stream );
	bool loaded() const { return loaded_; }
	const Qwen3VisionConfig& config() const { return cfg_; }
	const engine::loader::VisionParams& params() const { return params_; }
	int max_patches() const { return max_patches_; }

	// Encodes one prepared image into [tokens, out_hidden] FP16 rows (host). Blocks until done.
	bool encode( const engine::loader::PreparedImage& img, std::vector<backend::half_t>* out, backend::stream_t stream );

	// Host-side tables (exposed for tests)
	std::vector<float> position_embeddings( int grid_h, int grid_w ) const;                   // [h * w, hidden] merge order
	void rotary_tables( int grid_h, int grid_w, std::vector<float>* cos_t, std::vector<float>* sin_t ) const; // [h * w, head_dim]

private:
	struct Block {
		pod::TensorView norm1_w, norm1_b, norm2_w, norm2_b;
		pod::LinearWeight q, k, v, proj, fc1, fc2;
		pod::TensorView q_b, k_b, v_b, proj_b, fc1_b, fc2_b;
	};

	bool loaded_ = false;
	Qwen3VisionConfig cfg_;
	engine::loader::VisionParams params_;
	int max_patches_ = 0;
	std::vector<Block> blocks_;
	pod::TensorView patch_w_, patch_b_;
	std::vector<float> pos_table_; // [num_position_embeddings, hidden]
	pod::TensorView merge_norm_w_, merge_norm_b_;
	pod::LinearWeight merge_fc1_, merge_fc2_;
	pod::TensorView merge_fc1_b_, merge_fc2_b_;

	// Scratch (max_patches rows)
	backend::half_t* d_patches_ = nullptr;
	backend::half_t* d_pos_ = nullptr;
	float* d_cos_ = nullptr;
	float* d_sin_ = nullptr;
	float* d_hidden_ = nullptr;
	backend::half_t* d_ln_ = nullptr;
	backend::half_t* d_q_ = nullptr;
	backend::half_t* d_k_ = nullptr;
	backend::half_t* d_v_ = nullptr;
	backend::half_t* d_attn_ = nullptr;
	backend::half_t* d_tmp_ = nullptr;
	backend::half_t* d_mlp_ = nullptr;
	backend::half_t* d_merged_ = nullptr;
	backend::half_t* d_merged_fc1_ = nullptr;
	backend::half_t* d_out_ = nullptr;
};

} // namespace engine::models
