#pragma once

#include "engine/core/arena.h"
#include "engine/core/tensor.h"
#include "engine/modules/linear.h"
#include "engine/models/vision_tower.h"
#include <memory>
#include <string>
#include <vector>

namespace engine::models {

struct Glm5VisionConfig {
	int depth = 24;
	int hidden = 1024;
	int heads = 16;
	int intermediate = 4096;
	int out_hidden = 4096;
	int merger_intermediate = 10240;
	int patch_size = 14;
	int merge_size = 2;
	int temporal_patch_size = 2;
	float eps = 1e-5f;
	float swiglu_limit = 10.0f;

	int head_dim() const { return hidden / heads; }
	int patch_dim() const { return 3 * temporal_patch_size * patch_size * patch_size; }
};

// The tower's geometry and the checkpoint's image processor settings (config.json vision_config, processor_config.json) for images of up to
// max_patches patches; false without a usable vision_config. Shared by the GPU tower and the host one (host_vision.h)
bool glm5_vision_config( const char* model_dir, int max_patches, Glm5VisionConfig* cfg, engine::loader::VisionParams* params );

// The GLM-5 image encoder: patch embedding, RMSNorm + SwiGLU ViT blocks with per-head q / k norm and 2-D rotary, a stride-2
// convolution that merges 2x2 patches, and a SwiGLU projector into the language model's hidden size. Weights are the checkpoint's
// model.visual.* tensors (trellis projections, FP16 norms / biases).
class Glm5VisionTower : public VisionTower {
public:
	// False without a vision_config or without the quantized visual tensors.
	bool load( const char* model_dir, pod::MemoryArena& arena, int max_patches, backend::stream_t stream );
	bool loaded() const override { return loaded_; }
	const engine::loader::VisionParams& params() const override { return params_; }
	bool encode( const engine::loader::PreparedImage& img, std::vector<backend::half_t>* out, backend::stream_t stream ) override;

private:
	struct Block {
		pod::TensorView norm1_w, norm2_w, q_norm, k_norm;
		pod::LinearWeight q, k, v, proj, gate, up, down;
		pod::TensorView q_b, k_b, v_b, proj_b, gate_b, up_b, down_b;
	};

	bool loaded_ = false;
	Glm5VisionConfig cfg_;
	engine::loader::VisionParams params_;
	int max_patches_ = 0;
	std::vector<Block> blocks_;
	pod::TensorView patch_w_, patch_b_, post_norm_w_, down_w_, down_b_;
	pod::TensorView merge_norm_w_, merge_norm_b_;
	pod::LinearWeight merge_proj_, merge_gate_, merge_up_, merge_down_;

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
	backend::half_t* d_merged_ = nullptr;
	backend::half_t* d_mgate_ = nullptr;
	backend::half_t* d_mup_ = nullptr;
	backend::half_t* d_mmlp_ = nullptr;
	backend::half_t* d_out_ = nullptr;
};

// The tower on the host CPU (host_vision.h): the checkpoint's EXL3 projections decoded once into FP16, or the dense weights of the model's vision projector GGUF
std::unique_ptr<VisionTower> make_host_glm5_tower( const char* model_dir, const char* mmproj_dir, int max_tokens, const std::vector<int>& cpus, std::string* error );

} // namespace engine::models
