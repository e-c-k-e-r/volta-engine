#pragma once

#include "engine/core/arena.h"
#include "engine/core/tensor.h"
#include "engine/loader/audio.h"
#include "engine/modules/linear.h"
#include "engine/models/vision_tower.h"
#include <vector>

namespace engine::models {

struct Gemma4AudioConfig {
	int layers = 12;
	int hidden = 1024;
	int heads = 8;
	int conv_kernel = 5;
	int context_left = 13;       // attention_context_left: a row attends to the context_left - 1 rows that end at itself
	float logit_cap = 50.0f;
	float residual_weight = 0.5f;
	float eps = 1e-6f;
	int sub_channels[2] = { 128, 32 };
	int out_proj = 1536;
	int out_hidden = 2560;       // the language model's hidden size
	bool clipped = true;

	int head_dim() const { return hidden / heads; }
	int window() const { return context_left - 1; }
};

// Gemma 4's audio encoder (the small models' Universal Speech Model conformer): two stride-2 3x3 convolutions over the log-mel frames, each
// with a channel LayerNorm and ReLU (FP32), the input projection, then per block a half-step SiLU feed-forward, local attention
// over the row and the window - 1 rows before it with sinusoidal relative positions and a logit softcap, a light convolution (GLU, causal
// depthwise, norm, SiLU), a second half-step feed-forward and the block's norm; the output projection, an unweighted RMSNorm and the
// projection into the language model. The projections clamp their inputs and outputs to the checkpoint's ranges. Weights are the
// checkpoint's model.audio_tower.* and model.embed_audio.* tensors (or a GGUF directory's mmproj file), FP16 on the device.
class Gemma4AudioTower {
public:
	// False without an audio_config or the audio tensors, or without room; max_tokens: the longest chunk (sizes the scratch)
	bool load( const char* model_dir, pod::MemoryArena& arena, int max_tokens, backend::stream_t stream );
	bool loaded() const { return loaded_; }
	const engine::loader::AudioParams& params() const { return params_; }
	// One chunk into [tokens, out_hidden] FP16 rows (host). Blocks until done
	bool encode( const engine::loader::PreparedAudio& a, std::vector<backend::half_t>* out, backend::stream_t stream );

private:
	struct FeedForward {
		pod::TensorView pre, post;   // post: scaled by the residual weight
		pod::LinearWeight up, down;
		ClipRange cup, cdown;
	};
	struct Block {
		FeedForward ff1, ff2;
		pod::TensorView attn_pre, attn_post, conv_pre, conv_dw, conv_norm, out_norm;
		pod::LinearWeight q, k, v, post, pw1, pw2;
		ClipRange cq, ck, cv, cpost, cpw1, cpw2;
		float* q_scale = nullptr;    // [head_dim] FP32: head_dim^-0.5 / ln 2 * softplus( per_dim_scale )
		float* rel = nullptr;        // [window, hidden] FP32: the relative position keys of distances 0 .. window - 1
	};

	void clipped( pod::TensorView& in, const pod::LinearWeight& w, const ClipRange& c, pod::TensorView& out, backend::half_t* spare, backend::stream_t stream );
	void feed_forward( const FeedForward& f, int T, backend::stream_t stream, pod::TensorView* next = nullptr, const pod::TensorView* next_weight = nullptr );

	bool loaded_ = false;
	Gemma4AudioConfig cfg_;
	engine::loader::AudioParams params_;
	int max_tokens_ = 0;
	float* d_conv_w_[2] = {};   // the subsampler: [out, in, 3, 3] and the gains [out], FP32
	float* d_conv_g_[2] = {};
	float* d_mel_ = nullptr;    // [frames, mel] FP32
	float* d_sub_[2] = {};      // the subsampler's layers out, [rows, frequencies, channels] FP32
	std::vector<Block> blocks_;
	pod::LinearWeight in_proj_, out_proj_, embed_proj_;
	pod::TensorView out_bias_;

	float* d_hidden_ = nullptr;
	backend::half_t* d_zero_bias_ = nullptr;
	backend::half_t* d_ln_ = nullptr;
	backend::half_t* d_q_ = nullptr;
	backend::half_t* d_k_ = nullptr;
	backend::half_t* d_v_ = nullptr;
	backend::half_t* d_attn_ = nullptr;
	backend::half_t* d_tmp_ = nullptr;
	backend::half_t* d_ffn_ = nullptr;
	backend::half_t* d_proj_ = nullptr;
	backend::half_t* d_normed_ = nullptr;
	backend::half_t* d_out_ = nullptr;
};

} // namespace engine::models
