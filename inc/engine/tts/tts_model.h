// The vall-e v3 TTS model (vall_e/model.py V3, flow head, NAR demasking): a bidirectional transformer over
// [text | prompt patches | target patches] whose target positions feed a flow-matching head that generates each patch's
// 64 FSQ values (2 frames of the NeMo 44 kHz codec's 32), plus a length head. Weights come from a vall-e export
// (vall_e/export.py: an FP16 safetensors file with "config" and "vocab" JSON in its metadata).
#pragma once

#include "engine/core/arena.h"
#include "engine/loader/safetensors.h"
#include <cuda_fp16.h>
#include <string>
#include <vector>

namespace engine::tts {

struct TtsConfig {
	int n_text = 0, d_model = 0, layers = 0, heads = 0, patch = 2, n_modes = 3;
	int flow_width = 512, flow_depth = 4;
	float rot_frac = 1.0f, rope_theta = 10000.0f;
	bool gated = true, qk_norm = true, length = false;
	int text_prenet = 0;
	std::vector<std::string> vocab;  // phoneme id i + 1 is vocab[i] (0 is padding)

	int head_dim() const { return d_model / heads; }
	int ffn() const { return d_model * 8 / 3 / 64 * 64; }
	int values() const { return patch * 32; }  // FSQ scalars per position
};

struct TtsBlock {
	const float *n1 = nullptr, *n2 = nullptr, *q_norm = nullptr, *k_norm = nullptr;
	const half *qkv = nullptr, *gate = nullptr, *out = nullptr, *up = nullptr, *down = nullptr;
};

struct TtsFlowBlock {
	const half *mlp0 = nullptr, *mlp2 = nullptr, *mod = nullptr;
	const float *mlp0_b = nullptr, *mlp2_b = nullptr, *mod_b = nullptr;
};

struct TtsWeights {
	const half* text = nullptr;  // [n_text, d]
	const half* audio_in = nullptr;  // [d, values]
	const float *audio_in_b = nullptr, *mask_emb = nullptr, *len_q = nullptr, *segment = nullptr, *mode = nullptr, *norm = nullptr;
	std::vector<TtsBlock> blocks;
	// flow head
	const half *f_inp = nullptr, *f_cond = nullptr, *f_time0 = nullptr, *f_time2 = nullptr, *f_final_mod = nullptr, *f_out = nullptr;
	const float *f_inp_b = nullptr, *f_cond_b = nullptr, *f_time0_b = nullptr, *f_time2_b = nullptr, *f_final_mod_b = nullptr, *f_out_b = nullptr;
	std::vector<TtsFlowBlock> flow_blocks;
	// length head
	const float *length_w = nullptr, *length_b = nullptr;
};

class TtsModel {
public:
	bool load( const std::string& path, int device = 0 );
	const TtsConfig& config() const { return cfg_; }
	const TtsWeights& weights() const { return w_; }
	size_t weight_bytes() const { return arena_.offset_bytes; }
	~TtsModel();

private:
	TtsConfig cfg_;
	TtsWeights w_;
	pod::SafeTensorsFile st_;
	pod::MemoryArena arena_ {};
};

} // namespace engine::tts
