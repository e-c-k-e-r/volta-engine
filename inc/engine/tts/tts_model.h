// The vall-e v3 TTS model (vall_e/model.py V3, flow head, NAR demasking): a bidirectional transformer over
// [text | prompt patches | target patches] whose target positions feed a flow-matching head that generates each patch's
// 64 FSQ values (2 frames of the NeMo 44 kHz codec's 32), plus a length head. Weights come from a vall-e export
// (vall_e/export.py: an FP16 safetensors file with "config" and "vocab" JSON in its metadata).
#pragma once

#include "engine/core/arena.h"
#include "engine/loader/safetensors.h"
#include <cuda_fp16.h>
#include <cstdint>
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

// One sequence of the backbone (a row of vall_e/data.py pack, unpadded): [text | prompt | target], or [text | prompt |
// length query] for the length head. Digits are FSQ level indices, `values` per position.
struct TtsRow {
	std::vector<int> text;  // phoneme ids (1-based; 0 is padding)
	std::vector<uint8_t> prompt;  // [Lp * values]
	std::vector<uint8_t> target;  // [La * values]
	std::vector<uint8_t> masked;  // [La]: positions shown as the mask embedding
	int mode = 0;  // 0 NAR demask, 2 length
	bool len_query = false;  // a length-query position instead of the target
	int length() const;
};

class TtsModel {
public:
	bool load( const std::string& path, int device = 0 );
	// The final hidden states of every position of `row` into out (device, [L, d] FP32). taps: if given, the input of
	// every block (and finally the output, after the final norm) copied to the host, [layers + 1][L * d]
	void hidden( const TtsRow& row, float* out, std::vector<std::vector<float>>* taps = nullptr );
	static constexpr int MAX_LEN = 2560;  // positions per row (30 s targets with 10 s prompts and long texts fit)

	struct Sampling {
		int steps = 25, flow_steps = 8;
		float cfg = 6.0f, cfg_prompt = 2.0f, sway = -1.0f;
	};
	// One demasking step (vall_e/model.py demask_flow): the backbone over the guidance rows (cond; null without text and
	// prompt; promptless, with cfg_prompt) at the current target digits [La * values] / masked [La], then the guided flow
	// head from noise [k * values] at the `reveal` positions, whose snapped digits are written into digits (and their
	// masked flags cleared). vtaps: the guided velocity of every flow step, [flow_steps][k * values]
	void demask_step( const std::vector<int>& text, const std::vector<uint8_t>& prompt, std::vector<uint8_t>& digits, std::vector<uint8_t>& masked,
		const std::vector<int>& reveal, const float* noise, const Sampling& s, std::vector<std::vector<float>>* vtaps = nullptr );
	// A whole generation of n_frames frames: order [La] the reveal order and noise[step] each step's starting noise
	// (golden.py's draws, or the caller's own). Returns the frames' digits [n_frames * 32]
	std::vector<uint8_t> generate( const std::vector<int>& text, const std::vector<uint8_t>& prompt, int n_frames, const std::vector<int>& order,
		const std::vector<std::vector<float>>& noise, const Sampling& s );
	// How many patches step s of `steps` has revealed in total, of La
	static int revealed_after( int s, int steps, int La );
	const TtsConfig& config() const { return cfg_; }
	const TtsWeights& weights() const { return w_; }
	size_t weight_bytes() const { return arena_.offset_bytes; }
	~TtsModel();

private:
	void alloc_workspace();
	void alloc_flow_workspace();
	void gemm( const half* A, const half* W, void* C, int M, int N, int K, bool c32 );

	TtsConfig cfg_;
	TtsWeights w_;
	void* blas_ = nullptr;  // cublasHandle_t
	struct Workspace {
		float *x = nullptr, *a = nullptr, *S = nullptr;
		half *xn = nullptr, *qkv = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr, *g = nullptr, *y = nullptr, *up = nullptr, *act = nullptr, *P = nullptr, *vals = nullptr;
		int *ids = nullptr;
		uint8_t *kind = nullptr, *masked = nullptr, *digits = nullptr;
	} ws_;
	struct FlowWorkspace {
		float *h = nullptr, *cond = nullptr, *c = nullptr, *mod = nullptr, *xf = nullptr, *m2 = nullptr, *m1 = nullptr, *v = nullptr, *x = nullptr, *temb = nullptr, *t0 = nullptr;
		half *hh = nullptr, *sc = nullptr, *ln = nullptr, *m1h = nullptr, *xin = nullptr, *tvec = nullptr, *t0h = nullptr;
		uint8_t* snapped = nullptr;
	} fw_;
	pod::MemoryArena ws_arena_ {};
	pod::SafeTensorsFile st_;
	pod::MemoryArena arena_ {};
};

} // namespace engine::tts
