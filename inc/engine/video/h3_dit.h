#pragma once

#include "engine/config.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

// MiniMax-H3 diffusion transformer (video + audio latents, text conditioning). CUDA only.
// Weights: the Comfy INT8 "ConvRot" checkpoint stays INT8 on the device; each projection is dequantized (row scale) and
// un-rotated (the 256-group Hadamard folded into the stored matrix) into an FP16 scratch right before its GEMM.
// Numerics follow 1Cat's reference: FP32 residual stream, FP16 tensor-core GEMMs with FP32 accumulation, FP32 outputs for
// the wide-range projections (attention out, MLP down), pow-2 row scaling of the gated MLP activation.
namespace engine::video {

struct H3Config {
	int hidden = 5376;
	int layers = 50;
	int refiner_layers = 2;
	int heads = 56;
	int head_dim = 128;
	int ffn = 14336;
	int latents = 96;          // video patch width: 24 channels x (1 x 2 x 2)
	int audio_latents = 32;
	int text_dim = 5120;
	int adaln_grid = 1025;
	int adaln_dim = 8;
	int rope_freqs = 16;
	float eps = 1e-5f;
	float qk_eps = 1e-5f;
	float final_eps = 1e-5f;
	bool simple_attention = false; // bring-up kernel instead of the CUTLASS FMHA (tiny sequences only)
};

// One packed request in "used" coordinates (padding rows are dropped: they form their own attention document).
// Row order is the packed layout [text | condition | audio | video]; positions are row indices into it.
struct H3Request {
	int used = 0;
	int text_len = 0;
	int n_img = 0;             // video rows (condition rows first, then target rows)
	int n_audio = 0;
	// device pointers
	const float* video_rows = nullptr;   // [n_img, 96] fp32
	const float* audio_rows = nullptr;   // [n_audio, 32] fp32
	const float* prompt = nullptr;       // [text_len, 5120] fp32 (encoder hidden states)
	const float* position_ids = nullptr; // [used, 3] fp32 (t, h, w)
	const int* img_pos = nullptr;        // [n_img]
	const int* audio_pos = nullptr;      // [n_audio]
	const int* text_pos = nullptr;       // [text_len]
	const uint8_t* update_mask = nullptr; // [n_img] 1 = target row, 0 = condition row (output zeroed)
	const int* tags = nullptr;           // [used] modality 0 video, 1 text, 2 audio
	const int* inverse = nullptr;        // [used] index into unique_t
	std::vector<float> unique_t;         // distinct timesteps (t = 1 - sigma), ascending
};

class H3DiT {
public:
	H3DiT();
	~H3DiT();
	H3DiT( const H3DiT& ) = delete;
	H3DiT& operator=( const H3DiT& ) = delete;

	// Opens the Comfy checkpoint, allocates weights and scratch for at most max_rows packed rows.
	bool load( const char* checkpoint, int device, int max_rows, const H3Config& cfg = H3Config() );
	// Outputs: video_out [n_img, 96], audio_out [n_audio, 32] (device, fp32).
	void forward( const H3Request& req, float* video_out, float* audio_out, void* stream = nullptr );

	// Test hook: called with the FP32 residual stream [rows, hidden] after the embedding and after the listed blocks.
	std::function<void( const char* tag, const float* d_x, int rows )> probe;
	std::vector<int> probe_blocks;

	size_t weight_bytes() const;

private:
	struct Impl;
	Impl* p_;
};

} // namespace engine::video
