#pragma once

#include "engine/config.h"
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// MiniMax-H3 text-to-video-and-audio generation on one GPU: text encoder -> DiT denoise loop (optionally with LoRA adapters) -> video / audio VAE
// decoders -> MP4. Stages load their weights in turn so that the 32 GB card holds one at a time.
namespace engine::video {

// Packed token layout [text | condition | audio | video] in "used" coordinates (no padding rows), 3D positions for the rotary table
struct H3Layout {
	int text_len = 0, latent_t = 0, latent_h = 0, latent_w = 0, audio_t = 0;
	int used = 0, n_img = 0, n_audio = 0;
	std::vector<int> img_pos, audio_pos, text_pos, tags;
	std::vector<float> position_ids; // [used, 3]
	std::vector<uint8_t> update_mask; // per video row: 1 = target, 0 = condition
};

// Port of minimax_h3_packed_sequence for T2VA (and FL2VA keyframe condition rows when `keyframes` is non-empty: -1 = last frame, 0 = first)
H3Layout h3_build_layout( int text_len, int latent_t, int latent_h, int latent_w, int audio_t, const std::vector<int>& keyframes = {}, int frame_count = 0 );

// Output canvas (height, width) of the checkpoint's policy for an aspect ratio: 768 short edge, at most 768 x 1344 pixels, multiples of 32
void h3_output_canvas( double aspect, int* height, int* width );

// Frame / latent bookkeeping of the checkpoint (17n + 5 frames, 5n + 2 latent frames, 40 Hz audio latents)
int h3_align_frames( int frames );
int h3_video_latent_t( int frames );
int h3_audio_latent_t( double seconds );

struct H3Paths {
	std::string root; // FL2VA / Ref2VA component directory (text_encoder, video_vae, audio_vae, tokenizer)
	std::string dit;  // Comfy INT8 ConvRot transformer checkpoint
	int device = 0;
};

struct H3Lora {
	std::string path;
	float scale = 1.0f;
};

struct H3GenRequest {
	std::string prompt;
	int width = 1280, height = 736, frames = 124;
	int steps = 4;               // denoiser calls
	uint64_t seed = 1;
	float video_shift = 0.0f;    // <= 0: 6 when a 768p adapter is loaded, else 12
	float audio_shift = 3.0f;
	std::vector<H3Lora> loras;
	std::vector<std::string> images; // FL2VA keyframe images (1 or 2: first / last frame); empty = text-to-video
	std::vector<int> frame_indices;  // per image: 0 (first) or -1 (last); empty = [0] for one image, [0, -1] for two
	std::string out;             // .mp4 path
	bool keep_pcm = false;
};

struct H3GenResult {
	int frames = 0, width = 0, height = 0, latent_t = 0, rows = 0;
	double t_text = 0, t_load = 0, t_denoise = 0, t_decode = 0, t_mux = 0;
};

bool h3_generate( const H3Paths& paths, const H3GenRequest& req, H3GenResult* result = nullptr );

} // namespace engine::video
