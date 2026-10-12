#pragma once

#include "engine/config.h"
#include <atomic>
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
	std::vector<uint8_t> audio_update; // per audio row: 1 = target, 0 = reference audio (Ref2VA); empty = all targets
};

// A Ref2VA reference block, in request order (images, then videos, then standalone audio): packed before the target audio and video rows
struct H3RefBlock {
	enum Kind { Image, Audio, Video } kind = Image;
	int ref_audio_t = 0;  // Audio / Video: audio latent frames (a video without sound: 0)
	int latent_t = 1, latent_h = 0, latent_w = 0; // Image / Video: latent grid (an image has latent_t 1)
};

// Port of minimax_h3_packed_sequence_ref2va_blocks: [text | reference blocks | target audio | target video]; a video block packs its audio rows first
H3Layout h3_build_layout_ref( int text_len, int latent_t, int latent_h, int latent_w, int audio_t, const std::vector<H3RefBlock>& blocks );

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
	std::string dit;  // Comfy INT8 ConvRot transformer checkpoint (FL2VA: text-to-video and keyframes)
	std::string dit_ref2va; // the Ref2VA transformer checkpoint (reference images / audio); empty = use `dit`
	int device = 0;               // GPU for the encoders and decoders
	bool trim_scratch = true;     // H3Engine with a resident transformer: free its activation scratch while the text and decode stages run
	int resident_rows = 0;        // H3Engine with resident models: packed rows the resident transformer has room for (0 = what the first job needs)
	std::vector<int> devices;     // GPUs for the DiT (sequence parallel); empty = {device}
	std::string text_encoder;     // optional: an EXL3 conversion of the Qwen3-VL language model; kept resident and sharded over `devices` instead of streaming the BF16 layers from root/text_encoder for every job
	bool resident_decoders = false; // H3Engine with resident models: keep the video decoders and the audio VAE loaded between jobs too (about 6 GB per GPU; needs the room tensor parallelism leaves)
	bool tensor_parallel = false; // shard the DiT over `devices` by heads / MLP width instead of by rows (needs NCCL; less VRAM per GPU, a little slower)
};

// Up-front checks of a configuration (GPUs, tensor parallelism, the text tower checkpoint, whether the resident models fit in the GPU memory): returns false and sets
// `error` for a configuration that cannot work (the server refuses to start with it, h3_generate exits), and appends advice for ones that merely waste something to
// `warnings` (may be null). CUDA only.
bool h3_check_paths( const H3Paths& paths, std::string* error, std::vector<std::string>* warnings );

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
	std::vector<std::string> ref_images; // Ref2VA reference images (up to 9; resized to a 2048 px short edge)
	std::vector<std::string> ref_videos; // Ref2VA reference videos (up to 3; MP4 / MOV, 2 to 15 s each and in total; their soundtrack is a reference too)
	std::vector<std::string> ref_audios; // Ref2VA reference audio clips (up to 3; 2 to 15 s each, 15 s in total; cut to the clip duration)
	std::string out;             // .mp4 path
	bool keep_pcm = false;
	// progress(stage, done, total): stages "images", "text", "load", "denoise" (step done of steps), "decode", "mux"
	std::function<void( const char* stage, int done, int total )> progress;
	const std::atomic<bool>* cancel = nullptr; // checked between stages and denoising steps
};

struct H3GenResult {
	int frames = 0, width = 0, height = 0, latent_t = 0, rows = 0;
	bool cancelled = false;
	double t_text = 0, t_load = 0, t_denoise = 0, t_decode = 0, t_mux = 0;
};

// One generation with the models loaded for the run and freed stage by stage
bool h3_generate( const H3Paths& paths, const H3GenRequest& req, H3GenResult* result = nullptr );

// Generation that can keep the transformer loaded between jobs: with `resident` it stays on every GPU of paths.devices (weights and staged adapters; its activation
// scratch is freed during the text and decode stages) and a following job with the same checkpoint skips the ~15 s load. The VAEs, the image encoder and the
// language-model stack are loaded per stage: next to the transformer the first GPU has room for one of them at a time. Not thread safe: one job at a time.
class H3Engine {
public:
	H3Engine( const H3Paths& paths, bool resident );
	~H3Engine();
	H3Engine( const H3Engine& ) = delete;
	H3Engine& operator=( const H3Engine& ) = delete;

	bool generate( const H3GenRequest& req, H3GenResult* result = nullptr );
	void release(); // frees the resident transformer (the next job loads it again)

private:
	struct Impl;
	Impl* p_;
};

} // namespace engine::video
