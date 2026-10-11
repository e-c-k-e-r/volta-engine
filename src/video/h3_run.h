#pragma once

// Internal state and stages of one H3 generation (see h3_generate in h3_pipeline.cu). The stages run in this order, each reading what the earlier ones left in the members:
//   plan_shape -> encode_references -> encode_prompt -> plan_denoise -> denoise -> decode -> mux
#include "engine/config.h"
#include "engine/loader/image.h"
#include "engine/video/h3_image.h"
#include "engine/video/h3_pipeline.h"
#include "h3_components.h"
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace engine::video {

inline double seconds_since( std::chrono::steady_clock::time_point t0 ) {
	return std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
}

template <typename T>
T* upload( const std::vector<T>& v ) {
	T* d = nullptr;
	CUDA_CHECK( cudaMalloc( &d, std::max<size_t>( v.size(), 1 ) * sizeof( T ) ) );
	CUDA_CHECK( cudaMemcpy( d, v.data(), v.size() * sizeof( T ), cudaMemcpyHostToDevice ) );
	return d;
}

struct H3CanvasImage {
	std::vector<uint8_t> px; // RGB8 [h, w, 3]
	int h = 0, w = 0;
};

// A reference video as the language model sees it: its sampled frames cut into temporal blocks of vision-tower patches, with the block timestamps
struct H3SampledVideo {
	H3VideoBlocks vb;
	std::vector<double> stamps;
	bool has_audio = false;
};

struct H3Run {
	H3Run( const H3Paths& p, const H3GenRequest& r, H3Components& c ) : paths( p ), req( r ), comp( c ) {}

	const H3Paths& paths;
	const H3GenRequest& req;
	H3Components& comp; // the loaded models (resident between jobs or per job)
	H3GenResult res;

	// ---- plan_shape: the task and the sizes ----
	bool ref2va = false;                                       // references (Ref2VA) instead of keyframes (FL2VA) or nothing (T2VA)
	int width = 0, height = 0, frames = 0, latent_t = 0, audio_t = 0, lh = 0, lw = 0;
	std::vector<engine::loader::DecodedImage> refs;            // keyframe images
	std::vector<engine::loader::DecodedImage> ref_src;         // reference images
	std::vector<int> keyframes;                                // per keyframe image: 0 = first frame, -1 = last

	// ---- encode_references: latents of the keyframes / references ----
	std::vector<H3CanvasImage> canvas;                         // images as the encoders see them (the canvas for keyframes, a 2048 px short edge for references)
	std::vector<float> cond_rows;                              // [n_cond, 96] visual anchors in layout order
	std::vector<H3RefBlock> blocks;                            // Ref2VA reference blocks in layout order
	std::vector<float> ref_audio_rows;                         // [2 * frames, 32] per reference audio, concatenated
	std::vector<H3SampledVideo> vids;

	// ---- encode_prompt: the language-model conditioning ----
	std::vector<int> ids, text_tags;                           // presentation token ids and their modality tags
	std::vector<float> prompt;                                 // [ids.size(), 5120] hidden states (host)

	// ---- plan_denoise: layout, initial noise and the sigma schedules ----
	H3Layout lay;
	int n_cond = 0;                                            // leading video rows that are conditions
	int n_aref = 0;                                            // leading audio rows that are reference audio
	std::vector<float> vrows, arows;                           // [n_img, 96] / [n_audio, 32]: the state denoise() advances
	std::vector<double> sv, sa;                                // video / audio sigmas
	float video_shift = 12.0f;

	// ---- decode ----
	std::vector<uint8_t> rgb;                                  // [out_frames, height, width, 3]
	int out_frames = 0;
	std::vector<float> pcm;                                    // planar stereo at 32 kHz

	void report( const char* stage, int done, int total ) const {
		if ( req.progress ) { req.progress( stage, done, total ); }
	}
	bool cancelled() {
		if ( req.cancel && req.cancel->load() ) { res.cancelled = true; return true; }
		return false;
	}

	bool plan_shape();
	bool encode_references();
	bool encode_prompt();
	void plan_denoise();
	bool denoise();
	bool decode();
	bool mux();
};

} // namespace engine::video
