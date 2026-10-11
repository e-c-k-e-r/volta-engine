#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Reference media of Ref2VA (host side, through ffmpeg / ffprobe): reference videos at the checkpoint's canvas and 24 fps, audio as 32 kHz stereo.
namespace engine::video {

struct H3RefVideo {
	int width = 0, height = 0, frames = 0; // canvas size and frame count (24 fps, at most the clip's frame count)
	bool has_audio = false;
	double duration = 0;                   // seconds of the source used (after the 15 s total budget)
	std::vector<uint8_t> rgb;              // [frames, height, width, 3]
};

// Validates a reference video (MP4 / MOV, H.264 / H.265, 256 to 5760 px, aspect 0.4 to 2.5, 23.976 to 60 fps, 2 to 15 s; `budget` = seconds of the 15 s total left, reduced
// by the use) and decodes it onto the canvas (768 px short edge, at most 768 x 1344 pixels, 32 px multiples), at most `max_frames` frames.
bool h3_load_ref_video( const std::string& path, int max_frames, double* budget, H3RefVideo* out );

// Audio of a file (a standalone clip or the soundtrack of a video) as planar stereo samples at 32 kHz ([2][samples]); `max_seconds` > 0 cuts it. `full_seconds` receives the
// duration before the cut.
bool h3_load_ref_audio( const std::string& path, double max_seconds, std::vector<float>* planar, int* samples, double* full_seconds );

// The frames of a reference video that go to the language model: every 12th (2 fps of 24), their block timestamps (mean of each merged pair, the last frame repeated for an odd count)
std::vector<int> h3_video_sample_indices( int frames );
std::vector<double> h3_video_block_timestamps( int sampled );

} // namespace engine::video
