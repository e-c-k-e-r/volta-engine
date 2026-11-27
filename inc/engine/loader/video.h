#pragma once

#include "engine/loader/image.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace engine::loader {

// A video as frames: decoded images of one size with their times in seconds from the video's start
struct VideoFrames {
	std::vector<DecodedImage> frames;
	std::vector<double> times;
};

// A video cut for the vision tower: one prepared image per temporal patch (a pair of frames on one canvas; the tower attends within a step,
// as the checkpoints' towers do over a video's frames), each with its time: Qwen3-VL the mean of its two frames', GLM-5 its first frame's
struct PreparedVideo {
	std::vector<PreparedImage> steps;
	std::vector<double> times;
	int tokens_per_step( int merge_size ) const { return steps.empty() ? 0 : steps[0].tokens( merge_size ); }
};

// The canvas a video's frames are resized to and the frames its steps take
struct VideoPlan {
	int height = 0;      // the canvas (multiples of patch * merge)
	int width = 0;
	int content_h = 0;   // GLM-5: the resized frame in the canvas's top-left corner, the rest black; Qwen3-VL: the canvas
	int content_w = 0;
	std::vector<int> frames;   // indices of the input frames, two per step (an odd count's last frame repeated)
	int steps() const { return static_cast<int>( frames.size() / 2 ); }
};

// The frames the checkpoint's video processor samples from `total` frames at `fps` (duration: the container's, 0 = total / fps). Qwen3-VL:
// video_fps per second evenly spaced, within [video_min_frames, video_max_frames]; GLM-5: one per 1 / video_fps seconds up to
// video_max_frames, an even count
std::vector<int> sample_video_frames( const VisionParams& p, int total, double fps, double duration );

// The canvas of `count` frames of height x width and the frames that fit: the processor's budget (Qwen3-VL: frames x pixels within
// [video_min_pixels, video_max_pixels], at most video_frame_tokens tokens a frame; GLM-5: the canvas within [video_min_tokens,
// video_max_tokens]), the tower's largest step (max_pixels) and max_tokens language-model tokens for the whole video; past those, evenly
// spaced frames are dropped
bool plan_video( int count, int height, int width, const VisionParams& p, int max_tokens, VideoPlan* out, std::string* error );

// One step's patches from its two frames (both of the planned input size)
void prepare_video_step( const DecodedImage& a, const DecodedImage& b, const VideoPlan& plan, const VisionParams& p, PreparedImage* out );

// Frames given with their times (all of one size), planned and cut into steps
bool prepare_video( const VideoFrames& video, const VisionParams& p, int max_tokens, PreparedVideo* out, std::string* error );

// A video file (any container and codec FFmpeg's libraries read) sampled as the checkpoint's processor samples, decoded one frame at a time
// and cut into steps; false with *error set when the build has no FFmpeg
bool decode_video( const uint8_t* data, size_t size, const VisionParams& p, int max_tokens, PreparedVideo* out, std::string* error );
bool video_decoding_available();

} // namespace engine::loader
