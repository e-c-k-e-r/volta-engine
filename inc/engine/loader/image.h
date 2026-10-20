#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace engine::loader {

struct DecodedImage {
	int width = 0;
	int height = 0;
	std::vector<uint8_t> rgb; // [height, width, 3]
};

// JPEG, PNG, BMP, GIF (first frame) from memory. Alpha is composited onto white.
bool decode_image( const uint8_t* data, size_t size, DecodedImage* out, std::string* error );

// The image processor of the checkpoint (preprocessor_config.json + vision_config).
struct VisionParams {
	int patch_size = 16;
	int merge_size = 2;
	int temporal_patch_size = 2;
	int min_pixels = 65536;
	int max_pixels = 1048576;
	// GLM-5: aspect-preserving resize onto a zero-padded canvas aligned to patch * merge (instead of the Qwen2 warp)
	bool canvas = false;
	float mean[3] = { 0.5f, 0.5f, 0.5f };
	float stddev[3] = { 0.5f, 0.5f, 0.5f };
	// DeepSeek-V4.1: raster patches of one frame; the aligner merges downsample x downsample patches into a token and the model sees
	// [start] ([image] * llm_w + [newline]) * llm_h [end]; at most max_tokens of those; the resize pads onto a grey canvas
	bool deepseek = false;
	int downsample = 3;
	int max_tokens = 1024;
};

// An image resized to the patch grid and cut into patches, ready for the vision tower.
struct PreparedImage {
	int grid_h = 0; // patches per column / row (multiples of merge_size)
	int grid_w = 0;
	uint64_t hash = 0; // of the decoded pixels and the grid
	std::vector<float> patches; // [grid_h * grid_w, 3 * temporal * patch * patch], merge-block order (DeepSeek: raster order, 3 * patch * patch)
	int llm_h = 0; // DeepSeek: the aligner's token grid
	int llm_w = 0;
	int span_tokens() const { return llm_h > 0 ? llm_h * ( llm_w + 1 ) + 2 : 0; } // tokens the prompt carries for the image (DeepSeek)

	int num_patches() const { return grid_h * grid_w; }
	int tokens( int merge_size ) const { return num_patches() / ( merge_size * merge_size ); }
};

// Qwen2-VL smart resize: dimensions rounded to multiples of patch * merge inside [min_pixels, max_pixels].
void smart_resize( int height, int width, int factor, int min_pixels, int max_pixels, int* out_h, int* out_w );

// GLM-5 token-budget canvas (height, width) for a single image; budgets are single-frame pixels.
void glm_canvas( int height, int width, int factor, int min_pixels, int max_pixels, int* out_h, int* out_w );

bool prepare_image( const DecodedImage& img, const VisionParams& p, PreparedImage* out, std::string* error );

// Antialiased bicubic resize (PIL / torchvision antialias=True semantics), float output in [0, 255].
// round_mid: PIL keeps the horizontal pass as 8-bit values (rounded, clamped) before the vertical one; false keeps it in float
std::vector<float> resize_bicubic( const uint8_t* rgb, int in_h, int in_w, int out_h, int out_w, bool round_mid = false );

// Standard-alphabet base64 (padding and ASCII whitespace tolerated). False on any other character.
bool base64_decode( const std::string& in, std::string* out );

uint64_t hash_bytes( const void* data, size_t size, uint64_t seed = 0xcbf29ce484222325ULL );

} // namespace engine::loader
