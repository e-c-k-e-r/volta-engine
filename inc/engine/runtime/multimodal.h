#pragma once

#include "engine/backend/types.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::runtime {

// One image inside a prompt: `count` consecutive placeholder tokens from `start`, laid out on a grid_h x grid_w merged-patch grid.
struct MmImage {
	int start = 0;
	int count = 0;
	int grid_h = 0;
	int grid_w = 0;
	uint64_t hash = 0;
	// [count, hidden] embeddings from the vision tower; null when the rows are not needed (the prefix cache restored past them)
	std::shared_ptr<const std::vector<backend::half_t>> embeds;
};

// The images of one sequence in order. Image tokens carry 3-D rotary positions; text after an image continues from the image's
// largest grid extent, so rope positions run behind KV indices by a growing delta (Qwen M-RoPE).
struct MmLayout {
	std::vector<MmImage> images;

	bool empty() const { return images.empty(); }

	const MmImage* image_at( int idx ) const {
		for ( const MmImage& im : images ) {
			if ( idx >= im.start && idx < im.start + im.count ) return &im;
		}
		return nullptr;
	}

	// Rotary position (time, height, width) of the token at KV index idx
	void rope_at( int idx, int out[3] ) const {
		int delta = 0;
		for ( const MmImage& im : images ) {
			if ( idx < im.start ) break;
			if ( idx < im.start + im.count ) {
				const int base = im.start - delta;
				const int k = idx - im.start;
				out[0] = base;
				out[1] = base + k / im.grid_w;
				out[2] = base + k % im.grid_w;
				return;
			}
			delta += im.count - ( im.grid_h > im.grid_w ? im.grid_h : im.grid_w );
		}
		out[0] = out[1] = out[2] = idx - delta;
	}
};

// Placeholder id used in prefix-cache keys for token i of an image: distinct per image content, above any vocabulary id.
inline int mm_key_token( uint64_t image_hash, int i, int vocab_size ) {
	uint64_t h = image_hash ^ ( static_cast<uint64_t>( i + 1 ) * 0x9e3779b97f4a7c15ULL );
	h ^= h >> 31;
	h *= 0xd6e8feb86659fd93ULL;
	h ^= h >> 32;
	return vocab_size + static_cast<int>( h & 0x3fffffff );
}

} // namespace engine::runtime
