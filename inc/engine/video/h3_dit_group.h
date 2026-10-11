#pragma once

#include "engine/config.h"
#include <cstdint>
#include <memory>
#include <vector>

// The H3 DiT over several GPUs by sequence parallelism: every GPU holds the full (INT8) weights and the adapters, owns a contiguous quarter of the packed
// rows, and exchanges the K / V rows of every layer over NVLink (peer copies). A single GPU is the n = 1 case.
namespace engine::video {

struct H3HostRequest {
	int used = 0, text_len = 0, n_img = 0, n_audio = 0;
	std::vector<float> video_rows, audio_rows, prompt, position_ids;
	std::vector<int> img_pos, audio_pos, text_pos, tags, inverse;
	std::vector<uint8_t> update_mask;
	std::vector<float> unique_t;
};

class H3DiTGroup {
public:
	H3DiTGroup();
	~H3DiTGroup();
	H3DiTGroup( const H3DiTGroup& ) = delete;
	H3DiTGroup& operator=( const H3DiTGroup& ) = delete;

	bool load( const char* checkpoint, const std::vector<int>& devices, int max_rows );
	int add_lora( const char* path, float scale = 1.0f );
	void set_lora_scale( int id, float scale );
	int size() const;

	// video_out [n_img, 96] (condition rows zeroed), audio_out [n_audio, 32], host fp32
	bool forward( const H3HostRequest& req, std::vector<float>* video_out, std::vector<float>* audio_out );

private:
	struct Impl;
	Impl* p_;
};

} // namespace engine::video
