#pragma once

#include "engine/video/h3_text.h"
#include <memory>
#include <vector>

// The text tower over several GPUs by tensor parallelism: every GPU holds its shard of the resident EXL3 projections (a quarter of the heads and of the MLP width on
// four GPUs) and runs all tokens; the attention-output and MLP-down partial products are added over NCCL. One GPU is the plain resident case.
namespace engine::video {

class H3TextGroup {
public:
	H3TextGroup();
	~H3TextGroup();
	H3TextGroup( const H3TextGroup& ) = delete;
	H3TextGroup& operator=( const H3TextGroup& ) = delete;

	// dir: an EXL3 conversion of the Qwen3-VL language model (see H3TextEncoder::load); more than one device needs a build with NCCL
	// captioner: see H3TextEncoder::load (about 2 GB more per GPU)
	bool load( const char* dir, const std::vector<int>& devices, int max_tokens, int layers = 50, bool captioner = false );
	int size() const;
	int max_tokens() const;

	// As H3TextEncoder::encode_mm; d_out [ids.size(), 5120] fp32 lives on devices[0]
	bool encode_mm( const std::vector<int>& ids, int image_token_id, const std::vector<H3ImageFeatures>& images, float* d_out );

	// The model's answer (token ids) to the prompt; needs a group loaded with captioner = true
	bool generate( const std::vector<int>& ids, int image_token_id, const std::vector<H3ImageFeatures>& images, const H3CaptionOptions& opt, std::vector<int>* out_tokens );

private:
	struct Impl;
	Impl* p_;
};

} // namespace engine::video
