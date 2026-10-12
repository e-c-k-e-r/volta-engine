#pragma once

// The loaded models of H3 generation. A one-shot run loads each component when a stage first asks for it and frees it when the stage lets go of the lease
// (one component at a time on the 32 GB cards); with `resident` the transformer stays loaded between runs (see H3Engine).
#include "engine/video/h3_audio.h"
#include "engine/video/h3_dit_group.h"
#include "engine/video/h3_pipeline.h"
#include "engine/video/h3_text_group.h"
#include "engine/video/h3_vae.h"
#include "engine/video/h3_vae_enc.h"
#include <memory>
#include <string>
#include <vector>

namespace engine::video {

// Either a borrowed resident component or one the lease owns (freed with it)
template <typename T>
class H3Lease {
public:
	H3Lease() = default;
	explicit H3Lease( T* shared ) : ptr_( shared ) {}
	explicit H3Lease( std::unique_ptr<T> own ) : own_( std::move( own ) ), ptr_( own_.get() ) {}
	T* operator->() const { return ptr_; }
	T& operator*() const { return *ptr_; }
	explicit operator bool() const { return ptr_ != nullptr; }
	bool owned() const { return own_ != nullptr; }

private:
	std::unique_ptr<T> own_;
	T* ptr_ = nullptr;
};

class H3Components {
public:
	H3Components( const H3Paths& paths, bool resident ) : paths_( paths ), resident_( resident ) {}

	const H3VaeParams& video_params(); // latent statistics of the video VAE (read once)

	// Each returns an empty lease when the component cannot be loaded
	H3Lease<H3ImageEncoder> image_encoder();
	// One encoder per GPU of `devices` (the reference video encode spreads its tiles over them); fewer when a GPU cannot take one
	std::vector<H3Lease<H3ImageEncoder>> image_encoders( const std::vector<int>& devices );
	H3Lease<H3VideoVAE> video_vae();
	// One decoder per GPU of `devices` (paths.device first; the tiles of a clip are spread over them)
	std::vector<H3Lease<H3VideoVAE>> video_vaes( const std::vector<int>& devices );
	H3Lease<H3AudioVAE> audio_vae();
	// The transformer for `checkpoint` on `devices`, with room for `rows` packed rows. A resident one is reused while its checkpoint matches and its room is enough;
	// otherwise it is freed first (two transformers do not fit together) and loaded again.
	// The resident EXL3 text tower (paths.text_encoder) on `devices`, with room for `tokens` tokens; empty when paths.text_encoder is not set or it cannot be loaded.
	// Kept loaded between jobs when resident (reloaded if a longer prompt needs more room).
	H3Lease<H3TextGroup> text_group( const std::vector<int>& devices, int tokens );
	H3Lease<H3DiTGroup> dit( const std::string& checkpoint, const std::vector<int>& devices, int rows );
	// Makes `loras` the adapters of `dit`: a resident transformer keeps the staged adapters when the list is unchanged and only updates their scales.
	bool set_adapters( H3Lease<H3DiTGroup>& dit, const std::vector<H3Lora>& loras );

	// Frees the activation scratch of a resident transformer (its weights stay) so that the decoders have room on the same GPUs; dit() takes it back for the next job.
	void trim_transformer();
	void release(); // frees the resident transformer
	bool resident() const { return resident_; }

private:
	template <typename T>
	H3Lease<T> lease( std::unique_ptr<T>& slot, std::unique_ptr<T> fresh ) {
		if ( resident_ ) {
			slot = std::move( fresh );
			return H3Lease<T>( slot.get() );
		}
		return H3Lease<T>( std::move( fresh ) );
	}

	H3Paths paths_;
	bool resident_;
	bool have_params_ = false;
	H3VaeParams params_;
	std::vector<std::unique_ptr<H3VideoVAE>> vaes_; // resident decoders (paths.resident_decoders)
	std::vector<int> vaes_devices_;
	std::unique_ptr<H3AudioVAE> avae_;
	std::unique_ptr<H3TextGroup> text_;
	std::vector<int> text_devices_;
	int text_tokens_ = 0;
	std::unique_ptr<H3DiTGroup> dit_;
	std::string dit_checkpoint_;
	std::vector<int> dit_devices_;
	int dit_rows_ = 0;
	std::vector<std::string> adapter_paths_; // staged on the resident transformer, in order
};

} // namespace engine::video
