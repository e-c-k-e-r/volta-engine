// H3Engine: generation with the models kept loaded between jobs (H3Components) or loaded per job.
#include "engine/core/log.h"
#include "h3_components.h"
#include "h3_stager.h"
#include "h3_run.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <string>
#include <chrono>
#include <thread>

namespace engine::video {

const H3VaeParams& H3Components::video_params() {
	if ( !have_params_ ) {
		have_params_ = h3_read_vae_params( ( paths_.root + "/video_vae" ).c_str(), params_ );
	}
	return params_;
}

// The encoders and decoders are loaded for the stage that needs them and freed with the lease: next to a resident transformer the first GPU has room for one of them
// (and the language-model stack) at a time.
H3Lease<H3ImageEncoder> H3Components::image_encoder() {
	CUDA_CHECK( cudaSetDevice( paths_.device ) );
	auto enc = std::make_unique<H3ImageEncoder>();
	if ( !enc->load( ( paths_.root + "/video_vae" ).c_str(), paths_.device ) ) { return {}; }
	return H3Lease<H3ImageEncoder>( std::move( enc ) );
}

std::vector<H3Lease<H3ImageEncoder>> H3Components::image_encoders( const std::vector<int>& devices ) {
	// one thread per GPU: the weight loads (disk to device) overlap
	std::vector<std::unique_ptr<H3ImageEncoder>> made( devices.size() );
	std::vector<std::thread> threads;
	for ( size_t i = 0; i < devices.size(); ++i ) {
		threads.emplace_back( [&, i] {
			auto enc = std::make_unique<H3ImageEncoder>();
			if ( enc->load( ( paths_.root + "/video_vae" ).c_str(), devices[i] ) ) { made[i] = std::move( enc ); }
		} );
	}
	for ( std::thread& t : threads ) { t.join(); }
	std::vector<H3Lease<H3ImageEncoder>> out;
	for ( auto& m : made ) {
		if ( m ) { out.push_back( H3Lease<H3ImageEncoder>( std::move( m ) ) ); }
	}
	CUDA_CHECK( cudaSetDevice( paths_.device ) );
	return out;
}

H3Lease<H3VideoVAE> H3Components::video_vae() {
	CUDA_CHECK( cudaSetDevice( paths_.device ) );
	auto vae = std::make_unique<H3VideoVAE>();
	if ( !vae->load( ( paths_.root + "/video_vae" ).c_str(), paths_.device ) ) { return {}; }
	return H3Lease<H3VideoVAE>( std::move( vae ) );
}

std::vector<H3Lease<H3VideoVAE>> H3Components::video_vaes( const std::vector<int>& devices ) {
	const bool keep = resident_ && paths_.resident_decoders;
	if ( keep && !vaes_.empty() && vaes_devices_ == devices ) {
		std::vector<H3Lease<H3VideoVAE>> shared;
		for ( auto& v : vaes_ ) { shared.push_back( H3Lease<H3VideoVAE>( v.get() ) ); }
		return shared;
	}
	vaes_.clear();
	std::vector<int> order{ paths_.device };
	for ( int d : devices ) { if ( d != paths_.device ) { order.push_back( d ); } }
	std::vector<std::unique_ptr<H3VideoVAE>> made( order.size() );
	std::vector<std::thread> threads;
	H3StageGroup group( static_cast<int>( order.size() ) ); // one read of the weight file for all GPUs
	for ( size_t i = 0; i < order.size(); ++i ) {
		threads.emplace_back( [&, i] {
			auto vae = std::make_unique<H3VideoVAE>();
			if ( vae->load( ( paths_.root + "/video_vae" ).c_str(), order[i], 4096, &group, static_cast<int>( i ) ) ) { made[i] = std::move( vae ); }
		} );
	}
	for ( std::thread& t : threads ) { t.join(); }
	std::vector<H3Lease<H3VideoVAE>> out;
	if ( made[0] ) { // the primary (paths.device, first) is required
		for ( auto& m : made ) {
			if ( !m ) { continue; }
			if ( keep ) {
				out.push_back( H3Lease<H3VideoVAE>( m.get() ) );
				vaes_.push_back( std::move( m ) );
			} else {
				out.push_back( H3Lease<H3VideoVAE>( std::move( m ) ) );
			}
		}
		if ( keep ) { vaes_devices_ = devices; }
	}
	CUDA_CHECK( cudaSetDevice( paths_.device ) );
	return out;
}

H3Lease<H3TextGroup> H3Components::text_group( const std::vector<int>& devices, int tokens ) {
	if ( paths_.text_encoder.empty() ) { return {}; }
	if ( resident_ && text_ && text_devices_ == devices && text_tokens_ >= tokens ) { return H3Lease<H3TextGroup>( text_.get() ); }
	text_.reset();
	const int capacity = std::max( 8192, tokens );
	auto group = std::make_unique<H3TextGroup>();
	const auto t0 = std::chrono::steady_clock::now();
	if ( !group->load( paths_.text_encoder.c_str(), devices, capacity, 50 ) ) {
		LOG_ERROR( "H3: cannot load the EXL3 text tower from %s", paths_.text_encoder.c_str() );
		CUDA_CHECK( cudaSetDevice( paths_.device ) );
		return {};
	}
	LOG_INFO( "H3: EXL3 text tower resident on %zu GPU(s) in %.1f s", devices.size(), std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() );
	CUDA_CHECK( cudaSetDevice( paths_.device ) );
	if ( resident_ ) {
		text_devices_ = devices;
		text_tokens_ = capacity;
	}
	return lease( text_, std::move( group ) );
}

H3Lease<H3AudioVAE> H3Components::audio_vae() {
	const bool keep = resident_ && paths_.resident_decoders;
	if ( keep && avae_ ) { return H3Lease<H3AudioVAE>( avae_.get() ); }
	CUDA_CHECK( cudaSetDevice( paths_.device ) );
	auto vae = std::make_unique<H3AudioVAE>();
	if ( !vae->load( ( paths_.root + "/audio_vae" ).c_str(), paths_.device, 704 ) ) { return {}; } // 15 s of audio latents (600) and some slack
	if ( keep ) {
		avae_ = std::move( vae );
		return H3Lease<H3AudioVAE>( avae_.get() );
	}
	return H3Lease<H3AudioVAE>( std::move( vae ) );
}

H3Lease<H3DiTGroup> H3Components::dit( const std::string& checkpoint, const std::vector<int>& devices, int rows ) {
	if ( resident_ && dit_ && dit_checkpoint_ == checkpoint && dit_devices_ == devices && dit_rows_ >= rows ) {
		if ( !dit_->reserve() ) { return {}; } // the previous job trimmed the scratch (trim_for_decode)
		return H3Lease<H3DiTGroup>( dit_.get() );
	}
	dit_.reset(); // free the old one before loading the next
	adapter_paths_.clear();
	// a resident transformer is sized for the largest job the server accepts (paths.resident_rows), so a bigger one never forces a reload
	const int capacity = resident_ ? std::max( rows, paths_.resident_rows ) : rows;
	auto group = std::make_unique<H3DiTGroup>();
	if ( !group->load( checkpoint.c_str(), devices, capacity, paths_.tensor_parallel ) ) { return {}; }
	if ( resident_ ) {
		dit_checkpoint_ = checkpoint;
		dit_devices_ = devices;
		dit_rows_ = capacity;
	}
	return lease( dit_, std::move( group ) );
}

bool H3Components::set_adapters( H3Lease<H3DiTGroup>& dit, const std::vector<H3Lora>& loras ) {
	std::vector<std::string> wanted;
	for ( const H3Lora& l : loras ) { wanted.push_back( l.path ); }
	if ( !dit.owned() && wanted == adapter_paths_ ) { // the same adapters are staged: only their scales change
		for ( size_t i = 0; i < loras.size(); ++i ) { dit->set_lora_scale( static_cast<int>( i ), loras[i].scale ); }
		return true;
	}
	if ( !dit.owned() ) {
		dit->clear_loras();
		adapter_paths_.clear();
	}
	for ( const H3Lora& l : loras ) {
		if ( dit->add_lora( l.path.c_str(), l.scale ) < 0 ) {
			LOG_ERROR( "H3: cannot use LoRA %s", l.path.c_str() );
			if ( !dit.owned() ) { dit->clear_loras(); }
			return false;
		}
		if ( !dit.owned() ) { adapter_paths_.push_back( l.path ); }
	}
	return true;
}

void H3Components::trim_transformer() {
	if ( resident_ && dit_ && paths_.trim_scratch ) { dit_->trim(); }
}

void H3Components::release() {
	vaes_.clear();
	avae_.reset();
	text_.reset();
	dit_.reset();
	dit_checkpoint_.clear();
	adapter_paths_.clear();
	dit_rows_ = 0;
}

struct H3Engine::Impl {
	Impl( const H3Paths& p, bool resident ) : paths( p ), components( p, resident ) {}
	H3Paths paths;
	H3Components components;
};

H3Engine::H3Engine( const H3Paths& paths, bool resident ) : p_( new Impl( paths, resident ) ) {}

H3Engine::~H3Engine() {
	delete p_;
}

void H3Engine::release() {
	p_->components.release();
}

bool H3Engine::generate( const H3GenRequest& req, H3GenResult* result ) {
	CUDA_CHECK( cudaSetDevice( p_->paths.device ) );
	H3Run run( p_->paths, req, p_->components );
	bool ok = false;
	do {
		if ( !run.plan_shape() ) { break; }
		if ( !run.encode_references() || run.cancelled() ) { break; }
		if ( !run.encode_prompt() ) { break; }
		run.plan_denoise();
		if ( run.cancelled() || !run.denoise() || run.cancelled() ) { break; }
		if ( !run.decode() || !run.mux() ) { break; }
		ok = true;
	} while ( false );
	if ( result ) { *result = run.res; }
	return ok;
}

bool h3_generate( const H3Paths& paths, const H3GenRequest& req, H3GenResult* result ) {
	H3Engine engine( paths, false );
	return engine.generate( req, result );
}

} // namespace engine::video
