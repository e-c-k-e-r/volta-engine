// H3 stage: latents of the keyframes and the Ref2VA references (video VAE image / video encoder, audio VAE encoder).
#include "engine/core/log.h"
#include "engine/video/h3_audio.h"
#include "engine/video/h3_refs.h"
#include "engine/video/h3_vae.h"
#include "engine/video/h3_vae_enc.h"
#include "h3_run.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

namespace engine::video {

bool H3Run::encode_references() {
	report( "images", 0, 1 );
	auto t0 = std::chrono::steady_clock::now();
	const std::vector<engine::loader::DecodedImage>& src = ref2va ? ref_src : refs;
	for ( const auto& img : src ) {
		H3CanvasImage ci;
		if ( ref2va ) {
			const double scale = 2048.0 / std::min( img.width, img.height );
			auto align = []( double v ) { return std::max( 32, static_cast<int>( std::lround( v / 32.0 ) ) * 32 ); };
			ci.w = align( img.width * scale );
			ci.h = align( img.height * scale );
		} else {
			ci.w = width;
			ci.h = height;
		}
		ci.px = h3_resize_lanczos( img.rgb.data(), img.height, img.width, ci.h, ci.w );
		canvas.push_back( std::move( ci ) );
	}
	std::unique_ptr<H3AudioVAE> avae;
	double audio_budget = 15.0; // seconds of reference audio left
	// audio: encodes `path`'s sound (cut to `max_seconds`, 0 = whole) and appends its latent rows; returns the latent frame count or 0
	auto encode_audio = [&]( const std::string& path, double max_seconds, bool standalone ) -> int {
		if ( !avae ) {
			avae.reset( new H3AudioVAE );
			if ( !avae->load( ( paths.root + "/audio_vae" ).c_str(), paths.device, 16 ) ) { return 0; }
		}
		std::vector<float> planar;
		int samples = 0;
		double full = 0;
		if ( !h3_load_ref_audio( path, max_seconds, &planar, &samples, &full ) ) { LOG_ERROR( "H3: cannot read the audio of %s", path.c_str() ); return 0; }
		if ( standalone && ( full < 2.0 || full > 15.0 ) ) { LOG_ERROR( "H3: reference audio %s must be 2 to 15 seconds long (got %.2f)", path.c_str(), full ); return 0; }
		std::vector<float> rows;
		int fr = 0;
		if ( !avae->encode( planar, samples, &rows, &fr ) ) { LOG_ERROR( "H3: audio encode failed" ); return 0; }
		if ( fr < 80 || fr > 600 ) { LOG_ERROR( "H3: the audio of %s is %d latent frames after trimming to the clip duration (2 to 15 seconds are required)", path.c_str(), fr ); return 0; }
		audio_budget -= fr / 40.0;
		if ( audio_budget < -1e-6 ) { LOG_ERROR( "H3: reference audio must be at most 15 seconds in total" ); return 0; }
		ref_audio_rows.insert( ref_audio_rows.end(), rows.begin(), rows.end() );
		return fr;
	};
	if ( !canvas.empty() || !req.ref_videos.empty() ) {
		H3ImageEncoder enc;
		H3VaeParams vp;
		if ( !h3_read_vae_params( ( paths.root + "/video_vae" ).c_str(), vp ) || !enc.load( ( paths.root + "/video_vae" ).c_str(), paths.device ) ) { return false; }
		for ( const H3CanvasImage& ci : canvas ) {
			const int ih = ci.h / 16, iw = ci.w / 16;
			float *d_in = nullptr, *d_m = nullptr;
			CUDA_CHECK( cudaMalloc( &d_in, static_cast<size_t>( 3 ) * ci.h * ci.w * sizeof( float ) ) );
			CUDA_CHECK( cudaMalloc( &d_m, static_cast<size_t>( 48 ) * ih * iw * sizeof( float ) ) );
			const std::vector<float> in = h3_vae_input( ci.px.data(), ci.h, ci.w );
			CUDA_CHECK( cudaMemcpy( d_in, in.data(), in.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
			if ( !enc.encode( d_in, ci.h, ci.w, d_m ) ) { LOG_ERROR( "H3: image encode failed" ); return false; }
			std::vector<float> moments( static_cast<size_t>( 48 ) * ih * iw );
			CUDA_CHECK( cudaMemcpy( moments.data(), d_m, moments.size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
			cudaFree( d_in );
			cudaFree( d_m );
			const std::vector<float> rows = h3_keyframe_rows( moments, ih, iw, vp.mean, vp.stdev, ref2va ? req.seed : 42, 0.999f );
			cond_rows.insert( cond_rows.end(), rows.begin(), rows.end() );
			if ( ref2va ) { blocks.push_back( { H3RefBlock::Image, 0, 1, ih, iw } ); }
		}
		if ( !canvas.empty() ) { LOG_INFO( "H3: %zu %s image(s) encoded in %.1f s", canvas.size(), ref2va ? "reference" : "keyframe", seconds_since( t0 ) ); }
		double video_budget = 15.0;
		for ( const std::string& path : req.ref_videos ) {
			H3RefVideo rv;
			if ( !h3_load_ref_video( path, frames, &video_budget, &rv ) ) { return false; }
			const size_t fsz = static_cast<size_t>( rv.width ) * rv.height * 3;
			// the language model sees every 12th frame
			H3SampledVideo sv;
			const std::vector<int> idx = h3_video_sample_indices( rv.frames );
			std::vector<const uint8_t*> sampled;
			for ( int i : idx ) { sampled.push_back( rv.rgb.data() + static_cast<size_t>( i ) * fsz ); }
			sv.vb = h3_video_blocks( sampled, rv.height, rv.width );
			sv.stamps = h3_video_block_timestamps( static_cast<int>( idx.size() ) );
			sv.has_audio = rv.has_audio;
			// the VAE sees all frames
			std::vector<float> norm( static_cast<size_t>( rv.frames ) * fsz );
			for ( int f = 0; f < rv.frames; ++f ) {
				const std::vector<float> one = h3_vae_input( rv.rgb.data() + static_cast<size_t>( f ) * fsz, rv.height, rv.width );
				std::memcpy( &norm[static_cast<size_t>( f ) * fsz], one.data(), fsz * sizeof( float ) );
			}
			float* d_vid = upload( norm );
			norm = std::vector<float>();
			std::vector<float> moments;
			int tl = 0;
			if ( !enc.encode_video( d_vid, rv.frames, rv.height, rv.width, &moments, &tl ) ) { LOG_ERROR( "H3: video encode failed" ); return false; }
			cudaFree( d_vid );
			const std::vector<float> rows = h3_condition_rows( moments, tl, rv.height / 16, rv.width / 16, vp.mean, vp.stdev, req.seed, 0.999f );
			cond_rows.insert( cond_rows.end(), rows.begin(), rows.end() );
			int ref_audio_t = 0;
			if ( rv.has_audio ) {
				ref_audio_t = encode_audio( path, std::min( rv.duration, frames / 24.0 ), false );
				if ( ref_audio_t == 0 ) { return false; }
			}
			blocks.push_back( { H3RefBlock::Video, ref_audio_t, tl, rv.height / 16, rv.width / 16 } );
			vids.push_back( std::move( sv ) );
			LOG_INFO( "H3: reference video %s: %d frames %dx%d -> %d latent frames%s, %.1f s elapsed", path.c_str(), rv.frames, rv.width, rv.height, tl, rv.has_audio ? " + audio" : "", seconds_since( t0 ) );
		}
	}
	for ( const std::string& path : req.ref_audios ) {
		const int fr = encode_audio( path, frames / 24.0, true );
		if ( fr == 0 ) { return false; }
		blocks.push_back( { H3RefBlock::Audio, fr, 1, 0, 0 } );
	}
	if ( !req.ref_audios.empty() ) { LOG_INFO( "H3: %zu reference audio clip(s) encoded in %.1f s", req.ref_audios.size(), seconds_since( t0 ) ); }
	avae.reset();
	return true;
}

} // namespace engine::video
