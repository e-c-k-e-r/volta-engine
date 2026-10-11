// H3 stage: the language-model conditioning (Qwen3-VL text tower, vision tower for images and video blocks).
#include "engine/core/log.h"
#include "engine/loader/tokenizer.h"
#include "engine/video/h3_text.h"
#include "engine/video/h3_vision.h"
#include "h3_run.h"
#include "h3_util.h"
#include <algorithm>
#include <cstdio>

namespace engine::video {

bool H3Run::encode_prompt() {
	report( "text", 0, 1 );
	auto t0 = std::chrono::steady_clock::now();
	engine::loader::Tokenizer tok;
	if ( !tok.load( ( paths.root + "/tokenizer" ).c_str() ) ) { LOG_ERROR( "H3: tokenizer not found under %s", paths.root.c_str() ); return false; }
	std::vector<H3ImageFeatures> feats;
	const int image_pad = tok.get_special_token( "<|image_pad|>" );
	if ( canvas.empty() && vids.empty() ) {
		ids = tok.encode( req.prompt, false );
		text_tags.assign( ids.size(), 1 );
	} else {
		H3VisionTower tower;
		int max_patches = 8192;
		for ( const H3CanvasImage& ci : canvas ) { max_patches = std::max( max_patches, ( ci.h / 16 ) * ( ci.w / 16 ) ); }
		for ( const H3SampledVideo& v : vids ) { max_patches = std::max( max_patches, v.vb.grid_h * v.vb.grid_w ); }
		if ( !tower.load( ( paths.root + "/text_encoder" ).c_str(), paths.device, max_patches ) ) { return false; }
		engine::loader::VisionParams vparams;
		vparams.min_pixels = 65536;
		vparams.max_pixels = 16777216;
		auto push = [&]( const std::vector<int>& v, int tag ) {
			ids.insert( ids.end(), v.begin(), v.end() );
			text_tags.insert( text_tags.end(), v.size(), tag );
		};
		// one vision block: <|vision_start|> <|image_pad|> x n <|vision_end|>; video blocks use the same placeholder (the features replace it either way)
		auto vision = [&]( const std::vector<float>& patches, int gh, int gw ) -> bool {
			H3ImageFeatures f;
			f.grid_h = gh;
			f.grid_w = gw;
			if ( !tower.encode( patches, gh, gw, &f.merged, f.deepstack ) ) { LOG_ERROR( "H3: vision tower failed" ); return false; }
			const int n = static_cast<int>( f.merged.size() / 5120 );
			push( { tok.get_special_token( "<|vision_start|>" ) }, 0 );
			push( std::vector<int>( n, image_pad ), 0 );
			push( { tok.get_special_token( "<|vision_end|>" ) }, 0 );
			feats.push_back( std::move( f ) );
			return true;
		};
		for ( size_t i = 0; i < canvas.size(); ++i ) {
			engine::loader::DecodedImage di;
			di.width = canvas[i].w;
			di.height = canvas[i].h;
			di.rgb = canvas[i].px;
			engine::loader::PreparedImage prep;
			std::string err;
			if ( !engine::loader::prepare_image( di, vparams, &prep, &err ) ) { LOG_ERROR( "H3: image preparation failed: %s", err.c_str() ); return false; }
			push( tok.encode( "<Picture " + std::to_string( i + 1 ) + ">: ", false ), 1 );
			if ( !vision( prep.patches, prep.grid_h, prep.grid_w ) ) { return false; }
		}
		int audio_label = 0;
		for ( size_t k = 0; k < vids.size(); ++k ) {
			if ( vids[k].has_audio ) { push( tok.encode( "<Audio " + std::to_string( ++audio_label ) + ">: ", false ), 1 ); } // audio never enters the language model: label only
			push( tok.encode( "<Video " + std::to_string( k + 1 ) + ">: ", false ), 1 );
			for ( size_t b = 0; b < vids[k].vb.patches.size(); ++b ) {
				char stamp[64];
				std::snprintf( stamp, sizeof( stamp ), "<%.1f seconds>", vids[k].stamps[b] );
				push( tok.encode( stamp, false ), 1 );
				if ( !vision( vids[k].vb.patches[b], vids[k].vb.grid_h, vids[k].vb.grid_w ) ) { return false; }
			}
		}
		for ( size_t j = 0; j < req.ref_audios.size(); ++j ) { push( tok.encode( "<Audio " + std::to_string( ++audio_label ) + ">: ", false ), 1 ); }
		push( tok.encode( req.prompt, false ), 1 );
	}
	if ( ids.empty() ) { return false; }
	float* d_prompt = nullptr;
	CUDA_CHECK( cudaMalloc( &d_prompt, ids.size() * 5120 * sizeof( float ) ) );
	{
		H3TextEncoder enc;
		if ( !enc.load( ( paths.root + "/text_encoder" ).c_str(), paths.device, std::max<int>( 8192, static_cast<int>( ids.size() ) ), 50 ) || !enc.encode_mm( ids, image_pad, feats, d_prompt ) ) { return false; }
	}
	res.t_text = seconds_since( t0 );
	prompt.resize( ids.size() * 5120 );
	CUDA_CHECK( cudaMemcpy( prompt.data(), d_prompt, prompt.size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
	cudaFree( d_prompt );
	LOG_INFO( "H3: prompt fingerprint %016llx", static_cast<unsigned long long>( h3_fingerprint( prompt ) ) );
	LOG_INFO( "H3: prompt %d tokens (%zu image(s)) encoded in %.1f s", static_cast<int>( ids.size() ), feats.size(), res.t_text );
	return true;
}

} // namespace engine::video
