// H3 VAE encoder parity: moments of one image (tiled) or of a clip against the checkpoint's own encoder.
//   test_h3_vae_enc <video_vae dir> <golden> [gpus=1]     golden: tests/tools/h3_vae_enc_tiled_golden.py (enc_in) or h3_vae_enc_video_golden.py (vid_in)
#include "engine/config.h"
#include "engine/loader/safetensors.h"
#include "engine/video/h3_vae_enc.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

int main( int argc, char** argv ) {
	if ( argc < 3 ) { std::printf( "usage: test_h3_vae_enc <video_vae dir> <golden> [gpus]\n" ); return 1; }
	const int gpus = argc > 3 ? std::atoi( argv[3] ) : 1;
	pod::SafeTensorsFile st;
	ENGINE_ASSERT( engine::loader::safetensors_open( st, argv[2] ), "golden" );
	std::vector<std::unique_ptr<engine::video::H3ImageEncoder>> owned;
	std::vector<engine::video::H3ImageEncoder*> encs;
	for ( int d = 0; d < gpus; ++d ) {
		owned.emplace_back( new engine::video::H3ImageEncoder );
		ENGINE_ASSERT( owned.back()->load( argv[1], d ), "load" );
		encs.push_back( owned.back().get() );
	}
	const bool video = engine::loader::safetensors_find( st, "vid_in" ) != nullptr;
	const pod::SafeTensorMeta* in = engine::loader::safetensors_find( st, video ? "vid_in" : "enc_in" );
	const pod::SafeTensorMeta* mo = engine::loader::safetensors_find( st, video ? "vid_moments" : "enc_moments" );
	ENGINE_ASSERT( in && mo, "golden tensors" );
	const float* x = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( st, in ) );
	std::vector<float> got;
	int tl = 0, T = 1, H = 0, W = 0;
	auto t0 = std::chrono::steady_clock::now();
	if ( video ) {
		T = in->shape[0]; H = in->shape[2]; W = in->shape[3];
		ENGINE_ASSERT( engine::video::H3ImageEncoder::encode_video( encs, x, T, H, W, &got, &tl ), "encode_video" );
	} else {
		H = in->shape[1]; W = in->shape[2];
		ENGINE_ASSERT( engine::video::H3ImageEncoder::encode( encs, x, H, W, &got ), "encode" );
	}
	const double secs = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
	ENGINE_ASSERT( got.size() * 4 == mo->size_bytes, "moment count" );
	const float* ref = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( st, mo ) );
	double dot = 0, na = 0, nb = 0, nd = 0;
	for ( size_t i = 0; i < got.size(); ++i ) {
		dot += static_cast<double>( got[i] ) * ref[i]; na += static_cast<double>( got[i] ) * got[i]; nb += static_cast<double>( ref[i] ) * ref[i];
		nd += ( static_cast<double>( got[i] ) - ref[i] ) * ( static_cast<double>( got[i] ) - ref[i] );
	}
	const double cos = dot / std::sqrt( na * nb );
	std::printf( "%s %d x %d%s on %d GPU(s) (%.2f s): moments cosine %.6f rel %.4g\n", video ? "clip" : "image", W, H, video ? ( " x " + std::to_string( T ) + " frames -> " + std::to_string( tl ) + " latent frames" ).c_str() : "", gpus, secs, cos, std::sqrt( nd / nb ) );
	std::printf( cos > 0.9999 ? "ok\n" : "FAIL\n" );
	return cos > 0.9999 ? 0 : 1;
}
