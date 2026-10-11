// H3 audio VAE encoder parity against tests/tools/h3_audio_enc_golden.py.   test_h3_audio_enc <audio_vae dir> <golden.safetensors>
#include "engine/config.h"
#include "engine/loader/safetensors.h"
#include "engine/video/h3_audio.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

int main( int argc, char** argv ) {
	if ( argc < 3 ) { std::printf( "usage: test_h3_audio_enc <audio_vae dir> <golden>\n" ); return 1; }
	pod::SafeTensorsFile st;
	ENGINE_ASSERT( engine::loader::safetensors_open( st, argv[2] ), "golden" );
	const pod::SafeTensorMeta* wave = engine::loader::safetensors_find( st, "wave" );
	const pod::SafeTensorMeta* rows = engine::loader::safetensors_find( st, "rows" );
	ENGINE_ASSERT( wave && rows, "golden tensors" );
	const int samples = static_cast<int>( wave->shape[1] );
	const float* w = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( st, wave ) );
	engine::video::H3AudioVAE vae;
	ENGINE_ASSERT( vae.load( argv[1], 0, 64 ), "load" );
	std::vector<float> got;
	int frames = 0;
	auto t0 = std::chrono::steady_clock::now();
	ENGINE_ASSERT( vae.encode( std::vector<float>( w, w + static_cast<size_t>( 2 ) * samples ), samples, &got, &frames ), "encode" );
	const double secs = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
	const float* ref = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( st, rows ) );
	ENGINE_ASSERT( got.size() * 4 == rows->size_bytes, "row count" );
	double dot = 0, na = 0, nb = 0, nd = 0, mx = 0;
	for ( size_t i = 0; i < got.size(); ++i ) {
		dot += static_cast<double>( got[i] ) * ref[i]; na += static_cast<double>( got[i] ) * got[i]; nb += static_cast<double>( ref[i] ) * ref[i];
		nd += ( static_cast<double>( got[i] ) - ref[i] ) * ( static_cast<double>( got[i] ) - ref[i] );
		mx = std::fmax( mx, std::fabs( static_cast<double>( got[i] ) - ref[i] ) );
	}
	const double cos = dot / std::sqrt( na * nb + 1e-30 );
	std::printf( "audio encode %d samples -> %d frames in %.2f s: cosine %.6f rel %.4g max abs %.4g\n", samples, frames, secs, cos, std::sqrt( nd / ( nb + 1e-30 ) ), mx );
	std::printf( cos > 0.999 ? "ok\n" : "FAIL\n" );
	return cos > 0.999 ? 0 : 1;
}
