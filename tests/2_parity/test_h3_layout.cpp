// H3 packed-layout builder against the golden made from 1Cat's builder (tests/tools/h3_golden.py).   test_h3_layout <golden>
#include "engine/config.h"
#include "engine/loader/safetensors.h"
#include "engine/video/h3_pipeline.h"
#include <cmath>
#include <cstdio>

int main( int argc, char** argv ) {
	if ( argc < 2 ) { std::printf( "usage: test_h3_layout <golden>\n" ); return 1; }
	pod::SafeTensorsFile st;
	ENGINE_ASSERT( engine::loader::safetensors_open( st, argv[1] ), "golden" );
	auto get = [&]( const char* n ) { const pod::SafeTensorMeta* m = engine::loader::safetensors_find( st, n ); ENGINE_ASSERT( m, n ); return m; };
	const pod::SafeTensorMeta* pm = get( "img_position_ids" );
	const int text = get( "prompt_embeds" )->shape[0];
	// the golden is the default small case: 22 frames, 192x320, 32 text tokens
	const int frames = engine::video::h3_align_frames( 22 ), lt = engine::video::h3_video_latent_t( frames ), at = engine::video::h3_audio_latent_t( frames / 24.0 );
	const engine::video::H3Layout L = engine::video::h3_build_layout( text, lt, 12, 20, at );
	const float* pos = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( st, pm ) );
	const int* tags = static_cast<const int*>( engine::loader::safetensors_get_data_cpu( st, get( "token_tags" ) ) );
	const int* ip = static_cast<const int*>( engine::loader::safetensors_get_data_cpu( st, get( "img_pos" ) ) );
	const int* ap = static_cast<const int*>( engine::loader::safetensors_get_data_cpu( st, get( "audio_pos" ) ) );
	int bad = 0;
	double maxd = 0;
	for ( int i = 0; i < L.used; ++i ) {
		for ( int c = 0; c < 3; ++c ) { maxd = std::fmax( maxd, std::fabs( pos[static_cast<size_t>( i ) * 3 + c] - L.position_ids[static_cast<size_t>( i ) * 3 + c] ) ); }
		if ( tags[i] != L.tags[i] ) { ++bad; }
	}
	for ( int i = 0; i < L.n_img; ++i ) { if ( ip[i] != L.img_pos[i] ) { ++bad; } }
	for ( int i = 0; i < L.n_audio; ++i ) { if ( ap[i] != L.audio_pos[i] ) { ++bad; } }
	std::printf( "frames %d latent_t %d audio_t %d used %d (golden rows %d): position max diff %.3g, index/tag mismatches %d\n", frames, lt, at, L.used, pm->shape[0], maxd, bad );
	const bool ok = bad == 0 && maxd < 1e-4;
	std::printf( ok ? "ok\n" : "FAIL\n" );
	return ok ? 0 : 1;
}
