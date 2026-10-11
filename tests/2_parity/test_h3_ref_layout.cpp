// Ref2VA packed-layout builder against the golden from 1Cat's builder (tests/tools/h3_ref_layout_golden.py).   test_h3_ref_layout <golden>
#include "engine/config.h"
#include "engine/loader/safetensors.h"
#include "engine/video/h3_pipeline.h"
#include <cmath>
#include <cstdio>

int main( int argc, char** argv ) {
	if ( argc < 2 ) { std::printf( "usage: test_h3_ref_layout <golden>\n" ); return 1; }
	pod::SafeTensorsFile st;
	ENGINE_ASSERT( engine::loader::safetensors_open( st, argv[1] ), "golden" );
	auto get = [&]( const char* n ) { const pod::SafeTensorMeta* m = engine::loader::safetensors_find( st, n ); ENGINE_ASSERT( m, n ); return m; };
	using engine::video::H3RefBlock;
	std::vector<H3RefBlock> blocks = {
		{ H3RefBlock::Image, 0, 1, 24, 32 },
		{ H3RefBlock::Image, 0, 1, 16, 16 },
		{ H3RefBlock::Video, 60, 7, 16, 24 },
		{ H3RefBlock::Video, 0, 2, 12, 20 },
		{ H3RefBlock::Audio, 100, 1, 0, 0 },
	};
	const engine::video::H3Layout L = engine::video::h3_build_layout_ref( 20, 7, 12, 20, 50, blocks );
	const float* pos = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( st, get( "img_position_ids" ) ) );
	const int* tags = static_cast<const int*>( engine::loader::safetensors_get_data_cpu( st, get( "token_tags" ) ) );
	const int* ip = static_cast<const int*>( engine::loader::safetensors_get_data_cpu( st, get( "img_pos" ) ) );
	const int* ap = static_cast<const int*>( engine::loader::safetensors_get_data_cpu( st, get( "audio_pos" ) ) );
	const uint8_t* um = static_cast<const uint8_t*>( engine::loader::safetensors_get_data_cpu( st, get( "update_mask" ) ) );
	const uint8_t* aum = static_cast<const uint8_t*>( engine::loader::safetensors_get_data_cpu( st, get( "audio_update_mask" ) ) );
	const int used = *static_cast<const int*>( engine::loader::safetensors_get_data_cpu( st, get( "used" ) ) );
	int bad = L.used != used || static_cast<int>( get( "img_pos" )->shape[0] ) != L.n_img || static_cast<int>( get( "audio_pos" )->shape[0] ) != L.n_audio;
	double maxd = 0;
	for ( int i = 0; i < L.used; ++i ) {
		for ( int c = 0; c < 3; ++c ) { maxd = std::fmax( maxd, std::fabs( pos[static_cast<size_t>( i ) * 3 + c] - L.position_ids[static_cast<size_t>( i ) * 3 + c] ) ); }
		if ( tags[i] != L.tags[i] ) { ++bad; }
	}
	for ( int i = 0; i < L.n_img; ++i ) { if ( ip[i] != L.img_pos[i] || um[i] != L.update_mask[i] ) { ++bad; } }
	for ( int i = 0; i < L.n_audio; ++i ) { if ( ap[i] != L.audio_pos[i] || aum[i] != L.audio_update[i] ) { ++bad; } }
	std::printf( "used %d (golden %d), %d video rows, %d audio rows: position max diff %.3g, mismatches %d\n", L.used, used, L.n_img, L.n_audio, maxd, bad );
	const bool ok = bad == 0 && maxd < 1e-4;
	std::printf( ok ? "ok\n" : "FAIL\n" );
	return ok ? 0 : 1;
}
