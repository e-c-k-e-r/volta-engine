// H3 Lanczos resize against PIL (tests/tools/h3_image_golden.py).   test_h3_image <golden>
#include "engine/config.h"
#include "engine/loader/safetensors.h"
#include "engine/video/h3_image.h"
#include <cmath>
#include <cstdio>

int main( int argc, char** argv ) {
	if ( argc < 2 ) { std::printf( "usage: test_h3_image <golden>\n" ); return 1; }
	pod::SafeTensorsFile st;
	ENGINE_ASSERT( engine::loader::safetensors_open( st, argv[1] ), "golden" );
	auto get = [&]( const char* n ) { const pod::SafeTensorMeta* m = engine::loader::safetensors_find( st, n ); ENGINE_ASSERT( m, n ); return m; };
	const pod::SafeTensorMeta* sm = get( "src" );
	const uint8_t* src = static_cast<const uint8_t*>( engine::loader::safetensors_get_data_cpu( st, sm ) );
	int bad = 0;
	for ( const char* name : { "down", "up" } ) {
		const pod::SafeTensorMeta* dm = get( name );
		const uint8_t* ref = static_cast<const uint8_t*>( engine::loader::safetensors_get_data_cpu( st, dm ) );
		const std::vector<uint8_t> out = engine::video::h3_resize_lanczos( src, sm->shape[0], sm->shape[1], dm->shape[0], dm->shape[1] );
		int mx = 0;
		double se = 0;
		for ( size_t i = 0; i < out.size(); ++i ) { const int d = std::abs( static_cast<int>( out[i] ) - ref[i] ); mx = std::max( mx, d ); se += d * d; }
		std::printf( "%s %dx%d: max level diff %d, rms %.3f\n", name, dm->shape[1], dm->shape[0], mx, std::sqrt( se / out.size() ) );
		if ( mx > 2 ) { ++bad; }
	}
	std::printf( bad ? "FAIL\n" : "ok\n" );
	return bad ? 1 : 0;
}
