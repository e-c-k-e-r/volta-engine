// H3 vision tower parity against tests/tools/h3_vision_golden.py (HF transformers, FP32).   test_h3_vision <text_encoder dir> <golden>
#include "engine/config.h"
#include "engine/loader/safetensors.h"
#include "engine/video/h3_vision.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

static double cosine( const std::vector<float>& a, const float* b, double* rel ) {
	double dot = 0, na = 0, nb = 0, nd = 0;
	for ( size_t i = 0; i < a.size(); ++i ) { dot += static_cast<double>( a[i] ) * b[i]; na += static_cast<double>( a[i] ) * a[i]; nb += static_cast<double>( b[i] ) * b[i]; nd += ( static_cast<double>( a[i] ) - b[i] ) * ( static_cast<double>( a[i] ) - b[i] ); }
	*rel = std::sqrt( nd / nb );
	return dot / std::sqrt( na * nb );
}

int main( int argc, char** argv ) {
	if ( argc < 3 ) { std::printf( "usage: test_h3_vision <text_encoder dir> <golden>\n" ); return 1; }
	pod::SafeTensorsFile st;
	ENGINE_ASSERT( engine::loader::safetensors_open( st, argv[2] ), "golden" );
	auto get = [&]( const char* n ) { const pod::SafeTensorMeta* m = engine::loader::safetensors_find( st, n ); ENGINE_ASSERT( m, n ); return m; };
	const pod::SafeTensorMeta* pm = get( "pixel_values" );
	const int* grid = static_cast<const int*>( engine::loader::safetensors_get_data_cpu( st, get( "grid" ) ) );
	const float* px = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( st, pm ) );
	std::vector<float> patches( px, px + pm->size_bytes / 4 );
	engine::video::H3VisionTower tower;
	auto t0 = std::chrono::steady_clock::now();
	ENGINE_ASSERT( tower.load( argv[1], 0, 4096 ), "load" );
	const double tl = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
	std::vector<float> merged, ds[3];
	t0 = std::chrono::steady_clock::now();
	ENGINE_ASSERT( tower.encode( patches, grid[1], grid[2], &merged, ds ), "encode" );
	const double te = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
	int bad = 0;
	double rel;
	double c = cosine( merged, static_cast<const float*>( engine::loader::safetensors_get_data_cpu( st, get( "merged" ) ) ), &rel );
	std::printf( "grid %dx%d (load %.1f s, encode %.2f s): merged cosine %.6f rel %.4g\n", grid[1], grid[2], tl, te, c, rel );
	if ( c < 0.999 ) { ++bad; }
	for ( int i = 0; i < 3; ++i ) {
		const std::string n = "deepstack" + std::to_string( i );
		c = cosine( ds[i], static_cast<const float*>( engine::loader::safetensors_get_data_cpu( st, get( n.c_str() ) ) ), &rel );
		std::printf( "  %s cosine %.6f rel %.4g\n", n.c_str(), c, rel );
		if ( c < 0.999 ) { ++bad; }
	}
	std::printf( bad ? "FAIL\n" : "ok\n" );
	return bad ? 1 : 0;
}
