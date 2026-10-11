// H3 image-conditioned text encoding parity (vision tower + DeepStack + mRoPE) against tests/tools/h3_encoder_mm_golden.py.
//   test_h3_text_mm <text_encoder dir> <golden> [layers=50]
#include "engine/config.h"
#include "engine/loader/safetensors.h"
#include "engine/video/h3_text.h"
#include "engine/video/h3_vision.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main( int argc, char** argv ) {
	if ( argc < 3 ) { std::printf( "usage: test_h3_text_mm <text_encoder dir> <golden> [layers]\n" ); return 1; }
	const int layers = argc > 3 ? std::atoi( argv[3] ) : 50;
	pod::SafeTensorsFile st;
	ENGINE_ASSERT( engine::loader::safetensors_open( st, argv[2] ), "golden" );
	auto get = [&]( const char* n ) { const pod::SafeTensorMeta* m = engine::loader::safetensors_find( st, n ); ENGINE_ASSERT( m, n ); return m; };
	const int* gid = static_cast<const int*>( engine::loader::safetensors_get_data_cpu( st, get( "ids" ) ) );
	std::vector<int> ids( gid, gid + get( "ids" )->shape[0] );
	const int* grid = static_cast<const int*>( engine::loader::safetensors_get_data_cpu( st, get( "grid" ) ) );
	const pod::SafeTensorMeta* pm = get( "pixel_values" );
	const float* px = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( st, pm ) );
	std::vector<float> patches( px, px + pm->size_bytes / 4 );
	engine::video::H3ImageFeatures feat;
	feat.grid_h = grid[1];
	feat.grid_w = grid[2];
	{
		engine::video::H3VisionTower tower;
		ENGINE_ASSERT( tower.load( argv[1], 0, 4096 ), "vision load" );
		ENGINE_ASSERT( tower.encode( patches, grid[1], grid[2], &feat.merged, feat.deepstack ), "vision encode" );
	}
	engine::video::H3TextEncoder enc;
	ENGINE_ASSERT( enc.load( argv[1], 0, 4096, layers ), "text load" );
	float* d;
	const pod::SafeTensorMeta* hm = get( "hidden" );
	CUDA_CHECK( cudaMalloc( &d, hm->size_bytes ) );
	auto t0 = std::chrono::steady_clock::now();
	ENGINE_ASSERT( enc.encode_mm( ids, 151655, { feat }, d ), "encode_mm" );
	const double secs = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
	std::vector<float> got( hm->size_bytes / 4 );
	CUDA_CHECK( cudaMemcpy( got.data(), d, hm->size_bytes, cudaMemcpyDeviceToHost ) );
	const float* ref = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( st, hm ) );
	double dot = 0, na = 0, nb = 0, nd = 0;
	for ( size_t i = 0; i < got.size(); ++i ) {
		dot += static_cast<double>( got[i] ) * ref[i]; na += static_cast<double>( got[i] ) * got[i]; nb += static_cast<double>( ref[i] ) * ref[i];
		nd += ( static_cast<double>( got[i] ) - ref[i] ) * ( static_cast<double>( got[i] ) - ref[i] );
	}
	const double cos = dot / std::sqrt( na * nb );
	std::printf( "%zu tokens (%zu image), %d layers, %.1f s: cosine %.6f rel %.4g\n", ids.size(), feat.merged.size() / 5120, layers, secs, cos, std::sqrt( nd / nb ) );
	std::printf( cos > 0.999 ? "ok\n" : "FAIL\n" );
	return cos > 0.999 ? 0 : 1;
}
