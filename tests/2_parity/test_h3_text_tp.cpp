// The resident EXL3 text tower over N GPUs (H3TextGroup, tensor parallel) against the BF16 goldens: text only, then with an image.
//   test_h3_text_tp <exl3 dir> <encoder_golden> <encoder_mm_golden> <ngpus> [vision dir = exl3 dir]
#include "engine/config.h"
#include "engine/loader/safetensors.h"
#include "engine/video/h3_text_group.h"
#include "engine/video/h3_vision.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static double cosine( const std::vector<float>& a, const float* b, double* rel ) {
	double dot = 0, na = 0, nb = 0, nd = 0;
	for ( size_t i = 0; i < a.size(); ++i ) {
		dot += static_cast<double>( a[i] ) * b[i]; na += static_cast<double>( a[i] ) * a[i]; nb += static_cast<double>( b[i] ) * b[i];
		nd += ( static_cast<double>( a[i] ) - b[i] ) * ( static_cast<double>( a[i] ) - b[i] );
	}
	*rel = std::sqrt( nd / nb );
	return dot / std::sqrt( na * nb );
}

int main( int argc, char** argv ) {
	if ( argc < 5 ) { std::printf( "usage: test_h3_text_tp <exl3 dir> <encoder_golden> <encoder_mm_golden> <ngpus> [vision dir]\n" ); return 1; }
	const int ngpu = std::atoi( argv[4] );
	std::vector<int> devs;
	for ( int i = 0; i < ngpu; ++i ) { devs.push_back( i ); }
	pod::SafeTensorsFile g1, g2;
	ENGINE_ASSERT( engine::loader::safetensors_open( g1, argv[2] ) && engine::loader::safetensors_open( g2, argv[3] ), "goldens" );
	auto get = [&]( pod::SafeTensorsFile& st, const char* n ) { const pod::SafeTensorMeta* m = engine::loader::safetensors_find( st, n ); ENGINE_ASSERT( m, n ); return m; };
	engine::video::H3TextGroup grp;
	auto t0 = std::chrono::steady_clock::now();
	ENGINE_ASSERT( grp.load( argv[1], devs, 4096, 50 ), "load" );
	std::printf( "text tower loaded on %d GPU(s) in %.1f s\n", ngpu, std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() );
	int bad = 0;
	{ // text only
		const pod::SafeTensorMeta* im = get( g1, "ids" );
		const pod::SafeTensorMeta* hm = get( g1, "hidden" );
		const int* gid = static_cast<const int*>( engine::loader::safetensors_get_data_cpu( g1, im ) );
		std::vector<int> ids( gid, gid + im->shape[0] );
		float* d;
		CUDA_CHECK( cudaMalloc( &d, hm->size_bytes ) );
		auto t1 = std::chrono::steady_clock::now();
		ENGINE_ASSERT( grp.encode_mm( ids, -1, {}, d ), "encode" );
		const double secs = std::chrono::duration<double>( std::chrono::steady_clock::now() - t1 ).count();
		std::vector<float> got( hm->size_bytes / 4 );
		CUDA_CHECK( cudaMemcpy( got.data(), d, hm->size_bytes, cudaMemcpyDeviceToHost ) );
		double rel;
		const double cos = cosine( got, static_cast<const float*>( engine::loader::safetensors_get_data_cpu( g1, hm ) ), &rel );
		std::printf( "text only, %zu tokens: %.2f s, cosine %.6f rel %.4g\n", ids.size(), secs, cos, rel );
		if ( cos < 0.999 ) { ++bad; }
		(void) cudaFree( d );
	}
	{ // with an image
		const int* gid = static_cast<const int*>( engine::loader::safetensors_get_data_cpu( g2, get( g2, "ids" ) ) );
		std::vector<int> ids( gid, gid + get( g2, "ids" )->shape[0] );
		const int* grid = static_cast<const int*>( engine::loader::safetensors_get_data_cpu( g2, get( g2, "grid" ) ) );
		const pod::SafeTensorMeta* pm = get( g2, "pixel_values" );
		const float* px = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( g2, pm ) );
		std::vector<float> patches( px, px + pm->size_bytes / 4 );
		engine::video::H3ImageFeatures feat;
		feat.grid_h = grid[1];
		feat.grid_w = grid[2];
		{
			engine::video::H3VisionTower tower;
			CUDA_CHECK( cudaSetDevice( 0 ) );
			ENGINE_ASSERT( tower.load( argc > 5 ? argv[5] : argv[1], 0, 4096 ), "vision load" );
			ENGINE_ASSERT( tower.encode( patches, grid[1], grid[2], &feat.merged, feat.deepstack ), "vision encode" );
		}
		const pod::SafeTensorMeta* hm = get( g2, "hidden" );
		float* d;
		CUDA_CHECK( cudaMalloc( &d, hm->size_bytes ) );
		auto t1 = std::chrono::steady_clock::now();
		ENGINE_ASSERT( grp.encode_mm( ids, 151655, { feat }, d ), "encode_mm" );
		const double secs = std::chrono::duration<double>( std::chrono::steady_clock::now() - t1 ).count();
		std::vector<float> got( hm->size_bytes / 4 );
		CUDA_CHECK( cudaMemcpy( got.data(), d, hm->size_bytes, cudaMemcpyDeviceToHost ) );
		double rel;
		const double cos = cosine( got, static_cast<const float*>( engine::loader::safetensors_get_data_cpu( g2, hm ) ), &rel );
		std::printf( "with an image, %zu tokens (%zu image): %.2f s, cosine %.6f rel %.4g\n", ids.size(), feat.merged.size() / 5120, secs, cos, rel );
		if ( cos < 0.999 ) { ++bad; }
	}
	std::printf( bad ? "FAIL\n" : "ok\n" );
	return bad ? 1 : 0;
}
