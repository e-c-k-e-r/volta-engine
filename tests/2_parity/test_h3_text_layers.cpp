// Where two quantizations of the text tower drift apart: runs the image-conditioned golden prompt through both checkpoints and prints, per decoder layer, the cosine and the
// relative difference of the residual stream, its largest magnitude, and the non-finite count.   test_h3_text_layers <dir A> <dir B> <encoder_mm_golden> <vision dir>
#include "engine/config.h"
#include "engine/loader/safetensors.h"
#include "engine/video/h3_text.h"
#include "engine/video/h3_vision.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static std::vector<std::vector<float>> run( const char* dir, const std::vector<int>& ids, const engine::video::H3ImageFeatures& feat ) {
	std::vector<std::vector<float>> layers;
	engine::video::H3TextEncoder enc;
	ENGINE_ASSERT( enc.load( dir, 0, 4096, 50 ), "load" );
	enc.probe = [&]( int, const float* d_x, int S ) {
		std::vector<float> h( static_cast<size_t>( S ) * 5120 );
		CUDA_CHECK( cudaMemcpy( h.data(), d_x, h.size() * 4, cudaMemcpyDeviceToHost ) );
		layers.push_back( std::move( h ) );
	};
	float* d;
	CUDA_CHECK( cudaMalloc( &d, ids.size() * 5120 * 4 ) );
	ENGINE_ASSERT( enc.encode_mm( ids, 151655, { feat }, d ), "encode" );
	(void) cudaFree( d );
	return layers;
}

int main( int argc, char** argv ) {
	if ( argc < 5 ) { std::printf( "usage: test_h3_text_layers <dirA> <dirB> <encoder_mm_golden> <vision dir>\n" ); return 1; }
	pod::SafeTensorsFile g;
	ENGINE_ASSERT( engine::loader::safetensors_open( g, argv[3] ), "golden" );
	auto get = [&]( const char* n ) { const pod::SafeTensorMeta* m = engine::loader::safetensors_find( g, n ); ENGINE_ASSERT( m, n ); return m; };
	const int* gid = static_cast<const int*>( engine::loader::safetensors_get_data_cpu( g, get( "ids" ) ) );
	std::vector<int> ids( gid, gid + get( "ids" )->shape[0] );
	const int* grid = static_cast<const int*>( engine::loader::safetensors_get_data_cpu( g, get( "grid" ) ) );
	const pod::SafeTensorMeta* pm = get( "pixel_values" );
	const float* px = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( g, pm ) );
	std::vector<float> patches( px, px + pm->size_bytes / 4 );
	engine::video::H3ImageFeatures feat;
	feat.grid_h = grid[1];
	feat.grid_w = grid[2];
	{
		engine::video::H3VisionTower tower;
		ENGINE_ASSERT( tower.load( argv[4], 0, 4096 ), "vision load" );
		ENGINE_ASSERT( tower.encode( patches, grid[1], grid[2], &feat.merged, feat.deepstack ), "vision encode" );
	}
	const auto a = run( argv[1], ids, feat );
	const auto b = run( argv[2], ids, feat );
	std::printf( "layer  cosine      rel diff   max|A|    max|B|    non-finite(B)\n" );
	for ( size_t l = 0; l < a.size() && l < b.size(); ++l ) {
		double dot = 0, na = 0, nb = 0, nd = 0, ma = 0, mb = 0;
		size_t nonfin = 0;
		for ( size_t i = 0; i < a[l].size(); ++i ) {
			const double x = a[l][i], y = b[l][i];
			if ( !std::isfinite( y ) ) { ++nonfin; continue; }
			dot += x * y; na += x * x; nb += y * y; nd += ( x - y ) * ( x - y );
			ma = std::fmax( ma, std::fabs( x ) ); mb = std::fmax( mb, std::fabs( y ) );
		}
		std::printf( "%4zu  %.6f  %.4g  %8.1f  %8.1f  %zu\n", l, dot / std::sqrt( na * nb ), std::sqrt( nd / na ), ma, mb, nonfin );
	}
	// which tokens and channels carry the jump: the per-token difference at the layers around the largest increase
	for ( size_t l : { size_t( 42 ), size_t( 43 ) } ) {
		if ( l >= a.size() ) { continue; }
		const size_t S = a[l].size() / 5120;
		std::vector<std::pair<double, int>> dt;
		for ( size_t t = 0; t < S; ++t ) {
			double nd = 0;
			for ( int c = 0; c < 5120; ++c ) { const double d = static_cast<double>( a[l][t * 5120 + c] ) - b[l][t * 5120 + c]; nd += d * d; }
			dt.push_back( { std::sqrt( nd ), static_cast<int>( t ) } );
		}
		std::sort( dt.rbegin(), dt.rend() );
		std::printf( "layer %zu: tokens with the largest difference (norm, token, |A|, |B|):", l );
		for ( int k = 0; k < 5; ++k ) {
			double na = 0, nb = 0;
			for ( int c = 0; c < 5120; ++c ) { na += static_cast<double>( a[l][dt[k].second * 5120 + c] ) * a[l][dt[k].second * 5120 + c]; nb += static_cast<double>( b[l][dt[k].second * 5120 + c] ) * b[l][dt[k].second * 5120 + c]; }
			std::printf( " (%.0f, %d, %.0f, %.0f)", dt[k].first, dt[k].second, std::sqrt( na ), std::sqrt( nb ) );
		}
		std::printf( "\n" );
		int tk = dt[0].second;
		std::vector<std::pair<double, int>> dc;
		for ( int c = 0; c < 5120; ++c ) { dc.push_back( { std::fabs( static_cast<double>( a[l][tk * 5120 + c] ) - b[l][tk * 5120 + c] ), c } ); }
		std::sort( dc.rbegin(), dc.rend() );
		std::printf( "  token %d, channels with the largest difference (|diff|, channel, A, B):", tk );
		for ( int k = 0; k < 4; ++k ) { std::printf( " (%.0f, %d, %.0f, %.0f)", dc[k].first, dc[k].second, a[l][tk * 5120 + dc[k].second], b[l][tk * 5120 + dc[k].second] ); }
		std::printf( "\n" );
	}
	return 0;
}
