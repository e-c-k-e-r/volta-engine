// H3 video VAE parity against tests/tools/h3_vae_golden.py output: the raw ViT decoder on one tile, then the full tiled / chunked decode.
//   test_h3_vae <video_vae dir> <golden.safetensors>
#include "engine/config.h"
#include "engine/loader/safetensors.h"
#include "engine/video/h3_vae.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

static pod::SafeTensorsFile g_st;

static const pod::SafeTensorMeta* need( const char* name ) {
	const pod::SafeTensorMeta* m = engine::loader::safetensors_find( g_st, name );
	ENGINE_ASSERT( m, name );
	return m;
}

int main( int argc, char** argv ) {
	if ( argc < 3 ) { std::printf( "usage: test_h3_vae <video_vae dir> <golden>\n" ); return 1; }
	ENGINE_ASSERT( engine::loader::safetensors_open( g_st, argv[2] ), "golden" );
	engine::video::H3VideoVAE vae;
	auto t0 = std::chrono::steady_clock::now();
	ENGINE_ASSERT( vae.load( argv[1], 0 ), "load" );
	std::printf( "loaded %.2f GB in %.1f s\n", vae.weight_bytes() / 1e9, std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() );
	int bad = 0;
	{ // raw tile
		const pod::SafeTensorMeta* zi = need( "tile_in" );
		const pod::SafeTensorMeta* zo = need( "tile_out" );
		const int T = zi->shape[1], H = zi->shape[2], W = zi->shape[3];
		float *d_z, *d_p;
		CUDA_CHECK( cudaMalloc( &d_z, zi->size_bytes ) );
		CUDA_CHECK( cudaMalloc( &d_p, zo->size_bytes ) );
		CUDA_CHECK( cudaMemcpy( d_z, engine::loader::safetensors_get_data_cpu( g_st, zi ), zi->size_bytes, cudaMemcpyHostToDevice ) );
		auto t1 = std::chrono::steady_clock::now();
		vae.decode_tile( d_z, T, H, W, d_p );
		const double secs = std::chrono::duration<double>( std::chrono::steady_clock::now() - t1 ).count();
		std::vector<float> got( zo->size_bytes / 4 );
		CUDA_CHECK( cudaMemcpy( got.data(), d_p, zo->size_bytes, cudaMemcpyDeviceToHost ) );
		const float* ref = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( g_st, zo ) );
		double dot = 0, na = 0, nb = 0, nd = 0;
		for ( size_t i = 0; i < got.size(); ++i ) {
			dot += static_cast<double>( got[i] ) * ref[i]; na += static_cast<double>( got[i] ) * got[i]; nb += static_cast<double>( ref[i] ) * ref[i];
			nd += ( static_cast<double>( got[i] ) - ref[i] ) * ( static_cast<double>( got[i] ) - ref[i] );
		}
		std::printf( "tile %dx%dx%d (%.2f s): cosine %.6f rel %.4g\n", T, H, W, secs, dot / std::sqrt( na * nb ), std::sqrt( nd / nb ) );
		if ( dot / std::sqrt( na * nb ) < 0.999 ) { ++bad; }
	}
	{ // full decode
		const pod::SafeTensorMeta* z = need( "z" );
		const pod::SafeTensorMeta* fr = need( "frames" );
		const int T = z->shape[1], H = z->shape[2], W = z->shape[3];
		float* d_z;
		CUDA_CHECK( cudaMalloc( &d_z, z->size_bytes ) );
		CUDA_CHECK( cudaMemcpy( d_z, engine::loader::safetensors_get_data_cpu( g_st, z ), z->size_bytes, cudaMemcpyHostToDevice ) );
		std::vector<uint8_t> rgb;
		int frames = 0;
		auto t1 = std::chrono::steady_clock::now();
		ENGINE_ASSERT( vae.decode_video( d_z, T, H, W, rgb, frames ), "decode_video" );
		const double secs = std::chrono::duration<double>( std::chrono::steady_clock::now() - t1 ).count();
		const int RF = fr->shape[1], RH = fr->shape[2], RW = fr->shape[3];
		std::printf( "video: %d frames of %dx%d (golden %d of %dx%d) in %.2f s\n", frames, W * 16, H * 16, RF, RW, RH, secs );
		if ( frames != RF ) { ++bad; } else {
			const float* ref = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( g_st, fr ) );
			double se = 0, mx = 0;
			const size_t plane = static_cast<size_t>( RH ) * RW;
			for ( int f = 0; f < frames; ++f ) {
				double fse = 0;
				for ( size_t p = 0; p < plane; ++p ) {
					for ( int c = 0; c < 3; ++c ) {
						const double a = rgb[( static_cast<size_t>( f ) * plane + p ) * 3 + c] / 255.0, b = ref[( static_cast<size_t>( c ) * RF + f ) * plane + p];
						se += ( a - b ) * ( a - b );
						fse += ( a - b ) * ( a - b );
						mx = std::fmax( mx, std::fabs( a - b ) );
					}
				}
				if ( 10.0 * std::log10( 1.0 / ( fse / ( plane * 3 ) + 1e-12 ) ) < 40.0 ) { std::printf( "  frame %d is wrong (PSNR %.1f dB)\n", f, 10.0 * std::log10( 1.0 / ( fse / ( plane * 3 ) + 1e-12 ) ) ); }
			}
			const double mse = se / ( static_cast<double>( frames ) * plane * 3 );
			const double psnr = 10.0 * std::log10( 1.0 / ( mse + 1e-12 ) );
			std::printf( "video vs golden: PSNR %.2f dB, max abs diff %.4f\n", psnr, mx );
			if ( psnr < 40.0 ) { ++bad; }
		}
	}
	std::printf( bad ? "FAIL\n" : "ok\n" );
	return bad ? 1 : 0;
}
