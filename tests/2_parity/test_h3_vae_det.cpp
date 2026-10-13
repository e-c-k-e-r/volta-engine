// Run-to-run determinism of the VAE decoder tile at several shapes: the same latent decoded repeatedly must give identical pixels.   test_h3_vae_det <video_vae dir> [repeats=12]
#include "engine/config.h"
#include "engine/loader/safetensors.h"
#include "engine/video/h3_vae.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main( int argc, char** argv ) {
	if ( argc < 2 ) { std::printf( "usage: test_h3_vae_det <video_vae dir> [repeats]\n" ); return 1; }
	const int repeats = argc > 2 ? std::atoi( argv[2] ) : 12;
	engine::video::H3VideoVAE vae;
	ENGINE_ASSERT( vae.load( argv[1], 0 ), "load" );
	int bad = 0;
	const int shapes[][3] = { { 5, 20, 16 }, { 7, 16, 16 }, { 7, 16, 12 }, { 2, 9, 7 } };
	for ( const auto& sh : shapes ) {
		const int T = sh[0], H = sh[1], W = sh[2];
		const size_t zn = static_cast<size_t>( 24 ) * T * H * W, pn = static_cast<size_t>( 3 ) * T * 4 * H * 16 * W * 16;
		std::vector<float> z( zn );
		uint32_t s = 11;
		for ( float& x : z ) { s = s * 1664525u + 1013904223u; x = ( ( s >> 8 ) / 16777216.0f - 0.5f ) * 2.0f; }
		float *d_z, *d_p;
		CUDA_CHECK( cudaMalloc( &d_z, zn * 4 ) );
		CUDA_CHECK( cudaMalloc( &d_p, pn * 4 ) );
		CUDA_CHECK( cudaMemcpy( d_z, z.data(), zn * 4, cudaMemcpyHostToDevice ) );
		std::vector<float> first( pn ), cur( pn );
		int diffs = 0;
		for ( int r = 0; r < repeats; ++r ) {
			vae.decode_tile( d_z, T, H, W, d_p );
			CUDA_CHECK( cudaMemcpy( r == 0 ? first.data() : cur.data(), d_p, pn * 4, cudaMemcpyDeviceToHost ) );
			if ( r > 0 ) {
				size_t nd = 0;
				for ( size_t i = 0; i < pn; ++i ) { nd += first[i] != cur[i]; }
				if ( nd ) { ++diffs; }
			}
		}
		std::printf( "tile %dx%dx%d: %d of %d repeats differ from the first\n", T, H, W, diffs, repeats - 1 );
		bad += diffs ? 1 : 0;
		(void) cudaFree( d_z );
		(void) cudaFree( d_p );
	}
	if ( argc > 3 ) { // the full tiled decode of a golden latent, repeatedly: which frames and which pixels change
		pod::SafeTensorsFile st;
		ENGINE_ASSERT( engine::loader::safetensors_open( st, argv[3] ), "golden" );
		const pod::SafeTensorMeta* zm = engine::loader::safetensors_find( st, "z" );
		ENGINE_ASSERT( zm, "z" );
		const int T = zm->shape[1], H = zm->shape[2], W = zm->shape[3];
		float* d_z;
		CUDA_CHECK( cudaMalloc( &d_z, zm->size_bytes ) );
		CUDA_CHECK( cudaMemcpy( d_z, engine::loader::safetensors_get_data_cpu( st, zm ), zm->size_bytes, cudaMemcpyHostToDevice ) );
		std::vector<uint8_t> first, cur;
		int frames = 0;
		for ( int r = 0; r < repeats; ++r ) {
			ENGINE_ASSERT( vae.decode_video( d_z, T, H, W, r == 0 ? first : cur, frames ), "decode_video" );
			{
				uint64_t h = 1469598103934665603ull;
				for ( uint8_t b : r == 0 ? first : cur ) { h = ( h ^ b ) * 1099511628211ull; }
				std::printf( "full decode %d: hash %016llx\n", r, static_cast<unsigned long long>( h ) );
			}
			if ( r == 0 ) { continue; }
			const size_t plane = static_cast<size_t>( H ) * 16 * W * 16;
			int wrong_frames = 0, y0 = 1 << 30, y1 = -1, x0 = 1 << 30, x1 = -1;
			for ( int f = 0; f < frames; ++f ) {
				bool any = false;
				for ( size_t p = 0; p < plane; ++p ) {
					for ( int c = 0; c < 3; ++c ) {
						if ( first[( static_cast<size_t>( f ) * plane + p ) * 3 + c] != cur[( static_cast<size_t>( f ) * plane + p ) * 3 + c] ) {
							any = true;
							const int y = static_cast<int>( p / ( W * 16 ) ), x = static_cast<int>( p % ( W * 16 ) );
							y0 = std::min( y0, y ); y1 = std::max( y1, y ); x0 = std::min( x0, x ); x1 = std::max( x1, x );
						}
					}
				}
				wrong_frames += any ? 1 : 0;
			}
			std::printf( "full decode repeat %d: %d of %d frames differ from the first%s", r, wrong_frames, frames, wrong_frames ? "" : "\n" );
			if ( wrong_frames ) { std::printf( ", rows %d..%d cols %d..%d\n", y0, y1, x0, x1 ); ++bad; }
		}
	}
	std::printf( bad ? "FAIL\n" : "ok\n" );
	return bad ? 1 : 0;
}
