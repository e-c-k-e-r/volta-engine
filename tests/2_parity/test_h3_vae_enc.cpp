// H3 VAE image encoder parity (moments of the checkpoint's encoder on one frame).   test_h3_vae_enc <video_vae dir> <vae golden>
#include "engine/config.h"
#include "engine/loader/safetensors.h"
#include "engine/video/h3_vae_enc.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

int main( int argc, char** argv ) {
	if ( argc < 3 ) { std::printf( "usage: test_h3_vae_enc <video_vae dir> <golden>\n" ); return 1; }
	pod::SafeTensorsFile st;
	ENGINE_ASSERT( engine::loader::safetensors_open( st, argv[2] ), "golden" );
	if ( engine::loader::safetensors_find( st, "vid_in" ) ) { // clip golden (tests/tools/h3_vae_enc_video_golden.py)
		const pod::SafeTensorMeta* vi = engine::loader::safetensors_find( st, "vid_in" );
		const pod::SafeTensorMeta* vm = engine::loader::safetensors_find( st, "vid_moments" );
		const int T = vi->shape[0], H = vi->shape[2], W = vi->shape[3];
		engine::video::H3ImageEncoder enc;
		ENGINE_ASSERT( enc.load( argv[1], 0 ), "load" );
		float* d_in;
		CUDA_CHECK( cudaMalloc( &d_in, vi->size_bytes ) );
		CUDA_CHECK( cudaMemcpy( d_in, engine::loader::safetensors_get_data_cpu( st, vi ), vi->size_bytes, cudaMemcpyHostToDevice ) );
		std::vector<float> got;
		int tl = 0;
		auto t0 = std::chrono::steady_clock::now();
		ENGINE_ASSERT( enc.encode_video( d_in, T, H, W, &got, &tl ), "encode_video" );
		const double secs = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
		ENGINE_ASSERT( got.size() * 4 == vm->size_bytes, "moment count" );
		const float* ref = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( st, vm ) );
		double dot = 0, na = 0, nb = 0, nd = 0;
		for ( size_t i = 0; i < got.size(); ++i ) {
			dot += static_cast<double>( got[i] ) * ref[i]; na += static_cast<double>( got[i] ) * got[i]; nb += static_cast<double>( ref[i] ) * ref[i];
			nd += ( static_cast<double>( got[i] ) - ref[i] ) * ( static_cast<double>( got[i] ) - ref[i] );
		}
		const double cos = dot / std::sqrt( na * nb );
		std::printf( "clip %d frames %dx%d -> %d latent frames (%.2f s): moments cosine %.6f rel %.4g\n", T, W, H, tl, secs, cos, std::sqrt( nd / nb ) );
		std::printf( cos > 0.9999 ? "ok\n" : "FAIL\n" );
		return cos > 0.9999 ? 0 : 1;
	}
	const pod::SafeTensorMeta* in = engine::loader::safetensors_find( st, "enc_in" );
	const pod::SafeTensorMeta* mo = engine::loader::safetensors_find( st, "enc_moments" );
	ENGINE_ASSERT( in && mo, "golden tensors (regenerate with tests/tools/h3_vae_golden.py)" );
	const int H = in->shape[1], W = in->shape[2];
	engine::video::H3ImageEncoder enc;
	ENGINE_ASSERT( enc.load( argv[1], 0 ), "load" );
	float *d_in, *d_out;
	CUDA_CHECK( cudaMalloc( &d_in, in->size_bytes ) );
	CUDA_CHECK( cudaMalloc( &d_out, mo->size_bytes ) );
	CUDA_CHECK( cudaMemcpy( d_in, engine::loader::safetensors_get_data_cpu( st, in ), in->size_bytes, cudaMemcpyHostToDevice ) );
	auto t0 = std::chrono::steady_clock::now();
	ENGINE_ASSERT( enc.encode( d_in, H, W, d_out ), "encode" );
	const double secs = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
	std::vector<float> got( mo->size_bytes / 4 );
	CUDA_CHECK( cudaMemcpy( got.data(), d_out, mo->size_bytes, cudaMemcpyDeviceToHost ) );
	const float* ref = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( st, mo ) );
	double dot = 0, na = 0, nb = 0, nd = 0;
	for ( size_t i = 0; i < got.size(); ++i ) {
		dot += static_cast<double>( got[i] ) * ref[i]; na += static_cast<double>( got[i] ) * got[i]; nb += static_cast<double>( ref[i] ) * ref[i];
		nd += ( static_cast<double>( got[i] ) - ref[i] ) * ( static_cast<double>( got[i] ) - ref[i] );
	}
	const double cos = dot / std::sqrt( na * nb );
	std::printf( "image %dx%d (%.2f s): moments cosine %.6f rel %.4g\n", W, H, secs, cos, std::sqrt( nd / nb ) );
	std::printf( cos > 0.9999 ? "ok\n" : "FAIL\n" );
	return cos > 0.9999 ? 0 : 1;
}
