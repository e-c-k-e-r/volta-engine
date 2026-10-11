// H3 audio VAE parity against tests/tools/h3_audio_golden.py.   test_h3_audio <audio_vae dir> <golden.safetensors>
#include "engine/config.h"
#include "engine/loader/safetensors.h"
#include "engine/video/h3_audio.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

int main( int argc, char** argv ) {
	if ( argc < 3 ) { std::printf( "usage: test_h3_audio <audio_vae dir> <golden>\n" ); return 1; }
	pod::SafeTensorsFile st;
	ENGINE_ASSERT( engine::loader::safetensors_open( st, argv[2] ), "golden" );
	const pod::SafeTensorMeta* rows = engine::loader::safetensors_find( st, "rows" );
	const pod::SafeTensorMeta* wave = engine::loader::safetensors_find( st, "wave" );
	ENGINE_ASSERT( rows && wave, "golden tensors" );
	const int T = rows->shape[0] / 2;
	engine::video::H3AudioVAE vae;
	ENGINE_ASSERT( vae.load( argv[1], 0 ), "load" );
	float* d;
	CUDA_CHECK( cudaMalloc( &d, rows->size_bytes ) );
	CUDA_CHECK( cudaMemcpy( d, engine::loader::safetensors_get_data_cpu( st, rows ), rows->size_bytes, cudaMemcpyHostToDevice ) );
	std::vector<float> pcm;
	auto t0 = std::chrono::steady_clock::now();
	ENGINE_ASSERT( vae.decode( d, T, pcm ), "decode" );
	const double secs = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
	const float* ref = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( st, wave ) );
	ENGINE_ASSERT( pcm.size() * 4 == wave->size_bytes, "waveform length" );
	double dot = 0, na = 0, nb = 0, nd = 0, mx = 0;
	for ( size_t i = 0; i < pcm.size(); ++i ) {
		dot += static_cast<double>( pcm[i] ) * ref[i]; na += static_cast<double>( pcm[i] ) * pcm[i]; nb += static_cast<double>( ref[i] ) * ref[i];
		nd += ( static_cast<double>( pcm[i] ) - ref[i] ) * ( static_cast<double>( pcm[i] ) - ref[i] );
		mx = std::fmax( mx, std::fabs( static_cast<double>( pcm[i] ) - ref[i] ) );
	}
	const double cos = dot / std::sqrt( na * nb + 1e-30 );
	std::printf( "audio %d samples/channel in %.2f s: cosine %.6f rel %.4g max abs %.4g (|ref| rms %.4g)\n", T * 800, secs, cos, std::sqrt( nd / ( nb + 1e-30 ) ), mx, std::sqrt( nb / pcm.size() ) );
	std::printf( cos > 0.999 ? "ok\n" : "FAIL\n" );
	return cos > 0.999 ? 0 : 1;
}
