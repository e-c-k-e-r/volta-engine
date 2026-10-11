// H3 text encoder parity against tests/tools/h3_encoder_golden.py (and the engine tokenizer against the HF ids).
//   test_h3_text <text_encoder dir> <golden> [layers=50] [tokenizer dir] [prompt]
#include "engine/config.h"
#include "engine/loader/safetensors.h"
#include "engine/loader/tokenizer.h"
#include "engine/video/h3_text.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main( int argc, char** argv ) {
	if ( argc < 3 ) { std::printf( "usage: test_h3_text <text_encoder dir> <golden> [layers] [tokenizer dir] [prompt]\n" ); return 1; }
	const int layers = argc > 3 ? std::atoi( argv[3] ) : 50;
	pod::SafeTensorsFile st;
	ENGINE_ASSERT( engine::loader::safetensors_open( st, argv[2] ), "golden" );
	const pod::SafeTensorMeta* im = engine::loader::safetensors_find( st, "ids" );
	const pod::SafeTensorMeta* hm = engine::loader::safetensors_find( st, layers == 2 ? "hidden_l2" : "hidden" );
	ENGINE_ASSERT( im && hm, "golden tensors" );
	const int* gid = static_cast<const int*>( engine::loader::safetensors_get_data_cpu( st, im ) );
	std::vector<int> ids( gid, gid + im->shape[0] );
	if ( argc > 5 ) {
		engine::loader::Tokenizer tok;
		ENGINE_ASSERT( tok.load( argv[4] ), "tokenizer" );
		const std::vector<int> mine = tok.encode( argv[5], false );
		std::printf( "tokenizer: %zu ids vs golden %zu: %s\n", mine.size(), ids.size(), mine == ids ? "identical" : "DIFFERENT" );
		if ( mine != ids ) { return 1; }
	}
	engine::video::H3TextEncoder enc;
	auto t0 = std::chrono::steady_clock::now();
	ENGINE_ASSERT( enc.load( argv[1], 0, 2048, layers ), "load" );
	float* d;
	CUDA_CHECK( cudaMalloc( &d, hm->size_bytes ) );
	auto t1 = std::chrono::steady_clock::now();
	ENGINE_ASSERT( enc.encode( ids, d ), "encode" );
	const double secs = std::chrono::duration<double>( std::chrono::steady_clock::now() - t1 ).count();
	std::vector<float> got( hm->size_bytes / 4 );
	CUDA_CHECK( cudaMemcpy( got.data(), d, hm->size_bytes, cudaMemcpyDeviceToHost ) );
	const float* ref = static_cast<const float*>( engine::loader::safetensors_get_data_cpu( st, hm ) );
	double dot = 0, na = 0, nb = 0, nd = 0;
	for ( size_t i = 0; i < got.size(); ++i ) {
		dot += static_cast<double>( got[i] ) * ref[i]; na += static_cast<double>( got[i] ) * got[i]; nb += static_cast<double>( ref[i] ) * ref[i];
		nd += ( static_cast<double>( got[i] ) - ref[i] ) * ( static_cast<double>( got[i] ) - ref[i] );
	}
	const double cos = dot / std::sqrt( na * nb );
	std::printf( "%d tokens, %d layers: load %.1f s, encode %.1f s: cosine %.6f rel %.4g\n", static_cast<int>( ids.size() ), layers,
		std::chrono::duration<double>( t1 - t0 ).count(), secs, cos, std::sqrt( nd / nb ) );
	std::printf( cos > 0.999 ? "ok\n" : "FAIL\n" );
	return cos > 0.999 ? 0 : 1;
}
