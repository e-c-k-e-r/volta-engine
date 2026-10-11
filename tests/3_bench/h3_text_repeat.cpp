// Does the H3 text encoder give the same hidden states every time? Encodes one prompt several times in one process (a new encoder each time, and the same encoder twice)
// and prints a fingerprint of each result.   h3_text_repeat <text_encoder dir> <tokenizer dir> "prompt" [layers=50] [runs=3] [fill byte]
#include "engine/config.h"
#include "engine/loader/tokenizer.h"
#include "engine/video/h3_text.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint64_t fingerprint( const std::vector<float>& v ) {
	uint64_t h = 1469598103934665603ULL;
	for ( float f : v ) {
		uint32_t b;
		std::memcpy( &b, &f, 4 );
		h = ( h ^ b ) * 1099511628211ULL;
	}
	return h;
}

int main( int argc, char** argv ) {
	if ( argc < 4 ) { std::printf( "usage: h3_text_repeat <text_encoder dir> <tokenizer dir> \"prompt\" [layers] [runs]\n" ); return 1; }
	const int layers = argc > 4 ? std::atoi( argv[4] ) : 50, runs = argc > 5 ? std::atoi( argv[5] ) : 3;
	engine::loader::Tokenizer tok;
	ENGINE_ASSERT( tok.load( argv[2] ), "tokenizer" );
	const std::vector<int> ids = tok.encode( argv[3], false );
	// optional: fill most of the free device memory with a byte pattern and free it again, so later allocations see that pattern instead of zeros
	if ( argc > 6 ) {
		const int pattern = static_cast<int>( std::strtol( argv[6], nullptr, 0 ) );
		size_t free_b = 0, total_b = 0;
		CUDA_CHECK( cudaMemGetInfo( &free_b, &total_b ) );
		void* junk = nullptr;
		const size_t n = free_b - ( 2ULL << 30 );
		CUDA_CHECK( cudaMalloc( &junk, n ) );
		CUDA_CHECK( cudaMemset( junk, pattern, n ) );
		CUDA_CHECK( cudaFree( junk ) );
		std::printf( "filled %.1f GB with 0x%02x\n", n / 1e9, pattern & 255 );
	}
	float* d = nullptr;
	CUDA_CHECK( cudaMalloc( &d, ids.size() * 5120 * sizeof( float ) ) );
	std::vector<float> got( ids.size() * 5120 );
	for ( int r = 0; r < runs; ++r ) {
		engine::video::H3TextEncoder enc; // a new encoder per run, as the pipeline does
		ENGINE_ASSERT( enc.load( argv[1], 0, 8192, layers ), "load" );
		std::vector<uint64_t> per_layer;
		enc.probe = [&]( int layer, const float* d_x, int S ) {
			std::vector<float> h( static_cast<size_t>( S ) * 5120 );
			CUDA_CHECK( cudaMemcpy( h.data(), d_x, h.size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
			per_layer.push_back( fingerprint( h ) );
		};
		for ( int k = 0; k < 2; ++k ) {
			ENGINE_ASSERT( enc.encode( ids, d ), "encode" );
			CUDA_CHECK( cudaMemcpy( got.data(), d, got.size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
			std::printf( "run %d, encode %d: %zu tokens, fingerprint %016llx; layers:", r, k, ids.size(), static_cast<unsigned long long>( fingerprint( got ) ) );
			for ( size_t l = 0; l < std::min<size_t>( per_layer.size(), 4 ); ++l ) { std::printf( " %04llx", static_cast<unsigned long long>( per_layer[l] & 0xffff ) ); }
			std::printf( "\n" );
			per_layer.clear();
		}
	}
	return 0;
}
