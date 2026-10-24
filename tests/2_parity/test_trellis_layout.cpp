// The trellis dequantizer against a golden decode in the tile order the build expects: random trellis words and the FP16 weights they
// decode to, written by tests/tools/trellis_layout_golden.py (--layout stock for a normal build, volta for ENGINE_EXL3_VOLTA_LAYOUT=ON).
//   test_trellis_layout <golden_dir> <bits> <tiles_k> <tiles_n> [mcg]
// pass: the reconstructed matrix equals the golden bit for bit, for the plain and the Hadamard-fused dequantizer's input path.
#include "reconstruct.cuh"
#include "engine/config.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<char> read_all( const std::string& path ) {
	FILE* f = std::fopen( path.c_str(), "rb" );
	ENGINE_ASSERT( f, "cannot open golden file" );
	std::fseek( f, 0, SEEK_END );
	const long n = std::ftell( f );
	std::fseek( f, 0, SEEK_SET );
	std::vector<char> b( static_cast<size_t>( n ) );
	ENGINE_ASSERT( std::fread( b.data(), 1, b.size(), f ) == b.size(), "short read" );
	std::fclose( f );
	return b;
}

int main( int argc, char** argv ) {
	if ( argc < 5 ) {
		std::printf( "usage: test_trellis_layout <golden_dir> <bits> <tiles_k> <tiles_n> [mcg]\n" );
		return 2;
	}
	const std::string dir = argv[1];
	const int K = std::atoi( argv[2] ), tk = std::atoi( argv[3] ), tn = std::atoi( argv[4] );
	const bool mcg = argc > 5 && !std::strcmp( argv[5], "mcg" );
	const std::vector<char> words = read_all( dir + "/trellis.bin" ), expect = read_all( dir + "/expect.bin" );
	const size_t rows = 16 * static_cast<size_t>( tk ), cols = 16 * static_cast<size_t>( tn );
	ENGINE_ASSERT( words.size() == static_cast<size_t>( tk ) * tn * 16 * K * 2, "trellis.bin size" );
	ENGINE_ASSERT( expect.size() == rows * cols * 2, "expect.bin size" );
	CUDA_CHECK( cudaSetDevice( 0 ) );
	uint16_t* d_w = nullptr;
	half* d_out = nullptr;
	CUDA_CHECK( cudaMalloc( &d_w, words.size() ) );
	CUDA_CHECK( cudaMalloc( &d_out, expect.size() ) );
	CUDA_CHECK( cudaMemcpy( d_w, words.data(), words.size(), cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemset( d_out, 0xFF, expect.size() ) );
	reconstruct_slice( d_out, d_w, K, mcg, !mcg, tk, static_cast<int>( cols ), 0, nullptr );
	CUDA_CHECK( cudaDeviceSynchronize() );
	std::vector<char> got( expect.size() );
	CUDA_CHECK( cudaMemcpy( got.data(), d_out, got.size(), cudaMemcpyDeviceToHost ) );
	size_t bad = 0, first = 0;
	for ( size_t i = 0; i < got.size(); i += 2 ) {
		if ( std::memcmp( &got[i], &expect[i], 2 ) != 0 ) {
			if ( !bad ) first = i / 2;
			++bad;
		}
	}
	if ( bad ) std::printf( "[FAIL] %zu of %zu weights differ (first at row %zu col %zu)\n", bad, got.size() / 2, first / cols, first % cols );
	else std::printf( "[+] PASS reconstruct matches the golden (%d bits, %d x %d tiles)\n", K, tk, tn );
	cudaFree( d_w );
	cudaFree( d_out );
	return bad ? 1 : 0;
}
