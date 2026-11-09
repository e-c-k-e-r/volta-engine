// LinearTrellis::forward in both EXL3 tile orders (stock, Volta) on the same shapes and row counts, GPU 0. Random trellis words, timing only.
// Each shape rotates over enough weight copies to stay out of L2 (what decode sees), and the two orders alternate per repeat.
//   bench_trellis_order [--rows 1,2,4,8,16] [--reps 5]
#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/modules/trellis.h"
#include "trellis_context.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Shape { const char* name; int k, n, bits; };

pod::TensorView view( void* p, pod::DType dt, std::initializer_list<int> shape ) {
	pod::TensorView t{};
	t.data = p;
	t.dtype = dt;
	t.ndim = static_cast<int>( shape.size() );
	int i = 0;
	for ( int s : shape ) t.shape[i++] = s;
	return t;
}

} // namespace

int main( int argc, char** argv ) {
	std::vector<int> rows = { 1, 2, 4, 8, 16 };
	int reps = 5;
	for ( int a = 1; a < argc; ++a ) {
		if ( !std::strcmp( argv[a], "--rows" ) && a + 1 < argc ) {
			rows.clear();
			for ( const char* q = argv[++a]; *q; ) { rows.push_back( std::atoi( q ) ); while ( *q && *q != ',' ) ++q; if ( *q ) ++q; }
		}
		else if ( !std::strcmp( argv[a], "--reps" ) && a + 1 < argc ) reps = std::atoi( argv[++a] );
	}
	CUDA_CHECK( cudaSetDevice( 0 ) );
	if ( !engine::modules::trellis_volta_tiles_supported() ) {
		std::printf( "[*] bench_trellis_order: no Volta-order kernels on this build or device\n" );
		return 0;
	}
	engine::modules::trellis_context_init_all();
	cudaStream_t s;
	CUDA_CHECK( cudaStreamCreate( &s ) );
	const Shape shapes[] = {
		{ "flash attn/gdn in 2560x4096", 2560, 4096, 4 }, { "flash out 4096x2560", 4096, 2560, 4 },
		{ "27B mlp gate/up 5120x4352", 5120, 4352, 4 }, { "27B mlp down 4352x5120", 4352, 5120, 4 }, { "27B gdn qkv 5120x2560", 5120, 2560, 4 },
		{ "lm_head /4 5120x62080", 5120, 62080, 5 }, { "drafter gate/up 5120x17408", 5120, 17408, 5 },
	};
	const int mmax = *std::max_element( rows.begin(), rows.end() );
	std::printf( "%-30s %4s %10s %10s %8s\n", "shape", "rows", "stock us", "volta us", "volta/stock" );
	for ( const Shape& sh : shapes ) {
		const size_t words = static_cast<size_t>( sh.k / 16 ) * ( sh.n / 16 ) * 16 * sh.bits;
		const int copies = std::max( 2, static_cast<int>( ( 48ull << 20 ) / ( words * 2 ) ) );   // ~48 MB of weights per shape
		std::vector<uint16_t*> B( copies );
		std::vector<uint16_t> hb( words );
		uint32_t st = 12345u;
		for ( auto& w : hb ) { st = st * 1664525u + 1013904223u; w = static_cast<uint16_t>( st >> 16 ); }
		for ( auto& b : B ) { CUDA_CHECK( cudaMalloc( &b, words * 2 ) ); CUDA_CHECK( cudaMemcpy( b, hb.data(), words * 2, cudaMemcpyHostToDevice ) ); }
		half *suh, *svh, *x, *y;
		CUDA_CHECK( cudaMalloc( &suh, sh.k * 2 ) );
		CUDA_CHECK( cudaMalloc( &svh, sh.n * 2 ) );
		CUDA_CHECK( cudaMalloc( &x, static_cast<size_t>( mmax ) * sh.k * 2 ) );
		CUDA_CHECK( cudaMalloc( &y, static_cast<size_t>( mmax ) * sh.n * 2 ) );
		CUDA_CHECK( cudaMemset( suh, 0x3c, sh.k * 2 ) );
		CUDA_CHECK( cudaMemset( svh, 0x3c, sh.n * 2 ) );
		CUDA_CHECK( cudaMemset( x, 0x30, static_cast<size_t>( mmax ) * sh.k * 2 ) );
		for ( int m : rows ) {
			double best[2] = { 1e30, 1e30 };
			for ( int r = 0; r < reps; ++r )
				for ( int order = 0; order < 2; ++order ) {
					const int iters = 4 * copies;
					auto run = [&]( int i ) {
						pod::TensorView xt = view( x, pod::DType::FP16, { m, sh.k } ), ct = view( B[i % copies], pod::DType::UINT16, { sh.k / 16, sh.n / 16, 16 * sh.bits } );
						pod::TensorView su = view( suh, pod::DType::FP16, { sh.k } ), sv = view( svh, pod::DType::FP16, { sh.n } ), yt = view( y, pod::DType::FP16, { m, sh.n } );
						engine::modules::LinearTrellis::forward( xt, ct, su, sv, yt, sh.bits, false, true, s, order == 1 );
					};
					for ( int i = 0; i < copies; ++i ) run( i );
					cudaEvent_t a, b;
					CUDA_CHECK( cudaEventCreate( &a ) );
					CUDA_CHECK( cudaEventCreate( &b ) );
					CUDA_CHECK( cudaEventRecord( a, s ) );
					for ( int i = 0; i < iters; ++i ) run( i );
					CUDA_CHECK( cudaEventRecord( b, s ) );
					CUDA_CHECK( cudaEventSynchronize( b ) );
					float ms = 0;
					CUDA_CHECK( cudaEventElapsedTime( &ms, a, b ) );
					best[order] = std::min( best[order], 1000.0 * ms / iters );
					cudaEventDestroy( a );
					cudaEventDestroy( b );
				}
			std::printf( "%-30s %4d %10.1f %10.1f %8.3f\n", sh.name, m, best[0], best[1], best[1] / best[0] );
		}
		for ( auto& b : B ) cudaFree( b );
		cudaFree( suh ); cudaFree( svh ); cudaFree( x ); cudaFree( y );
	}
	return 0;
}
