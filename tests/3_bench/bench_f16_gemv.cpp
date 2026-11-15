// Plain FP16 GEMV (dflash_f16_gemv_nt: FP16 linears, the drafter's projections, GLM's indexer) at decode / verify rows: time per call over distinct
// weight copies (cycled so the L2 does not hold them) in graph replay, the GB/s of the weight, and a hash of the outputs (bit for bit).
//   bench_f16_gemv [N K ...]   default GLM's indexer projections at TP=4 (4096 x 1536, 512 x 4096) and a drafter shape (4096 x 4096)
#include "engine/core/arena.h"
#include "engine/spec/dflash.h"
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	std::vector<int> shapes;
	for ( int i = 1; i + 1 < argc; i += 2 ) { shapes.push_back( std::atoi( argv[i] ) ); shapes.push_back( std::atoi( argv[i + 1] ) ); }
	if ( shapes.empty() ) shapes = { 4096, 1536, 512, 4096, 4096, 4096 };
	CUDA_CHECK( cudaSetDevice( 0 ) );
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );
	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 3ULL * 1024 * 1024 * 1024 ), "arena" );
	uint32_t seed = 4321u;
	auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return ( ( seed >> 8 ) / 16777216.0f ) * 2.0f - 1.0f; };
	cudaEvent_t e0, e1;
	CUDA_CHECK( cudaEventCreate( &e0 ) );
	CUDA_CHECK( cudaEventCreate( &e1 ) );
	const int rows_list[] = { 1, 2, 4, 8, 12, 16 };
	for ( size_t s = 0; s + 1 < shapes.size(); s += 2 ) {
		const int n = shapes[s], k = shapes[s + 1];
		const size_t wbytes = static_cast<size_t>( n ) * k * sizeof( half );
		const int copies = static_cast<int>( std::max<size_t>( 2, ( 64ull << 20 ) / wbytes + 1 ) );   // past the L2
		std::vector<half> hw( static_cast<size_t>( n ) * k );
		std::vector<half*> w( copies );
		for ( int c = 0; c < copies; ++c ) {
			for ( auto& v : hw ) v = __float2half( rnd() * 0.05f );
			w[c] = static_cast<half*>( engine::core::arena_alloc( arena, wbytes, 256 ) );
			CUDA_CHECK( cudaMemcpy( w[c], hw.data(), wbytes, cudaMemcpyHostToDevice ) );
		}
		std::vector<half> hx( static_cast<size_t>( 16 ) * k );
		for ( auto& v : hx ) v = __float2half( rnd() );
		half* x = static_cast<half*>( engine::core::arena_alloc( arena, hx.size() * sizeof( half ), 256 ) );
		CUDA_CHECK( cudaMemcpy( x, hx.data(), hx.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		half* out = static_cast<half*>( engine::core::arena_alloc( arena, static_cast<size_t>( 16 ) * n * sizeof( half ), 256 ) );
		for ( int m : rows_list ) {
			auto run = [&]() { for ( int c = 0; c < copies; ++c ) engine::spec::dflash_f16_gemv_nt( w[c], x, out, m, n, k, stream ); };
			run();
			CUDA_CHECK( cudaStreamSynchronize( stream ) );
			cudaGraph_t graph;
			cudaGraphExec_t exec;
			CUDA_CHECK( cudaStreamBeginCapture( stream, cudaStreamCaptureModeGlobal ) );
			run();
			CUDA_CHECK( cudaStreamEndCapture( stream, &graph ) );
			CUDA_CHECK( cudaGraphInstantiate( &exec, graph, nullptr, nullptr, 0 ) );
			const int iters = 20;
			CUDA_CHECK( cudaEventRecord( e0, stream ) );
			for ( int i = 0; i < iters; ++i ) CUDA_CHECK( cudaGraphLaunch( exec, stream ) );
			CUDA_CHECK( cudaEventRecord( e1, stream ) );
			CUDA_CHECK( cudaEventSynchronize( e1 ) );
			float ms = 0.0f;
			CUDA_CHECK( cudaEventElapsedTime( &ms, e0, e1 ) );
			const double us = ms * 1000.0 / iters / copies;
			std::vector<half> ho( static_cast<size_t>( m ) * n );
			CUDA_CHECK( cudaMemcpy( ho.data(), out, ho.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
			uint64_t hash = 1469598103934665603ull;
			for ( size_t i = 0; i < ho.size() * sizeof( half ); ++i ) hash = ( hash ^ reinterpret_cast<const unsigned char*>( ho.data() )[i] ) * 1099511628211ull;
			std::printf( "N %5d K %5d M %2d: %7.2f us per call, %6.1f GB/s, hash %016llx\n", n, k, m, us, wbytes / us / 1e3, static_cast<unsigned long long>( hash ) );
			CUDA_CHECK( cudaGraphExecDestroy( exec ) );
			CUDA_CHECK( cudaGraphDestroy( graph ) );
		}
	}
	return 0;
}
