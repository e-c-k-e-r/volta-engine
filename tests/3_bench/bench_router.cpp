// Flash-Next decode router: time of MoERouter::route_softmax_topk (logits + softmax/top-k select) for 1..8 rows over 48 distinct weight
// copies (cycled so the L2 does not hold them), the way a decode step meets 48 layers.
//   bench_router [rows ...]
#include "engine/core/arena.h"
#include "engine/modules/moe.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	const bool glm = std::getenv( "GLM" ) != nullptr;   // GLM: sigmoid scores with a selection bias (route_topk)
	const int H = glm ? 4096 : 2560, E = glm ? 256 : 512, K = glm ? 8 : 10, LAYERS = 48;
	CUDA_CHECK( cudaSetDevice( 0 ) );
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );
	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 1ULL * 1024 * 1024 * 1024 ), "arena" );
	std::vector<int> rows_list;
	for ( int i = 1; i < argc; ++i ) rows_list.push_back( std::atoi( argv[i] ) );
	if ( rows_list.empty() ) rows_list = { 1, 4, 8 };
	std::vector<half> hw( static_cast<size_t>( H ) * E );
	uint32_t seed = 1234u;
	auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return ( ( seed >> 8 ) / 16777216.0f ) * 2.0f - 1.0f; };
	std::vector<pod::TensorView> w( LAYERS );
	int ws[2] = { H, E };
	for ( int l = 0; l < LAYERS; ++l ) {
		for ( auto& v : hw ) v = __float2half( rnd() * 0.05f );
		w[l] = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ws, 2 );
		CUDA_CHECK( cudaMemcpy( w[l].data, hw.data(), hw.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
	}
	cudaEvent_t e0, e1;
	CUDA_CHECK( cudaEventCreate( &e0 ) );
	CUDA_CHECK( cudaEventCreate( &e1 ) );
	for ( int rows : rows_list ) {
		int xs[2] = { rows, H };
		auto x = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
		std::vector<half> hx( static_cast<size_t>( rows ) * H );
		for ( auto& v : hx ) v = __float2half( rnd() );
		CUDA_CHECK( cudaMemcpy( x.data, hx.data(), hx.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		int* idx = static_cast<int*>( engine::core::arena_alloc( arena, rows * K * sizeof( int ) ) );
		half* wt = static_cast<half*>( engine::core::arena_alloc( arena, rows * K * sizeof( half ) ) );
		int bs[1] = { E };
		auto bias = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, bs, 1 );
		{
			std::vector<half> hb( E );
			for ( auto& v : hb ) v = __float2half( rnd() * 0.1f );
			CUDA_CHECK( cudaMemcpy( bias.data, hb.data(), hb.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		}
		auto run = [&]() {
			for ( int l = 0; l < LAYERS; ++l ) {
				if ( glm ) engine::modules::MoERouter::route_topk( x, w[l], bias, idx, wt, H, E, K, 2.5f, stream, rows );
				else engine::modules::MoERouter::route_softmax_topk( x, w[l], idx, wt, H, E, K, stream, rows );
			}
		};
		for ( int i = 0; i < 5; ++i ) run();
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		const int iters = 50;
		cudaGraph_t graph;
		cudaGraphExec_t exec;
		CUDA_CHECK( cudaStreamBeginCapture( stream, cudaStreamCaptureModeGlobal ) );
		run();
		CUDA_CHECK( cudaStreamEndCapture( stream, &graph ) );
		CUDA_CHECK( cudaGraphInstantiate( &exec, graph, nullptr, nullptr, 0 ) );
		CUDA_CHECK( cudaEventRecord( e0, stream ) );
		for ( int i = 0; i < iters; ++i ) CUDA_CHECK( cudaGraphLaunch( exec, stream ) );
		CUDA_CHECK( cudaEventRecord( e1, stream ) );
		CUDA_CHECK( cudaEventSynchronize( e1 ) );
		float ms = 0.0f;
		CUDA_CHECK( cudaEventElapsedTime( &ms, e0, e1 ) );
		std::printf( "rows %d: %.1f us per layer (%.2f ms per 48 layers), graph replay\n", rows, ms * 1000.0f / iters / LAYERS, ms / iters );
		std::vector<int> hi( rows * K );
		CUDA_CHECK( cudaMemcpy( hi.data(), idx, hi.size() * sizeof( int ), cudaMemcpyDeviceToHost ) );
		std::printf( "  row 0 top-k:" );
		for ( int k = 0; k < K; ++k ) std::printf( " %d", hi[k] );
		std::printf( "\n" );
	}
	return 0;
}
