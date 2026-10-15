// GLM decode KDA recurrent step: time of GDN::recurrent_step_kda_rows for 1..8 rows over 34 distinct layer states (cycled so the L2 does not hold
// them), the way a decode step meets 34 linear layers; 16 heads of 128 (one rank of four). For a profiler: BENCH_ITERS=1 runs a single pass.
//   bench_kda [rows ...]
#include "engine/core/arena.h"
#include "engine/core/batch.h"
#include "engine/modules/gdn.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	const int HEADS = 16, LAYERS = 34, D = 128, C = HEADS * D;
	CUDA_CHECK( cudaSetDevice( 0 ) );
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );
	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 1ULL * 1024 * 1024 * 1024 ), "arena" );
	std::vector<int> rows_list;
	for ( int i = 1; i < argc; ++i ) rows_list.push_back( std::atoi( argv[i] ) );
	if ( rows_list.empty() ) rows_list = { 1, 4, 8 };
	const size_t state_floats = static_cast<size_t>( HEADS ) * D * D;
	std::vector<float*> state( LAYERS );
	for ( int l = 0; l < LAYERS; ++l ) {
		state[l] = static_cast<float*>( engine::core::arena_alloc( arena, state_floats * sizeof( float ) ) );
		CUDA_CHECK( cudaMemset( state[l], 0, state_floats * sizeof( float ) ) );
	}
	uint32_t seed = 99u;
	auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return ( ( seed >> 8 ) / 16777216.0f ) * 2.0f - 1.0f; };
	auto make = [&]( pod::DType t, int r, int c ) {
		int sh[2] = { r, c };
		return engine::core::arena_alloc_tensor( arena, t, sh, 2 );
	};
	const int iters_env = std::getenv( "BENCH_ITERS" ) ? std::atoi( std::getenv( "BENCH_ITERS" ) ) : 50;
	cudaEvent_t e0, e1;
	CUDA_CHECK( cudaEventCreate( &e0 ) );
	CUDA_CHECK( cudaEventCreate( &e1 ) );
	for ( int rows : rows_list ) {
		auto qkv = make( pod::DType::FP16, rows, 3 * C );
		auto g = make( pod::DType::FP32, rows, C );
		auto beta = make( pod::DType::FP32, rows, HEADS );
		auto z = make( pod::DType::FP16, rows, C );
		auto onorm = make( pod::DType::FP16, 1, D );
		auto out = make( pod::DType::FP16, rows, C );
		std::vector<half> hq( static_cast<size_t>( rows ) * 3 * C ), hz( static_cast<size_t>( rows ) * C ), hn( D );
		std::vector<float> hg( static_cast<size_t>( rows ) * C ), hb( static_cast<size_t>( rows ) * HEADS );
		for ( auto& v : hq ) v = __float2half( rnd() * 0.3f );
		for ( auto& v : hz ) v = __float2half( rnd() );
		for ( auto& v : hn ) v = __float2half( 1.0f + rnd() * 0.1f );
		for ( auto& v : hg ) v = -0.05f - 0.05f * ( rnd() + 1.0f );
		for ( auto& v : hb ) v = 0.5f + 0.25f * rnd();
		CUDA_CHECK( cudaMemcpy( qkv.data, hq.data(), hq.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( z.data, hz.data(), hz.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( onorm.data, hn.data(), hn.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( g.data, hg.data(), hg.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( beta.data, hb.data(), hb.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
		pod::BatchRows br;
		br.count = rows;
		for ( int m = 0; m < rows; ++m ) br.slot[m] = 0;
		pod::BatchRows* d_rows = static_cast<pod::BatchRows*>( engine::core::arena_alloc( arena, sizeof( pod::BatchRows ) ) );
		CUDA_CHECK( cudaMemcpy( d_rows, &br, sizeof( br ), cudaMemcpyHostToDevice ) );
		auto run = [&]() {
			for ( int l = 0; l < LAYERS; ++l ) {
				engine::modules::GDN::recurrent_step_kda_rows( state[l], qkv, g, beta, z, onorm, out, HEADS, D, 1e-5f, rows, d_rows, state_floats, nullptr, 0, 0, nullptr, stream );
			}
		};
		run();
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		if ( iters_env <= 1 ) continue;
		cudaGraph_t graph;
		cudaGraphExec_t exec;
		CUDA_CHECK( cudaStreamBeginCapture( stream, cudaStreamCaptureModeGlobal ) );
		run();
		CUDA_CHECK( cudaStreamEndCapture( stream, &graph ) );
		CUDA_CHECK( cudaGraphInstantiate( &exec, graph, nullptr, nullptr, 0 ) );
		for ( int i = 0; i < 5; ++i ) CUDA_CHECK( cudaGraphLaunch( exec, stream ) );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		CUDA_CHECK( cudaEventRecord( e0, stream ) );
		for ( int i = 0; i < iters_env; ++i ) CUDA_CHECK( cudaGraphLaunch( exec, stream ) );
		CUDA_CHECK( cudaEventRecord( e1, stream ) );
		CUDA_CHECK( cudaEventSynchronize( e1 ) );
		float ms = 0.0f;
		CUDA_CHECK( cudaEventElapsedTime( &ms, e0, e1 ) );
		std::printf( "rows %d: %.1f us per layer (%.2f ms per %d layers), graph replay\n", rows, ms * 1000.0f / iters_env / LAYERS, ms / iters_env, LAYERS );
	}
	return 0;
}
