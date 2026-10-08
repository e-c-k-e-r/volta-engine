// Routed-MoE prefill throughput on real GLM-5.3-Flash layer-3 experts (rank-0 partition, 72 of 288 experts):
// rows routed uniformly over all 288 experts, time per MoE::forward at several chunk sizes.
//   bench_moe_prefill [rows ...]
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/linear.h"
#include "engine/modules/moe.h"
#include "engine/modules/swiglu.h"
#include "trellis_context.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	const char* dir = "/home/mrq/Documents/models/GLM-5.3-Flash-exl3-3.05bpw";
	const bool shard = std::getenv( "BENCH_SHARD" ) != nullptr; // every expert a quarter wide (rank 0 of 4), all 288 local
	const int H = 4096, I = 2048, E = shard ? 288 : 72, K = 8, L = 3;
	CUDA_CHECK( cudaSetDevice( 0 ) );
	engine::modules::trellis_context_init_all();
	engine::modules::trellis_get_workspace( 0, 16 * 1024 * 1024 );
	engine::modules::trellis_get_locks( 0 );
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );
	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 2ULL * 1024 * 1024 * 1024 ), "arena" );

	engine::loader::SafeTensorsRepo repo;
	ENGINE_ASSERT( repo.open( dir ), "open model dir" );
	pod::MoELayerWeights mw;
	mw.type = pod::QuantType::TRELLIS;
	mw.num_local_experts = E;
	char name[256];
	for ( int e = 0; e < E; ++e ) {
		std::snprintf( name, sizeof( name ), "model.language_model.layers.%d.mlp.experts.%d.gate_proj", L, e );
		mw.gate[e] = shard ? repo.load_linear_slice( name, arena, 4, 1, 0, 4, stream ) : repo.load_linear( name, arena, 4, stream );
		std::snprintf( name, sizeof( name ), "model.language_model.layers.%d.mlp.experts.%d.up_proj", L, e );
		mw.up[e] = shard ? repo.load_linear_slice( name, arena, 4, 1, 0, 4, stream ) : repo.load_linear( name, arena, 4, stream );
		std::snprintf( name, sizeof( name ), "model.language_model.layers.%d.mlp.experts.%d.down_proj", L, e );
		mw.down[e] = shard ? repo.load_linear_slice( name, arena, 4, 0, 0, 4, stream ) : repo.load_linear( name, arena, 4, stream );
	}
	engine::modules::MoE::init_device_ptrs( mw, arena, stream );
	std::printf( "[*] experts: bits g/u/d = %d/%d/%d, mul1 g/u/d = %d/%d/%d\n",
		mw.gate[0].bits, mw.up[0].bits, mw.down[0].bits, mw.gate[0].mul1, mw.up[0].mul1, mw.down[0].mul1 );

	// Reference: the dense decode kernels on one expert matrix (gate: 4096 -> 2048, 3 bit), rows 1..16
	for ( int Rg : { 1, 4, 8, 16 } ) {
		int xs[2] = { Rg, H }, ys[2] = { Rg, I };
		auto xg = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
		auto yg = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ys, 2 );
		CUDA_CHECK( cudaMemset( xg.data, 0x3c, static_cast<size_t>( Rg ) * H * 2 ) );
		cudaEvent_t g0, g1;
		CUDA_CHECK( cudaEventCreate( &g0 ) );
		CUDA_CHECK( cudaEventCreate( &g1 ) );
		for ( int it = 0; it < 3; ++it ) engine::modules::Linear::forward( xg, mw.gate[0], yg, stream );
		CUDA_CHECK( cudaEventRecord( g0, stream ) );
		for ( int it = 0; it < 50; ++it ) engine::modules::Linear::forward( xg, mw.gate[it % E], yg, stream );
		CUDA_CHECK( cudaEventRecord( g1, stream ) );
		CUDA_CHECK( cudaEventSynchronize( g1 ) );
		float gms = 0.0f;
		CUDA_CHECK( cudaEventElapsedTime( &gms, g0, g1 ) );
		std::printf( "  dense Linear::forward on an expert gate matrix, %2d rows: %.1f us (%.0f GB/s of 3.2 MB)\n", Rg, gms * 1000.0f / 50.0f, 3.1e6 / ( gms * 1000.0f / 50.0f ) * 1e-3 );
	}
	std::vector<int> row_list;
	for ( int i = 1; i < argc; ++i ) row_list.push_back( std::atoi( argv[i] ) );
	if ( row_list.empty() ) row_list = { 16, 64, 128, 256, 512 };
	uint32_t seed = 12345u;
	cudaEvent_t e0, e1;
	CUDA_CHECK( cudaEventCreate( &e0 ) );
	CUDA_CHECK( cudaEventCreate( &e1 ) );
	for ( int R : row_list ) {
		int rs[2] = { R, H };
		auto xr = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, rs, 2 );
		auto outr = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, rs, 2 );
		int* d_idx = static_cast<int*>( engine::core::arena_alloc( arena, R * K * sizeof( int ) ) );
		half* d_w = static_cast<half*>( engine::core::arena_alloc( arena, R * K * sizeof( half ) ) );
		std::vector<half> hx( static_cast<size_t>( R ) * H ), hw( R * K );
		std::vector<int> hi( R * K );
		for ( auto& v : hx ) {
			seed = seed * 1664525u + 1013904223u;
			v = __float2half( ( ( seed >> 8 ) / 16777216.0f ) * 2.0f - 1.0f );
		}
		int local_routes = 0;
		for ( int r = 0; r < R; ++r ) {
			int used[288] = {};
			for ( int k = 0; k < K; ++k ) {
				int e;
				do {
					seed = seed * 1664525u + 1013904223u;
					e = ( seed >> 8 ) % 288;
				} while ( used[e] );
				used[e] = 1;
				hi[r * K + k] = e;
				hw[r * K + k] = __float2half( 0.125f );
				local_routes += e < E ? 1 : 0;
			}
		}
		CUDA_CHECK( cudaMemcpy( xr.data, hx.data(), hx.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( d_idx, hi.data(), hi.size() * sizeof( int ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( d_w, hw.data(), hw.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		auto scratch = engine::modules::MoE::alloc_scratch( arena, E, K, H, I, R );
		for ( int it = 0; it < 2; ++it ) engine::modules::MoE::forward( xr, mw, d_idx, d_w, outr, 0, E, scratch, 10.0f, stream );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		const int iters = 10;
		CUDA_CHECK( cudaEventRecord( e0, stream ) );
		for ( int it = 0; it < iters; ++it ) engine::modules::MoE::forward( xr, mw, d_idx, d_w, outr, 0, E, scratch, 10.0f, stream );
		CUDA_CHECK( cudaEventRecord( e1, stream ) );
		CUDA_CHECK( cudaEventSynchronize( e1 ) );
		float ms = 0.0f;
		CUDA_CHECK( cudaEventElapsedTime( &ms, e0, e1 ) );
		std::printf( "  rows %4d: %7.3f ms per layer (%d local routes, %.2f us per local route)\n", R, ms / iters, local_routes, 1000.0 * ms / iters / MAX( 1, local_routes ) );
	}
	return 0;
}
