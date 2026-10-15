// Flash-Next decode MoE: time of one MoE::forward (1 row, top 10) on rank 0's 128 experts as the number of routes that
// land on this rank varies (the rest point to experts of other ranks), plus the router-free floor of 0 local routes.
//   bench_moe_routes [layer] [rows ...]   (with rows: prefill-sized calls, tokens routed uniformly over all 512 experts, rank 0's 128 are local)
//   BENCH_GLM=1 bench_moe_routes [layer] [rows ...]   GLM-5.3-Flash's experts instead (H 4096, I 2048, 72 of 288 experts on rank 0, top 8); per-route times
//   are also printed as GB/s of packed expert weights
#include "engine/runtime/test_paths.h"
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/moe.h"
#include "trellis_context.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	const bool glm = std::getenv( "BENCH_GLM" ) != nullptr;
	const char* dir = engine::runtime::test::model_path( glm ? "GLM-5.3-Flash-exl3-3.05bpw" : "Swift-Qwen3.8-Flash-Next-exl3-4.00bpw", glm ? nullptr : "ENGINE_TEST_FLASH_DIR" );
	const int BITS = std::getenv( "BENCH_BITS" ) ? std::atoi( std::getenv( "BENCH_BITS" ) ) : 4;   // the experts' bit width (Qwen3.8-Flash-Next-exl3: 2)
	const int L = argc > 1 ? std::atoi( argv[1] ) : ( glm ? 3 : 10 );
	const int H = glm ? 4096 : 2560, I = glm ? 2048 : 640, E = glm ? 72 : 128, K = glm ? 8 : 10;
	const int TOTAL = glm ? 288 : 512;
	CUDA_CHECK( cudaSetDevice( 0 ) );
	engine::modules::trellis_context_init_all();
	engine::modules::trellis_get_workspace( 0, 16 * 1024 * 1024 );
	engine::modules::trellis_get_locks( 0 );
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );
	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 4ULL * 1024 * 1024 * 1024 ), "arena" );
	engine::loader::SafeTensorsRepo repo;
	ENGINE_ASSERT( repo.open( dir ), "open model dir" );
	pod::MoELayerWeights mw;
	mw.type = pod::QuantType::TRELLIS;
	mw.num_local_experts = E;
	mw.top_k = K;
	char name[256];
	for ( int e = 0; e < E; ++e ) {
		std::snprintf( name, sizeof( name ), "model.language_model.layers.%d.mlp.experts.%d.gate_proj", L, e ); mw.gate[e] = repo.load_linear( name, arena, BITS, stream );
		std::snprintf( name, sizeof( name ), "model.language_model.layers.%d.mlp.experts.%d.up_proj", L, e ); mw.up[e] = repo.load_linear( name, arena, BITS, stream );
		std::snprintf( name, sizeof( name ), "model.language_model.layers.%d.mlp.experts.%d.down_proj", L, e ); mw.down[e] = repo.load_linear( name, arena, BITS, stream );
	}
	engine::modules::MoE::init_device_ptrs( mw, arena, stream );
	std::printf( "[*] layer %d experts: bits %d, mul1 %d\n", L, mw.gate[0].bits, mw.gate[0].mul1 );
	const double expert_bytes = 3.0 * H * I * mw.gate[0].bits / 8.0;   // packed trellis weights of one expert (gate + up + down)
	std::printf( "[*] one expert: %.2f MB packed\n", expert_bytes / 1e6 );
	int hs[2] = { 1, H };
	auto x = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, hs, 2 );
	auto out = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, hs, 2 );
	std::vector<half> hx( H );
	uint32_t seed = 777u;
	for ( auto& v : hx ) {
		seed = seed * 1664525u + 1013904223u;
		v = __float2half( ( ( seed >> 8 ) / 16777216.0f ) * 2.0f - 1.0f );
	}
	CUDA_CHECK( cudaMemcpy( x.data, hx.data(), H * sizeof( half ), cudaMemcpyHostToDevice ) );
	auto scratch = engine::modules::MoE::alloc_scratch( arena, E, K, H, I, 1 );
	// 32 distinct route sets per count, cycled so the L2 does not hold the experts between calls
	const int SETS = 32;
	int* d_idx = static_cast<int*>( engine::core::arena_alloc( arena, SETS * K * sizeof( int ) ) );
	half* d_w = static_cast<half*>( engine::core::arena_alloc( arena, K * sizeof( half ) ) );
	std::vector<half> hw( K, __float2half( 0.1f ) );
	CUDA_CHECK( cudaMemcpy( d_w, hw.data(), K * sizeof( half ), cudaMemcpyHostToDevice ) );
	cudaEvent_t e0, e1;
	CUDA_CHECK( cudaEventCreate( &e0 ) );
	CUDA_CHECK( cudaEventCreate( &e1 ) );
	std::vector<int> counts = { 0, 1, 2, 3, 4, 5, 6, 8, 10 };
	if ( const char* only = std::getenv( "BENCH_ROUTES" ) ) counts = { std::atoi( only ) };   // a single route count (for a profiler)
	for ( int n : counts ) {
		std::vector<int> hi( SETS * K );
		for ( int s = 0; s < SETS; ++s ) {
			int used[512] = {};   // TOTAL <= 512
			for ( int k = 0; k < K; ++k ) {
				int e;
				do {
					seed = seed * 1664525u + 1013904223u;
					e = k < n ? ( seed >> 8 ) % E : E + ( seed >> 8 ) % ( TOTAL - E );
				} while ( used[e] );
				used[e] = 1;
				hi[s * K + k] = e;
			}
		}
		CUDA_CHECK( cudaMemcpy( d_idx, hi.data(), hi.size() * sizeof( int ), cudaMemcpyHostToDevice ) );
		for ( int it = 0; it < 4; ++it ) engine::modules::MoE::forward( x, mw, d_idx + ( it % SETS ) * K, d_w, out, 0, E, scratch, 0.0f, stream );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		const int iters = 256;
		CUDA_CHECK( cudaEventRecord( e0, stream ) );
		for ( int it = 0; it < iters; ++it ) engine::modules::MoE::forward( x, mw, d_idx + ( it % SETS ) * K, d_w, out, 0, E, scratch, 0.0f, stream );
		CUDA_CHECK( cudaEventRecord( e1, stream ) );
		CUDA_CHECK( cudaEventSynchronize( e1 ) );
		float ms = 0.0f;
		CUDA_CHECK( cudaEventElapsedTime( &ms, e0, e1 ) );
		std::printf( "  %2d local routes: %7.1f us per forward (%.0f GB/s of packed weights)\n", n, 1000.0f * ms / iters, n * expert_bytes / ( 1000.0 * ms / iters * 1e-6 ) / 1e9 );
	}
	for ( int ai = 2; ai < argc; ++ai ) {
		const int Rn = std::atoi( argv[ai] );
		int xs[2] = { Rn, H };
		auto xr = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
		auto outr = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
		std::vector<half> hxr( static_cast<size_t>( Rn ) * H );
		for ( auto& v : hxr ) { seed = seed * 1664525u + 1013904223u; v = __float2half( ( ( seed >> 8 ) / 16777216.0f ) * 2.0f - 1.0f ); }
		CUDA_CHECK( cudaMemcpy( xr.data, hxr.data(), hxr.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		int* ri = static_cast<int*>( engine::core::arena_alloc( arena, static_cast<size_t>( Rn ) * K * sizeof( int ) ) );
		half* rw = static_cast<half*>( engine::core::arena_alloc( arena, static_cast<size_t>( Rn ) * K * sizeof( half ) ) );
		std::vector<int> hri( static_cast<size_t>( Rn ) * K );
		std::vector<half> hrw( static_cast<size_t>( Rn ) * K, __float2half( 0.1f ) );
		int local = 0;
		for ( int r = 0; r < Rn; ++r ) {
			int used[512] = {};   // TOTAL <= 512
			for ( int k = 0; k < K; ++k ) {
				int e;
				do { seed = seed * 1664525u + 1013904223u; e = ( seed >> 8 ) % TOTAL; } while ( used[e] );
				used[e] = 1;
				hri[static_cast<size_t>( r ) * K + k] = e;
				local += e < E ? 1 : 0;
			}
		}
		CUDA_CHECK( cudaMemcpy( ri, hri.data(), hri.size() * sizeof( int ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( rw, hrw.data(), hrw.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		auto sc = engine::modules::MoE::alloc_scratch( arena, E, K, H, I, Rn );
		for ( int it = 0; it < 2; ++it ) engine::modules::MoE::forward( xr, mw, ri, rw, outr, 0, E, sc, 0.0f, stream );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		CUDA_CHECK( cudaEventRecord( e0, stream ) );
		for ( int it = 0; it < 8; ++it ) engine::modules::MoE::forward( xr, mw, ri, rw, outr, 0, E, sc, 0.0f, stream );
		CUDA_CHECK( cudaEventRecord( e1, stream ) );
		CUDA_CHECK( cudaEventSynchronize( e1 ) );
		float ms = 0.0f;
		CUDA_CHECK( cudaEventElapsedTime( &ms, e0, e1 ) );
		std::printf( "  rows %5d: %7.3f ms per layer (%d local routes, %.1f us per route, %.2f us per row, %.0f GB/s if every route read its own expert)\n", Rn, ms / 8, local, 1000.0 * ms / 8 / MAX( 1, local ),
			1000.0 * ms / 8 / Rn, local * expert_bytes / ( ms / 8 * 1e-3 ) / 1e9 );
	}
	return 0;
}
