// Flash-Next decode MoE: time of one MoE::forward (1 row, top 10) on rank 0's 128 experts as the number of routes that
// land on this rank varies (the rest point to experts of other ranks), plus the router-free floor of 0 local routes.
//   bench_moe_routes [layer] [rows ...]   (with rows: prefill-sized calls, tokens routed uniformly over all 512 experts, rank 0's 128 are local)
//   BENCH_GLM=1 bench_moe_routes [layer] [rows ...]   GLM-5.3-Flash's experts instead (H 4096, I 2048, 72 of 288 experts on rank 0, top 8); per-route times
//   are also printed as GB/s of packed expert weights
//   BENCH_DSV41=1 bench_moe_routes [layer] [rows ...]   DeepSeek-V4.1-Flash's EXL3 experts (H 5120, I 2304, rank 0's 96 of 384, top 6), each projection at
//   its pack's width (the 1.68 bpw pack: 2 bits in some layers, 1.5 in the others)
//   BENCH_CHECK=1: per route count, the default tile's output against the tensor-core tile's (moe_coop_mode 0)
#include "engine/runtime/test_paths.h"
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/moe.h"
#include "engine/modules/trellis_bits.h"
#include "engine/runtime/options.h"
#include "trellis_context.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	const bool ds = std::getenv( "BENCH_DSV41" ) != nullptr;
	const bool glm = !ds && std::getenv( "BENCH_GLM" ) != nullptr;
	const char* dir = engine::runtime::test::model_path( ds ? "ENGINE_TEST_DSV41_DIR" : glm ? "ENGINE_TEST_GLM_DIR" : "ENGINE_TEST_FLASH_DIR" );
	int BITS = std::getenv( "BENCH_BITS" ) ? std::atoi( std::getenv( "BENCH_BITS" ) ) : 4;   // the experts' bit width (Qwen3.8-Flash-Next-exl3: 2; DeepSeek: from the pack)
	const int L = argc > 1 ? std::atoi( argv[1] ) : ( ds ? 3 : glm ? 3 : 10 );
	const int H = ds ? 5120 : glm ? 4096 : 2560, I = ds ? 2304 : glm ? 2048 : 640, E = ds ? 96 : glm ? 72 : 128, K = ds ? 6 : glm ? 8 : 10;
	const int TOTAL = ds ? 384 : glm ? 288 : 512;
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
	if ( ds ) {
		std::snprintf( name, sizeof( name ), "layers.%d.ffn.experts.0.w1.trellis", L );
		const auto* meta = repo.find( name );
		ENGINE_ASSERT( meta != nullptr && meta->ndim == 3, "DeepSeek expert tensor" );
		BITS = engine::modules::trellis_code_from_u16( meta->shape[2] );
	}
	for ( int e = 0; e < E && ds; ++e ) {
		const char* w[3] = { "w1", "w3", "w2" };
		pod::LinearWeight* dst[3] = { &mw.gate[e], &mw.up[e], &mw.down[e] };
		for ( int q = 0; q < 3; ++q ) {
			std::snprintf( name, sizeof( name ), "layers.%d.ffn.experts.%d.%s", L, e, w[q] );
			*dst[q] = repo.load_linear( name, arena, BITS, stream );
		}
	}
	for ( int e = 0; e < E && !ds; ++e ) {
		std::snprintf( name, sizeof( name ), "model.language_model.layers.%d.mlp.experts.%d.gate_proj", L, e ); mw.gate[e] = repo.load_linear( name, arena, BITS, stream );
		std::snprintf( name, sizeof( name ), "model.language_model.layers.%d.mlp.experts.%d.up_proj", L, e ); mw.up[e] = repo.load_linear( name, arena, BITS, stream );
		std::snprintf( name, sizeof( name ), "model.language_model.layers.%d.mlp.experts.%d.down_proj", L, e ); mw.down[e] = repo.load_linear( name, arena, BITS, stream );
	}
	engine::modules::MoE::init_device_ptrs( mw, arena, stream );
	std::printf( "[*] layer %d experts: bits %d, mul1 %d\n", L, mw.gate[0].bits, mw.gate[0].mul1 );
	const double expert_bytes = 3.0 * H * I * engine::modules::trellis_k2( mw.gate[0].bits ) / 16.0;   // packed trellis weights of one expert (gate + up + down)
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
	std::vector<int> counts = ds ? std::vector<int>{ 0, 1, 2, 3, 4, 5, 6 } : std::vector<int>{ 0, 1, 2, 3, 4, 5, 6, 8, 10 };
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
		if ( std::getenv( "BENCH_CHECK" ) && n > 0 ) {
			// the default tile against the tensor-core one (moe_coop_mode 0) on the same routes
			std::vector<half> o0( H ), o1( H );
			auto& mode = engine::runtime::options().moe_coop_mode;
			const int keep = mode;
			mode = 0;
			engine::modules::MoE::forward( x, mw, d_idx, d_w, out, 0, E, scratch, 0.0f, stream );
			CUDA_CHECK( cudaMemcpy( o0.data(), out.data, H * sizeof( half ), cudaMemcpyDeviceToHost ) );
			mode = keep;
			engine::modules::MoE::forward( x, mw, d_idx, d_w, out, 0, E, scratch, 0.0f, stream );
			CUDA_CHECK( cudaMemcpy( o1.data(), out.data, H * sizeof( half ), cudaMemcpyDeviceToHost ) );
			double mx = 0, num = 0, den = 0;
			for ( int i = 0; i < H; ++i ) {
				const double a = __half2float( o0[i] ), b = __half2float( o1[i] );
				mx = std::max( mx, std::abs( a - b ) ); num += ( a - b ) * ( a - b ); den += a * a;
			}
			std::printf( "  check %d routes: max |diff| %.3g, relative rms %.3g\n", n, mx, std::sqrt( num / std::max( den, 1e-30 ) ) );
		}
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
