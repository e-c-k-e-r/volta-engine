// Per-shape GEMV vs GEMM timing on Qwen TP4 per-rank shapes (GPU 0). Random trellis data:
// timing only, numerics meaningless.
#include <vector>
#include <cmath>
#include "engine/config.h"
#include "engine/core/arena.h"
#include "exl3_gemv.cuh"
#include "exl3_gemm.cuh"
#include "trellis_context.h"
#include <cstdio>

namespace impl {

struct Shape { const char* name; int k, n, bits; };

template <typename F>
float time_ms( F&& f, cudaStream_t s, int iters ) {
	for ( int i = 0; i < 5; ++i ) f();
	cudaEvent_t a, b;
	CUDA_CHECK( cudaEventCreate( &a ) );
	CUDA_CHECK( cudaEventCreate( &b ) );
	CUDA_CHECK( cudaEventRecord( a, s ) );
	for ( int i = 0; i < iters; ++i ) f();
	CUDA_CHECK( cudaEventRecord( b, s ) );
	CUDA_CHECK( cudaEventSynchronize( b ) );
	float ms = 0.0f;
	CUDA_CHECK( cudaEventElapsedTime( &ms, a, b ) );
	return ms / iters;
}

} // namespace impl

int main() {
	CUDA_CHECK( cudaSetDevice( 0 ) );
	engine::modules::trellis_context_init_all();
	exl3_gemm_init_all_attrs( 0 );
	cudaStream_t s;
	CUDA_CHECK( cudaStreamCreate( &s ) );
	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 4ULL << 30 ), "arena" );
	half* A_had = engine::modules::trellis_get_workspace( 0, 16ULL * 8192 * sizeof( half ) );
	engine::modules::trellis_get_locks( 0 );

	const impl::Shape shapes[] = {
		{ "mlp gate/up", 5120, 4352, 4 }, { "mlp down", 4352, 5120, 4 },
		{ "gdn qkv", 5120, 2560, 4 }, { "gdn z", 5120, 1536, 4 }, { "gdn out", 1536, 5120, 4 },
		{ "attn q+gate", 5120, 3072, 4 }, { "mlp gate/up 5b", 5120, 4352, 5 },
		{ "lm_head full", 5120, 248320, 5 }, { "lm_head /4", 5120, 62080, 5 },
		// DFlash2 drafter backbone (5-bit, unsharded): per layer q, k/v, o, gate/up, down; plus fc
		{ "drafter gate/up", 5120, 17408, 5 }, { "drafter down", 17408, 5120, 5 },
		{ "drafter q", 5120, 4096, 5 }, { "drafter o", 4096, 5120, 5 }, { "drafter k/v", 5120, 1024, 5 },
		{ "drafter fc", 25600, 5120, 5 },
	};
	const int ms_list[] = { 1, 2, 4, 8 };
	for ( const auto& sh : shapes ) {
		const size_t words = static_cast<size_t>( sh.k / 16 ) * ( sh.n / 16 ) * 16 * sh.bits;
		auto* B = static_cast<uint16_t*>( engine::core::arena_alloc( arena, words * 2 ) );
		auto* suh = static_cast<half*>( engine::core::arena_alloc( arena, sh.k * 2 ) );
		auto* svh = static_cast<half*>( engine::core::arena_alloc( arena, sh.n * 2 ) );
		auto* A = static_cast<half*>( engine::core::arena_alloc( arena, 8ULL * sh.k * 2 ) );
		auto* C = static_cast<half*>( engine::core::arena_alloc( arena, 8ULL * sh.n * 2 ) );
		{
			// Pseudo-random trellis words and activations so the row-consistency check below is not trivial.
			std::vector<uint16_t> hb( words );
			uint32_t st = 12345u;
			for ( auto& w : hb ) { st = st * 1664525u + 1013904223u; w = static_cast<uint16_t>( st >> 16 ); }
			CUDA_CHECK( cudaMemcpy( B, hb.data(), words * 2, cudaMemcpyHostToDevice ) );
		}
		CUDA_CHECK( cudaMemset( suh, 0x3c, sh.k * 2 ) );
		CUDA_CHECK( cudaMemset( svh, 0x3c, sh.n * 2 ) );
		{
			std::vector<half> ha( 8ULL * sh.k );
			uint32_t st = 777u;
			for ( auto& v : ha ) { st = st * 1664525u + 1013904223u; v = __float2half( ( static_cast<int>( st >> 16 ) % 2001 - 1000 ) / 1000.0f ); }
			CUDA_CHECK( cudaMemcpy( A, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice ) );
		}
		const double mb = words * 2 / 1e6;
		for ( int m : ms_list ) {
			const float tv = impl::time_ms( [&]() { exl3_gemv( A, B, C, suh, A_had, svh, m, sh.k, sh.n, sh.bits, false, true, false, s ); }, s, 50 );
			const float tg = impl::time_ms( [&]() { exl3_gemm( A, B, C, suh, A_had, svh, m, sh.k, sh.n, sh.bits, false, true, false, s ); }, s, 50 );
			std::printf( "%-15s k%-5d n%-6d %db M=%d  %7.1f MB | gemv %8.1f us %6.0f GB/s | gemm %8.1f us %6.0f GB/s\n",
				sh.name, sh.k, sh.n, sh.bits, m, mb, tv * 1e3, mb / tv, tg * 1e3, mb / tg );
		}
		// Row consistency: row r of an M=2 call against the same row run alone at M=1.
		{
			const size_t cn = 2ULL * sh.n;
			std::vector<half> c8( cn ), c1( sh.n );
			exl3_gemv( A, B, C, suh, A_had, svh, 2, sh.k, sh.n, sh.bits, false, true, false, s );
			CUDA_CHECK( cudaStreamSynchronize( s ) );
			CUDA_CHECK( cudaMemcpy( c8.data(), C, cn * 2, cudaMemcpyDeviceToHost ) );
			int bad = 0;
			double maxd = 0.0;
			for ( int r = 0; r < 2; ++r ) {
				exl3_gemv( A + static_cast<size_t>( r ) * sh.k, B, C, suh, A_had, svh, 1, sh.k, sh.n, sh.bits, false, true, false, s );
				CUDA_CHECK( cudaStreamSynchronize( s ) );
				CUDA_CHECK( cudaMemcpy( c1.data(), C, sh.n * 2, cudaMemcpyDeviceToHost ) );
				for ( int j = 0; j < sh.n; ++j ) {
					const double d = std::fabs( static_cast<double>( __half2float( c8[r * sh.n + j] ) ) - __half2float( c1[j] ) );
					if ( d != 0.0 ) ++bad;
					maxd = d > maxd ? d : maxd;
				}
			}
			std::printf( "%-15s row-consistency M=2 vs M=1: %d / %zu elements differ, max |d| %.4g\n", sh.name, bad, cn, maxd );
		}
		engine::core::arena_reset( arena );
		A_had = engine::modules::trellis_get_workspace( 0, 16ULL * 8192 * sizeof( half ) );
	}
	engine::core::arena_destroy( arena );
	engine::modules::trellis_context_destroy();
	cudaDeviceReset();
	return 0;
}
