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
		{ "glm k4096 n6144", 4096, 6144, 5 }, { "glm k4096 n4096", 4096, 4096, 5 }, { "glm k2048 n4096", 2048, 4096, 5 },
		{ "glm k1536 n4096", 1536, 4096, 5 }, { "glm k4096 n2048", 4096, 2048, 5 }, { "glm k4096 n1536", 4096, 1536, 5 },
		{ "glm k4096 n512", 4096, 512, 5 },
		{ "glm expert g/u", 4096, 2048, 3 }, { "glm expert down", 2048, 4096, 3 },
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
		// Tile shape sweep of the GEMM at M=4 and M=8 over enough distinct weight copies to defeat the L2 (as in the model, where
		// every layer's matrix is cold). Shape 0 = default selection, gemv = the GEMV kernel.
		{
			const int nb = static_cast<int>( MAX( 2ULL, MIN( 24ULL, ( 48ULL << 20 ) / ( words * 2 ) + 1 ) ) );
			std::vector<uint16_t*> Bc( nb );
			Bc[0] = B;
			for ( int b = 1; b < nb; ++b ) {
				Bc[b] = static_cast<uint16_t*>( engine::core::arena_alloc( arena, words * 2 ) );
				CUDA_CHECK( cudaMemcpy( Bc[b], B, words * 2, cudaMemcpyDeviceToDevice ) );
			}
			for ( int m : { 4, 8 } ) {
				float t[5], tg;
				int it = 0;
				for ( int pass = 0; pass < 2; ++pass ) {
					for ( int sh_i = 0; sh_i <= 4; ++sh_i ) {
						if ( sh_i > 0 && ( sh.k % ( sh_i == 1 || sh_i == 4 ? 16 : 32 ) || sh.n % ( sh_i == 1 || sh_i == 2 ? 128 : sh_i == 3 ? 256 : 512 ) ) ) { t[sh_i] = 0; continue; }
						t[sh_i] = impl::time_ms( [&]() { exl3_gemm( A, Bc[it++ % nb], C, suh, A_had, svh, m, sh.k, sh.n, sh.bits, false, true, false, s, sh_i ); }, s, 100 );
					}
					tg = impl::time_ms( [&]() { exl3_gemv( A, Bc[it++ % nb], C, suh, A_had, svh, m, sh.k, sh.n, sh.bits, false, true, false, s ); }, s, 100 );
				}
				std::printf( "%-15s cold sweep M=%d  gemv %6.1f | default-gemm %6.1f | shape1 %6.1f shape2 %6.1f shape3 %6.1f shape4 %6.1f us\n", sh.name, m, tg * 1e3, t[0] * 1e3, t[1] * 1e3, t[2] * 1e3, t[3] * 1e3, t[4] * 1e3 );
			}
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

	// Cold-weight probe: cycle through 72 distinct expert matrices (223 MB, far above the 6 MB L2) so every call streams
	// its weights from HBM, as the MoE does; the loop above re-reads one matrix from L2.
	{
		const impl::Shape sh = { "glm expert cold", 4096, 2048, 3 };
		const int N = 72;
		const size_t words = static_cast<size_t>( sh.k / 16 ) * ( sh.n / 16 ) * 16 * sh.bits;
		std::vector<uint16_t*> Bs( N );
		std::vector<uint16_t> hb( words );
		uint32_t st = 99u;
		for ( auto& w : hb ) { st = st * 1664525u + 1013904223u; w = static_cast<uint16_t>( st >> 16 ); }
		for ( auto& b : Bs ) {
			b = static_cast<uint16_t*>( engine::core::arena_alloc( arena, words * 2 ) );
			CUDA_CHECK( cudaMemcpy( b, hb.data(), words * 2, cudaMemcpyHostToDevice ) );
		}
		auto* suh = static_cast<half*>( engine::core::arena_alloc( arena, sh.k * 2 ) );
		auto* svh = static_cast<half*>( engine::core::arena_alloc( arena, sh.n * 2 ) );
		auto* A = static_cast<half*>( engine::core::arena_alloc( arena, 16ULL * sh.k * 2 ) );
		auto* C = static_cast<half*>( engine::core::arena_alloc( arena, 16ULL * sh.n * 2 ) );
		CUDA_CHECK( cudaMemset( suh, 0x3c, sh.k * 2 ) );
		CUDA_CHECK( cudaMemset( svh, 0x3c, sh.n * 2 ) );
		CUDA_CHECK( cudaMemset( A, 0x3c, 16ULL * sh.k * 2 ) );
		A_had = engine::modules::trellis_get_workspace( 0, 16ULL * 8192 * sizeof( half ) );
		const double mb = words * 2 / 1e6;
		for ( int m : { 1, 4, 8, 16 } ) {
			int i = 0;
			const float tv = impl::time_ms( [&]() { exl3_gemv( A, Bs[i++ % N], C, suh, A_had, svh, m, sh.k, sh.n, sh.bits, false, true, false, s ); }, s, 144 );
			const float tg = impl::time_ms( [&]() { exl3_gemm( A, Bs[i++ % N], C, suh, A_had, svh, m, sh.k, sh.n, sh.bits, false, true, false, s ); }, s, 144 );
			std::printf( "%-15s k%-5d n%-6d %db M=%d  %7.1f MB | gemv %8.1f us %6.0f GB/s | gemm %8.1f us %6.0f GB/s\n",
				sh.name, sh.k, sh.n, sh.bits, m, mb, tv * 1e3, mb / tv, tg * 1e3, mb / tg );
		}
		engine::core::arena_reset( arena );
	}
	// Concurrency probe: two independent GEMVs of one shape (own weights, own Hadamard workspace) back to
	// back on one stream versus forked onto two streams. M >= 4 only: the M <= 2 int8 path shares one
	// device workspace and cannot run concurrently with itself.
	{
		cudaStream_t s2;
		CUDA_CHECK( cudaStreamCreateWithFlags( &s2, cudaStreamNonBlocking ) );
		cudaEvent_t fork_ev, join_ev;
		CUDA_CHECK( cudaEventCreateWithFlags( &fork_ev, cudaEventDisableTiming ) );
		CUDA_CHECK( cudaEventCreateWithFlags( &join_ev, cudaEventDisableTiming ) );
		const impl::Shape pair_shapes[] = {
			{ "mlp gate+up", 5120, 4352, 4 }, { "gdn qkv+z", 5120, 2560, 4 }, { "drafter gate+up", 5120, 17408, 5 },
		};
		for ( const auto& sh : pair_shapes ) {
			const size_t words = static_cast<size_t>( sh.k / 16 ) * ( sh.n / 16 ) * 16 * sh.bits;
			uint16_t* B[2];
			half* Ah[2];
			half* C[2];
			half* suh = static_cast<half*>( engine::core::arena_alloc( arena, sh.k * 2 ) );
			half* svh = static_cast<half*>( engine::core::arena_alloc( arena, sh.n * 2 ) );
			half* A = static_cast<half*>( engine::core::arena_alloc( arena, 8ULL * sh.k * 2 ) );
			CUDA_CHECK( cudaMemset( suh, 0x3c, sh.k * 2 ) );
			CUDA_CHECK( cudaMemset( svh, 0x3c, sh.n * 2 ) );
			CUDA_CHECK( cudaMemset( A, 0x30, 8ULL * sh.k * 2 ) );
			for ( int i = 0; i < 2; ++i ) {
				B[i] = static_cast<uint16_t*>( engine::core::arena_alloc( arena, words * 2 ) );
				C[i] = static_cast<half*>( engine::core::arena_alloc( arena, 8ULL * sh.n * 2 ) );
				Ah[i] = static_cast<half*>( engine::core::arena_alloc( arena, 8ULL * sh.k * 2 ) );
				CUDA_CHECK( cudaMemset( B[i], 0x5a + i, words * 2 ) );
			}
			for ( int m : { 4, 8 } ) {
				auto one = [&]( int i, cudaStream_t st ) {
					exl3_gemv( A, B[i], C[i], suh, Ah[i], svh, m, sh.k, sh.n, sh.bits, false, true, false, st );
				};
				const float t_seq = impl::time_ms( [&]() { one( 0, s ); one( 1, s ); }, s, 50 );
				const float t_par = impl::time_ms( [&]() {
					CUDA_CHECK( cudaEventRecord( fork_ev, s ) );
					CUDA_CHECK( cudaStreamWaitEvent( s2, fork_ev, 0 ) );
					one( 0, s );
					one( 1, s2 );
					CUDA_CHECK( cudaEventRecord( join_ev, s2 ) );
					CUDA_CHECK( cudaStreamWaitEvent( s, join_ev, 0 ) );
				}, s, 50 );
				std::printf( "%-16s M=%d pair: sequential %7.1f us | forked %7.1f us (%.2fx)\n", sh.name, m, t_seq * 1e3, t_par * 1e3, t_seq / t_par );
			}
			engine::core::arena_reset( arena );
			A_had = engine::modules::trellis_get_workspace( 0, 16ULL * 8192 * sizeof( half ) );
		}
	}
	// Large-M GEMM scaling (prefill chunks): time per call as M grows, on the Qwen MLP shapes.
	{
		const impl::Shape big[] = { { "mlp gate/up", 5120, 4352, 4 }, { "mlp down", 4352, 5120, 4 }, { "gdn qkv", 5120, 2560, 4 } };
		const int ms_big[] = { 16, 32, 64, 128, 256 };
		for ( const auto& sh : big ) {
			const size_t words = static_cast<size_t>( sh.k / 16 ) * ( sh.n / 16 ) * 16 * sh.bits;
			auto* B = static_cast<uint16_t*>( engine::core::arena_alloc( arena, words * 2 ) );
			auto* suh = static_cast<half*>( engine::core::arena_alloc( arena, sh.k * 2 ) );
			auto* svh = static_cast<half*>( engine::core::arena_alloc( arena, sh.n * 2 ) );
			auto* A = static_cast<half*>( engine::core::arena_alloc( arena, 256ULL * sh.k * 2 ) );
			auto* C = static_cast<half*>( engine::core::arena_alloc( arena, 256ULL * sh.n * 2 ) );
			half* Ah = engine::modules::trellis_get_workspace( 0, 256ULL * 8192 * sizeof( half ) );
			CUDA_CHECK( cudaMemset( B, 0x5a, words * 2 ) );
			CUDA_CHECK( cudaMemset( suh, 0x3c, sh.k * 2 ) );
			CUDA_CHECK( cudaMemset( svh, 0x3c, sh.n * 2 ) );
			CUDA_CHECK( cudaMemset( A, 0x30, 256ULL * sh.k * 2 ) );
			for ( int m : ms_big ) {
				const float t = impl::time_ms( [&]() { exl3_gemm( A, B, C, suh, Ah, svh, m, sh.k, sh.n, sh.bits, false, true, false, s ); }, s, 20 );
				const double flop = 2.0 * m * sh.k * sh.n;
				std::printf( "%-12s gemm M=%3d: %8.1f us  (%.1f us/row, %.1f TFLOP/s)\n", sh.name, m, t * 1e3, t * 1e3 / m, flop / ( t * 1e-3 ) / 1e12 );
			}
			engine::core::arena_reset( arena );
			A_had = engine::modules::trellis_get_workspace( 0, 16ULL * 8192 * sizeof( half ) );
		}
	}
	engine::core::arena_destroy( arena );
	engine::modules::trellis_context_destroy();
	cudaDeviceReset();
	return 0;
}
