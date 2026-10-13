// Prefill attention kernel against the decode kernel on random K/V: a chunk of rows at consecutive positions over a
// cache that already holds earlier positions, GQA at head_dim 256: 6 q-heads / 2 kv-heads (the per-rank Qwen shape) and 24 / 4.
//   pass: cosine >= 0.9999 and max relative error <= 0.5% per case.
#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/core/batch.h"
#include "engine/modules/attention.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
	CUDA_CHECK( cudaSetDevice( 0 ) );
	cudaStream_t s;
	CUDA_CHECK( cudaStreamCreate( &s ) );
	const int hd = 256, max_seq = 8192;  // cases keep pos0 + rows <= max_seq
	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 1ULL << 30 ), "arena" );
	const size_t kv_elems = static_cast<size_t>( max_seq ) * 4 * hd;   // one layer, one slot, up to 4 kv heads
	auto* k_cache = static_cast<half*>( engine::core::arena_alloc( arena, kv_elems * 2 ) );
	auto* v_cache = static_cast<half*>( engine::core::arena_alloc( arena, kv_elems * 2 ) );
	std::vector<half> hk( kv_elems ), hv( kv_elems );
	uint32_t st = 31u;
	for ( size_t i = 0; i < kv_elems; ++i ) {
		st = st * 1664525u + 1013904223u; hk[i] = __float2half( ( static_cast<int>( st >> 16 ) % 2001 - 1000 ) / 700.0f );
		st = st * 1664525u + 1013904223u; hv[i] = __float2half( ( static_cast<int>( st >> 16 ) % 2001 - 1000 ) / 500.0f );
	}
	CUDA_CHECK( cudaMemcpy( k_cache, hk.data(), kv_elems * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( v_cache, hv.data(), kv_elems * 2, cudaMemcpyHostToDevice ) );
	auto* d_rows = static_cast<pod::BatchRows*>( engine::core::arena_alloc( arena, sizeof( pod::BatchRows ) ) );

	struct Case { int rows, pos0; bool per_row = false; };
	const Case cases[] = { { 20, 0 }, { 33, 5 }, { 128, 0 }, { 128, 700 }, { 256, 1900 }, { 200, 4000 }, { 256, 7936 }, { 256, 7700 },
		{ 1, 5000 }, { 2, 3000 }, { 5, 6000 }, { 8, 7000 }, { 8, 40 }, { 3, 0 },
		{ 1, 5000, true }, { 5, 6000, true }, { 8, 7000, true }, { 8, 40, true } }; // decode / verify shapes (key_hint 0, as the graphs pass)
	int fails = 0;
	for ( const int shape : { 0, 1 } ) {
	const int qh = shape ? 24 : 6, kvh = shape ? 4 : 2;   // per-rank TP4 shape, and the whole model on one GPU
	std::printf( "q heads %d, kv heads %d\n", qh, kvh );
	for ( const auto& c : cases ) {
		const size_t qn = static_cast<size_t>( c.rows ) * qh * hd;
		std::vector<half> hq( qn ), hg( qn );
		for ( size_t i = 0; i < qn; ++i ) {
			st = st * 1664525u + 1013904223u; hq[i] = __float2half( ( static_cast<int>( st >> 16 ) % 2001 - 1000 ) / 400.0f );
			st = st * 1664525u + 1013904223u; hg[i] = __float2half( ( static_cast<int>( st >> 16 ) % 2001 - 1000 ) / 600.0f );
		}
		int qs[3] = { c.rows, qh, hd };
		pod::TensorView q = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, qs, 3 );
		pod::TensorView g = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, qs, 3 );
		pod::TensorView o_ref = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, qs, 3 );
		pod::TensorView o_new = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, qs, 3 );
		CUDA_CHECK( cudaMemcpy( q.data, hq.data(), qn * 2, cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( g.data, hg.data(), qn * 2, cudaMemcpyHostToDevice ) );
		const auto rows = pod::BatchRows::causal( c.rows, c.pos0, 0 );
		CUDA_CHECK( cudaMemcpy( d_rows, &rows, sizeof( rows ), cudaMemcpyHostToDevice ) );
		engine::modules::Attention::forward_decode( pod::kv_ref_f16( k_cache, v_cache ), q, g, o_ref, 0, 0, d_rows, c.rows, qh, kvh, hd, max_seq, s );
		engine::modules::Attention::forward_prefill( pod::kv_ref_f16( k_cache, v_cache ), q, g, o_new, 0, 0, d_rows, c.rows, qh, kvh, hd, max_seq, s, c.rows <= pod::MAX_BATCH ? 0 : c.pos0 + c.rows, c.per_row );
		CUDA_CHECK( cudaStreamSynchronize( s ) );
		std::vector<half> r1( qn ), r2( qn );
		CUDA_CHECK( cudaMemcpy( r1.data(), o_ref.data, qn * 2, cudaMemcpyDeviceToHost ) );
		CUDA_CHECK( cudaMemcpy( r2.data(), o_new.data, qn * 2, cudaMemcpyDeviceToHost ) );
		double dot = 0, n1 = 0, n2 = 0, maxd = 0, maxabs = 0;
		for ( size_t i = 0; i < qn; ++i ) {
			const double a = __half2float( r1[i] ), b = __half2float( r2[i] );
			dot += a * b; n1 += a * a; n2 += b * b;
			maxd = std::fmax( maxd, std::fabs( a - b ) );
			maxabs = std::fmax( maxabs, std::fabs( a ) );
		}
		const double cosv = dot / ( std::sqrt( n1 ) * std::sqrt( n2 ) + 1e-30 );
		const bool ok = cosv >= 0.9999 && maxd <= 0.005 * maxabs + 1e-3;
		fails += ok ? 0 : 1;
		// timing
		cudaEvent_t e0, e1;
		CUDA_CHECK( cudaEventCreate( &e0 ) ); CUDA_CHECK( cudaEventCreate( &e1 ) );
		CUDA_CHECK( cudaEventRecord( e0, s ) );
		for ( int i = 0; i < 5; ++i ) engine::modules::Attention::forward_decode( pod::kv_ref_f16( k_cache, v_cache ), q, g, o_ref, 0, 0, d_rows, c.rows, qh, kvh, hd, max_seq, s );
		CUDA_CHECK( cudaEventRecord( e1, s ) ); CUDA_CHECK( cudaEventSynchronize( e1 ) );
		float t_dec = 0; CUDA_CHECK( cudaEventElapsedTime( &t_dec, e0, e1 ) );
		CUDA_CHECK( cudaEventRecord( e0, s ) );
		for ( int i = 0; i < 5; ++i ) engine::modules::Attention::forward_prefill( pod::kv_ref_f16( k_cache, v_cache ), q, g, o_new, 0, 0, d_rows, c.rows, qh, kvh, hd, max_seq, s, c.rows <= pod::MAX_BATCH ? 0 : c.pos0 + c.rows, c.per_row );
		CUDA_CHECK( cudaEventRecord( e1, s ) ); CUDA_CHECK( cudaEventSynchronize( e1 ) );
		float t_pre = 0; CUDA_CHECK( cudaEventElapsedTime( &t_pre, e0, e1 ) );
		std::printf( "  rows %3d at pos %4d: cos %.6f max|d| %.4g (max %.3g) | decode kernel %8.1f us  prefill kernel %8.1f us (%.1fx)  %s\n",
			c.rows, c.pos0, cosv, maxd, maxabs, t_dec * 200.0f, t_pre * 200.0f, t_dec / t_pre, ok ? "PASS" : "FAIL" );
	}
	}
	std::printf( "=== ATTENTION PREFILL: %s ===\n", fails ? "FAIL" : "PASS" );
	return fails ? 1 : 0;
}
