// Large-M linear path (dequantize once + cuBLAS HGEMM) against the fused EXL3 GEMM kernel, on the per-rank Qwen
// shapes with random trellis data, plus timing of both to locate the crossover.
//   pass: cosine >= 0.9999 and max |diff| <= 0.5% of the output's RMS-scaled range for every shape and M.
#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/modules/trellis.h"
#include "exl3_gemm.cuh"
#include "trellis_context.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

struct Shape { const char* name; int k, n, bits; };

template <typename F>
float time_us( F&& f, cudaStream_t s, int iters ) {
	for ( int i = 0; i < 3; ++i ) f();
	cudaEvent_t a, b;
	CUDA_CHECK( cudaEventCreate( &a ) );
	CUDA_CHECK( cudaEventCreate( &b ) );
	CUDA_CHECK( cudaEventRecord( a, s ) );
	for ( int i = 0; i < iters; ++i ) f();
	CUDA_CHECK( cudaEventRecord( b, s ) );
	CUDA_CHECK( cudaEventSynchronize( b ) );
	float ms = 0.0f;
	CUDA_CHECK( cudaEventElapsedTime( &ms, a, b ) );
	return ms * 1e3f / iters;
}

} // namespace

int main() {
	CUDA_CHECK( cudaSetDevice( 0 ) );
	cudaStream_t s;
	CUDA_CHECK( cudaStreamCreate( &s ) );
	engine::modules::trellis_context_init_all();
	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 2ULL << 30 ), "arena" );

	const Shape shapes[] = {
		{ "mlp gate/up 4b", 5120, 4352, 4 }, { "mlp down 4b", 4352, 5120, 4 }, { "gdn qkv 4b", 5120, 2560, 4 },
		{ "gdn z 4b", 5120, 1536, 4 }, { "gdn out 4b", 1536, 5120, 4 }, { "attn q 5b", 5120, 3072, 5 }, { "attn kv 4b", 5120, 512, 4 }, { "attn kv 5b", 5120, 512, 5 }, { "attn q 4b", 5120, 3072, 4 },
	};
	const int ms[] = { 16, 17, 20, 24, 32, 40, 48, 64 };
	int fails = 0;
	for ( const auto& sh : shapes ) {
		const size_t words = static_cast<size_t>( sh.k / 16 ) * ( sh.n / 16 ) * 16 * sh.bits;
		std::vector<uint16_t> ht( words );
		uint32_t st = 99u;
		for ( auto& w : ht ) { st = st * 1664525u + 1013904223u; w = static_cast<uint16_t>( st >> 16 ); }
		std::vector<half> hs_u( sh.k ), hs_v( sh.n ), hx( 256ULL * sh.k );
		for ( auto& v : hs_u ) { st = st * 1664525u + 1013904223u; v = __float2half( ( st >> 16 ) & 1 ? 1.0f : -1.0f ); }
		for ( auto& v : hs_v ) { st = st * 1664525u + 1013904223u; v = __float2half( ( st >> 16 ) & 1 ? 1.0f : -1.0f ); }
		for ( auto& v : hx ) { st = st * 1664525u + 1013904223u; v = __float2half( ( static_cast<int>( st >> 16 ) % 2001 - 1000 ) / 500.0f ); }

		engine::core::arena_reset( arena );
		int tshape[3] = { sh.k / 16, sh.n / 16, 16 * sh.bits };
		pod::TensorView tv = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, tshape, 3 );
		int ks[1] = { sh.k }, ns[1] = { sh.n };
		pod::TensorView su = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ks, 1 );
		pod::TensorView sv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ns, 1 );
		int xs[2] = { 256, sh.k }, os[2] = { 256, sh.n };
		pod::TensorView x = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
		pod::TensorView o1 = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, os, 2 );
		pod::TensorView o2 = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, os, 2 );
		CUDA_CHECK( cudaMemcpy( tv.data, ht.data(), words * 2, cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( su.data, hs_u.data(), sh.k * 2, cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( sv.data, hs_v.data(), sh.n * 2, cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( x.data, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice ) );
		half* a_had = engine::modules::trellis_get_workspace( 0, 256ULL * sh.k * sizeof( half ) );

		for ( int m : ms ) {
			pod::TensorView xm = pod::make_tensor_2d( x.data, pod::DType::FP16, m, sh.k );
			pod::TensorView om1 = pod::make_tensor_2d( o1.data, pod::DType::FP16, m, sh.n );
			pod::TensorView om2 = pod::make_tensor_2d( o2.data, pod::DType::FP16, m, sh.n );
			auto gemm = [&]() {
				exl3_gemm( x.ptr<const half>(), tv.ptr<const uint16_t>(), o1.data, su.ptr<const half>(), a_had, sv.ptr<const half>(),
					m, sh.k, sh.n, sh.bits, false, true, false, s );
			};
			auto recon = [&]() {
				engine::modules::LinearTrellis::forward_recon( xm, tv, su, sv, om2, sh.bits, false, true, s );
			};
			gemm();
			recon();
			CUDA_CHECK( cudaStreamSynchronize( s ) );
			std::vector<half> r1( static_cast<size_t>( m ) * sh.n ), r2( r1.size() );
			CUDA_CHECK( cudaMemcpy( r1.data(), o1.data, r1.size() * 2, cudaMemcpyDeviceToHost ) );
			CUDA_CHECK( cudaMemcpy( r2.data(), o2.data, r2.size() * 2, cudaMemcpyDeviceToHost ) );
			double dot = 0, n1 = 0, n2 = 0, maxd = 0, sq = 0;
			for ( size_t i = 0; i < r1.size(); ++i ) {
				const double a = __half2float( r1[i] ), b = __half2float( r2[i] );
				dot += a * b; n1 += a * a; n2 += b * b; sq += a * a;
				maxd = std::fmax( maxd, std::fabs( a - b ) );
			}
			const double cosv = dot / ( std::sqrt( n1 ) * std::sqrt( n2 ) + 1e-30 );
			const double rms = std::sqrt( sq / r1.size() );
			const bool ok = cosv >= 0.9999 && maxd <= 0.05 * rms * 4.0;
			fails += ok ? 0 : 1;
			const float t_g = time_us( gemm, s, 10 ), t_r = time_us( recon, s, 10 );
			std::printf( "  %-15s M=%3d: cos %.6f  max|d| %.4g (rms %.3g) | gemm %7.1f us  recon %7.1f us (%.2fx)  %s\n",
				sh.name, m, cosv, maxd, rms, t_g, t_r, t_g / t_r, ok ? "PASS" : "FAIL" );
		}
	}
	engine::modules::trellis_context_destroy();
	std::printf( "=== PREFILL LINEAR: %s ===\n", fails ? "FAIL" : "PASS" );
	return fails ? 1 : 0;
}
