// Gated residual tensor-core path (hi/lo fp16 GEMM pairs) against the fp32 row-tile kernel, and row-shift invariance:
// a row's mixed output must not depend on its position in the batch or on how many rows share the call.
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/modules/gated_residual.h"
#include "engine/backend/runtime.h"
#include "trellis_context.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using engine::modules::GatedResidual;
using half_t = engine::backend::half_t;
namespace backend = engine::backend;

namespace {

struct Buffers {
	float *x, *normed, *post, *scratch, *z, *u;
	half_t *hi, *lo, *thi, *tlo, *mixed, *mixed_tc, *w_norm, *down, *up, *inject, *down_cat;
};

} // namespace

int main() {
	CUDA_CHECK( cudaSetDevice( 0 ) );
	engine::modules::trellis_context_init_all();
	cudaStream_t s;
	CUDA_CHECK( cudaStreamCreate( &s ) );
	const int H = 2560, hc = 4, flat = hc * H, rank = 320, R = 512;
	auto dalloc = []( size_t bytes ) { void* p = nullptr; CUDA_CHECK( cudaMalloc( &p, bytes ) ); return p; };
	Buffers b;
	b.x = static_cast<float*>( dalloc( sizeof( float ) * R * flat ) );
	b.normed = static_cast<float*>( dalloc( sizeof( float ) * R * flat ) );
	b.post = static_cast<float*>( dalloc( sizeof( float ) * R * hc ) );
	b.scratch = static_cast<float*>( dalloc( sizeof( float ) * GatedResidual::scratch_floats( R, H, hc, rank ) ) );
	b.z = static_cast<float*>( dalloc( sizeof( float ) * R * ( rank + 8 ) ) );
	b.u = static_cast<float*>( dalloc( sizeof( float ) * R * flat ) );
	b.hi = static_cast<half_t*>( dalloc( 2ull * R * flat ) );
	b.lo = static_cast<half_t*>( dalloc( 2ull * R * flat ) );
	b.thi = static_cast<half_t*>( dalloc( 2ull * R * rank ) );
	b.tlo = static_cast<half_t*>( dalloc( 2ull * R * rank ) );
	b.mixed = static_cast<half_t*>( dalloc( 2ull * R * H ) );
	b.mixed_tc = static_cast<half_t*>( dalloc( 2ull * R * H ) );
	b.w_norm = static_cast<half_t*>( dalloc( 2ull * flat ) );
	b.down = static_cast<half_t*>( dalloc( 2ull * rank * flat ) );
	b.up = static_cast<half_t*>( dalloc( 2ull * flat * rank ) );
	b.inject = static_cast<half_t*>( dalloc( 2ull * hc * flat ) );
	b.down_cat = static_cast<half_t*>( dalloc( 2ull * ( rank + 8 ) * flat ) );

	uint32_t st = 4242u;
	auto rnd = [&]() { st = st * 1664525u + 1013904223u; return ( ( st >> 8 ) / 16777216.0f ) * 2.0f - 1.0f; };
	auto fill_h = [&]( half_t* d, size_t n, float scale ) {
		std::vector<__half> h( n );
		for ( auto& v : h ) v = __float2half( rnd() * scale );
		CUDA_CHECK( cudaMemcpy( d, h.data(), n * 2, cudaMemcpyHostToDevice ) );
	};
	{
		std::vector<float> hx( static_cast<size_t>( R ) * flat );
		for ( auto& v : hx ) v = rnd() * 3.0f;
		CUDA_CHECK( cudaMemcpy( b.x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice ) );
	}
	fill_h( b.w_norm, flat, 0.1f );
	fill_h( b.down, static_cast<size_t>( rank ) * flat, 0.02f );
	fill_h( b.up, static_cast<size_t>( flat ) * rank, 0.1f );
	fill_h( b.inject, static_cast<size_t>( hc ) * flat, 0.02f );
	CUDA_CHECK( cudaMemset( b.down_cat, 0, 2ull * ( rank + 8 ) * flat ) );
	CUDA_CHECK( cudaMemcpy( b.down_cat, b.down, 2ull * rank * flat, cudaMemcpyDeviceToDevice ) );
	CUDA_CHECK( cudaMemcpy( b.down_cat + static_cast<size_t>( rank ) * flat, b.inject, 2ull * hc * flat, cudaMemcpyDeviceToDevice ) );

	auto run_tc = [&]( const float* xin, int rows, half_t* out ) {
		GatedResidual::norm( const_cast<float*>( xin ), b.w_norm, b.scratch, b.hi, b.lo, rows, H, hc, 1e-6f, nullptr, nullptr, s );
		GatedResidual::gemm_f32( b.hi, b.down_cat, b.z, R, rank + 8, flat, false, s );
		GatedResidual::gemm_f32( b.lo, b.down_cat, b.z, R, rank + 8, flat, true, s );
		GatedResidual::gate_act( b.z, rank + 8, b.thi, b.tlo, b.post, rows, rank, hc, true, s );
		GatedResidual::gemm_f32( b.thi, b.up, b.u, R, flat, rank, false, s );
		GatedResidual::gemm_f32( b.tlo, b.up, b.u, R, flat, rank, true, s );
		GatedResidual::up_epilogue( b.u, b.scratch, out, rows, H, hc, s );
	};

	// 1. against the fp32 kernels
	GatedResidual::mix( b.x, b.w_norm, b.down, b.up, b.inject, b.mixed, b.post, b.scratch, R, H, hc, rank, 1e-6f, s );
	run_tc( b.x, R, b.mixed_tc );
	CUDA_CHECK( cudaStreamSynchronize( s ) );
	std::vector<__half> a( static_cast<size_t>( R ) * H ), c( static_cast<size_t>( R ) * H );
	CUDA_CHECK( cudaMemcpy( a.data(), b.mixed, a.size() * 2, cudaMemcpyDeviceToHost ) );
	CUDA_CHECK( cudaMemcpy( c.data(), b.mixed_tc, c.size() * 2, cudaMemcpyDeviceToHost ) );
	double num = 0, den = 0, maxd = 0;
	size_t diff_elems = 0;
	for ( size_t i = 0; i < a.size(); ++i ) {
		const double d = static_cast<double>( __half2float( a[i] ) ) - __half2float( c[i] );
		num += d * d; den += static_cast<double>( __half2float( a[i] ) ) * __half2float( a[i] );
		maxd = std::fmax( maxd, std::fabs( d ) );
		diff_elems += d != 0.0 ? 1 : 0;
	}
	std::printf( "[*] tensor-core vs fp32 kernels: rel L2 %.3g, max |d| %.3g, %zu of %zu fp16 outputs differ\n", std::sqrt( num / den ), maxd, diff_elems, a.size() );
	bool ok = std::sqrt( num / den ) < 2e-3;

	// 2. row-shift invariance: rows [100, 100 + n) of x run alone as rows [0, n) (different counts and positions)
	for ( int n : { 9, 64, 300 } ) {
		run_tc( b.x + static_cast<size_t>( 100 ) * flat, n, b.mixed_tc );
		CUDA_CHECK( cudaStreamSynchronize( s ) );
		std::vector<__half> d2( static_cast<size_t>( n ) * H );
		CUDA_CHECK( cudaMemcpy( d2.data(), b.mixed_tc, d2.size() * 2, cudaMemcpyDeviceToHost ) );
		size_t bad = 0;
		for ( int r = 0; r < n; ++r )
			for ( int d = 0; d < H; ++d ) bad += __half2float( d2[static_cast<size_t>( r ) * H + d] ) != __half2float( c[static_cast<size_t>( 100 + r ) * H + d] ) ? 1 : 0;
		std::printf( "[%c] rows 100..%d alone vs inside the 512-row call: %zu of %zu outputs differ\n", bad ? '!' : '+', 100 + n - 1, bad, d2.size() );
		if ( bad ) ok = false;
	}
	std::printf( ok ? "[+] GR TENSOR-CORE PASS\n" : "[-] GR TENSOR-CORE FAIL\n" );
	return ok ? 0 : 1;
}
