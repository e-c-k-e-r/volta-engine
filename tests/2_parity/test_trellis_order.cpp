// Tier 2: both EXL3 tile orders on the same random codes (LinearWeight::volta_tiles: the stock order, and the Volta order of SM_70 builds).
//   (a) reconstruct: the Volta decode equals the stock decode with every tile's elements moved by the order's permutation, bit for bit
//   (b) LinearTrellis::forward (GEMV, int8 GEMV, GEMM, reconstruct + GEMM, by row count) against x @ W in double precision, W the order's
//       own reconstruction with the rotations applied
//   test_trellis_order [--quick]   (--quick: 4-bit mul1 only)
#include "engine/config.h"
#include "engine/modules/trellis.h"
#include "engine/modules/trellis_bits.h"
#include "engine/core/tensor.h"
#include "reconstruct.cuh"
#include "trellis_context.h"
#include "exl3_vl.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

// Tile element (row k, column n) that stream position p decodes to
void stock_pos( int p, int* r, int* c ) {
	const int L = p >> 3, i = p & 7;
	*r = ( L & 3 ) * 2 + ( i & 1 ) + 8 * ( ( i >> 1 ) & 1 );
	*c = 2 * ( ( L >> 3 ) + 4 * ( i >> 2 ) ) + ( ( L >> 2 ) & 1 );
}

void volta_pos( int p, int* r, int* c ) {
	const int L = p >> 3, i = p & 7, q = ( L >> 2 ) & 3;
	*r = 4 * ( 2 * ( q >> 1 ) + ( i >> 2 ) ) + ( i & 3 );
	*c = 8 * ( q & 1 ) + ( L & 3 ) + 4 * ( L >> 4 );
}

pod::TensorView view( void* p, pod::DType dt, std::initializer_list<int> shape ) {
	pod::TensorView t{};
	t.data = p;
	t.dtype = dt;
	t.ndim = static_cast<int>( shape.size() );
	int i = 0;
	for ( int s : shape ) t.shape[i++] = s;
	return t;
}

struct Case { int bits; int cb; };

int run_case( const Case& cs, bool quick ) {
	const int k = 1024, n = 512;
	const bool mcg = cs.cb == 1, mul1 = cs.cb == 2;
	const int words = engine::modules::trellis_tile_u16( cs.bits );   // uint16 words per 16x16 tile
	const size_t code_n = static_cast<size_t>( k / 16 ) * ( n / 16 ) * words;
	std::mt19937 rng( 1234u + cs.bits * 7u + cs.cb );
	std::vector<uint16_t> codes( code_n );
	for ( auto& w : codes ) w = static_cast<uint16_t>( rng() );
	std::uniform_real_distribution<float> sgn( -1.0f, 1.0f );
	std::vector<half> suh( k ), svh( n );
	for ( auto& v : suh ) v = __float2half( sgn( rng ) > 0 ? 1.0f : -1.0f );
	for ( auto& v : svh ) v = __float2half( 0.5f + 0.5f * std::fabs( sgn( rng ) ) );

	uint16_t* d_codes; half *d_suh, *d_svh, *d_w;
	CUDA_CHECK( cudaMalloc( &d_codes, code_n * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_suh, k * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_svh, n * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_w, static_cast<size_t>( k ) * n * 2 ) );
	CUDA_CHECK( cudaMemcpy( d_codes, codes.data(), code_n * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_suh, suh.data(), k * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_svh, svh.data(), n * 2, cudaMemcpyHostToDevice ) );

	int fails = 0;
	// (a) the plain decode of both orders
	std::vector<uint16_t> w_stock( static_cast<size_t>( k ) * n ), w_volta( w_stock.size() );
	EXL3_TILES_CALL( false, reconstruct_slice, d_w, d_codes, cs.bits, mcg, mul1, k / 16, n, 0, nullptr );
	CUDA_CHECK( cudaMemcpy( w_stock.data(), d_w, w_stock.size() * 2, cudaMemcpyDeviceToHost ) );
	EXL3_TILES_CALL( true, reconstruct_slice, d_w, d_codes, cs.bits, mcg, mul1, k / 16, n, 0, nullptr );
	CUDA_CHECK( cudaMemcpy( w_volta.data(), d_w, w_volta.size() * 2, cudaMemcpyDeviceToHost ) );
	size_t mism = 0;
	for ( int kt = 0; kt < k / 16; ++kt )
		for ( int nt = 0; nt < n / 16; ++nt )
			for ( int p = 0; p < 256; ++p ) {
				int rs, ccs, rv, cv;
				stock_pos( p, &rs, &ccs );
				volta_pos( p, &rv, &cv );
				const size_t is = static_cast<size_t>( kt * 16 + rs ) * n + nt * 16 + ccs, iv = static_cast<size_t>( kt * 16 + rv ) * n + nt * 16 + cv;
				mism += w_stock[is] != w_volta[iv];
			}
	std::printf( "  bits %2d cb %d  reconstruct: %zu of %d elements differ from the permuted stock decode %s\n", cs.bits, cs.cb, mism, k * n,
		mism ? "FAIL" : "ok" );
	fails += mism != 0;

	// (b) the fused paths of both orders against their own rotated reconstruction (half-integer rates only have the reconstruct path)
	const std::vector<int> rows = quick ? std::vector<int>{ 1, 2, 4, 8, 16, 48 } : std::vector<int>{ 1, 2, 3, 4, 5, 8, 12, 16, 24, 36, 48 };
	const int mmax = 48;
	std::vector<half> x( static_cast<size_t>( mmax ) * k );
	std::normal_distribution<float> nd( 0.0f, 1.0f );
	for ( auto& v : x ) v = __float2half( nd( rng ) );
	half *d_x, *d_out;
	CUDA_CHECK( cudaMalloc( &d_x, x.size() * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_out, static_cast<size_t>( mmax ) * n * 2 ) );
	CUDA_CHECK( cudaMemcpy( d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice ) );
	for ( int order = 0; order < 2; ++order ) {
		const bool volta = order == 1;
		std::vector<half> w( static_cast<size_t>( k ) * n );
		EXL3_TILES_CALL( volta, reconstruct_had_slice, d_w, d_codes, d_suh, d_svh, cs.bits, mcg, mul1, k, n, 0, nullptr );
		CUDA_CHECK( cudaMemcpy( w.data(), d_w, w.size() * 2, cudaMemcpyDeviceToHost ) );
		std::vector<float> wf( w.size() );
		for ( size_t i = 0; i < w.size(); ++i ) wf[i] = __half2float( w[i] );
		double worst = 1.0;
		int worst_m = 0;
		for ( int m : rows ) {
			pod::TensorView xt = view( d_x, pod::DType::FP16, { m, k } );
			pod::TensorView ct = view( d_codes, pod::DType::UINT16, { k / 16, n / 16, words } );
			pod::TensorView st = view( d_suh, pod::DType::FP16, { k } );
			pod::TensorView vt = view( d_svh, pod::DType::FP16, { n } );
			pod::TensorView ot = view( d_out, pod::DType::FP16, { m, n } );
			engine::modules::LinearTrellis::forward( xt, ct, st, vt, ot, cs.bits, mcg, mul1, nullptr, volta );
			CUDA_CHECK( cudaDeviceSynchronize() );
			std::vector<half> y( static_cast<size_t>( m ) * n );
			CUDA_CHECK( cudaMemcpy( y.data(), d_out, y.size() * 2, cudaMemcpyDeviceToHost ) );
			double dot = 0, ng = 0, nr = 0;
			for ( int r = 0; r < m; ++r )
				for ( int c = 0; c < n; ++c ) {
					double ref = 0;
					for ( int i = 0; i < k; ++i ) ref += static_cast<double>( __half2float( x[static_cast<size_t>( r ) * k + i] ) ) * wf[static_cast<size_t>( i ) * n + c];
					const double g = __half2float( y[static_cast<size_t>( r ) * n + c] );
					dot += g * ref; ng += g * g; nr += ref * ref;
				}
			const double cos = dot / ( std::sqrt( ng * nr ) + 1e-30 );
			if ( cos < worst ) { worst = cos; worst_m = m; }
		}
		const bool ok = worst > 0.9999;
		std::printf( "  bits %2d cb %d  %s order: forward vs x @ W over %zu row counts, worst cosine %.6f (m=%d) %s\n", cs.bits, cs.cb, volta ? "volta" : "stock",
			rows.size(), worst, worst_m, ok ? "ok" : "FAIL" );
		fails += !ok;
	}
	cudaFree( d_x ); cudaFree( d_out ); cudaFree( d_codes ); cudaFree( d_suh ); cudaFree( d_svh ); cudaFree( d_w );
	return fails;
}

} // namespace

int main( int argc, char** argv ) {
	const bool quick = argc > 1 && !std::strcmp( argv[1], "--quick" );
#if !ENGINE_EXL3_VL
	std::printf( "[*] test_trellis_order: this build has no Volta-order kernels (no SM_70 target): skipped\n" );
	return 0;
#endif
	if ( !engine::modules::trellis_volta_tiles_supported() ) {
		std::printf( "[*] test_trellis_order: the devices are not SM_70: skipped\n" );
		return 0;
	}
	engine::modules::trellis_context_init_all();
	std::vector<Case> cases;
	if ( quick ) cases = { { 4, 2 } };
	else {
		for ( int b = 1; b <= 8; ++b ) cases.push_back( { b, 2 } );
		for ( int b : { 9, 10, 11 } ) cases.push_back( { b, 2 } );   // 1.5 / 2.5 / 3.5 bpw
		for ( int b : { 2, 4, 6 } ) cases.push_back( { b, 1 } );
		cases.push_back( { 4, 0 } );
	}
	int fails = 0;
	for ( const Case& c : cases ) fails += run_case( c, quick );
	std::printf( fails ? "=== TRELLIS ORDER: %d FAILURE(S) ===\n" : "=== TRELLIS ORDER: PASS ===\n", fails );
	return fails ? 1 : 0;
}
