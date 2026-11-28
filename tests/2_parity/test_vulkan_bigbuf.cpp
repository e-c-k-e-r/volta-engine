// Vulkan: tensors deep inside a large arena. The arena is one allocation of 17 GiB, which the Vulkan backend serves as a sparse buffer over 4 GiB memory
// chunks (a driver caps one memory allocation near 16 GiB), and its bound-buffer shaders (llama.cpp's mat-vec and matmul) read a tensor through a
// descriptor window over the arena (a driver reads a descriptor into a buffer past 4 GiB wrongly). FP16 weights are placed at offsets before, across and past
// the 4, 8, 12 and 16 GiB marks, uploaded, and multiplied at 1 row (the mat-vec) and 40 rows (the matmul), against an FP64 host product.
// Skips where the device cannot hold the arena.
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/backend/tiers.h"
#include "engine/core/arena.h"
#include "engine/modules/linear.h"
#include "../../src/backend/vulkan/mod_common.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace engine;
using backend::half_t;

int main() {
	constexpr uint64_t kGiB = uint64_t( 1 ) << 30, kMiB = uint64_t( 1 ) << 20;
	constexpr uint64_t kArena = 17 * kGiB;
	if ( backend::device_free_bytes( 0 ) < kArena + kGiB ) {
		std::printf( "[skip] test_vulkan_bigbuf: %.1f GiB free, needs %.1f\n", backend::device_free_bytes( 0 ) / double( kGiB ), ( kArena + kGiB ) / double( kGiB ) );
		return 0;
	}
	pod::MemoryArena arena{};
	ENGINE_ASSERT( core::arena_create( arena, 0, kArena ), "arena" );
	std::printf( "=== tensors deep in a %.0f GiB arena (backend %s) ===\n", kArena / double( kGiB ), ENGINE_BACKEND_NAME );

	const int k = 1024, n = 2048;   // a 4 MiB weight
	const uint64_t offsets[] = { 0, 3 * kGiB + 512 * kMiB, 4 * kGiB - 2 * kMiB, 4 * kGiB + 256 * kMiB, 8 * kGiB - 2 * kMiB, 9 * kGiB + 100 * kMiB,
		12 * kGiB - 2 * kMiB, 14 * kGiB + 777 * kMiB, 16 * kGiB - 2 * kMiB, 16 * kGiB + 512 * kMiB };
	uint32_t state = 12345u;
	auto frand = [&]( float scale ) { state = state * 1664525u + 1013904223u; return ( static_cast<int>( state >> 16 ) % 2001 - 1000 ) / 1000.0f * scale; };
	int fails = 0;
	for ( uint64_t off : offsets ) {
		arena.offset_bytes = off;
		int ws[2] = { k, n };   // the dense FP16 weight's shape; its memory is n rows of k (out = x W^T)
		pod::TensorView w = core::arena_alloc_tensor( arena, pod::DType::FP16, ws, 2 );
		std::vector<float> wf( static_cast<size_t>( n ) * k );
		std::vector<half_t> wh( wf.size() );
		for ( size_t i = 0; i < wf.size(); ++i ) { wh[i] = backend::from_float_half( frand( 0.25f ) ); wf[i] = backend::to_float( wh[i] ); }
		backend::memcpy_sync( w.data, wh.data(), wh.size() * sizeof( half_t ), backend::Copy::H2D );
		const pod::LinearWeight lw = pod::LinearWeight::make_fp16( w );
		for ( int m : { 1, 40 } ) {
			int xs[2] = { m, k }, os[2] = { m, n };
			pod::TensorView x = core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
			pod::TensorView out = core::arena_alloc_tensor( arena, pod::DType::FP16, os, 2 );
			std::vector<float> xf( static_cast<size_t>( m ) * k );
			std::vector<half_t> xh( xf.size() );
			for ( size_t i = 0; i < xf.size(); ++i ) { xh[i] = backend::from_float_half( frand( 1.0f ) ); xf[i] = backend::to_float( xh[i] ); }
			backend::memcpy_sync( x.data, xh.data(), xh.size() * sizeof( half_t ), backend::Copy::H2D );
			modules::Linear::forward( x, lw, out );
			std::vector<half_t> got( static_cast<size_t>( m ) * n );
			backend::memcpy_sync( got.data(), out.data, got.size() * sizeof( half_t ), backend::Copy::D2H );
			double dot = 0, na = 0, nb = 0, maxd = 0, maxabs = 0;
			for ( int r = 0; r < m; ++r ) {
				for ( int c = 0; c < n; ++c ) {
					double ref = 0;
					for ( int j = 0; j < k; ++j ) ref += static_cast<double>( xf[static_cast<size_t>( r ) * k + j] ) * wf[static_cast<size_t>( c ) * k + j];
					const double g = backend::to_float( got[static_cast<size_t>( r ) * n + c] );
					dot += ref * g; na += ref * ref; nb += g * g;
					maxd = std::fmax( maxd, std::fabs( ref - g ) );
					maxabs = std::fmax( maxabs, std::fabs( ref ) );
				}
			}
			const double cosv = dot / ( std::sqrt( na ) * std::sqrt( nb ) + 1e-30 );
			const bool ok = std::isfinite( cosv ) && cosv >= 0.9999 && maxd <= 0.005 * maxabs + 1e-3;
			std::printf( "  weight at %6.2f GiB, %2d rows: cos %.7f max|d| %.3g (max %.3g)  %s\n", off / double( kGiB ), m, cosv, maxd, maxabs, ok ? "PASS" : "FAIL" );
			fails += ok ? 0 : 1;
		}
	}
	// the matmul at prefill row counts over the widths of a model's projections (tile edges: widths that are not multiples of a tile, a width below
	// one); every output checked at 400 sampled (row, column) pairs
	for ( int m : { 1536, 2048 } ) {
		for ( int nn : { 48, 100, 128, 640, 5120, 6144, 17408 } ) {
			arena.offset_bytes = 0;
			const int kk = 1024;
			int ws2[2] = { kk, nn }, xs2[2] = { m, kk }, os2[2] = { m, nn };
			pod::TensorView w2 = core::arena_alloc_tensor( arena, pod::DType::FP16, ws2, 2 );
			pod::TensorView x2 = core::arena_alloc_tensor( arena, pod::DType::FP16, xs2, 2 );
			pod::TensorView out2 = core::arena_alloc_tensor( arena, pod::DType::FP16, os2, 2 );
			std::vector<float> wf2( static_cast<size_t>( nn ) * kk ), xf2( static_cast<size_t>( m ) * kk );
			std::vector<half_t> wh2( wf2.size() ), xh2( xf2.size() );
			for ( size_t q = 0; q < wf2.size(); ++q ) { wh2[q] = backend::from_float_half( frand( 0.25f ) ); wf2[q] = backend::to_float( wh2[q] ); }
			for ( size_t q = 0; q < xf2.size(); ++q ) { xh2[q] = backend::from_float_half( frand( 1.0f ) ); xf2[q] = backend::to_float( xh2[q] ); }
			backend::memcpy_sync( w2.data, wh2.data(), wh2.size() * sizeof( half_t ), backend::Copy::H2D );
			backend::memcpy_sync( x2.data, xh2.data(), xh2.size() * sizeof( half_t ), backend::Copy::H2D );
			modules::Linear::forward( x2, pod::LinearWeight::make_fp16( w2 ), out2 );
			std::vector<half_t> got2( static_cast<size_t>( m ) * nn );
			backend::memcpy_sync( got2.data(), out2.data, got2.size() * sizeof( half_t ), backend::Copy::D2H );
			double maxd = 0, maxabs = 0;
			for ( int s = 0; s < 400; ++s ) {
				// sampled pairs, the first the last row and column
				state = state * 1664525u + 1013904223u;
				const int r = s == 0 ? m - 1 : static_cast<int>( ( state >> 8 ) % m );
				state = state * 1664525u + 1013904223u;
				const int c = s == 0 ? nn - 1 : static_cast<int>( ( state >> 8 ) % nn );
				double ref = 0;
				for ( int j = 0; j < kk; ++j ) ref += static_cast<double>( xf2[static_cast<size_t>( r ) * kk + j] ) * wf2[static_cast<size_t>( c ) * kk + j];
				maxd = std::fmax( maxd, std::fabs( ref - backend::to_float( got2[static_cast<size_t>( r ) * nn + c] ) ) );
				maxabs = std::fmax( maxabs, std::fabs( ref ) );
			}
			const bool ok2 = maxd <= 0.005 * maxabs + 1e-3;
			std::printf( "  %4d rows x %5d wide: max|d| %.3g (max %.3g)  %s\n", m, nn, maxd, maxabs, ok2 ? "PASS" : "FAIL" );
			fails += ok2 ? 0 : 1;
		}
	}
	// the same matmul through its split-K form (what the NVFP4 and trellis dequantize-then-matmul linears call: a long K over few output tiles is cut into FP32 slabs
	// that are summed), and with FP32 output
	for ( int nn : { 48, 100, 640, 5120 } ) {
		for ( int kk : { 2048, 5120 } ) {
			for ( int mode = 0; mode < 2; ++mode ) {   // 0: matmul_split (FP16 out), 1: the matmul with FP32 out
				const int m = 2048;
				arena.offset_bytes = 0;
				int ws2[2] = { nn, kk }, xs2[2] = { m, kk }, os2[2] = { m, nn };
				pod::TensorView w2 = core::arena_alloc_tensor( arena, pod::DType::FP16, ws2, 2 );
				pod::TensorView x2 = core::arena_alloc_tensor( arena, pod::DType::FP16, xs2, 2 );
				pod::TensorView out2 = core::arena_alloc_tensor( arena, mode ? pod::DType::FP32 : pod::DType::FP16, os2, 2 );
				std::vector<float> wf2( static_cast<size_t>( nn ) * kk ), xf2( static_cast<size_t>( m ) * kk );
				std::vector<half_t> wh2( wf2.size() ), xh2( xf2.size() );
				for ( size_t q = 0; q < wf2.size(); ++q ) { wh2[q] = backend::from_float_half( frand( 0.25f ) ); wf2[q] = backend::to_float( wh2[q] ); }
				for ( size_t q = 0; q < xf2.size(); ++q ) { xh2[q] = backend::from_float_half( frand( 1.0f ) ); xf2[q] = backend::to_float( xh2[q] ); }
				backend::memcpy_sync( w2.data, wh2.data(), wh2.size() * sizeof( half_t ), backend::Copy::H2D );
				backend::memcpy_sync( x2.data, xh2.data(), xh2.size() * sizeof( half_t ), backend::Copy::H2D );
				const bool ran = mode ? backend::impl::ggml_f16_matmul( w2.data, x2.data, out2.data, m, nn, kk, true, nullptr )
					: backend::impl::ggml_f16_matmul_split( w2.data, x2.data, out2.data, m, nn, kk, nullptr );
				std::vector<float> got2( static_cast<size_t>( m ) * nn );
				if ( mode ) backend::memcpy_sync( got2.data(), out2.data, got2.size() * sizeof( float ), backend::Copy::D2H );
				else {
					std::vector<half_t> gh( got2.size() );
					backend::memcpy_sync( gh.data(), out2.data, gh.size() * sizeof( half_t ), backend::Copy::D2H );
					for ( size_t q = 0; q < gh.size(); ++q ) got2[q] = backend::to_float( gh[q] );
				}
				double maxd = 0, maxabs = 0;
				for ( int s = 0; s < 400; ++s ) {
					state = state * 1664525u + 1013904223u;
					const int r = s == 0 ? m - 1 : static_cast<int>( ( state >> 8 ) % m );
					state = state * 1664525u + 1013904223u;
					const int c = s == 0 ? nn - 1 : static_cast<int>( ( state >> 8 ) % nn );
					double ref = 0;
					for ( int j = 0; j < kk; ++j ) ref += static_cast<double>( xf2[static_cast<size_t>( r ) * kk + j] ) * wf2[static_cast<size_t>( c ) * kk + j];
					maxd = std::fmax( maxd, std::fabs( ref - got2[static_cast<size_t>( r ) * nn + c] ) );
					maxabs = std::fmax( maxabs, std::fabs( ref ) );
				}
				const bool ok2 = ran && maxd <= 0.005 * maxabs + 1e-3;
				std::printf( "  %s %4d rows x %5d wide, K %4d: max|d| %.3g (max %.3g)  %s\n", mode ? "fp32-out  " : "split-K   ", m, nn, kk, maxd, maxabs, ok2 ? "PASS" : "FAIL" );
				fails += ok2 ? 0 : 1;
			}
		}
	}
	core::arena_destroy( arena );
	std::printf( "%s\n", fails ? "[FAIL] test_vulkan_bigbuf" : "[PASS] test_vulkan_bigbuf" );
	return fails ? 1 : 0;
}
