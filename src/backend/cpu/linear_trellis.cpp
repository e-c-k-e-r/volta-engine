// CPU backend: EXL3 trellis linear (rotate, dequantize, GEMM, rotate) on the E1 reference Slang host kernels
// (src/kernels/slang/ref_gemm.slang, ref_trellis.slang). Bit-exact dequantization vs the engine's reconstruct; the fused CUDA
// kernels and the large-M recon path differ from this only by FP16/FP32 rounding order.
// cpu_kernels.h first: the Slang prelude defines int4, which the backend shim only declares when the prelude is absent
#include "cpu_kernels.h"
#include "engine/modules/trellis.h"
#include "engine/backend/numeric.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <cstdlib>

extern "C" {
void ref_gemm_f16( ComputeVaryingInput*, void*, void* );
void ref_trellis_reconstruct( ComputeVaryingInput*, void*, void* );
void ref_had128( ComputeVaryingInput*, void*, void* );
}

struct RefGemmParams { half* x; half* w; half* out_; int m; int k; int n; };
struct RefReconParams { uint16_t* codes; half* out_; int k_dim; int n_dim; int bits; int cb; };
struct RefHadParams { half* src; half* dst; half* pre; half* post; int cols; float r_scale; };

namespace engine::modules {

namespace {

constexpr float kRScale = 0.08838834764831845f; // 1 / sqrt( 128 )

// Blockwise 128-point Hadamard over the last dim of [rows, cols]: dst = post * ( r_scale * H( pre * src ) ); dst != src
void had_rows( const half* src, half* dst, const half* pre, const half* post, int rows, int cols ) {
	RefHadParams p{ const_cast<half*>( src ), dst, const_cast<half*>( pre ), const_cast<half*>( post ), cols, kRScale };
	impl::cpu_dispatch_mt( ref_had128, &p, static_cast<uint32_t>( cols / 128 ), static_cast<uint32_t>( rows ) );
}

// Codebook index of the reference decoder: 0 = 3INST, 1 = MCG, 2 = MUL1
int codebook( bool mcg, bool mul1 ) {
	return mcg ? 1 : ( mul1 ? 2 : 0 );
}

float half_to_float( half h ) {
	return static_cast<float>( h );
}

// IEEE binary16 bit pattern to float; the codebooks only produce finite values
inline float hbits_to_float( uint32_t h ) {
	const float sign = ( h & 0x8000u ) ? -1.0f : 1.0f;
	const int e = static_cast<int>( ( h >> 10 ) & 31u );
	const int m = static_cast<int>( h & 1023u );
	if ( e == 0 ) return sign * static_cast<float>( m ) * ( 1.0f / 16777216.0f );
	const uint32_t bits = ( static_cast<uint32_t>( e + 112 ) << 23 ) | ( static_cast<uint32_t>( m ) << 13 );
	float f;
	std::memcpy( &f, &bits, 4 );
	return sign * f;
}

inline uint16_t float_to_hbits( float f ) {
	const half h = static_cast<half>( f );
	uint16_t b;
	std::memcpy( &b, &h, 2 );
	return b;
}

// One 16-bit codebook window -> FP16 bit pattern (same math as decode_3inst_ref in ref_trellis.slang)
inline uint16_t decode_window( uint32_t w, int cb ) {
	if ( cb == 2 ) {
		const uint32_t x = w * 0x83DCD12Du;
		const uint32_t s = ( x & 255u ) + ( ( x >> 8 ) & 255u ) + ( ( x >> 16 ) & 255u ) + ( x >> 24 );
		static const float c1 = hbits_to_float( 0x1eeeu ), c2 = hbits_to_float( 0xc931u );
		return float_to_hbits( std::fmaf( static_cast<float>( 1024u + s ), c1, c2 ) );
	}
	const uint32_t x = ( cb == 0 ) ? w * 89226354u + 64248484u : w * 0xCBAC1FEDu;
	const uint32_t y = ( x & 0x8fff8fffu ) ^ 0x3b603b60u;
	return float_to_hbits( hbits_to_float( y & 0xffffu ) + hbits_to_float( y >> 16 ) );
}

// Native reconstruction of the dequantized [k, n] weight (same result as the ref_trellis_reconstruct kernel, which costs ~100 ns per
// element on the host): each 16x16 tile's cyclic MSB-first bit stream is laid out as bytes once, then a window is one unaligned read
void reconstruct_native( const uint16_t* codes, half* out, int k, int n, int bits, int cb ) {
	static_assert( sizeof( half ) == 2, "reconstruct_native: half must be 16 bits" );
	const int tiles_n = n / 16;
	const int total = 256 * bits;
	impl::cpu_parallel_for( k / 16, 1, [&]( int64_t kt0, int64_t kt1 ) {
		std::vector<uint8_t> stream( static_cast<size_t>( total / 8 ) + 4 );
		uint16_t vals[256];
		for ( int64_t kt = kt0; kt < kt1; ++kt ) {
			for ( int nt = 0; nt < tiles_n; ++nt ) {
				const uint16_t* tile = codes + ( kt * tiles_n + nt ) * 16 * bits;
				for ( int wd = 0; wd < total / 32; ++wd ) {
					const uint32_t word = static_cast<uint32_t>( tile[wd * 2] ) | ( static_cast<uint32_t>( tile[wd * 2 + 1] ) << 16 );
					stream[wd * 4 + 0] = static_cast<uint8_t>( word >> 24 );
					stream[wd * 4 + 1] = static_cast<uint8_t>( word >> 16 );
					stream[wd * 4 + 2] = static_cast<uint8_t>( word >> 8 );
					stream[wd * 4 + 3] = static_cast<uint8_t>( word );
				}
				for ( int e = 0; e < 4; ++e ) stream[total / 8 + e] = stream[e];   // cyclic extension
				for ( int p = 0; p < 256; ++p ) {
					const int q0 = ( p * bits + bits - 16 + total ) % total;
					const uint8_t* b = stream.data() + ( q0 >> 3 );
					const uint32_t v = ( static_cast<uint32_t>( b[0] ) << 24 ) | ( static_cast<uint32_t>( b[1] ) << 16 ) | ( static_cast<uint32_t>( b[2] ) << 8 ) | b[3];
					vals[p] = decode_window( ( v >> ( 16 - ( q0 & 7 ) ) ) & 0xffffu, cb );
				}
				for ( int p = 0; p < 256; ++p ) {
					const int L = p >> 3, i = p & 7;
					const int row = ( L & 3 ) * 2 + ( i & 1 ) + 8 * ( ( i >> 1 ) & 1 );
					const int col = 2 * ( ( L >> 3 ) + 4 * ( i >> 2 ) ) + ( ( L >> 2 ) & 1 );
					std::memcpy( &out[( kt * 16 + row ) * n + nt * 16 + col], &vals[p], 2 );
				}
			}
		}
	} );
}

// acc[m, n] = xh[m, k] * w[k, n] in FP32, split over column ranges so each thread streams contiguous slices of every weight row
void gemm_f32( const std::vector<half>& xh, const std::vector<half>& w, std::vector<float>& acc, int m, int k, int n ) {
	acc.assign( static_cast<size_t>( m ) * n, 0.0f );
	impl::cpu_parallel_for( n, 256, [&]( int64_t c0, int64_t c1 ) {
		for ( int i = 0; i < k; ++i ) {
			const half* wr = w.data() + static_cast<size_t>( i ) * n;
			for ( int r = 0; r < m; ++r ) {
				const float xv = half_to_float( xh[static_cast<size_t>( r ) * k + i] );
				float* a = acc.data() + static_cast<size_t>( r ) * n;
				for ( int64_t c = c0; c < c1; ++c ) a[c] += xv * half_to_float( wr[c] );
			}
		}
	} );
}

// FP32 output (out.dtype == FP32): FP32 GEMM over the dequantized weight and an FP32 Hadamard, no FP16 rounding of the result
void linear_fp32_out( const std::vector<half>& xh, const std::vector<half>& w, const half* svh, float* out, int m, int k, int n ) {
	std::vector<float> acc;
	gemm_f32( xh, w, acc, m, k, n );
	for ( int r = 0; r < m; ++r ) {
		float* row = acc.data() + static_cast<size_t>( r ) * n;
		for ( int b = 0; b < n / 128; ++b ) {
			float* v = row + b * 128;
			for ( int len = 1; len < 128; len <<= 1 ) {
				for ( int i = 0; i < 128; i += 2 * len ) {
					for ( int j = i; j < i + len; ++j ) {
						const float a = v[j], c = v[j + len];
						v[j] = a + c;
						v[j + len] = a - c;
					}
				}
			}
		}
		for ( int c = 0; c < n; ++c ) out[static_cast<size_t>( r ) * n + c] = row[c] * kRScale * half_to_float( svh[c] );
	}
}

void trellis_linear_impl( const pod::TensorView& x, const pod::TensorView& trellis, const pod::TensorView& suh, const pod::TensorView& svh, pod::TensorView& out, int bits, bool mcg, bool mul1 ) {
	const int m = x.shape[0];
	const int k = trellis.shape[0] * 16;
	const int n = trellis.shape[1] * 16;
	ENGINE_ASSERT( k % 128 == 0 && n % 128 == 0, "LinearTrellis (CPU): dims must be multiples of 128" );
	ENGINE_ASSERT( x.dtype == pod::DType::FP16 && x.shape[1] == k, "LinearTrellis (CPU): x must be FP16 [m, k]" );
	ENGINE_ASSERT( out.dtype == pod::DType::FP16 || out.dtype == pod::DType::FP32, "LinearTrellis (CPU): FP16 or FP32 output only" );
	const int cb = codebook( mcg, mul1 );
	std::vector<half> xh( static_cast<size_t>( m ) * k );
	std::vector<half> w( static_cast<size_t>( k ) * n );
	had_rows( x.ptr<const half>(), xh.data(), suh.ptr<const half>(), nullptr, m, k );
	reconstruct_native( trellis.ptr<const uint16_t>(), w.data(), k, n, bits, cb );
	if ( out.dtype == pod::DType::FP32 ) {
		linear_fp32_out( xh, w, svh.ptr<const half>(), out.ptr<float>(), m, k, n );
		return;
	}
	std::vector<float> acc;
	gemm_f32( xh, w, acc, m, k, n );
	std::vector<half> y( static_cast<size_t>( m ) * n );
	for ( size_t i = 0; i < y.size(); ++i ) y[i] = static_cast<half>( acc[i] );
	had_rows( y.data(), out.ptr<half>(), nullptr, svh.ptr<const half>(), m, n );
}

// ENGINE_DEBUG_CPU_TRACE=1: one line per call with the elapsed time and the weight elements dequantized so far
void trellis_linear( const pod::TensorView& x, const pod::TensorView& trellis, const pod::TensorView& suh, const pod::TensorView& svh, pod::TensorView& out, int bits, bool mcg, bool mul1 ) {
	static const bool trace = std::getenv( "ENGINE_DEBUG_CPU_TRACE" ) != nullptr;
	if ( !trace ) {
		trellis_linear_impl( x, trellis, suh, svh, out, bits, mcg, mul1 );
		return;
	}
	static double total_elems = 0.0;
	const auto t0 = std::chrono::steady_clock::now();
	trellis_linear_impl( x, trellis, suh, svh, out, bits, mcg, mul1 );
	total_elems += static_cast<double>( trellis.shape[0] ) * 16 * trellis.shape[1] * 16;
	std::fprintf( stderr, "[cpu-trace] linear %dx%d m=%d %.0f ms, %.2f G elems so far\n", trellis.shape[0] * 16, trellis.shape[1] * 16, x.shape[0],
		std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count(), total_elems * 1e-9 );
}

} // namespace

void LinearTrellis::forward( const pod::TensorView& x, const pod::TensorView& trellis, const pod::TensorView& suh, const pod::TensorView& svh, pod::TensorView& out, int bits, bool mcg, bool mul1, backend::stream_t ) {
	trellis_linear( x, trellis, suh, svh, out, bits, mcg, mul1 );
}

void LinearTrellis::forward_recon( const pod::TensorView& x, const pod::TensorView& trellis, const pod::TensorView& suh, const pod::TensorView& svh, pod::TensorView& out, int bits, bool mcg, bool mul1, backend::stream_t ) {
	trellis_linear( x, trellis, suh, svh, out, bits, mcg, mul1 );
}

} // namespace engine::modules
