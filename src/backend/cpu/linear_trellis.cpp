// CPU backend: EXL3 trellis linear (rotate, dequantize, GEMM, rotate) on the E1 reference Slang host kernels
// (src/kernels/slang/ref_gemm.slang, ref_trellis.slang). Bit-exact dequantization vs the engine's reconstruct; the fused CUDA
// kernels and the large-M recon path differ from this only by FP16/FP32 rounding order.
// cpu_kernels.h first: the Slang prelude defines int4, which the backend shim only declares when the prelude is absent
#include "cpu_kernels.h"
#include "engine/modules/trellis.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif
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

#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
// The 256 codebook values of a tile, eight at a time: the 32-bit windows are gathered from the byte stream (big-endian), shifted to the 16-bit window and
// decoded with the integer hash and the fp16 arithmetic of decode_window (F16C converts exactly, the fused multiply-add and the round to nearest even are the
// scalar ones), so the result is bit-identical to the scalar path
void decode_tile_avx2( const uint8_t* stream, int total, int bits, int cb, uint16_t* vals ) {
	const __m256i bswap = _mm256_setr_epi8( 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12 );
	const __m256i m16 = _mm256_set1_epi32( 0xffff );
	const __m256i mul1_k = _mm256_set1_epi32( static_cast<int>( 0x83DCD12Du ) );
	const __m256i k_mul = _mm256_set1_epi32( cb == 0 ? 89226354 : static_cast<int>( 0xCBAC1FEDu ) );
	const __m256i k_add = _mm256_set1_epi32( cb == 0 ? 64248484 : 0 );
	const __m256i y_and = _mm256_set1_epi32( static_cast<int>( 0x8fff8fffu ) );
	const __m256i y_xor = _mm256_set1_epi32( 0x3b603b60 );
	const __m256 c1 = _mm256_set1_ps( hbits_to_float( 0x1eeeu ) ), c2 = _mm256_set1_ps( hbits_to_float( 0xc931u ) );
	alignas( 32 ) int q0s[8];
	for ( int p0 = 0; p0 < 256; p0 += 8 ) {
		for ( int l = 0; l < 8; ++l ) q0s[l] = ( ( p0 + l ) * bits + bits - 16 + total ) % total;
		const __m256i q0 = _mm256_load_si256( reinterpret_cast<const __m256i*>( q0s ) );
		const __m256i idx = _mm256_srli_epi32( q0, 3 );
		const __m256i shift = _mm256_sub_epi32( _mm256_set1_epi32( 16 ), _mm256_and_si256( q0, _mm256_set1_epi32( 7 ) ) );
		__m256i v = _mm256_i32gather_epi32( reinterpret_cast<const int*>( stream ), idx, 1 );
		v = _mm256_shuffle_epi8( v, bswap );
		const __m256i w = _mm256_and_si256( _mm256_srlv_epi32( v, shift ), m16 );
		__m128i out;
		if ( cb == 2 ) {
			const __m256i x = _mm256_mullo_epi32( w, mul1_k );
			const __m256i b0 = _mm256_and_si256( x, _mm256_set1_epi32( 255 ) ), b1 = _mm256_and_si256( _mm256_srli_epi32( x, 8 ), _mm256_set1_epi32( 255 ) );
			const __m256i b2 = _mm256_and_si256( _mm256_srli_epi32( x, 16 ), _mm256_set1_epi32( 255 ) ), b3 = _mm256_srli_epi32( x, 24 );
			const __m256i sum = _mm256_add_epi32( _mm256_add_epi32( b0, b1 ), _mm256_add_epi32( b2, b3 ) );
			const __m256 f = _mm256_cvtepi32_ps( _mm256_add_epi32( sum, _mm256_set1_epi32( 1024 ) ) );
			out = _mm256_cvtps_ph( _mm256_fmadd_ps( f, c1, c2 ), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC );
		}
		else {
			const __m256i x = _mm256_add_epi32( _mm256_mullo_epi32( w, k_mul ), k_add );
			const __m256i y = _mm256_xor_si256( _mm256_and_si256( x, y_and ), y_xor );
			const __m256i lo = _mm256_and_si256( y, m16 ), hi = _mm256_srli_epi32( y, 16 );
			// Eight 32-bit lanes holding 16-bit values -> eight packed halves
			const __m128i lo16 = _mm256_castsi256_si128( _mm256_permute4x64_epi64( _mm256_packus_epi32( lo, lo ), 0xD8 ) );
			const __m128i hi16 = _mm256_castsi256_si128( _mm256_permute4x64_epi64( _mm256_packus_epi32( hi, hi ), 0xD8 ) );
			out = _mm256_cvtps_ph( _mm256_add_ps( _mm256_cvtph_ps( lo16 ), _mm256_cvtph_ps( hi16 ) ), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC );
		}
		_mm_storeu_si128( reinterpret_cast<__m128i*>( vals + p0 ), out );
	}
}
#endif

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
#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
				decode_tile_avx2( stream.data(), total, bits, cb, vals );
#else
				for ( int p = 0; p < 256; ++p ) {
					const int q0 = ( p * bits + bits - 16 + total ) % total;
					const uint8_t* b = stream.data() + ( q0 >> 3 );
					const uint32_t v = ( static_cast<uint32_t>( b[0] ) << 24 ) | ( static_cast<uint32_t>( b[1] ) << 16 ) | ( static_cast<uint32_t>( b[2] ) << 8 ) | b[3];
					vals[p] = decode_window( ( v >> ( 16 - ( q0 & 7 ) ) ) & 0xffffu, cb );
				}
#endif
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
void gemm_f32( const std::vector<half>& xh, const half* w, std::vector<float>& acc, int m, int k, int n ) {
	acc.assign( static_cast<size_t>( m ) * n, 0.0f );
	impl::cpu_parallel_for( n, 256, [&]( int64_t c0, int64_t c1 ) {
		for ( int i = 0; i < k; ++i ) {
			const half* wr = w + static_cast<size_t>( i ) * n;
			for ( int r = 0; r < m; ++r ) {
				const float xv = half_to_float( xh[static_cast<size_t>( r ) * k + i] );
				float* a = acc.data() + static_cast<size_t>( r ) * n;
				for ( int64_t c = c0; c < c1; ++c ) a[c] += xv * half_to_float( wr[c] );
			}
		}
	} );
}

// FP32 output (out.dtype == FP32): FP32 GEMM over the dequantized weight and an FP32 Hadamard, no FP16 rounding of the result
void linear_fp32_out( const std::vector<half>& xh, const half* w, const half* svh, float* out, int m, int k, int n ) {
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
	// The dequantized weight is the largest buffer of the call and every call needs one: keep it per calling thread instead of faulting in fresh pages
	thread_local std::vector<half> wbuf;
	if ( wbuf.size() < static_cast<size_t>( k ) * n ) wbuf.resize( static_cast<size_t>( k ) * n );
	const half* w = wbuf.data();
	had_rows( x.ptr<const half>(), xh.data(), suh.ptr<const half>(), nullptr, m, k );
	reconstruct_native( trellis.ptr<const uint16_t>(), wbuf.data(), k, n, bits, cb );
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
