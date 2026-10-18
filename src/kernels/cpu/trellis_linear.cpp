// EXL3 trellis (MUL1 codebook, whole bits per weight) linear on the host CPU: repack and the AVX2 tile GEMV (see trellis_linear.h)
#include "engine/kernels/cpu/trellis_linear.h"
#include "engine/kernels/cpu/cpu_util.h"
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <utility>
#if defined( __x86_64__ )
#include <immintrin.h>
#define CPUX_X86 1
#define CPUX_AVX2 __attribute__( ( target( "avx2,fma,f16c" ) ) )
#define CPUX_INLINE __attribute__( ( always_inline ) ) inline
#endif

namespace engine::kernels::cpu {

namespace {

struct Mat {
	const uint32_t* data = nullptr;    // [nt][kt][bits][8] u32
	int kt = 0, nt = 0, bits = 0;
};

using TilesFn = void ( * )( const Mat&, int, const float*, const float*, int, int, float* );

constexpr uint32_t kMul1 = 0x83DCD12Du;

// fp16 bit patterns of the codebook constants: value = ( 1024 + byte_sum ) * c1 + c2
const float kC1 = half_bits_to_float( 0x1eee );
const float kC2 = half_bits_to_float( 0xc931 );
const float kBias = static_cast<float>( 1024.0 * static_cast<double>( kC1 ) + static_cast<double>( kC2 ) );   // value = c1 * byte_sum + kBias

// ---- tile format (the tensor-core order the EXL3 tiles are stored in) ----
constexpr std::array<uint16_t, 256> tc_perm() {
	std::array<uint16_t, 256> p{};
	for ( int t = 0; t < 32; ++t ) {
		const int r0 = ( t % 4 ) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9, c0 = t / 4, c1 = c0 + 8;
		p[t * 8 + 0] = static_cast<uint16_t>( r0 * 16 + c0 ); p[t * 8 + 1] = static_cast<uint16_t>( r1 * 16 + c0 );
		p[t * 8 + 2] = static_cast<uint16_t>( r2 * 16 + c0 ); p[t * 8 + 3] = static_cast<uint16_t>( r3 * 16 + c0 );
		p[t * 8 + 4] = static_cast<uint16_t>( r0 * 16 + c1 ); p[t * 8 + 5] = static_cast<uint16_t>( r1 * 16 + c1 );
		p[t * 8 + 6] = static_cast<uint16_t>( r2 * 16 + c1 ); p[t * 8 + 7] = static_cast<uint16_t>( r3 * 16 + c1 );
	}
	return p;
}
// end of state t's 16-bit window in the tile's bit ring (biased by whole rings so it stays positive)
constexpr int state_end( int b, int t ) { return ( ( t + 1 ) * b ) + 256 * b; }

// State i = l * 32 + v sits in lane l of vector v: all eight lanes of a vector then share the window's word offset relative to the lane's own
// b words, the shift, the input row and which half (columns 0-7 or 8-15) of the tile's 16 columns the vector accumulates into; lane l is column l (+8)
template <int B> struct Plan {
	int c0[32], c1[32], sh[32], row[32], hi[32];
	bool ok;
	constexpr Plan() : c0(), c1(), sh(), row(), hi(), ok( true ) {
		const auto perm = tc_perm();
		for ( int v = 0; v < 32; ++v ) {
			int c0v = 0, c1v = 0, shv = 0, rowv = 0, hiv = 0;
			for ( int l = 0; l < 8; ++l ) {
				const int i = l * 32 + v, b1 = state_end( B, i ), b0 = b1 - 16;
				const int w0 = b0 / 32 - l * B - 8 * B, w1 = ( b1 - 1 ) / 32 - l * B - 8 * B, s = ( ( b1 - 1 ) / 32 + 1 ) * 32 - b1;
				const int p = perm[i], r = p / 16, c = p % 16;
				if ( l == 0 ) { c0v = w0; c1v = w1; shv = s; rowv = r; hiv = c >= 8; }
				else if ( c0v != w0 || c1v != w1 || shv != s || rowv != r || ( c >= 8 ) != ( hiv != 0 ) || ( c % 8 ) != l ) ok = false;
			}
			c0[v] = c0v; c1[v] = c1v; sh[v] = shv; row[v] = rowv; hi[v] = hiv;
		}
	}
};

inline uint32_t ld32( const uint16_t* p, int i ) { uint32_t v; std::memcpy( &v, p + i * 2, 4 ); return v; }

// ---- scalar model of one tile's 256 weights: sum[i] for state i, in the tile's position order (the repack's source of truth) ----
uint32_t byte_sum( uint32_t window ) {
	const uint32_t x = window * kMul1;
	return ( x & 255u ) + ( ( x >> 8 ) & 255u ) + ( ( x >> 16 ) & 255u ) + ( x >> 24 );
}

// Scalar tiles [nt0, nt1) over a repacked matrix: the vector kernel's arithmetic, lane by lane (the path without AVX2)
template <int B>
void gemv_tiles_scalar_b( const Mat& m, int rows, const float* xin, const float*, int nt0, int nt1, float* out ) {
	static constexpr Plan<B> P{};
	static_assert( Plan<B>().ok, "lane layout" );
	const int k = m.kt * 16, n = m.nt * 16;
	for ( int r = 0; r < rows; ++r ) {
		const float* x = xin + static_cast<size_t>( r ) * k;
		double xsum = 0;
		for ( int i = 0; i < k; ++i ) xsum += x[i];
		for ( int nt = nt0; nt < nt1; ++nt ) {
			double acc[16] = {};
			for ( int kt = 0; kt < m.kt; ++kt ) {
				const uint32_t* tp = m.data + ( static_cast<size_t>( nt ) * m.kt + kt ) * B * 8;
				auto word = [&]( int l, int c ) { const int j = ( ( l * B + c ) % ( 8 * B ) + 8 * B ) % ( 8 * B ); return tp[( j % B ) * 8 + j / B]; };
				for ( int v = 0; v < 32; ++v )
					for ( int l = 0; l < 8; ++l ) {
						const uint64_t merged = ( static_cast<uint64_t>( word( l, P.c0[v] ) ) << 32 ) | word( l, P.c1[v] );
						const uint32_t w = static_cast<uint32_t>( merged >> P.sh[v] ) & 0xffffu;
						acc[l + 8 * P.hi[v]] += static_cast<double>( byte_sum( w ) ) * x[kt * 16 + P.row[v]];
					}
			}
			for ( int c = 0; c < 16; ++c ) out[static_cast<size_t>( r ) * n + nt * 16 + c] = static_cast<float>( kC1 * acc[c] + kBias * xsum );
		}
	}
}

#if CPUX_X86
#define CPUX_AVX2 __attribute__( ( target( "avx2,fma,f16c" ) ) )
#define CPUX_INLINE __attribute__( ( always_inline ) ) inline

// One vector (v) of a tile: eight windows, byte sums, multiply-accumulate into the MB rows' accumulators
template <int B, int V, int MB, int NA>
CPUX_AVX2 CPUX_INLINE void step( const __m256i* R, const float* const* xk, __m256 ( &acc )[MB][NA] ) {
	constexpr Plan<B> P{};
	constexpr int ci0 = P.c0[V] + 1, ci1 = P.c1[V] + 1, sh = P.sh[V], row = P.row[V], hi = P.hi[V];
	const __m256i mulk = _mm256_set1_epi32( static_cast<int>( kMul1 ) ), m16 = _mm256_set1_epi32( 0xffff );
	const __m256i ones8 = _mm256_set1_epi8( 1 ), ones16 = _mm256_set1_epi16( 1 );
	__m256i w;
	if constexpr ( ci0 == ci1 ) w = _mm256_srli_epi32( R[ci0], sh );
	else w = _mm256_or_si256( _mm256_srli_epi32( R[ci1], sh ), _mm256_slli_epi32( R[ci0], 32 - sh ) );
	w = _mm256_and_si256( w, m16 );
	const __m256i xm = _mm256_mullo_epi32( w, mulk );
	const __m256 f = _mm256_cvtepi32_ps( _mm256_madd_epi16( _mm256_maddubs_epi16( xm, ones8 ), ones16 ) );
	constexpr int a = NA == 4 ? hi * 2 + ( V & 1 ) : hi;
	for ( int m = 0; m < MB; ++m ) acc[m][a] = _mm256_fmadd_ps( f, _mm256_broadcast_ss( xk[m] + row ), acc[m][a] );
}
template <int B, int MB, int NA, int... Vs>
CPUX_AVX2 CPUX_INLINE void steps( const __m256i* R, const float* const* xk, __m256 ( &acc )[MB][NA], std::integer_sequence<int, Vs...> ) {
	( step<B, Vs, MB, NA>( R, xk, acc ), ... );
}

// n-tiles [nt0, nt1) of the matrix for MB rows starting at row r0
template <int B, int MB>
CPUX_AVX2 void gemv_tiles_mb( const Mat& mat, const float* xin, const float* xsum, int r0, int nt0, int nt1, float* out ) {
	static_assert( Plan<B>().ok, "lane layout" );
	constexpr int NA = MB == 1 ? 4 : 2;
	const int k = mat.kt * 16, n = mat.nt * 16;
	const __m256i rot = _mm256_setr_epi32( 1, 2, 3, 4, 5, 6, 7, 0 ), rotb = _mm256_setr_epi32( 7, 0, 1, 2, 3, 4, 5, 6 );
	for ( int nt = nt0; nt < nt1; ++nt ) {
		__m256 acc[MB][NA];
		for ( int m = 0; m < MB; ++m ) for ( int a = 0; a < NA; ++a ) acc[m][a] = _mm256_setzero_ps();
		const uint32_t* tp = mat.data + static_cast<size_t>( nt ) * mat.kt * B * 8;
		const float* xr[MB];
		for ( int m = 0; m < MB; ++m ) xr[m] = xin + static_cast<size_t>( r0 + m ) * k;
		for ( int kt = 0; kt < mat.kt; ++kt, tp += B * 8 ) {
			__m256i R[B + 2];
			for ( int c = 0; c < B; ++c ) R[c + 1] = _mm256_loadu_si256( reinterpret_cast<const __m256i*>( tp + c * 8 ) );
			R[0] = _mm256_permutevar8x32_epi32( R[B], rotb );
			R[B + 1] = _mm256_permutevar8x32_epi32( R[1], rot );
			const float* xk[MB];
			for ( int m = 0; m < MB; ++m ) xk[m] = xr[m] + kt * 16;
			steps<B, MB, NA>( R, xk, acc, std::make_integer_sequence<int, 32>{} );
		}
		const __m256 c1 = _mm256_set1_ps( kC1 );
		for ( int m = 0; m < MB; ++m ) {
			__m256 lo, hi;
			if constexpr ( NA == 4 ) { lo = _mm256_add_ps( acc[m][0], acc[m][1] ); hi = _mm256_add_ps( acc[m][2], acc[m][3] ); }
			else { lo = acc[m][0]; hi = acc[m][1]; }
			const __m256 bias = _mm256_set1_ps( kBias * xsum[r0 + m] );
			float* o = out + static_cast<size_t>( r0 + m ) * n + nt * 16;
			_mm256_storeu_ps( o, _mm256_fmadd_ps( lo, c1, bias ) );
			_mm256_storeu_ps( o + 8, _mm256_fmadd_ps( hi, c1, bias ) );
		}
	}
}

template <int B>
void gemv_tiles_b( const Mat& mat, int rows, const float* xin, const float* xsum, int nt0, int nt1, float* out ) {
	int r = 0;
	for ( ; r + 4 <= rows; r += 4 ) gemv_tiles_mb<B, 4>( mat, xin, xsum, r, nt0, nt1, out );
	switch ( rows - r ) {
		case 3: gemv_tiles_mb<B, 3>( mat, xin, xsum, r, nt0, nt1, out ); break;
		case 2: gemv_tiles_mb<B, 2>( mat, xin, xsum, r, nt0, nt1, out ); break;
		case 1: gemv_tiles_mb<B, 1>( mat, xin, xsum, r, nt0, nt1, out ); break;
		default: break;
	}
}
#endif   // CPUX_X86


TilesFn tiles_fn( int bits, bool avx2 ) {
#if CPUX_X86
	if ( avx2 ) {
		switch ( bits ) {
			case 1: return gemv_tiles_b<1>; case 2: return gemv_tiles_b<2>; case 3: return gemv_tiles_b<3>; case 4: return gemv_tiles_b<4>;
			case 5: return gemv_tiles_b<5>; case 6: return gemv_tiles_b<6>; case 7: return gemv_tiles_b<7>; case 8: return gemv_tiles_b<8>;
			default: return nullptr;
		}
	}
#else
	(void) avx2;
#endif
	switch ( bits ) {
		case 1: return gemv_tiles_scalar_b<1>; case 2: return gemv_tiles_scalar_b<2>; case 3: return gemv_tiles_scalar_b<3>; case 4: return gemv_tiles_scalar_b<4>;
		case 5: return gemv_tiles_scalar_b<5>; case 6: return gemv_tiles_scalar_b<6>; case 7: return gemv_tiles_scalar_b<7>; case 8: return gemv_tiles_scalar_b<8>;
		default: return nullptr;
	}
}

}   // namespace
bool TrellisLinear::supported( int bits ) { return bits >= 1 && bits <= 8; }

size_t TrellisLinear::repacked_bytes( int kt, int nt, int bits ) { return static_cast<size_t>( kt ) * nt * bits * 32; }

TrellisLinear::TrellisLinear( const uint16_t* trellis, int kt, int nt, int bits, const uint16_t* suh, const uint16_t* svh, void* storage )
	: kt_( kt ), nt_( nt ), bits_( bits ) {
	if ( storage ) data_ = static_cast<uint32_t*>( storage );
	else { owned_.resize( repacked_bytes( kt, nt, bits ) / 4 ); data_ = owned_.data(); }
	uint32_t* out = const_cast<uint32_t*>( data_ );
	const int words = 8 * bits;
	for ( int n = 0; n < nt; ++n )
		for ( int k = 0; k < kt; ++k ) {
			const uint16_t* tile = trellis + ( static_cast<size_t>( k ) * nt + n ) * 16 * bits;
			uint32_t* d = out + ( static_cast<size_t>( n ) * kt + k ) * bits * 8;
			for ( int c = 0; c < bits; ++c )
				for ( int l = 0; l < 8; ++l ) d[c * 8 + l] = ld32( tile, ( l * bits + c ) % words );
		}
	suh_.resize( static_cast<size_t>( kt ) * 16 );
	svh_.resize( static_cast<size_t>( nt ) * 16 );
	half_to_float( suh, suh_.data(), suh_.size() );
	half_to_float( svh, svh_.data(), svh_.size() );
}

void TrellisLinear::prepare_block( const float* x, int b, float* xin ) const {
	for ( int i = 0; i < kBlock; ++i ) xin[i] = x[i] * suh_[static_cast<size_t>( b ) * kBlock + i];
	hadamard128( xin );
	for ( int i = 0; i < kBlock; ++i ) xin[i] *= kRScale;
}

void TrellisLinear::finish_block( float* v, int b ) const {
	hadamard128( v );
	for ( int i = 0; i < kBlock; ++i ) v[i] *= kRScale * svh_[static_cast<size_t>( b ) * kBlock + i];
}

void TrellisLinear::run( int rows, const float* xin, const float* row_sum, int u0, int u1, float* raw ) const {
	Mat m; m.data = data_; m.kt = kt_; m.nt = nt_; m.bits = bits_;
	tiles_fn( bits_, avx2_available() )( m, rows, xin, row_sum, u0, u1, raw );
}

}   // namespace engine::kernels::cpu
