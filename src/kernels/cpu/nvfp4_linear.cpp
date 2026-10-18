// NVFP4 W4A16 linear on the host CPU (see nvfp4_linear.h)
#include "engine/kernels/cpu/nvfp4_linear.h"
#include "engine/kernels/cpu/cpu_util.h"
#include "engine/quant/fp4_fp8.h"
#include <cmath>
#include <cstring>
#if defined( __x86_64__ )
#include <immintrin.h>
#define CPUX_X86 1
#define CPUX_AVX2 __attribute__( ( target( "avx2,fma,f16c" ) ) )
#endif

namespace engine::kernels::cpu {

namespace {

// signed 2 x E2M1 value of a 4-bit code
constexpr int8_t kVal2[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

constexpr int kGroup = 16;
constexpr int kGroupsPerBlock = kBlock / kGroup;

// Activation layout of one 128-value block in the kernel's input (xin): a float scale, then for each of the 8 groups and each half h of its K values
// (k = 8h .. 8h + 7) eight int16 in the pair order the kernel multiplies them with the weights: the nibbles of a column's dword (j = 0 .. 7) are the
// K values 0 2 4 6 1 3 5 7 of the half, taken as the pairs (j0, j1), (j2, j3), (j4, j5), (j6, j7).
constexpr int kDataOffsetFloats = 4;                              // the int16 data start 16 bytes into the block
constexpr int kKPos[8] = { 0, 4, 1, 5, 2, 6, 3, 7 };              // K value within a half -> its slot in the 8 int16
constexpr float kQMax = 32767.0f;

// The prepacked [tile][group][lane] x 8 bytes are read in place. 8 columns of a tile are the chunks of lanes qp*4 .. +3 (32 bytes) and 16 + qp*4 .. +3
// (32 bytes); a shuffle of the two loads gives one dword per column, in the order 0 1 4 5 2 3 6 7 (the "position" p of a column)
constexpr int kScr[8] = { 0, 1, 4, 5, 2, 3, 6, 7 };              // position -> column
// After the lane sums the 8 columns sit in the order of positions 0 4 2 6 1 5 3 7, i.e. columns 0 2 4 6 1 3 5 7; kNat[c] is the lane of column c
constexpr int kNat[8] = { 0, 4, 1, 5, 2, 6, 3, 7 };

#if CPUX_X86
// MB rows. Per group: the 16 nibbles of 8 columns become int16 weights by a byte shuffle (2 x the E2M1 value), are multiplied with the int16
// activations pairwise (madd) into int32 partials, summed per column, converted and scaled once per group
template <int MB>
CPUX_AVX2 void run_mb( const uint8_t* codes, const uint8_t* scales, const float* lut, int groups, int k, const float* xin, int r0, int u0, int u1, int n, float* raw ) {
	const __m128i tab = _mm_loadu_si128( reinterpret_cast<const __m128i*>( kVal2 ) );
	const __m256i lut16 = _mm256_broadcastsi128_si256( tab ), m0f = _mm256_set1_epi8( 0x0f );
	const __m256i nat = _mm256_setr_epi32( kNat[0], kNat[1], kNat[2], kNat[3], kNat[4], kNat[5], kNat[6], kNat[7] );
	for ( int u = u0; u < u1; ++u ) {
		const uint8_t* tile = codes + static_cast<size_t>( u >> 1 ) * groups * 256;
		for ( int vec = 0; vec < 2; ++vec ) {
			const int qp = ( ( u & 1 ) * 16 + vec * 8 ) >> 3;
			__m256 acc[MB];
			for ( int m = 0; m < MB; ++m ) acc[m] = _mm256_setzero_ps();
			const uint8_t* sp = scales + static_cast<size_t>( u ) * groups * 16 + vec * 8;
			for ( int g = 0; g < groups; ++g, sp += 16 ) {
				const uint8_t* gp = tile + static_cast<size_t>( g ) * 256 + qp * 32;
				const __m256 a = _mm256_castsi256_ps( _mm256_loadu_si256( reinterpret_cast<const __m256i*>( gp ) ) );
				const __m256 b = _mm256_castsi256_ps( _mm256_loadu_si256( reinterpret_cast<const __m256i*>( gp + 128 ) ) );
				const __m256i vhs[2] = { _mm256_castps_si256( _mm256_shuffle_ps( a, b, 0x88 ) ), _mm256_castps_si256( _mm256_shuffle_ps( a, b, 0xDD ) ) };
				const int blk = g / kGroupsPerBlock, gi = g % kGroupsPerBlock;
				const int16_t* xq[MB];
				for ( int m = 0; m < MB; ++m ) xq[m] = reinterpret_cast<const int16_t*>( xin + static_cast<size_t>( r0 + m ) * k + blk * kBlock + kDataOffsetFloats ) + gi * 16;
				__m256i part[MB][4];
				for ( int m = 0; m < MB; ++m ) for ( int q = 0; q < 4; ++q ) part[m][q] = _mm256_setzero_si256();
				for ( int h = 0; h < 2; ++h ) {
					const __m256i vh = vhs[h];
					const __m256i lw = _mm256_shuffle_epi8( lut16, _mm256_and_si256( vh, m0f ) );                           // nibbles 0 2 4 6 of every dword
					const __m256i hw = _mm256_shuffle_epi8( lut16, _mm256_and_si256( _mm256_srli_epi16( vh, 4 ), m0f ) );    // nibbles 1 3 5 7
					const __m256i u0b = _mm256_unpacklo_epi8( lw, hw ), u1b = _mm256_unpackhi_epi8( lw, hw );                // (j0, j1) (j2, j3) ... pairs, 2 columns per 16 bytes
					const __m256i w[4] = { _mm256_cvtepi8_epi16( _mm256_castsi256_si128( u0b ) ), _mm256_cvtepi8_epi16( _mm256_extracti128_si256( u0b, 1 ) ),
					                       _mm256_cvtepi8_epi16( _mm256_castsi256_si128( u1b ) ), _mm256_cvtepi8_epi16( _mm256_extracti128_si256( u1b, 1 ) ) };   // positions (0 1) (4 5) (2 3) (6 7)
					for ( int m = 0; m < MB; ++m ) {
						const __m256i xv = _mm256_broadcastsi128_si256( _mm_loadu_si128( reinterpret_cast<const __m128i*>( xq[m] + 8 * h ) ) );
						for ( int q = 0; q < 4; ++q ) part[m][q] = _mm256_add_epi32( part[m][q], _mm256_madd_epi16( w[q], xv ) );
					}
				}
				const __m256 s = _mm256_i32gather_ps( lut, _mm256_cvtepu8_epi32( _mm_loadl_epi64( reinterpret_cast<const __m128i*>( sp ) ) ), 4 );
				for ( int m = 0; m < MB; ++m ) {
					// lanes: 4 partials of a column each; hadd twice -> positions 0 4 2 6 | 1 5 3 7
					const __m256i h1 = _mm256_hadd_epi32( part[m][0], part[m][1] ), h2 = _mm256_hadd_epi32( part[m][2], part[m][3] );
					const __m256 f = _mm256_cvtepi32_ps( _mm256_hadd_epi32( h1, h2 ) );
					const __m256 xs = _mm256_broadcast_ss( xin + static_cast<size_t>( r0 + m ) * k + blk * kBlock );   // the block's activation scale
					acc[m] = _mm256_fmadd_ps( f, _mm256_mul_ps( s, xs ), acc[m] );
				}
			}
			for ( int m = 0; m < MB; ++m ) _mm256_storeu_ps( raw + static_cast<size_t>( r0 + m ) * n + u * 16 + vec * 8, _mm256_permutevar8x32_ps( acc[m], nat ) );
		}
	}
}
#endif

// Portable version of the same arithmetic
void run_scalar( const uint8_t* codes, const uint8_t* scales, const float* lut, int groups, int k, const float* xin, int rows, int u0, int u1, int n, float* raw ) {
	for ( int u = u0; u < u1; ++u )
		for ( int r = 0; r < rows; ++r )
			for ( int c = 0; c < 16; ++c ) {
				const int col = u * 16 + c, t = col / 32, j = col % 32, qp = j >> 3, rr = j & 7, lane = qp * 4 + ( rr & 3 ) + ( rr >= 4 ? 16 : 0 );
				const int vec = c >> 3;
				float acc = 0;
				for ( int g = 0; g < groups; ++g ) {
					uint32_t cw[2];
					std::memcpy( cw, codes + ( static_cast<size_t>( t ) * groups + g ) * 256 + lane * 8, 8 );
					const float* blk = xin + static_cast<size_t>( r ) * k + ( g / kGroupsPerBlock ) * kBlock;
					const int16_t* xq = reinterpret_cast<const int16_t*>( blk + kDataOffsetFloats ) + ( g % kGroupsPerBlock ) * 16;
					int sum = 0;
					for ( int h = 0; h < 2; ++h )
						for ( int jj = 0; jj < 8; ++jj ) {
							const int code = ( cw[h] >> ( 4 * jj ) ) & 15;
							sum += kVal2[code] * xq[8 * h + kKPos[jj < 4 ? 2 * jj : 2 * ( jj - 4 ) + 1]];
						}
					acc += static_cast<float>( sum ) * lut[scales[static_cast<size_t>( u ) * groups * 16 + g * 16 + vec * 8 + kNat[c & 7]]] * blk[0];
				}
				raw[static_cast<size_t>( r ) * n + col] = acc;
			}
}

}   // namespace

Nvfp4Linear::Nvfp4Linear( const uint8_t* codes, const uint8_t* scales, float global_scale, int k, int n ) : k_( k ), n_( n ), codes_( codes ) {
	const int groups = k / kGroup, units = n / 16;
	scales_.assign( static_cast<size_t>( units ) * groups * 16, 0 );
	for ( int i = 0; i < 256; ++i ) lut_[i] = quant::e4m3_to_float( static_cast<unsigned>( i ) ) * global_scale * 0.5f;
	for ( int col = 0; col < n; ++col ) {
		const int t = col / 32, j = col % 32, qp = j >> 3, r = j & 7, lane = qp * 4 + ( r & 3 ) + ( r >= 4 ? 16 : 0 );
		const int u = col / 16, c = col % 16, vec = c >> 3;
		for ( int g = 0; g < groups; ++g ) scales_[static_cast<size_t>( u ) * groups * 16 + g * 16 + vec * 8 + kNat[c & 7]] = scales[( static_cast<size_t>( t ) * groups + g ) * 32 + lane];
	}
}

// 128 activations -> a scale and int16 values (15 bits of the block's largest), in the pair order of the kernel
void Nvfp4Linear::prepare_block( const float* x, int, float* xin ) const {
	float mx = 0.0f;
	for ( int i = 0; i < kBlock; ++i ) mx = std::fmax( mx, std::fabs( x[i] ) );
	const float scale = mx > 0.0f ? mx / kQMax : 1.0f, inv = 1.0f / scale;
	xin[0] = scale;
	int16_t* q = reinterpret_cast<int16_t*>( xin + kDataOffsetFloats );
	for ( int g = 0; g < kGroupsPerBlock; ++g )
		for ( int h = 0; h < 2; ++h )
			for ( int kk = 0; kk < 8; ++kk )
				q[g * 16 + h * 8 + kKPos[kk]] = static_cast<int16_t>( std::lrintf( x[g * kGroup + h * 8 + kk] * inv ) );
}

void Nvfp4Linear::finish_block( float*, int ) const {}

void Nvfp4Linear::run( int rows, const float* xin, const float*, int u0, int u1, float* raw ) const {
	const int groups = k_ / kGroup;
#if CPUX_X86
	if ( avx2_available() ) {
		int r = 0;
		for ( ; r + 2 <= rows; r += 2 ) run_mb<2>( codes_, scales_.data(), lut_, groups, k_, xin, r, u0, u1, n_, raw );
		if ( r < rows ) run_mb<1>( codes_, scales_.data(), lut_, groups, k_, xin, r, u0, u1, n_, raw );
		return;
	}
#endif
	run_scalar( codes_, scales_.data(), lut_, groups, k_, xin, rows, u0, u1, n_, raw );
}

}   // namespace engine::kernels::cpu
