// Host CPU helpers shared by the CPU kernels (see cpu_util.h)
#include "engine/kernels/cpu/cpu_util.h"
#include <cmath>
#include <cstring>
#if defined( __x86_64__ )
#include <immintrin.h>
#define CPUX_X86 1
#define CPUX_AVX2 __attribute__( ( target( "avx2,fma,f16c" ) ) )
#endif

namespace engine::kernels::cpu {

float half_bits_to_float( uint16_t h ) {
	const int e = ( h >> 10 ) & 31, m = h & 1023;
	const float f = e == 0 ? std::ldexp( static_cast<float>( m ), -24 ) : std::ldexp( 1.0f + m / 1024.0f, e - 15 );
	return ( h & 0x8000 ) ? -f : f;
}

bool avx2_available() {
#if CPUX_X86
	static const bool ok = __builtin_cpu_supports( "avx2" ) && __builtin_cpu_supports( "fma" ) && __builtin_cpu_supports( "f16c" );
	return ok;
#else
	return false;
#endif
}

void half_to_float( const uint16_t* src, float* dst, size_t n ) {
	size_t i = 0;
#if CPUX_X86
	if ( avx2_available() ) {
		struct F { CPUX_AVX2 static void run( const uint16_t* s, float* d, size_t cnt, size_t& i ) {
			for ( ; i + 8 <= cnt; i += 8 ) _mm256_storeu_ps( d + i, _mm256_cvtph_ps( _mm_loadu_si128( reinterpret_cast<const __m128i*>( s + i ) ) ) );
		} };
		F::run( src, dst, n, i );
	}
#endif
	for ( ; i < n; ++i ) dst[i] = half_bits_to_float( src[i] );
}

namespace {
uint16_t f2h_scalar( float f ) {
	uint32_t b; std::memcpy( &b, &f, 4 );
	const uint32_t sign = ( b >> 16 ) & 0x8000u;
	int e = static_cast<int>( ( b >> 23 ) & 255 ) - 127 + 15;
	uint32_t m = b & 0x7fffffu;
	if ( ( b & 0x7fffffffu ) == 0 ) return static_cast<uint16_t>( sign );
	if ( e >= 31 ) return static_cast<uint16_t>( sign | 0x7c00u );
	if ( e <= 0 ) {
		if ( e < -10 ) return static_cast<uint16_t>( sign );
		m |= 0x800000u;
		const int shift = 14 - e;
		uint32_t h = m >> shift;
		const uint32_t rem = m & ( ( 1u << shift ) - 1 ), half = 1u << ( shift - 1 );
		if ( rem > half || ( rem == half && ( h & 1 ) ) ) ++h;
		return static_cast<uint16_t>( sign | h );
	}
	uint32_t h = ( static_cast<uint32_t>( e ) << 10 ) | ( m >> 13 );
	const uint32_t rem = m & 0x1fffu;
	if ( rem > 0x1000u || ( rem == 0x1000u && ( h & 1 ) ) ) ++h;
	return static_cast<uint16_t>( sign | h );
}
}   // namespace

void float_to_half( const float* src, uint16_t* dst, size_t n ) {
	size_t i = 0;
#if CPUX_X86
	if ( avx2_available() ) {
		struct F { CPUX_AVX2 static void run( const float* s, uint16_t* d, size_t cnt, size_t& i ) {
			for ( ; i + 8 <= cnt; i += 8 ) _mm_storeu_si128( reinterpret_cast<__m128i*>( d + i ), _mm256_cvtps_ph( _mm256_loadu_ps( s + i ), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
		} };
		F::run( src, dst, n, i );
	}
#endif
	for ( ; i < n; ++i ) dst[i] = f2h_scalar( src[i] );
}

void round_to_half( float* v, int n ) {
	uint16_t h[128];
	float_to_half( v, h, static_cast<size_t>( n ) );
	half_to_float( h, v, static_cast<size_t>( n ) );
}

void had128_scalar( float* v ) {
	for ( int len = 1; len < 128; len <<= 1 )
		for ( int i = 0; i < 128; i += 2 * len )
			for ( int j = i; j < i + len; ++j ) { const float a = v[j], b = v[j + len]; v[j] = a + b; v[j + len] = a - b; }
}

#if CPUX_X86
CPUX_AVX2 void had128_avx2( float* v ) {
	__m256 r[16];
	for ( int i = 0; i < 16; ++i ) r[i] = _mm256_loadu_ps( v + i * 8 );
	const __m256 neg = _mm256_set1_ps( 0.0f );
	(void) neg;
	for ( int i = 0; i < 16; ++i ) {
		__m256 a = r[i], b = _mm256_permute2f128_ps( a, a, 0x01 );
		a = _mm256_blend_ps( _mm256_add_ps( a, b ), _mm256_sub_ps( b, a ), 0xF0 );
		b = _mm256_shuffle_ps( a, a, 0x4E );
		a = _mm256_blend_ps( _mm256_add_ps( a, b ), _mm256_sub_ps( b, a ), 0xCC );
		b = _mm256_shuffle_ps( a, a, 0xB1 );
		r[i] = _mm256_blend_ps( _mm256_add_ps( a, b ), _mm256_sub_ps( b, a ), 0xAA );
	}
	for ( int stride = 1; stride < 16; stride <<= 1 )
		for ( int i = 0; i < 16; i += 2 * stride )
			for ( int j = i; j < i + stride; ++j ) { const __m256 a = r[j], b = r[j + stride]; r[j] = _mm256_add_ps( a, b ); r[j + stride] = _mm256_sub_ps( a, b ); }
	for ( int i = 0; i < 16; ++i ) _mm256_storeu_ps( v + i * 8, r[i] );
}
#endif


void hadamard128( float* v ) {
#if CPUX_X86
	if ( avx2_available() ) { had128_avx2( v ); return; }
#endif
	had128_scalar( v );
}

}   // namespace engine::kernels::cpu
