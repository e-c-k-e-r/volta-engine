// 8-bit block-wise Adam moments: blocks of kBlock values share an FP32 absmax; a moment is stored as a small float of its value over the
// block's absmax (3 mantissa bits; m: a sign and 4 exponent bits, codes 0..255 with bit 7 the sign; the second moment as its square root,
// 5 exponent bits, codes 0..255), so a block spans 2^15 (m) or 2^31 (sqrt v) below its largest value at a relative step of 1/8. The root
// keeps v's range above m's (v itself would need twice the exponent range: a v rounded to 0 under a nonzero m makes the step m / eps), and
// the 8-bit step's ratio m / sqrt v is held to kRatio (Adam's own bound is about 7.3 at beta1 0.9, beta2 0.999), so a rounding cannot make
// a step larger than Adam does. Codes round stochastically (a hash of the index and the step: unbiased, the same draws on the device and
// the host). The same steps run on the device (grad.cu) and on the host (the optimizer state in host memory).
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined( __CUDACC__ )
#define ADAM8_HD __host__ __device__ __forceinline__
#else
#define ADAM8_HD inline
#endif

namespace engine::train::adam8 {

constexpr int kBlock = 256;
constexpr uint32_t kSaltM = 0x6d6d6d6du, kSaltV = 0x76767676u;   // the rounding draws of m and of v

// The magnitude of a code (exponent bits e, 3 mantissa bits; e = 0 subnormal), over the largest code's: 1 at the top
ADAM8_HD float mag( int c, int ebits ) {
	const int emax = ( 1 << ebits ) - 1, E = c >> 3, M = c & 7;
	const float y = E == 0 ? ldexpf( M * 0.125f, 1 - emax ) : ldexpf( 1.0f + M * 0.125f, E - emax );
	return y * ( 1.0f / 1.875f );
}

// A uniform draw in [0, 1) from a parameter's index and the step (the same on the device and the host)
ADAM8_HD float draw( size_t i, int step, uint32_t salt ) {
	uint32_t h = static_cast<uint32_t>( i ) * 0x9E3779B1u ^ static_cast<uint32_t>( i >> 32 ) * 0x85EBCA77u ^ static_cast<uint32_t>( step ) * 0xC2B2AE3Du ^ salt;
	h ^= h >> 16;
	h *= 0x7FEB352Du;
	h ^= h >> 15;
	h *= 0x846CA68Bu;
	h ^= h >> 16;
	return static_cast<float>( h >> 8 ) * ( 1.0f / 16777216.0f );
}

// The code of y in [0, 1] (the value over its block's absmax), rounded stochastically with the draw u (0.5: to the nearest): unbiased, so
// that a moment moving by less than a step per update (v at beta2 0.999 moves 0.1%) follows on average instead of sticking
ADAM8_HD int code( float y, int ebits, float u = 0.5f ) {
	const int emax = ( 1 << ebits ) - 1, top = ( ( emax + 1 ) << 3 ) - 1;
	const float z = y * 1.875f;   // the code's float value
	if ( !( z > 0.0f ) ) return 0;
	int ex = 0;
	const float f = frexpf( z, &ex );   // z = f 2^ex, f in [0.5, 1)
	int E = ex - 1 + emax, M;
	if ( E >= 1 ) {
		M = static_cast<int>( floorf( ( 2.0f * f - 1.0f ) * 8.0f + u ) );
		if ( M == 8 ) {
			++E;
			M = 0;
		}
	}
	else {
		M = static_cast<int>( floorf( ldexpf( z, emax - 1 ) * 8.0f + u ) );   // subnormal: z / 2^(1 - emax) in eighths
		E = 0;
		if ( M == 8 ) {
			E = 1;
			M = 0;
		}
	}
	const int c = E * 8 + M;
	return c > top ? top : c;
}

ADAM8_HD float dec_m( uint8_t q, float absmax ) {
	const float y = mag( q & 127, 4 ) * absmax;
	return ( q & 128 ) ? -y : y;
}
ADAM8_HD uint8_t enc_m( float m, float absmax, float u = 0.5f ) {
	if ( !( absmax > 0.0f ) ) return 0;
	const int c = code( fabsf( m ) / absmax, 4, u );
	return static_cast<uint8_t>( c | ( m < 0.0f && c > 0 ? 128 : 0 ) );
}
// v from the code of its root; rmax: the block's largest root
ADAM8_HD float dec_v( uint8_t q, float rmax ) {
	const float r = mag( q, 5 ) * rmax;
	return r * r;
}
ADAM8_HD uint8_t enc_v( float v, float rmax, float u = 0.5f ) { return rmax > 0.0f ? static_cast<uint8_t>( code( sqrtf( v ) / rmax, 5, u ) ) : 0; }

constexpr float kRatio = 8.0f;

// One Adam step on one parameter: the moments and its value a from the gradient g (q8: the 8-bit form's ratio bound)
ADAM8_HD void step( float g, float& m, float& v, float& a, float lr, float b1, float b2, float eps, float c1, float c2, bool q8 = false ) {
	m = b1 * m + ( 1.0f - b1 ) * g;
	v = b2 * v + ( 1.0f - b2 ) * g * g;
	float u = ( m / c1 ) / ( sqrtf( v / c2 ) + eps );
	if ( q8 ) u = fminf( fmaxf( u, -kRatio ), kRatio );
	a = a - lr * u;
}

} // namespace engine::train::adam8
