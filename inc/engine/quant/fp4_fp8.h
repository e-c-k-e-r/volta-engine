#pragma once

// The 4-bit E2M1 (NVFP4 / KV cache nibbles) and 8-bit E4M3 codes, decoded and encoded in one place for the host loaders, the CPU backend and
// the CUDA / HIP kernels (the slang kernels keep their own switch tables: slang has no constant arrays).
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
	#define ENGINE_QUANT_FN __host__ __device__ inline
#else
	#define ENGINE_QUANT_FN inline
#endif

namespace engine::quant {

// E2M1 code (sign bit 3, magnitude index 0..7 = 0, 0.5, 1, 1.5, 2, 3, 4, 6) to float; code 8 is -0
ENGINE_QUANT_FN float e2m1_to_float( int code ) {
	const float mag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
	const float v = mag[code & 7];
	return ( code & 8 ) ? -v : v;
}

// E4M3 (bias 7, no infinity; the NaN code 0x7F decodes as 480) byte to float
ENGINE_QUANT_FN float e4m3_to_float( unsigned b ) {
	const int e = static_cast<int>( ( b >> 3 ) & 15u ), m = static_cast<int>( b & 7u );
	const float v = ( e == 0 ) ? ldexpf( m / 8.0f, -6 ) : ldexpf( 1.0f + m / 8.0f, e - 7 );
	return ( b & 0x80u ) ? -v : v;
}

// Magnitude index (0..7) of the E2M1 value nearest to a = |x|, ties to the even code (round to nearest even on the grid: the midpoints
// 0.25, 1.25, 2.5 and 5.0 fall to the lower value, 0.75, 1.75 and 3.5 to the upper one)
ENGINE_QUANT_FN int e2m1_index( float a ) {
	return a <= 0.25f ? 0 : a < 0.75f ? 1 : a <= 1.25f ? 2 : a < 1.75f ? 3 : a <= 2.5f ? 4 : a < 3.5f ? 5 : a <= 5.0f ? 6 : 7;
}

// Nearest E2M1 code of x with the sign bit set for every negative input (a negative value that rounds to zero is code 8, -0)
ENGINE_QUANT_FN int e2m1_code( float x ) {
	return e2m1_index( fabsf( x ) ) | ( x < 0.0f ? 8 : 0 );
}

} // namespace engine::quant
