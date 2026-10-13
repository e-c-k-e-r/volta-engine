#pragma once
// ROCM configuration: the one FP8 conversion the kernels use (float -> E4M3, saturating to the finite range, round to nearest even),
// written out: HIP's FP8 types follow the target's native format (FNUZ on some parts), the KV cache stores OCP E4M3 everywhere
#include "../compat.h"

enum __nv_saturation_t { __NV_NOSAT, __NV_SATFINITE };
enum __nv_fp8_interpretation_t { __NV_E4M3, __NV_E5M2 };
typedef unsigned char __nv_fp8_storage_t;

__host__ __device__ inline __nv_fp8_storage_t __nv_cvt_float_to_fp8( float x, __nv_saturation_t sat, __nv_fp8_interpretation_t fmt ) {
	(void) fmt;   // E4M3 only (the single call site)
	uint32_t u;
	__builtin_memcpy( &u, &x, 4 );
	const uint32_t sign = ( u >> 24 ) & 0x80u;
	u &= 0x7fffffffu;
	if ( u > 0x7f800000u ) return static_cast<__nv_fp8_storage_t>( sign | 0x7f );                      // NaN
	if ( u >= 0x43e00000u ) return static_cast<__nv_fp8_storage_t>( sign | ( sat == __NV_SATFINITE ? 0x7e : 0x7f ) );   // >= 448
	float a;
	__builtin_memcpy( &a, &u, 4 );
	if ( a < 0.015625f ) {   // subnormal range: steps of 2^-9, round to nearest even
		const float q = a * 512.0f;
		float r = rintf( q );
		return static_cast<__nv_fp8_storage_t>( sign | static_cast<uint32_t>( r ) );
	}
	// normal: keep 3 mantissa bits, round to nearest even on the 20 dropped bits, rebias the exponent (127 -> 7)
	const uint32_t lsb = ( u >> 20 ) & 1u;
	u += 0x7ffffu + lsb;
	const uint32_t e = ( u >> 23 ) - 120u, m = ( u >> 20 ) & 7u;
	if ( e > 15u || ( e == 15u && m == 7u ) ) return static_cast<__nv_fp8_storage_t>( sign | 0x7e );   // rounded up past 448
	return static_cast<__nv_fp8_storage_t>( sign | ( e << 3 ) | m );
}
