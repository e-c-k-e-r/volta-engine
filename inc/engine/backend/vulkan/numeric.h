#pragma once

#include "engine/backend/types.h"
#include <cstring>

// Vulkan backend: software fp16/bf16 conversions, round to nearest even (bit-identical to cuda_fp16 / cuda_bf16 for finite values)
namespace engine::backend {

inline float to_float( half_t h ) {
	const uint32_t sign = ( uint32_t( h.bits ) & 0x8000u ) << 16;
	uint32_t e = ( h.bits >> 10 ) & 0x1fu, m = h.bits & 0x3ffu, u;
	if ( e == 0x1f ) {
		u = sign | 0x7f800000u | ( m << 13 );
	} else if ( e != 0 ) {
		u = sign | ( ( e + 112 ) << 23 ) | ( m << 13 );
	} else if ( m == 0 ) {
		u = sign;
	} else { // subnormal: normalize
		e = 113;
		while ( !( m & 0x400u ) ) { m <<= 1; --e; }
		u = sign | ( e << 23 ) | ( ( m & 0x3ffu ) << 13 );
	}
	float f;
	std::memcpy( &f, &u, 4 );
	return f;
}
inline float to_float( bf16_t b ) {
	const uint32_t u = uint32_t( b.bits ) << 16;
	float f;
	std::memcpy( &f, &u, 4 );
	return f;
}

// float -> fp16, round to nearest even
inline half_t from_float_half( float f ) {
	uint32_t u;
	std::memcpy( &u, &f, 4 );
	const uint32_t sign = ( u >> 16 ) & 0x8000u, m = u & 0x7fffffu;
	const int e = int( ( u >> 23 ) & 0xff );
	if ( e == 0xff ) return half_t{ uint16_t( sign | 0x7c00u | ( m ? ( 0x200u | ( m >> 13 ) ) : 0u ) ) };
	const int exp = e - 127 + 15;
	if ( exp >= 31 ) return half_t{ uint16_t( sign | 0x7c00u ) };
	if ( exp <= 0 ) {
		if ( exp < -10 ) return half_t{ uint16_t( sign ) };
		const uint32_t mm = m | 0x800000u;
		const int shift = 14 - exp;
		uint32_t h = mm >> shift;
		const uint32_t rem = mm & ( ( 1u << shift ) - 1 ), mid = 1u << ( shift - 1 );
		if ( rem > mid || ( rem == mid && ( h & 1 ) ) ) ++h;
		return half_t{ uint16_t( sign | h ) };
	}
	uint32_t h = ( uint32_t( exp ) << 10 ) | ( m >> 13 );
	const uint32_t rem = m & 0x1fffu;
	if ( rem > 0x1000u || ( rem == 0x1000u && ( h & 1 ) ) ) ++h;
	return half_t{ uint16_t( sign | h ) };
}
// float -> bf16, round to nearest even
inline bf16_t from_float_bf16( float f ) {
	uint32_t u;
	std::memcpy( &u, &f, 4 );
	if ( ( u & 0x7fffffffu ) > 0x7f800000u ) return bf16_t{ uint16_t( ( u >> 16 ) | 0x40u ) };
	u += 0x7fffu + ( ( u >> 16 ) & 1u );
	return bf16_t{ uint16_t( u >> 16 ) };
}

} // namespace engine::backend
