#pragma once

#include "engine/backend/types.h"

// CPU backend numeric: host conversions, round to nearest even like the CUDA intrinsics
namespace engine::backend {

inline float to_float( half_t h ) { return static_cast<float>( h ); }
inline float to_float( bf16_t b ) {
	const uint32_t u = static_cast<uint32_t>( b.bits ) << 16;
	float f;
	std::memcpy( &f, &u, sizeof( f ) );
	return f;
}

// float -> fp16, round to nearest even (compiler conversion)
inline half_t from_float_half( float f ) { return static_cast<half_t>( f ); }
// float -> bf16, round to nearest even; NaN stays NaN
inline bf16_t from_float_bf16( float f ) {
	uint32_t u;
	std::memcpy( &u, &f, sizeof( u ) );
	if ( ( u & 0x7fffffffu ) > 0x7f800000u ) return bf16_t{ static_cast<uint16_t>( ( u >> 16 ) | 0x40u ) };
	u += 0x7fffu + ( ( u >> 16 ) & 1u );
	return bf16_t{ static_cast<uint16_t>( u >> 16 ) };
}

} // namespace engine::backend
