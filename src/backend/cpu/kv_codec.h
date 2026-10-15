// Host-side KV element decoding (the plain C++ twin of the load half of src/kernels/slang/kv_codec.slang), for the CPU paths that
// read the cache without a Slang kernel (QSA sparse attention, selected latent rows). Format codes are pod::KVQ.
#pragma once

#include "engine/core/kv_format.h"
#include "engine/quant/fp4_fp8.h"
#include "engine/backend/types.h"
#include <cmath>
#include <cstdint>
#include <cstring>

namespace engine::modules {

inline float kv_h2f( uint16_t bits ) {
	backend::half_t h;
	std::memcpy( &h, &bits, sizeof( bits ) );
	return static_cast<float>( h );
}

inline float kv_e2m1( unsigned nib ) {
	return quant::e2m1_to_float( static_cast<int>( nib ) );
}

inline float kv_e4m3( unsigned b ) {
	return quant::e4m3_to_float( b );
}

// Element i (flat within the pool) of a cache tensor of format q; scales as in kv_format.h
inline float kv_load( const void* data, const void* scales, pod::KVQ q, size_t i ) {
	const uint8_t* d = static_cast<const uint8_t*>( data );
	switch ( q ) {
		case pod::KVQ::F16: {
			uint16_t b;
			std::memcpy( &b, d + 2 * i, 2 );
			return kv_h2f( b );
		}
		case pod::KVQ::E5M2: return kv_h2f( static_cast<uint16_t>( d[i] ) << 8 );
		case pod::KVQ::I8: {
			uint16_t sb;
			std::memcpy( &sb, static_cast<const uint8_t*>( scales ) + 2 * ( i >> 6 ), 2 );
			return static_cast<float>( static_cast<int8_t>( d[i] ) ) * kv_h2f( sb );
		}
		case pod::KVQ::Q4: {
			uint16_t sb;
			std::memcpy( &sb, static_cast<const uint8_t*>( scales ) + 2 * ( i >> 6 ), 2 );
			return static_cast<float>( static_cast<int>( ( d[i >> 1] >> ( ( i & 1 ) * 4 ) ) & 15 ) - 8 ) * kv_h2f( sb );
		}
		default: return kv_e2m1( ( d[i >> 1] >> ( ( i & 1 ) * 4 ) ) & 15u ) * kv_e4m3( static_cast<const uint8_t*>( scales )[i >> 4] );
	}
}

} // namespace engine::modules
