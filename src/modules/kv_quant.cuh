// KV cache element codecs shared by the attention and MLA kernels (see inc/engine/core/kv_format.h).
#pragma once

#include "engine/core/kv_format.h"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace engine::modules {

using pod::KVQ;

__device__ __forceinline__ float kv_e2m1_value( int nib ) {
	const float mag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
	const float v = mag[nib & 7];
	return ( nib & 8 ) ? -v : v;
}

__device__ __forceinline__ int kv_e2m1_encode( float x ) {
	const float a = fabsf( x );
	int idx = a < 0.25f ? 0 : a < 0.75f ? 1 : a < 1.25f ? 2 : a < 1.75f ? 3 : a < 2.5f ? 4 : a < 3.5f ? 5 : a < 5.0f ? 6 : 7;
	return idx | ( ( x < 0.0f && idx != 0 ) ? 8 : 0 );
}

__device__ __forceinline__ float kv_e4m3_value( uint8_t b ) {
	const int e = ( b >> 3 ) & 15, m = b & 7;
	const float v = ( e == 0 ) ? ldexpf( m * 0.125f, -6 ) : ldexpf( 1.0f + m * 0.125f, e - 7 );
	return ( b & 0x80 ) ? -v : v;
}

__device__ __forceinline__ uint8_t kv_e4m3_encode( float x ) {
	return static_cast<uint8_t>( __nv_cvt_float_to_fp8( x, __NV_SATFINITE, __NV_E4M3 ) );
}

// FP16 -> E5M2: keep the top 8 bits with round to nearest even; saturate instead of overflowing to infinity.
__device__ __forceinline__ uint8_t kv_e5m2_encode( half h ) {
	const uint32_t bits = __half_as_ushort( h );
	const uint32_t sign = bits & 0x8000u, mag = bits & 0x7FFFu;
	if ( mag >= 0x7C00u ) return static_cast<uint8_t>( ( sign >> 8 ) | ( mag > 0x7C00u ? 0x7Eu : 0x7Bu ) );
	uint32_t r = ( mag + 0x7Fu + ( ( mag >> 8 ) & 1u ) ) >> 8;
	if ( r > 0x7Bu ) r = 0x7Bu;
	return static_cast<uint8_t>( ( sign >> 8 ) | r );
}

// Element idx (flat within the slot-relative tensor) of format Q.
template <KVQ Q>
__device__ __forceinline__ float kv_load1( const void* data, const void* scales, size_t idx ) {
	if constexpr ( Q == KVQ::F16 ) {
		return __half2float( static_cast<const half*>( data )[idx] );
	}
	else if constexpr ( Q == KVQ::E5M2 ) {
		return __half2float( __ushort_as_half( static_cast<unsigned short>( static_cast<const uint8_t*>( data )[idx] ) << 8 ) );
	}
	else if constexpr ( Q == KVQ::I8 ) {
		return static_cast<float>( static_cast<const int8_t*>( data )[idx] ) * __half2float( static_cast<const half*>( scales )[idx >> 6] );
	}
	else if constexpr ( Q == KVQ::Q4 ) {
		const int nib = ( static_cast<const uint8_t*>( data )[idx >> 1] >> ( ( idx & 1 ) * 4 ) ) & 15;
		return static_cast<float>( nib - 8 ) * __half2float( static_cast<const half*>( scales )[idx >> 6] );
	}
	else {
		const int nib = ( static_cast<const uint8_t*>( data )[idx >> 1] >> ( ( idx & 1 ) * 4 ) ) & 15;
		return kv_e2m1_value( nib ) * kv_e4m3_value( static_cast<const uint8_t*>( scales )[idx >> 4] );
	}
}

// Eight consecutive elements starting at idx (idx % 8 == 0).
template <KVQ Q>
__device__ __forceinline__ void kv_load8( const void* data, const void* scales, size_t idx, float out[8] ) {
	if constexpr ( Q == KVQ::F16 ) {
		const uint4 raw = *reinterpret_cast<const uint4*>( static_cast<const half*>( data ) + idx );
		const half2* h2 = reinterpret_cast<const half2*>( &raw );
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			const float2 f = __half22float2( h2[i] );
			out[2 * i] = f.x;
			out[2 * i + 1] = f.y;
		}
	}
	else if constexpr ( Q == KVQ::E5M2 ) {
		const uint2 raw = *reinterpret_cast<const uint2*>( static_cast<const uint8_t*>( data ) + idx );
		const uint8_t* b = reinterpret_cast<const uint8_t*>( &raw );
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) out[i] = __half2float( __ushort_as_half( static_cast<unsigned short>( b[i] ) << 8 ) );
	}
	else if constexpr ( Q == KVQ::I8 ) {
		const uint2 raw = *reinterpret_cast<const uint2*>( static_cast<const int8_t*>( data ) + idx );
		const int8_t* b = reinterpret_cast<const int8_t*>( &raw );
		const float s = __half2float( static_cast<const half*>( scales )[idx >> 6] );
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) out[i] = static_cast<float>( b[i] ) * s;
	}
	else {
		const uint32_t raw = *reinterpret_cast<const uint32_t*>( static_cast<const uint8_t*>( data ) + ( idx >> 1 ) );
		const float s = ( Q == KVQ::Q4 ) ? __half2float( static_cast<const half*>( scales )[idx >> 6] )
		                                 : kv_e4m3_value( static_cast<const uint8_t*>( scales )[idx >> 4] );
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) {
			const int nib = ( raw >> ( 4 * i ) ) & 15;
			out[i] = ( Q == KVQ::Q4 ) ? static_cast<float>( nib - 8 ) * s : kv_e2m1_value( nib ) * s;
		}
	}
}

// Elements per quantization unit of a store: one thread quantizes a whole unit.
template <KVQ Q>
__host__ __device__ constexpr int kv_unit() {
	return ( Q == KVQ::I8 || Q == KVQ::Q4 ) ? 64 : ( Q == KVQ::FP4 ? 16 : 1 );
}

// In-place orthonormal 64-point Walsh-Hadamard transform (its own inverse).
__device__ __forceinline__ void kv_fwht64( float* v ) {
	#pragma unroll
	for ( int h = 1; h < 64; h <<= 1 ) {
		#pragma unroll
		for ( int i = 0; i < 64; ++i ) {
			if ( ( i & h ) == 0 ) {
				const float a = v[i], b = v[i | h];
				v[i] = a + b;
				v[i | h] = a - b;
			}
		}
	}
	#pragma unroll
	for ( int i = 0; i < 64; ++i ) v[i] *= 0.125f;
}

// Quantize one unit of kv_unit<Q>() source elements to idx0 (unit aligned), writing data and scale.
template <KVQ Q>
__device__ __forceinline__ void kv_store_unit( void* data, void* scales, size_t idx0, const half* src ) {
	if constexpr ( Q == KVQ::F16 ) {
		static_cast<half*>( data )[idx0] = src[0];
	}
	else if constexpr ( Q == KVQ::E5M2 ) {
		static_cast<uint8_t*>( data )[idx0] = kv_e5m2_encode( src[0] );
	}
	else if constexpr ( Q == KVQ::I8 || Q == KVQ::Q4 ) {
		float v[64];
		float amax = 0.0f;
		#pragma unroll
		for ( int i = 0; i < 64; ++i ) v[i] = __half2float( src[i] );
		if constexpr ( Q == KVQ::Q4 ) kv_fwht64( v );
		#pragma unroll
		for ( int i = 0; i < 64; ++i ) amax = fmaxf( amax, fabsf( v[i] ) );
		const half sh = __float2half_rn( amax / ( Q == KVQ::I8 ? 127.0f : 7.0f ) );
		const float s = __half2float( sh );
		const float inv = s > 0.0f ? 1.0f / s : 0.0f;
		static_cast<half*>( scales )[idx0 >> 6] = sh;
		if constexpr ( Q == KVQ::I8 ) {
			int8_t* d = static_cast<int8_t*>( data ) + idx0;
			#pragma unroll
			for ( int i = 0; i < 64; ++i ) d[i] = static_cast<int8_t>( max( -127, min( 127, __float2int_rn( v[i] * inv ) ) ) );
		}
		else {
			uint8_t* d = static_cast<uint8_t*>( data ) + ( idx0 >> 1 );
			#pragma unroll
			for ( int i = 0; i < 64; i += 2 ) {
				const int lo = max( -8, min( 7, __float2int_rn( v[i] * inv ) ) ) + 8;
				const int hi = max( -8, min( 7, __float2int_rn( v[i + 1] * inv ) ) ) + 8;
				d[i >> 1] = static_cast<uint8_t>( lo | ( hi << 4 ) );
			}
		}
	}
	else {
		float v[16];
		float amax = 0.0f;
		#pragma unroll
		for ( int i = 0; i < 16; ++i ) {
			v[i] = __half2float( src[i] );
			amax = fmaxf( amax, fabsf( v[i] ) );
		}
		const uint8_t sb = kv_e4m3_encode( amax / 6.0f );
		const float s = kv_e4m3_value( sb );
		const float inv = s > 0.0f ? 1.0f / s : 0.0f;
		static_cast<uint8_t*>( scales )[idx0 >> 4] = sb;
		uint8_t* d = static_cast<uint8_t*>( data ) + ( idx0 >> 1 );
		#pragma unroll
		for ( int i = 0; i < 16; i += 2 ) {
			d[i >> 1] = static_cast<uint8_t>( kv_e2m1_encode( v[i] * inv ) | ( kv_e2m1_encode( v[i + 1] * inv ) << 4 ) );
		}
	}
}

} // namespace engine::modules
