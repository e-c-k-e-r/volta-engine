#pragma once

#include "util.cuh"

// lop3.b32 with LUT 0x6a: (x & 0x8fff8fff) ^ 0x3b603b60
#if defined(USE_ROCM)
	#define EXL3_CB_LOP3( x ) ( (x) = ( (x) & 0x8fff8fffu ) ^ 0x3b603b60u )
#else
	#define EXL3_CB_LOP3( x ) asm ( "lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"( x ) )
#endif

__device__ inline half2 decode_mul1_product_2( uint32_t x0, uint32_t x1 ) {
	const uint32_t acc = 0x6400u;
	uint32_t sum0 = __dp4a( x0, 0x01010101u, acc );
	uint32_t sum1 = __dp4a( x1, 0x01010101u, acc );
	half2 k_inv_h2 = __half2half2( __ushort_as_half( 0x1eee ) );
	half2 k_bias_h2 = __half2half2( __ushort_as_half( 0xc931 ) );
	half_uint16 h0( (uint16_t) sum0 );
	half_uint16 h1( (uint16_t) sum1 );
	return __hfma2( __halves2half2( h0.as_half, h1.as_half ), k_inv_h2, k_bias_h2 );
}

__device__ inline half2 decode_mcg_product_2( uint32_t x0, uint32_t x1 ) {
	EXL3_CB_LOP3( x0 );
	EXL3_CB_LOP3( x1 );
	half2_uint32 xu0( x0 );
	half2_uint32 xu1( x1 );
	half2 d0 = __lows2half2( xu0.as_half2, xu1.as_half2 );
	half2 d1 = __highs2half2( xu0.as_half2, xu1.as_half2 );
	return __hadd2( d0, d1 );
}

template <int cb>
__device__ inline half decode_3inst( uint32_t x ) {
	if constexpr ( cb == 0 ) {
		x *= 89226354u;
		x += 64248484u;
		EXL3_CB_LOP3( x );
		half2_uint32 xu( x );
		return __hadd( __low2half( xu.as_half2 ), __high2half( xu.as_half2 ) );
	}
	if constexpr ( cb == 1 ) {
		x *= 0xCBAC1FEDu;
		EXL3_CB_LOP3( x );
		half2_uint32 xu( x );
		return __hadd( __low2half( xu.as_half2 ), __high2half( xu.as_half2 ) );
	}
	if constexpr ( cb == 2 ) {
		x *= 0x83DCD12Du;
		const uint32_t acc = 0x6400u;
		uint32_t sum = __dp4a( x, 0x01010101u, acc );
		const __half k_inv_h = __ushort_as_half( 0x1eee );
		const __half k_bias_h = __ushort_as_half( 0xc931 );
		half_uint16 h( (uint16_t) sum );
		return __hfma( h.as_half, k_inv_h, k_bias_h );
	}
}

template <int cb>
__device__ inline half2 decode_3inst_2( uint32_t x0, uint32_t x1 ) {
	if constexpr ( cb == 0 ) {
		x0 *= 89226354u;
		x1 *= 89226354u;
		x0 += 64248484u;
		x1 += 64248484u;
		EXL3_CB_LOP3( x0 );
		EXL3_CB_LOP3( x1 );
		half2_uint32 xu0( x0 );
		half2_uint32 xu1( x1 );
		half2 d0 = __lows2half2( xu0.as_half2, xu1.as_half2 );
		half2 d1 = __highs2half2( xu0.as_half2, xu1.as_half2 );
		return __hadd2( d0, d1 );
	}
	if constexpr ( cb == 1 ) {
		x0 *= 0xCBAC1FEDu;
		x1 *= 0xCBAC1FEDu;
		return decode_mcg_product_2( x0, x1 );
	}
	if constexpr ( cb == 2 ) {
		x0 *= 0x83DCD12Du;
		x1 *= 0x83DCD12Du;
		return decode_mul1_product_2( x0, x1 );
	}
}
