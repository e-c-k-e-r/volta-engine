// RDNA form of the trellis pair decode (decode_3inst_2, src/kernels/cuda/trellis/codebook.cuh includes it on ROCm): the same values bit for bit
#pragma once

// x * C + D mod 2^32 for a 16-bit x, on RDNA without v_mul_lo_u32 (quarter rate): with x < 2^16,
// x * C = x * C_lo16 + ( ( x * C_hi16 ) mod 2^16 ) << 16. The 16-bit vector product is one v_pk_mul_lo_u16 / v_pk_mad_u16 (low lane
// D_lo16, high lane x * C_hi16 + D_hi16) and the rest one v_mad_u32_u24, both full rate (after upstream exllamav3's RDNA GEMV, which
// writes them as inline asm; asm is convergent to the compiler and stops it unrolling the k loops)
template <uint32_t C, uint32_t D>
__device__ __forceinline__ uint32_t rdna_hash16( uint32_t x ) {
	typedef uint16_t u16x2 __attribute__( ( ext_vector_type( 2 ) ) );
	const uint16_t xl = static_cast<uint16_t>( x );
	const u16x2 r = u16x2{ xl, xl } * u16x2{ 0, static_cast<uint16_t>( C >> 16 ) } + u16x2{ static_cast<uint16_t>( D ), static_cast<uint16_t>( D >> 16 ) };
	return static_cast<uint32_t>( xl ) * ( C & 0xffffu ) + __builtin_bit_cast( uint32_t, r );
}

// The pair decode of decode_3inst_2 below, bit for bit: the mul1 byte sums are v_sad_u8 / v_sad_hi_u8, which leave the pair packed as
// [ 0x6400 + s0 | 0x6400 + s1 ]
template <int cb>
__device__ inline half2 decode_3inst_2( uint32_t x0, uint32_t x1 ) {
	if constexpr ( cb == 0 ) {
		x0 = rdna_hash16<89226354u, 64248484u>( x0 );
		x1 = rdna_hash16<89226354u, 64248484u>( x1 );
		EXL3_CB_LOP3( x0 );
		EXL3_CB_LOP3( x1 );
		half2_uint32 xu0( x0 );
		half2_uint32 xu1( x1 );
		half2 d0 = __lows2half2( xu0.as_half2, xu1.as_half2 );
		half2 d1 = __highs2half2( xu0.as_half2, xu1.as_half2 );
		return __hadd2( d0, d1 );
	}
	if constexpr ( cb == 1 ) {
		return decode_mcg_product_2( rdna_hash16<0xCBAC1FEDu, 0u>( x0 ), rdna_hash16<0xCBAC1FEDu, 0u>( x1 ) );
	}
	if constexpr ( cb == 2 ) {
		const uint32_t p0 = rdna_hash16<0x83DCD12Du, 0u>( x0 ), p1 = rdna_hash16<0x83DCD12Du, 0u>( x1 );
#if defined(__HIP_DEVICE_COMPILE__)
		const uint32_t s = __builtin_amdgcn_sad_hi_u8( p1, 0u, __builtin_amdgcn_sad_u8( p0, 0u, 0x64006400u ) );
#else
		const uint32_t s = 0x64006400u + ( __dp4a( p0, 0x01010101u, 0u ) | ( __dp4a( p1, 0x01010101u, 0u ) << 16 ) );
#endif
		half2_uint32 u( s );
		return __hfma2( u.as_half2, __half2half2( __ushort_as_half( 0x1eee ) ), __half2half2( __ushort_as_half( 0xc931 ) ) );
	}
}
