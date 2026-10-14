#pragma once
// ROCm only (RDNA3, wave32): vector types of v_wmma_f32_16x16x16_f16 and a 16-half fragment load from shared memory.
// Lane layouts (wave32): A row = lane & 15 and B column = lane & 15, each with its 16 k values; accumulator element i of lane l is
// row 2 i + (l >> 4), column l & 15. The two half-waves feed the even and odd output rows (they hold duplicates in a plain product).
typedef _Float16 wm_v16h __attribute__( ( ext_vector_type( 16 ) ) );
typedef _Float16 wm_v8h __attribute__( ( ext_vector_type( 8 ) ) );
typedef _Float16 wm_v2h __attribute__( ( ext_vector_type( 2 ) ) );
typedef float wm_v8f __attribute__( ( ext_vector_type( 8 ) ) );

__device__ __forceinline__ wm_v16h wm_ld16( const half* p ) {
	const uint4 a = *reinterpret_cast<const uint4*>( p );
	const uint4 b = *reinterpret_cast<const uint4*>( p + 8 );
	wm_v8h lo = __builtin_bit_cast( wm_v8h, a ), hi = __builtin_bit_cast( wm_v8h, b );
	return __builtin_shufflevector( lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );
}

// The same from a row that is only 8-byte aligned (shared-memory rows padded to 36 halves)
__device__ __forceinline__ wm_v16h wm_ld16u( const half* p ) {
	const uint2* q = reinterpret_cast<const uint2*>( p );
	const uint2 a = q[0], b = q[1], c = q[2], d = q[3];
	wm_v8h lo = __builtin_bit_cast( wm_v8h, make_uint4( a.x, a.y, b.x, b.y ) ), hi = __builtin_bit_cast( wm_v8h, make_uint4( c.x, c.y, d.x, d.y ) );
	return __builtin_shufflevector( lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );
}
