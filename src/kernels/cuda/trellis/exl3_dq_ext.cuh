#pragma once

// The extended trellis' decode: exl3_dq.cuh's helpers with the two optional per-tile companions. A start state replaces the wrapped read of a
// tile's first windows (its low 16 - K bits are the bits before the tile's stream, instead of its end), a gain code c multiplies the decoded
// values by (896 + c) / 1024. Kept apart from exl3_dq.cuh so that kernels decoding plain trellises compile exactly as before.
#include "exl3_dq.cuh"

template <int bits, int cb>
__device__ __forceinline__ half dq_ext( const uint32_t* ptr, int t_offset, uint32_t start = 0, bool has_start = false ) {
	int b0 = t_offset * bits + bits - 16 + 256 * bits;
	int b1 = b0 + 16;
	int i0 = b0 / 32;
	int i1 = ( b1 - 1 ) / 32;
	int s0 = ( i1 + 1 ) * 32 - b1;

	uint32_t a = has_start && ( t_offset + 1 ) * bits < 16 ? start : ptr[i0 % ( bits * 256 / 32 )];
	uint32_t b = ptr[i1 % ( bits * 256 / 32 )];

	uint32_t w0 = __funnelshift_r( b, a, s0 ) & 0xffff;
	return decode_3inst<cb>( w0 );
}

template <int bits, int cb>
__device__ __forceinline__ half2 dq2_ext( const uint32_t* ptr, int t_offset, uint32_t start = 0, bool has_start = false ) {
	int b0 = t_offset * bits + bits - 16 + 256 * bits;
	int b1 = b0 + 16;
	int i0 = b0 / 32;
	int i1 = ( b1 - 1 ) / 32;
	int s0 = ( i1 + 1 ) * 32 - b1;

	uint32_t a = has_start && ( t_offset + 1 ) * bits < 16 ? start : ptr[i0 % ( bits * 256 / 32 )];
	uint32_t b = ptr[i1 % ( bits * 256 / 32 )];

	uint32_t w1 = __funnelshift_r( b, a, s0 ) & 0xffff;
	uint32_t w0 = __funnelshift_r( b, a, s0 + bits ) & 0xffff;
	return decode_3inst_2<cb>( w0, w1 );
}

template <int bits, int cb>
__device__ __forceinline__ void dq4_ext( const uint32_t* ptr, int t_offset, FragB& frag, uint32_t start = 0, bool has_start = false ) {
	int b0 = ( t_offset + 257 ) * bits - 16;
	int b1 = b0 + 3 * bits;
	int b2 = b1 + 16;
	int i0 = b0 / 32;
	int i2 = ( b2 - 1 ) / 32;
	int s2 = ( i2 + 1 ) * 32 - b2;

	uint32_t a = has_start && ( t_offset + 1 ) * bits < 16 ? start : ptr[i0 % ( bits * 256 / 32 )];
	uint32_t b = ptr[i2 % ( bits * 256 / 32 )];
	uint32_t w3 = fshift( b, a, s2 ) & 0xffff;
	uint32_t w2 = fshift( b, a, s2 + bits ) & 0xffff;
	uint32_t w1 = fshift( b, a, s2 + bits * 2 ) & 0xffff;
	uint32_t w0 = fshift( b, a, s2 + bits * 3 ) & 0xffff;
	half2 d0d1 = decode_3inst_2<cb>( w0, w1 );
	half2 d2d3 = decode_3inst_2<cb>( w2, w3 );
	frag[0] = d0d1;
	frag[1] = d2d3;
}

template <int bits, int cb>
__device__ __forceinline__ void dq2x2_ext( const uint32_t* ptr, int t_offset, FragB& frag, uint32_t start = 0, bool has_start = false ) {
	#pragma unroll
	for ( int i = 0; i < 2; ++i ) {
		int b0 = ( t_offset + 2 * i + 257 ) * bits - 16;
		int b1 = b0 + 1 * bits;
		int b2 = b1 + 16;
		int i0 = b0 / 32;
		int i2 = ( b2 - 1 ) / 32;
		int s2 = ( i2 + 1 ) * 32 - b2;

		uint32_t a = has_start && ( t_offset + 2 * i + 1 ) * bits < 16 ? start : ptr[i0 % ( bits * 256 / 32 )];
		uint32_t b = ptr[i2 % ( bits * 256 / 32 )];
		uint32_t w1 = fshift( b, a, s2 ) & 0xffff;
		uint32_t w0 = fshift( b, a, s2 + bits ) & 0xffff;
		half2 d0d1 = decode_3inst_2<cb>( w0, w1 );
		frag[i] = d0d1;
	}
}

template <int bits, int cb, int align>
__device__ __forceinline__ void dq8_ext( const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1, uint32_t start = 0, bool has_start = false ) {
	int b1 = ( t_offset + 257 ) * bits;
	int b0 = b1 - 16;
	int b2 = b1 + bits * 7;
	int i0 = b0 / 32;
	int i2 = ( b2 - 1 ) / 32;
	int s2 = ( i2 + 1 ) * 32 - b2;

	uint32_t a = has_start && ( t_offset + 1 ) * bits < 16 ? start : ptr[i0 % ( bits * 256 / 32 )];
	uint32_t b = ptr[i2 % ( bits * 256 / 32 )];
	uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
	if constexpr ( align == 1 ) {
		w7 = fshift( b, a, s2 );
		w6 = fshift( b, a, s2 + bits );
		w5 = fshift( b, a, s2 + bits * 2 );
		w4 = fshift( b, a, s2 + bits * 3 );
		w3 = fshift( b, a, s2 + bits * 4 );
		w2 = fshift( b, a, s2 + bits * 5 );
		w1 = fshift( b, a, s2 + bits * 6 );
		w0 = fshift( b, a, s2 + bits * 7 );
	}
	if constexpr ( align == 2 ) {
		w7 = fshift( b, a, s2 );
		w6 = w7 >> bits;
		w5 = fshift( b, a, s2 + bits * 2 );
		w4 = w5 >> bits;
		w3 = fshift( b, a, s2 + bits * 4 );
		w2 = w3 >> bits;
		w1 = fshift( b, a, s2 + bits * 6 );
		w0 = w1 >> bits;
	}
	if constexpr ( align == 4 ) {
		w7 = fshift( b, a, s2 );
		w6 = w7 >> bits;
		w5 = w6 >> bits;
		w4 = w5 >> bits;
		w3 = fshift( b, a, s2 + bits * 4 );
		w2 = w3 >> bits;
		w1 = w2 >> bits;
		w0 = w1 >> bits;
	}
	if constexpr ( align == 8 ) {
		w7 = fshift( b, a, s2 );
		w6 = w7 >> bits;
		w5 = w6 >> bits;
		w4 = w5 >> bits;
		w3 = w4 >> bits;
		w2 = w3 >> bits;
		w1 = w2 >> bits;
		w0 = w1 >> bits;
	}
	half2 d0d1 = decode_3inst_2<cb>( w0 & 0xffff, w1 & 0xffff );
	half2 d2d3 = decode_3inst_2<cb>( w2 & 0xffff, w3 & 0xffff );
	half2 d4d5 = decode_3inst_2<cb>( w4 & 0xffff, w5 & 0xffff );
	half2 d6d7 = decode_3inst_2<cb>( w6 & 0xffff, w7 & 0xffff );
	frag0[0] = d0d1;
	frag0[1] = d2d3;
	frag1[0] = d4d5;
	frag1[1] = d6d7;
}

template <int cb>
__device__ __forceinline__ void dq8_aligned_4bits_ext( const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1, uint32_t start = 0,
	bool has_start = false ) {
	uint32_t i0, i1, a, b, s, w0, w1, w2, w3, w4, w5, w6, w7;
	i1 = t_offset >> 3;
	i0 = ( i1 + 31 ) & 31;
	a = has_start && i1 == 0 ? start : ptr[i0];
	b = ptr[i1];
	FSHF_IMM( s, b, a, 20 );
	w7 = b & 0xffff;
	BFE16_IMM( w6, b, 4 );
	BFE16_IMM( w5, b, 8 );
	BFE16_IMM( w4, b, 12 );
	BFE16_IMM( w3, b, 16 );
	w2 = s & 0xffff;
	BFE16_IMM( w1, s, 4 );
	BFE16_IMM( w0, s, 8 );
	frag0[0] = decode_3inst_2<cb>( w0, w1 );
	frag0[1] = decode_3inst_2<cb>( w2, w3 );
	frag1[0] = decode_3inst_2<cb>( w4, w5 );
	frag1[1] = decode_3inst_2<cb>( w6, w7 );
}

template <int cb>
__device__ __forceinline__ void dq8_aligned_2bits_ext( const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1, uint32_t start = 0,
	bool has_start = false ) {
	uint32_t i0, i1, a, b, w0, w1, w2, w3, w4, w5, w6, w7;
	i1 = t_offset >> 4;
	i0 = ( i1 + 15 ) & 15;
	a = has_start && i1 == 0 ? start : ptr[i0];
	b = ptr[i1];
	b = fshift( b, a, ( ( ~t_offset ) & 8 ) << 1 );
	w7 = b & 0xffff;
	BFE16_IMM( w6, b, 2 );
	BFE16_IMM( w5, b, 4 );
	BFE16_IMM( w4, b, 6 );
	BFE16_IMM( w3, b, 8 );
	BFE16_IMM( w2, b, 10 );
	BFE16_IMM( w1, b, 12 );
	BFE16_IMM( w0, b, 14 );
	frag0[0] = decode_3inst_2<cb>( w0, w1 );
	frag0[1] = decode_3inst_2<cb>( w2, w3 );
	frag1[0] = decode_3inst_2<cb>( w4, w5 );
	frag1[1] = decode_3inst_2<cb>( w6, w7 );
}

template <int cb>
__device__ __forceinline__ void dq8_aligned_1bit_ext( const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1, uint32_t start = 0,
	bool has_start = false ) {
	uint32_t i0, i1, a, b, w0, w1, w2, w3, w4, w5, w6, w7;
	i1 = t_offset >> 5;
	i0 = ( i1 + 7 ) & 7;
	a = has_start && i1 == 0 ? start : ptr[i0];
	b = ptr[i1];
	b = fshift( b, a, ( ( ~t_offset ) & 24 ) );
	w7 = b & 0xffff;
	BFE16_IMM( w6, b, 1 );
	BFE16_IMM( w5, b, 2 );
	BFE16_IMM( w4, b, 3 );
	BFE16_IMM( w3, b, 4 );
	BFE16_IMM( w2, b, 5 );
	BFE16_IMM( w1, b, 6 );
	BFE16_IMM( w0, b, 7 );
	frag0[0] = decode_3inst_2<cb>( w0, w1 );
	frag0[1] = decode_3inst_2<cb>( w2, w3 );
	frag1[0] = decode_3inst_2<cb>( w4, w5 );
	frag1[1] = decode_3inst_2<cb>( w6, w7 );
}

// Half-integer rates K + 0.5 (KA = K, mul1 codebook): the windows of a tile advance by KA + 1 and KA bits alternately (one extra bit per
// pair), so eight consecutive positions starting at an even index take 8 * KA + 4 bits and a lane's window ends are fixed. Two groups of
// four windows, each spanning 18 + 3 * KA <= 27 bits = at most two 32-bit words: one funnel shift per group, the other three windows
// derived by shifting the 32-bit result. A tile is 4 * (2 * KA + 1) words.
template <int KA, int cb>
__device__ __forceinline__ void dq8_half_ext( const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1, uint32_t start = 0, bool has_start = false ) {
	constexpr int bits2 = 2 * KA + 1;                              // bits per two positions
	constexpr int words = 4 * bits2;                               // uint32 per 256-weight tile
	constexpr int gspan = 18 + 3 * KA;                             // bits spanned by four windows
	const int e7 = ( ( t_offset >> 1 ) + 4 ) * bits2 + 128 * bits2;   // end bit of window 7 (+ one tile for the wrap)
	const int e3 = e7 - 2 * bits2;                                 // end bit of window 3
	uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
	{
		int hi = ( e7 - 1 ) / 32, lo = ( e7 - gspan ) / 32, s = ( hi + 1 ) * 32 - e7;
		uint32_t a, b = ptr[hi % words];
		a = has_start && e7 - gspan < 128 * bits2 ? start : ptr[lo % words];
		w7 = fshift( b, a, s ); w6 = w7 >> ( KA + 1 ); w5 = w6 >> KA; w4 = w5 >> ( KA + 1 );
	}
	{
		int hi = ( e3 - 1 ) / 32, lo = ( e3 - gspan ) / 32, s = ( hi + 1 ) * 32 - e3;
		uint32_t a, b = ptr[hi % words];
		a = has_start && e3 - gspan < 128 * bits2 ? start : ptr[lo % words];
		w3 = fshift( b, a, s ); w2 = w3 >> ( KA + 1 ); w1 = w2 >> KA; w0 = w1 >> ( KA + 1 );
	}
	half2 d0d1 = decode_3inst_2<cb>( w0 & 0xffff, w1 & 0xffff );
	half2 d2d3 = decode_3inst_2<cb>( w2 & 0xffff, w3 & 0xffff );
	half2 d4d5 = decode_3inst_2<cb>( w4 & 0xffff, w5 & 0xffff );
	half2 d6d7 = decode_3inst_2<cb>( w6 & 0xffff, w7 & 0xffff );
	frag0[0] = d0d1;
	frag0[1] = d2d3;
	frag1[0] = d4d5;
	frag1[1] = d6d7;
}

template <int bits, int cb>
__device__ __forceinline__ void dq_dispatch_ext( const uint32_t* ptr, int idx, FragB& frag0, FragB& frag1, uint32_t start = 0, bool has_start = false,
	bool gained = false, half2 gain = __half2half2( __ushort_as_half( 0x3c00 ) ) ) {
	if constexpr ( bits > 8 ) {
		// width code 8 + K: the half-integer rate K + 0.5 (see engine/modules/trellis_bits.h)
		dq8_half_ext<bits - 8, cb>( ptr, idx, frag0, frag1, start, has_start );
	}
	else if constexpr ( bits == 1 ) {
		dq8_aligned_1bit_ext<cb>( ptr, idx, frag0, frag1, start, has_start );
	}
	else if constexpr ( bits == 2 ) {
		dq8_aligned_2bits_ext<cb>( ptr, idx, frag0, frag1, start, has_start );
	}
	else if constexpr ( bits == 3 ) {
		dq8_ext<bits, cb, 4>( ptr, idx, frag0, frag1, start, has_start );
	}
	else if constexpr ( bits == 4 ) {
		dq8_aligned_4bits_ext<cb>( ptr, idx, frag0, frag1, start, has_start );
	}
	else if constexpr ( bits == 5 ) {
		dq4_ext<bits, cb>( ptr, idx, frag0, start, has_start );
		dq4_ext<bits, cb>( ptr, idx + 4, frag1, start, has_start );
	}
	else if constexpr ( bits == 6 ) {
		dq4_ext<bits, cb>( ptr, idx, frag0, start, has_start );
		dq4_ext<bits, cb>( ptr, idx + 4, frag1, start, has_start );
	}
	else if constexpr ( bits == 7 ) {
		dq2x2_ext<bits, cb>( ptr, idx, frag0, start, has_start );
		dq2x2_ext<bits, cb>( ptr, idx + 4, frag1, start, has_start );
	}
	else if constexpr ( bits == 8 ) {
		dq4_ext<bits, cb>( ptr, idx, frag0, start, has_start );
		dq4_ext<bits, cb>( ptr, idx + 4, frag1, start, has_start );
	}
	if ( gained ) {
		frag0[0] = __hmul2( frag0[0], gain );
		frag0[1] = __hmul2( frag0[1], gain );
		frag1[0] = __hmul2( frag1[0], gain );
		frag1[1] = __hmul2( frag1[1], gain );
	}
}

// A tile's gain code as the value it multiplies by: (896 + c) / 1024, exact in FP16
__device__ __forceinline__ half2 tile_gain2( uint8_t c ) { return __half2half2( __float2half_rn( ( 896.0f + c ) * ( 1.0f / 1024.0f ) ) ); }
