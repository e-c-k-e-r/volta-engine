// The trellis decode of exl3_dq.cuh (dq_dispatch) split into a per-lane load plan and a decode from the loaded words, for loops that walk
// many tiles with one lane layout: the plan's byte offsets within a tile are fixed for the loop, so a tile's words load through the
// wave-uniform tile pointer plus a 32-bit lane offset (no per-lane 64-bit address arithmetic per load), and the loads of several tiles
// can be issued before the first is decoded. The window arithmetic is exl3_dq.cuh's, so the halves are dq_dispatch's bit for bit
// (test_trellis_decode). After upstream exllamav3's RDNA "tiles" GEMV core.
#pragma once

template <int bits>
struct Exl3TilePlan {
	// words a lane reads per tile: the aligned widths 2, the 3-bit and half-integer forms 2 / 4, dq4 x 2 (5, 6, 8 bits) 4, dq2x2 x 2 (7 bits) 8
	static constexpr int R = bits > 8 ? 4 : ( bits <= 4 ? 2 : ( bits == 7 ? 8 : 4 ) );
	static constexpr int S = R / 2;
	uint32_t off[R];   // byte offsets in the tile, pairs ( a = the lower-index word, b )
	int sh[S];         // the funnel shift of each pair
};

template <int bits>
__device__ __forceinline__ Exl3TilePlan<bits> exl3_tile_plan( int lane ) {
	Exl3TilePlan<bits> p;
	const int t_offset = lane << 3;
	constexpr int words = bits > 8 ? 4 * ( 2 * ( bits - 8 ) + 1 ) : bits * 256 / 32;
	if constexpr ( bits > 8 ) {
		constexpr int KA = bits - 8, bits2 = 2 * KA + 1, gspan = 18 + 3 * KA;
		const int e7 = ( ( t_offset >> 1 ) + 4 ) * bits2 + 128 * bits2;
		const int e3 = e7 - 2 * bits2;
		p.off[0] = ( ( e7 - gspan ) / 32 % words ) * 4;
		p.off[1] = ( ( e7 - 1 ) / 32 % words ) * 4;
		p.sh[0] = ( ( e7 - 1 ) / 32 + 1 ) * 32 - e7;
		p.off[2] = ( ( e3 - gspan ) / 32 % words ) * 4;
		p.off[3] = ( ( e3 - 1 ) / 32 % words ) * 4;
		p.sh[1] = ( ( e3 - 1 ) / 32 + 1 ) * 32 - e3;
	}
	else if constexpr ( bits == 1 || bits == 2 || bits == 4 ) {
		constexpr int per = 32 / bits;   // weights per word
		const int i1 = t_offset / per;
		p.off[1] = i1 * 4;
		p.off[0] = ( ( i1 + words - 1 ) % words ) * 4;
		p.sh[0] = bits == 4 ? 20 : ( bits == 2 ? ( ( ~t_offset ) & 8 ) << 1 : ( ~t_offset ) & 24 );
	}
	else if constexpr ( bits == 3 ) {
		const int b1 = ( t_offset + 257 ) * bits, b0 = b1 - 16, b2 = b1 + bits * 7;
		p.off[0] = ( b0 / 32 % words ) * 4;
		p.off[1] = ( ( b2 - 1 ) / 32 % words ) * 4;
		p.sh[0] = ( ( b2 - 1 ) / 32 + 1 ) * 32 - b2;
	}
	else if constexpr ( bits == 7 ) {
		#pragma unroll
		for ( int g = 0; g < 4; ++g ) {   // dq2x2( idx ), dq2x2( idx + 4 ): pairs at t_offset + 0, 2, 4, 6
			const int b0 = ( t_offset + 2 * g + 257 ) * bits - 16, b2 = b0 + bits + 16;
			p.off[2 * g] = ( b0 / 32 % words ) * 4;
			p.off[2 * g + 1] = ( ( b2 - 1 ) / 32 % words ) * 4;
			p.sh[g] = ( ( b2 - 1 ) / 32 + 1 ) * 32 - b2;
		}
	}
	else {
		#pragma unroll
		for ( int g = 0; g < 2; ++g ) {   // dq4( idx ), dq4( idx + 4 )
			const int b0 = ( t_offset + 4 * g + 257 ) * bits - 16, b2 = b0 + 3 * bits + 16;
			p.off[2 * g] = ( b0 / 32 % words ) * 4;
			p.off[2 * g + 1] = ( ( b2 - 1 ) / 32 % words ) * 4;
			p.sh[g] = ( ( b2 - 1 ) / 32 + 1 ) * 32 - b2;
		}
	}
	return p;
}

// The tile's words of this lane, from the wave-uniform tile pointer; the tile must be in global memory (on ROCm the loads are in the global
// address space: shared-memory tiles take dq_dispatch)
#if defined(USE_ROCM)
typedef __attribute__( ( address_space( 1 ) ) ) const char exl3_global_char;
typedef __attribute__( ( address_space( 1 ) ) ) const uint32_t exl3_global_u32;
#endif
template <int bits>
__device__ __forceinline__ void exl3_tile_load( const char* tile, const Exl3TilePlan<bits>& p, uint32_t* raw ) {
	#pragma unroll
	for ( int i = 0; i < Exl3TilePlan<bits>::R; ++i ) {
#if defined(USE_ROCM)
		// global address space, scalar base + 32-bit lane offset (global_load saddr)
		raw[i] = *(exl3_global_u32*) ( (exl3_global_char*) tile + p.off[i] );   // C cast: the address-space conversion
#else
		raw[i] = *reinterpret_cast<const uint32_t*>( tile + p.off[i] );
#endif
	}
}

// dq_dispatch's fragments from the loaded words (raw[2 g] = a, raw[2 g + 1] = b of pair g)
template <int bits, int cb>
__device__ __forceinline__ void exl3_tile_decode( const uint32_t* raw, const Exl3TilePlan<bits>& p, FragB& frag0, FragB& frag1 ) {
	if constexpr ( bits > 8 ) {
		constexpr int KA = bits - 8;
		uint32_t w7 = fshift( raw[1], raw[0], p.sh[0] ), w6 = w7 >> ( KA + 1 ), w5 = w6 >> KA, w4 = w5 >> ( KA + 1 );
		uint32_t w3 = fshift( raw[3], raw[2], p.sh[1] ), w2 = w3 >> ( KA + 1 ), w1 = w2 >> KA, w0 = w1 >> ( KA + 1 );
		frag0[0] = decode_3inst_2<cb>( w0 & 0xffff, w1 & 0xffff );
		frag0[1] = decode_3inst_2<cb>( w2 & 0xffff, w3 & 0xffff );
		frag1[0] = decode_3inst_2<cb>( w4 & 0xffff, w5 & 0xffff );
		frag1[1] = decode_3inst_2<cb>( w6 & 0xffff, w7 & 0xffff );
	}
	else if constexpr ( bits == 1 || bits == 2 || bits == 4 ) {
		uint32_t b = raw[1], s = 0;
		if constexpr ( bits == 4 ) s = __funnelshift_r( b, raw[0], 20 );
		else b = fshift( b, raw[0], p.sh[0] );
		uint32_t w[8];
		if constexpr ( bits == 4 ) {
			w[7] = b; w[6] = b >> 4; w[5] = b >> 8; w[4] = b >> 12; w[3] = b >> 16; w[2] = s; w[1] = s >> 4; w[0] = s >> 8;
		}
		else {
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) w[7 - i] = b >> ( i * bits );
		}
		frag0[0] = decode_3inst_2<cb>( w[0] & 0xffff, w[1] & 0xffff );
		frag0[1] = decode_3inst_2<cb>( w[2] & 0xffff, w[3] & 0xffff );
		frag1[0] = decode_3inst_2<cb>( w[4] & 0xffff, w[5] & 0xffff );
		frag1[1] = decode_3inst_2<cb>( w[6] & 0xffff, w[7] & 0xffff );
	}
	else if constexpr ( bits == 3 ) {
		const uint32_t a = raw[0], b = raw[1];
		const int s2 = p.sh[0];
		uint32_t w7 = fshift( b, a, s2 ), w6 = w7 >> bits, w5 = w6 >> bits, w4 = w5 >> bits;
		uint32_t w3 = fshift( b, a, s2 + bits * 4 ), w2 = w3 >> bits, w1 = w2 >> bits, w0 = w1 >> bits;
		frag0[0] = decode_3inst_2<cb>( w0 & 0xffff, w1 & 0xffff );
		frag0[1] = decode_3inst_2<cb>( w2 & 0xffff, w3 & 0xffff );
		frag1[0] = decode_3inst_2<cb>( w4 & 0xffff, w5 & 0xffff );
		frag1[1] = decode_3inst_2<cb>( w6 & 0xffff, w7 & 0xffff );
	}
	else if constexpr ( bits == 7 ) {
		#pragma unroll
		for ( int g = 0; g < 4; ++g ) {
			const uint32_t a = raw[2 * g], b = raw[2 * g + 1];
			const uint32_t w1 = fshift( b, a, p.sh[g] ) & 0xffff, w0 = fshift( b, a, p.sh[g] + bits ) & 0xffff;
			( g < 2 ? frag0 : frag1 )[g & 1] = decode_3inst_2<cb>( w0, w1 );
		}
	}
	else {
		#pragma unroll
		for ( int g = 0; g < 2; ++g ) {
			const uint32_t a = raw[2 * g], b = raw[2 * g + 1];
			const int s2 = p.sh[g];
			const uint32_t w3 = fshift( b, a, s2 ) & 0xffff, w2 = fshift( b, a, s2 + bits ) & 0xffff;
			const uint32_t w1 = fshift( b, a, s2 + bits * 2 ) & 0xffff, w0 = fshift( b, a, s2 + bits * 3 ) & 0xffff;
			FragB& f = g == 0 ? frag0 : frag1;
			f[0] = decode_3inst_2<cb>( w0, w1 );
			f[1] = decode_3inst_2<cb>( w2, w3 );
		}
	}
}

// Load and decode in one call, from a tile in global memory
template <int bits, int cb>
__device__ __forceinline__ void exl3_tile_dq( const void* tile, const Exl3TilePlan<bits>& p, FragB& frag0, FragB& frag1 ) {
	uint32_t raw[Exl3TilePlan<bits>::R];
	exl3_tile_load<bits>( static_cast<const char*>( tile ), p, raw );
	exl3_tile_decode<bits, cb>( raw, p, frag0, frag1 );
}
