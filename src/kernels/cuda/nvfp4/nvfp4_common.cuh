#pragma once
// Shared NVFP4 device helpers: FP8 E4M3 scale decode, 8-code dequant to FP16 pairs, Volta m8n8k4 MMA
#include <cuda_fp16.h>
#define DEV_INLINE __device__ __forceinline__

#define MMA_8N8K4(C, A0, A1, B0, B1) \
	asm volatile ( \
		"mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 " \
		"{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, " \
		"{%0,%1,%2,%3,%4,%5,%6,%7};\n" \
		: "+f"( C[0] ), "+f"( C[1] ), "+f"( C[2] ), "+f"( C[3] ), "+f"( C[4] ), \
		  "+f"( C[5] ), "+f"( C[6] ), "+f"( C[7] ) \
		: "r"( A0 ), "r"( A1 ), "r"( B0 ), "r"( B1 ) \
	)

DEV_INLINE half2 fp8e4m3_to_half2( unsigned char b ) {
	const unsigned short hb =
		( ( (unsigned short) b & 0x80u ) << 8 ) | ( ( (unsigned short) b & 0x7Fu ) << 7 );
	const half hs = __hmul( __ushort_as_half( hb ), __ushort_as_half( 0x5C00 ) ); // *256
	return __halves2half2( hs, hs );
}

DEV_INLINE void dequant8_tm( unsigned q, half2 sc2p, half2 out[4] ) {
	constexpr unsigned S = 0x80008000u, EM = 0x0E000E00u;
	unsigned v0 = ( ( q << 12 ) & S ) | ( ( q << 9 ) & EM );
	unsigned v1 = ( ( q << 8 ) & S ) | ( ( q << 5 ) & EM );
	unsigned v2 = ( ( q << 4 ) & S ) | ( ( q << 1 ) & EM );
	unsigned v3 = ( q & S ) | ( ( q >> 3 ) & EM );
	out[0] = __hmul2( *reinterpret_cast<half2*>( &v0 ), sc2p );
	out[1] = __hmul2( *reinterpret_cast<half2*>( &v1 ), sc2p );
	out[2] = __hmul2( *reinterpret_cast<half2*>( &v2 ), sc2p );
	out[3] = __hmul2( *reinterpret_cast<half2*>( &v3 ), sc2p );
}
