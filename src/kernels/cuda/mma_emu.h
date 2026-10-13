#pragma once
// mma.sync emulation for targets without NVIDIA's tensor-core instructions (the ROCM configuration): the m16n8k16 / m16n8k8 / m8n8k4
// forms the kernels use, keeping NVIDIA's per-lane fragment layouts so every kernel built on them runs unchanged. Each lane gathers the
// A rows and B columns its outputs need from the lanes that hold them (warp shuffles, 32 lanes) and sums the products in FP32: the
// products of FP16 inputs are exact in FP32, as on the tensor cores; only the summation order differs. The same idea as upstream
// exllamav3's rocm/ptx_rocm.cuh.
//
// Written over a shuffle callable shfl(const uint32_t* reg, int src_lane), returning register *reg of lane src_lane, and the lane index,
// so tests/2_parity/test_mma_frag.cpp runs them on a simulated warp (lane by lane, against the PTX layout models; the pointer tells it
// which register) on the CPU; device code passes rocm_shfl below. Operands are raw 32-bit registers holding FP16 pairs (low half
// first), as the PTX operands, passed as arrays.
//
// Layouts (PTX ISA), g = lane / 4, t = lane % 4:
//   m16n8k16 A: a[r] = A[g + 8 (r & 1)][2t + 8 (r >> 1) + {0, 1}]   B: b[r] = B[2t + 8r + {0, 1}][g]   C: c[r] = C[g + 8 (r >> 1)][2t + (r & 1)]
//   m16n8k8  A: a[r] = A[g + 8r][2t + {0, 1}]                        B: b = B[2t + {0, 1}][g]           C: as m16n8k16 (f16: c[0] row g, c[1] row g + 8)
//   m8n8k4 (quad pair q = (lane >> 2) & 3 alone, lanes 4q..4q+3 and 4q+16..4q+19, row / column index e = (lane & 3) + 4 (lane >= 16)):
//            A: lane holds row e, k 0..3 (a0 = k 0, 1; a1 = k 2, 3)   B: lane holds column e, k 0..3   D (f32): d[i] = D[(lane & 1) + (i & 2) + 4 (lane >= 16)][(i & 1) + (lane & 2) + (i & 4)]

#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIPCC__)
	#define MMA_EMU_HD __host__ __device__ __forceinline__
#else
	#define MMA_EMU_HD inline
#endif

// FP16 bits -> float (portable: the same function on the device and in the CPU test)
MMA_EMU_HD float mma_emu_h2f( uint32_t h ) {
	const uint32_t s = ( h & 0x8000u ) << 16, e = ( h >> 10 ) & 0x1fu, m = h & 0x3ffu;
	uint32_t u;
	if ( e == 0 ) {
		if ( m == 0 ) u = s;
		else {   // subnormal: normalize
			int ee = 113;
			uint32_t mm = m;
			while ( !( mm & 0x400u ) ) { mm <<= 1; --ee; }
			u = s | ( static_cast<uint32_t>( ee ) << 23 ) | ( ( mm & 0x3ffu ) << 13 );
		}
	}
	else if ( e == 31 ) u = s | 0x7f800000u | ( m << 13 );
	else u = s | ( ( e + 112 ) << 23 ) | ( m << 13 );
	float f;
	memcpy( &f, &u, 4 );
	return f;
}
// float -> FP16 bits, round to nearest even (the f16-accumulator forms round each result once)
MMA_EMU_HD uint32_t mma_emu_f2h( float f ) {
	uint32_t u;
	memcpy( &u, &f, 4 );
	const uint32_t s = ( u >> 16 ) & 0x8000u;
	const uint32_t a = u & 0x7fffffffu;
	if ( a > 0x7f800000u ) return s | 0x7e00u;
	if ( a >= 0x477ff000u ) return s | 0x7c00u;   // rounds to >= 65520: infinity
	if ( a < 0x38800000u ) {   // below the smallest normal (2^-14): subnormal steps of 2^-24
		float af;
		memcpy( &af, &a, 4 );
		const float q = af * 16777216.0f;
		uint32_t r = static_cast<uint32_t>( q );
		const float frac = q - static_cast<float>( r );
		if ( frac > 0.5f || ( frac == 0.5f && ( r & 1u ) ) ) ++r;
		return s | r;
	}
	const uint32_t lsb = ( a >> 13 ) & 1u;
	const uint32_t r = a + 0xfffu + lsb;
	return s | ( ( ( r >> 23 ) - 112 ) << 10 ) | ( ( r >> 13 ) & 0x3ffu );
}
MMA_EMU_HD float mma_emu_lo( uint32_t v ) { return mma_emu_h2f( v & 0xffffu ); }
MMA_EMU_HD float mma_emu_hi( uint32_t v ) { return mma_emu_h2f( v >> 16 ); }
MMA_EMU_HD float mma_emu_dot2( uint32_t a, uint32_t b ) { return mma_emu_lo( a ) * mma_emu_lo( b ) + mma_emu_hi( a ) * mma_emu_hi( b ); }

// m16n8k16, FP32 accumulator: c[0..3] += A B
template <typename Shfl>
MMA_EMU_HD void mma_emu_m16n8k16_f32( const uint32_t a[4], const uint32_t b[2], float c[4], int lane, Shfl shfl ) {
	const int g = lane >> 2, t = lane & 3;
	// Row g of A sits in lanes 4g..4g+3 (a[0] / a[2]: k pairs i and i + 4), row g + 8 in a[1] / a[3] of the same lanes; column n of B
	// in lanes 4n..4n+3 (b[0] / b[1]: k pairs i and i + 4). This lane's outputs: rows g, g + 8 by columns 2t, 2t + 1
	float s[4] = { c[0], c[1], c[2], c[3] };
	for ( int i = 0; i < 4; ++i ) {
		const uint32_t r0lo = shfl( &a[0], 4 * g + i ), r0hi = shfl( &a[2], 4 * g + i );
		const uint32_t r1lo = shfl( &a[1], 4 * g + i ), r1hi = shfl( &a[3], 4 * g + i );
		const uint32_t c0lo = shfl( &b[0], 8 * t + i ), c0hi = shfl( &b[1], 8 * t + i );
		const uint32_t c1lo = shfl( &b[0], 8 * t + 4 + i ), c1hi = shfl( &b[1], 8 * t + 4 + i );
		s[0] += mma_emu_dot2( r0lo, c0lo ) + mma_emu_dot2( r0hi, c0hi );
		s[1] += mma_emu_dot2( r0lo, c1lo ) + mma_emu_dot2( r0hi, c1hi );
		s[2] += mma_emu_dot2( r1lo, c0lo ) + mma_emu_dot2( r1hi, c0hi );
		s[3] += mma_emu_dot2( r1lo, c1lo ) + mma_emu_dot2( r1hi, c1hi );
	}
	for ( int r = 0; r < 4; ++r ) c[r] = s[r];
}

// m16n8k16, FP16 accumulator (c[0] = row g, c[1] = row g + 8, each a pair of columns): products summed in FP32, rounded once into c
template <typename Shfl>
MMA_EMU_HD void mma_emu_m16n8k16_f16( const uint32_t a[4], const uint32_t b[2], uint32_t c[2], int lane, Shfl shfl ) {
	float s[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	mma_emu_m16n8k16_f32( a, b, s, lane, shfl );
	c[0] = mma_emu_f2h( mma_emu_lo( c[0] ) + s[0] ) | ( mma_emu_f2h( mma_emu_hi( c[0] ) + s[1] ) << 16 );
	c[1] = mma_emu_f2h( mma_emu_lo( c[1] ) + s[2] ) | ( mma_emu_f2h( mma_emu_hi( c[1] ) + s[3] ) << 16 );
}

// m16n8k8, FP16 accumulator (A: rows g, g + 8 at k pair t; B: k pair t of column g)
template <typename Shfl>
MMA_EMU_HD void mma_emu_m16n8k8_f16( const uint32_t a[2], const uint32_t b[1], uint32_t c[2], int lane, Shfl shfl ) {
	const int g = lane >> 2, t = lane & 3;
	float s[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	for ( int i = 0; i < 4; ++i ) {
		const uint32_t r0 = shfl( &a[0], 4 * g + i ), r1 = shfl( &a[1], 4 * g + i );
		const uint32_t c0 = shfl( &b[0], 8 * t + i ), c1 = shfl( &b[0], 8 * t + 4 + i );
		s[0] += mma_emu_dot2( r0, c0 );
		s[1] += mma_emu_dot2( r0, c1 );
		s[2] += mma_emu_dot2( r1, c0 );
		s[3] += mma_emu_dot2( r1, c1 );
	}
	c[0] = mma_emu_f2h( mma_emu_lo( c[0] ) + s[0] ) | ( mma_emu_f2h( mma_emu_hi( c[0] ) + s[1] ) << 16 );
	c[1] = mma_emu_f2h( mma_emu_lo( c[1] ) + s[2] ) | ( mma_emu_f2h( mma_emu_hi( c[1] ) + s[3] ) << 16 );
}

// m8n8k4, FP32 accumulator (row.col), each quad pair its own 8 x 8 x 4 product
template <typename Shfl>
MMA_EMU_HD void mma_emu_m8n8k4_f32( const uint32_t a[2], const uint32_t b[2], float d[8], int lane, Shfl shfl ) {
	const int q = ( lane >> 2 ) & 3, hi = lane >> 4;
	auto owner = [&]( int e ) { return 4 * q + ( e & 3 ) + 16 * ( e >> 2 ); };   // lane of the quad pair holding row / column e
	uint32_t ra[2][2], cb[4][2];
	for ( int rr = 0; rr < 2; ++rr ) {
		const int src = owner( ( lane & 1 ) + 2 * rr + 4 * hi );
		ra[rr][0] = shfl( &a[0], src );
		ra[rr][1] = shfl( &a[1], src );
	}
	for ( int cc = 0; cc < 4; ++cc ) {
		const int src = owner( ( lane & 2 ) + ( cc & 1 ) + 4 * ( cc >> 1 ) );
		cb[cc][0] = shfl( &b[0], src );
		cb[cc][1] = shfl( &b[1], src );
	}
	for ( int i = 0; i < 8; ++i ) {
		const int rr = ( i >> 1 ) & 1, cc = ( i & 1 ) + 2 * ( i >> 2 );
		d[i] += mma_emu_dot2( ra[rr][0], cb[cc][0] ) + mma_emu_dot2( ra[rr][1], cb[cc][1] );
	}
}

#if defined(__HIPCC__)
// Device shuffle for the emulations (wave32)
struct rocm_shfl {
	__device__ __forceinline__ uint32_t operator()( const uint32_t* reg, int src ) const { return __shfl( *reg, src ); }
};
#endif
