#pragma once
// Shared NVFP4 device helpers: FP8 E4M3 scale decode, 8-code dequant to FP16 pairs, Volta m8n8k4 MMA, and the SM_80+ m16n8k16 step
// (fragment maps and their CPU check: nvfp4_frag.h, tests/2_parity/test_nvfp4_frag.cpp)
#include <cuda_fp16.h>
#include "nvfp4_frag.h"
#if defined(USE_ROCM)
	#include "../mma_emu.h"
#endif
#define DEV_INLINE __device__ __forceinline__

// Token rows the MoE kernels stage per chunk (their input rows, K halves each, sit in shared memory): 8 on CUDA; on ROCm as many as fit
// the 64 KB a workgroup gets (6 at K = 5120)
#if defined(USE_ROCM)
	__host__ __device__ inline int nvfp4_moe_chunk_rows( int K ) { const int r = 65536 / ( 2 * K ); return r < 1 ? 1 : ( r > 8 ? 8 : r ); }
#else
	#define nvfp4_moe_chunk_rows( K ) 8
#endif

// SM_80+ has no native m8n8k4 (ptxas expands one into ~80 FFMA / SHFL instructions): the skinny and MoE kernels take the swapped
// m16n8k16 step there. Per device pass
#if defined(USE_ROCM) || ( defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) )
	#define NVFP4_MMA_SM80 1
#else
	#define NVFP4_MMA_SM80 0
#endif

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

// Token row this lane supplies to the step, and accumulator register i of lane -> (token row 0..7, column 0..31 of the tile)
DEV_INLINE int nvfp4_x_row( int lane ) { return NVFP4_MMA_SM80 ? ( lane >> 2 ) : ( lane & 3 ) + ( ( lane & 16 ) ? 4 : 0 ); }
DEV_INLINE int nvfp4_acc_row( int i, int lane ) { return NVFP4_MMA_SM80 ? nvfp4_acc_row_sm80( i, lane ) : nvfp4_acc_row_sm70( i, lane ); }
DEV_INLINE int nvfp4_acc_col( int i, int lane ) { return NVFP4_MMA_SM80 ? nvfp4_acc_col_sm80( i, lane ) : nvfp4_acc_col_sm70( i, lane ); }

#if NVFP4_MMA_SM80
DEV_INLINE void mma_16816_f32( float* c, const unsigned a[4], const unsigned b[2] ) {
#if defined(USE_ROCM)
	mma_emu_m16n8k16_f32( a, b, c, threadIdx.x & 31, rocm_shfl() );
#else
	asm volatile (
		"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
		"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
		: "+f"( c[0] ), "+f"( c[1] ), "+f"( c[2] ), "+f"( c[3] )
		: "r"( a[0] ), "r"( a[1] ), "r"( a[2] ), "r"( a[3] ), "r"( b[0] ), "r"( b[1] )
	);
#endif
}

// Weight A fragments of one k16 group for both MMAs. q / s point at this lane's slot of the group (as the SM_70 loop reads them); the
// lane reads the four columns it needs (16j + g + 8h) from their slots, which the warp's other loads of the group cover, and decodes
// only the k pairs it feeds (t and t + 4). The same FP16 operations as dequant8_tm, so the weights are bit-identical
template <bool STREAM_SCALES>
DEV_INLINE void nvfp4_wfrag( const uint2* q, const uint8_t* s, half2 gm, unsigned a[2][4] ) {
	const int lane = threadIdx.x & 31, t = lane & 3;
	q -= lane;
	s -= lane;
	#pragma unroll
	for ( int j = 0; j < 2; ++j ) {
		#pragma unroll
		for ( int h = 0; h < 2; ++h ) {
			const int slot = nvfp4_col_slot( 16 * j + ( lane >> 2 ) + 8 * h );
			const uint2 w = __ldcs( q + slot );
			const half2 sc = __hmul2( fp8e4m3_to_half2( STREAM_SCALES ? __ldcs( s + slot ) : __ldg( s + slot ) ), gm );
			unsigned lo = nvfp4_pair_bits( w.x, t ), hi = nvfp4_pair_bits( w.y, t );
			const half2 vlo = __hmul2( *reinterpret_cast<half2*>( &lo ), sc ), vhi = __hmul2( *reinterpret_cast<half2*>( &hi ), sc );
			a[j][h] = *reinterpret_cast<const unsigned*>( &vlo );
			a[j][h + 2] = *reinterpret_cast<const unsigned*>( &vhi );
		}
	}
}

// Token B fragment of one k16 group: xr = this lane's token row (nvfp4_x_row) at the group's first k, or nullptr past the last row
DEV_INLINE void nvfp4_xfrag( const half* xr, unsigned b[2] ) {
	const int t = threadIdx.x & 3;
	b[0] = xr != nullptr ? *reinterpret_cast<const unsigned*>( xr + 2 * t ) : 0u;
	b[1] = xr != nullptr ? *reinterpret_cast<const unsigned*>( xr + 2 * t + 8 ) : 0u;
}

// c (registers 4j..4j+3 = MMA j) += the 8 token rows x the tile's 32 columns over one k16 group
DEV_INLINE void nvfp4_step( float c[8], const unsigned a[2][4], const unsigned b[2] ) {
	mma_16816_f32( c, a[0], b );
	mma_16816_f32( c + 4, a[1], b );
}
#endif
