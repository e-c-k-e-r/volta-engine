#pragma once
// SM_80+ tensor-core helpers for kernels whose SM_70 form uses mma.m8n8k4 (not native on SM_80+: ptxas expands each into ~80 FFMA /
// SHFL instructions): mma.m16n8k16 with FP32 accumulation and its operand loads from shared memory, by the maps of mma_frag.h
#include <cuda_fp16.h>
#include <cstdint>
#include "mma_frag.h"
#if defined(USE_ROCM)
	#include "../rocm/mma_emu.h"
#endif

// ROCm takes these paths too, with the m16n8k16 emulated (mma_emu.h)
#if defined(USE_ROCM) || ( defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) )
	#define MMA_SM80 1
#else
	#define MMA_SM80 0
#endif

#if MMA_SM80
// c (4 floats, frag_c layout) += A (16 x 16) B (16 x 8)
__device__ __forceinline__ void mma16816( float* c, const uint32_t a[4], const uint32_t b[2] ) {
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

// A fragment of the 16 x 16 tile at p: row-major halves, ld halves per row (even, so every pair is 4-byte aligned)
__device__ __forceinline__ void lds_frag_a( const half* p, int ld, uint32_t a[4] ) {
	const int lane = threadIdx.x & 31;
	#pragma unroll
	for ( int r = 0; r < 4; ++r ) a[r] = *reinterpret_cast<const uint32_t*>( p + frag_a_row( r, lane ) * ld + 2 * frag_a_kpair( r, lane ) );
}

// B fragment of the 16 (k) x 8 (n) tile at p, stored n-major: row n holds its 16 k contiguously, ld halves per row (even)
__device__ __forceinline__ void lds_frag_b( const half* p, int ld, uint32_t b[2] ) {
	const int lane = threadIdx.x & 31;
	#pragma unroll
	for ( int r = 0; r < 2; ++r ) b[r] = *reinterpret_cast<const uint32_t*>( p + frag_b_col( lane ) * ld + 2 * frag_b_kpair( r, lane ) );
}
#endif
