#pragma once

// Shared by the GGUF kernels (gguf_linear.cu, gguf_moe.cu): the block types' traits around llama.cpp's vec_dot functions, and the q8_1
// activation quantizer they read.
#include "engine/modules/gguf.h"
#include "ggml/vecdotq.cuh"
#include <cuda_fp16.h>

namespace engine::modules {

// The block geometry table of gguf_types.cpp against the vendored layouts
static_assert( sizeof( block_q8_0 ) == 34 && sizeof( block_q4_K ) == 144 && sizeof( block_q5_K ) == 176 && sizeof( block_q6_K ) == 210, "block sizes" );
static_assert( sizeof( block_q2_K ) == 84 && sizeof( block_q3_K ) == 110 && sizeof( block_iq4_xs ) == 136 && sizeof( block_iq4_nl ) == 18, "block sizes" );
static_assert( sizeof( block_iq2_xxs ) == 66 && sizeof( block_iq2_xs ) == 74 && sizeof( block_iq2_s ) == 82 && sizeof( block_iq3_xxs ) == 98, "block sizes" );
static_assert( sizeof( block_iq3_s ) == 110 && sizeof( block_iq1_s ) == 50 && sizeof( block_iq1_m ) == 56 && sizeof( block_q8_1 ) == 36, "block sizes" );
static_assert( sizeof( block_q4_0 ) == 18 && sizeof( block_q4_1 ) == 20 && sizeof( block_q5_0 ) == 22 && sizeof( block_q5_1 ) == 24, "block sizes" );
static_assert( sizeof( block_q2_0 ) == 18, "block sizes" );

// Values per block, quant ints per block, ints per vec_dot call (llama.cpp's type traits and MMVQ VDR) and bytes per block of each type
template <GgmlType T> struct Traits;
#define GGUF_TRAITS( T, B, QK, QI, VDR, FN ) \
	template <> struct Traits<GgmlType::T> { \
		static constexpr int qk = QK, qi = QI, vdr = VDR, bytes = sizeof( B ); \
		static __device__ __forceinline__ float dot( const void* x, const block_q8_1* y, int kbx, int kqs ) { return FN( x, y, kbx, kqs ); } \
	};
GGUF_TRAITS( Q4_0, block_q4_0, QK4_0, QI4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1 )
GGUF_TRAITS( Q4_1, block_q4_1, QK4_1, QI4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1 )
GGUF_TRAITS( Q5_0, block_q5_0, QK5_0, QI5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1 )
GGUF_TRAITS( Q5_1, block_q5_1, QK5_1, QI5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1 )
GGUF_TRAITS( Q8_0, block_q8_0, QK8_0, QI8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1 )
GGUF_TRAITS( Q2_0, block_q2_0, QK2_0, QI2_0, VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1 )
GGUF_TRAITS( Q2_K, block_q2_K, QK_K, QI2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1 )
GGUF_TRAITS( Q3_K, block_q3_K, QK_K, QI3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1 )
GGUF_TRAITS( Q4_K, block_q4_K, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1 )
GGUF_TRAITS( Q5_K, block_q5_K, QK_K, QI5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1 )
GGUF_TRAITS( Q6_K, block_q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1 )
GGUF_TRAITS( IQ2_XXS, block_iq2_xxs, QK_K, QI2_XXS, VDR_IQ2_XXS_Q8_1_MMVQ, vec_dot_iq2_xxs_q8_1 )
GGUF_TRAITS( IQ2_XS, block_iq2_xs, QK_K, QI2_XS, VDR_IQ2_XS_Q8_1_MMVQ, vec_dot_iq2_xs_q8_1 )
GGUF_TRAITS( IQ2_S, block_iq2_s, QK_K, QI2_S, VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1 )
GGUF_TRAITS( IQ3_XXS, block_iq3_xxs, QK_K, QI3_XXS, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1 )
GGUF_TRAITS( IQ3_S, block_iq3_s, QK_K, QI3_S, VDR_IQ3_S_Q8_1_MMVQ, vec_dot_iq3_s_q8_1 )
GGUF_TRAITS( IQ1_S, block_iq1_s, QK_K, QI1_S, VDR_IQ1_S_Q8_1_MMVQ, vec_dot_iq1_s_q8_1 )
GGUF_TRAITS( IQ1_M, block_iq1_m, QK_K, QI1_M, VDR_IQ1_M_Q8_1_MMVQ, vec_dot_iq1_m_q8_1 )
GGUF_TRAITS( IQ4_NL, block_iq4_nl, QK4_NL, QI4_NL, VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1 )
GGUF_TRAITS( IQ4_XS, block_iq4_xs, QK_K, QI4_XS, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1 )
#undef GGUF_TRAITS

// A warp asks the L2 for all of a weight row's lines at once (a line a lane), so the row's later blocks do not wait on DRAM one vec_dot
// step after another
static __device__ __forceinline__ void prefetch_row( const void* row, int bytes, int lane ) {
#if defined(USE_ROCM)
	( void ) row; ( void ) bytes; ( void ) lane;   // only a hint: RDNA3 has no L2 prefetch instruction
#else
	for ( int o = lane * 128; o < bytes; o += 32 * 128 ) asm volatile( "prefetch.global.L2 [%0];" :: "l"( static_cast<const char*>( row ) + o ) );
#endif
}

// The activation column a weight column reads: tiled [v][head][dim] against grouped [head][v][dim]
static __device__ __forceinline__ int tiled_source( int i, GgufColTiling t, int k ) {
	if ( t.heads == 0 ) return i;
	const int per_v = t.heads * t.dim, v = i / per_v, head = i % per_v / t.dim, d = i % t.dim;
	return ( head * ( k / per_v ) + v ) * t.dim + d;
}

// x [m, k] FP16 -> y [m][k / 32] q8_1 (llama.cpp's quantize_q8_1: scale amax / 127, the block sum of the unquantized values), in the
// weight's column order
// Values i of row `row` (a thread each, whole warps on whole q8_1 blocks)
static __device__ __forceinline__ void quantize_q8_1_at( const half* __restrict__ x, block_q8_1* __restrict__ y, int k, GgufColTiling t, int row, int i ) {
	if ( i >= k ) return;
	const float xi = __half2float( x[static_cast<size_t>( row ) * k + tiled_source( i, t, k )] );
	float amax = fabsf( xi ), sum = xi;
	#pragma unroll
	for ( int o = 16; o > 0; o >>= 1 ) {
		amax = fmaxf( amax, __shfl_xor_sync( 0xffffffff, amax, o ) );
		sum += __shfl_xor_sync( 0xffffffff, sum, o );
	}
	const float d = amax / 127.0f;
	const int8_t q = amax == 0.0f ? 0 : static_cast<int8_t>( roundf( xi / d ) );
	block_q8_1& b = y[static_cast<size_t>( row ) * ( k / QK8_1 ) + i / QK8_1];
	b.qs[i % QK8_1] = q;
	if ( i % QK8_1 == 0 ) b.ds = make_half2( d, sum );
}

static __global__ void quantize_q8_1_kernel( const half* __restrict__ x, block_q8_1* __restrict__ y, int k, GgufColTiling t ) {
	quantize_q8_1_at( x, y, k, t, blockIdx.y, blockIdx.x * blockDim.x + threadIdx.x );
}

} // namespace engine::modules
