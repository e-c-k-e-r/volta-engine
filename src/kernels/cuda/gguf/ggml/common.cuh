#pragma once
// Stand-in for llama.cpp's ggml-cuda/common.cuh: only what vecdotq.cuh, convert.cuh and dequantize.cuh use (block layouts and grids from ggml-common.h, dp4a, the
// FP8 scale decoders), CUDA on SM61 and later
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

// ggml.h's type ids (the GGUF tensor types), which convert.cuh's declarations name
enum ggml_type {
	GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1, GGML_TYPE_Q4_0 = 2, GGML_TYPE_Q4_1 = 3, GGML_TYPE_Q5_0 = 6, GGML_TYPE_Q5_1 = 7, GGML_TYPE_Q8_0 = 8,
	GGML_TYPE_Q8_1 = 9, GGML_TYPE_Q2_K = 10, GGML_TYPE_Q3_K = 11, GGML_TYPE_Q4_K = 12, GGML_TYPE_Q5_K = 13, GGML_TYPE_Q6_K = 14,
	GGML_TYPE_Q8_K = 15, GGML_TYPE_IQ2_XXS = 16, GGML_TYPE_IQ2_XS = 17, GGML_TYPE_IQ3_XXS = 18, GGML_TYPE_IQ1_S = 19, GGML_TYPE_IQ4_NL = 20,
	GGML_TYPE_IQ3_S = 21, GGML_TYPE_IQ2_S = 22, GGML_TYPE_IQ4_XS = 23, GGML_TYPE_I8 = 24, GGML_TYPE_I16 = 25, GGML_TYPE_I32 = 26,
	GGML_TYPE_I64 = 27, GGML_TYPE_F64 = 28, GGML_TYPE_IQ1_M = 29, GGML_TYPE_BF16 = 30, GGML_TYPE_TQ1_0 = 34, GGML_TYPE_TQ2_0 = 35,
	GGML_TYPE_MXFP4 = 39, GGML_TYPE_NVFP4 = 40, GGML_TYPE_Q1_0 = 41, GGML_TYPE_Q2_0 = 42, GGML_TYPE_COUNT = 43,
};

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#define GGML_UNUSED( x ) (void) ( x )

static __device__ __forceinline__ int ggml_cuda_dp4a( const int a, const int b, int c ) { return __dp4a( a, b, c ); }

static __device__ __forceinline__ float ggml_cuda_e8m0_to_fp32( uint8_t x ) {
	uint32_t bits = x == 0 ? 0x00400000u : static_cast<uint32_t>( x ) << 23;
	float result;
	memcpy( &result, &bits, sizeof( float ) );
	return result;
}

static __device__ __forceinline__ float ggml_cuda_ue4m3_to_fp32( uint8_t x ) {
	// upstream's software path (no FP8 types): NaN (0x7F) decodes to 0, like the CPU implementation
	if ( x == 0 || ( x == 0x7F && x != 0xFF ) ) return 0.0f;
	const int exp = ( x >> 3 ) & 0xF;
	const int man = x & 0x7;
	float raw;
	if ( exp == 0 ) raw = ldexpf( static_cast<float>( man ), -9 );
	else raw = ldexpf( 1.0f + static_cast<float>( man ) / 8.0f, exp - 7 );
	return static_cast<float>( raw / 2 );
}

#include <cstdio>
#include <cstdlib>
#include <limits>

#define WARP_SIZE 32
#define GGML_ASSERT( x ) do { if ( !( x ) ) { std::fprintf( stderr, "GGML_ASSERT failed: %s (%s:%d)\n", #x, __FILE__, __LINE__ ); std::abort(); } } while ( 0 )

typedef void ( *dequantize_kernel_t )( const void* vx, const int64_t ib, const int iqs, float2& v );

// llama.cpp's integer division by a constant (common.cuh, unchanged)
static const uint3 init_fastdiv_values( uint64_t d_64 ) {
	GGML_ASSERT( d_64 != 0 );
	GGML_ASSERT( d_64 <= std::numeric_limits<uint32_t>::max() );
	uint32_t d = static_cast<uint32_t>( d_64 );
	uint32_t L = 0;
	while ( L < 32 && ( uint32_t{ 1 } << L ) < d ) L++;
	uint32_t mp = static_cast<uint32_t>( ( uint64_t{ 1 } << 32 ) * ( ( uint64_t{ 1 } << L ) - d ) / d + 1 );
	return make_uint3( mp, L, d );
}

static __device__ __forceinline__ uint32_t fastdiv( uint32_t n, const uint3 fastdiv_values ) {
	const uint32_t hi = __umulhi( n, fastdiv_values.x );
	return ( hi + n ) >> fastdiv_values.y;
}

static __device__ __forceinline__ uint32_t fastmodulo( uint32_t n, const uint3 fastdiv_values ) { return n - fastdiv( n, fastdiv_values ) * fastdiv_values.z; }

static __device__ __forceinline__ uint2 fast_div_modulo( uint32_t n, const uint3 fastdiv_values ) {
	const uint32_t div_val = fastdiv( n, fastdiv_values );
	return make_uint2( div_val, n - div_val * fastdiv_values.z );
}

// convert.cu asks the device whether FP16 arithmetic is available to pick its Q8_0 path: this backend builds for SM70 and later, where it is
struct ggml_cuda_device_info { struct { int cc = 700; } devices[64]; };
static const ggml_cuda_device_info& ggml_cuda_info() { static ggml_cuda_device_info info; return info; }
static int ggml_cuda_get_device() { int d = 0; cudaGetDevice( &d ); return d; }
static bool fp16_available( const int ) { return true; }
