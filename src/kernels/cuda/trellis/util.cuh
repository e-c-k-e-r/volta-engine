#pragma once

#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "engine/config.h"

typedef struct __align__( 8 ) half4 {
	half2 x;
	half2 y;
	__device__ half4() = default;
	__device__ half4( half2 x_, half2 y_ ) : x( x_ ), y( y_ ) {}
	__device__ half4( half h0, half h1, half h2, half h3 ) :
		 x( __halves2half2( h0, h1 ) ),
		 y( __halves2half2( h2, h3 ) ) {}
} half4;

typedef struct __align__( 8 ) bfloat164 {
	__nv_bfloat162 x;
	__nv_bfloat162 y;
	__device__ bfloat164() = default;
	__device__ bfloat164( __nv_bfloat162 x_, __nv_bfloat162 y_ ) : x( x_ ), y( y_ ) {}
	__device__ bfloat164( __nv_bfloat16 b0, __nv_bfloat16 b1, __nv_bfloat16 b2, __nv_bfloat16 b3 ) :
		x( __halves2bfloat162( b0, b1 ) ),
		y( __halves2bfloat162( b2, b3 ) ) {}
} bfloat164;

typedef struct __align__( 16 ) half8 {
	half2 x;
	half2 y;
	half2 z;
	half2 w;
	__device__ half8() = default;
	__device__ half8( half2 x_, half2 y_, half2 z_, half2 w_ ) : x( x_ ), y( y_ ), z( z_ ), w( w_ ) {}
	__device__ half8( half h0, half h1, half h2, half h3, half h4, half h5, half h6, half h7 ) :
		 x( __halves2half2( h0, h1 ) ),
		 y( __halves2half2( h2, h3 ) ),
		 z( __halves2half2( h4, h5 ) ),
		 w( __halves2half2( h6, h7 ) ) {}
} half8;

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CEIL_DIVIDE(x, size) (((x) + (size) - 1) / (size))

union half2_uint32 {
	uint32_t as_uint32;
	half2 as_half2;
	__device__ half2_uint32( uint32_t val ) : as_uint32( val ) {}
	__device__ half2_uint32( half2 val ) : as_half2( val ) {}
	__device__ half2_uint32() : as_uint32( 0 ) {}
};

union half_uint16 {
	uint16_t as_uint16;
	half as_half;
	__device__ half_uint16( uint16_t val ) : as_uint16( val ) {}
	__device__ half_uint16( half val ) : as_half( val ) {}
	__device__ half_uint16() : as_uint16( 0 ) {}
};

