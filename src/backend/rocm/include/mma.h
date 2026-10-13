#pragma once
// ROCM configuration: the nvcuda::wmma subset the kernels use (16 x 16 x 16, half inputs, float accumulator), on RDNA's WMMA.
//
// gfx11 (RDNA3) v_wmma_f32_16x16x16_f16, wave32, with the operands passed (B, A) as upstream exllamav3 does (rocm/rdna_wmma.cuh,
// validated there on hardware): lane L holds row L % 16 of A (all 16 k) and column L % 16 of B (all 16 k), lanes 16-31 repeating
// lanes 0-15; the accumulator lane L holds row L % 16, columns 2i + L / 16 (i = 0..7). gfx12 takes 8 halves per lane and lays the
// accumulator out as columns i + 8 * (L / 16): converted around the instruction (upstream's form, also hardware-validated). Other
// targets compute the same layout with shuffles.
//
// The accumulator's element order differs from CUDA's (which the CUDA API leaves unspecified anyway): code that reads fragment
// elements directly (.x[i]) instead of through load / store has a layout assumption to port (only the MiniMax-H3 attention does).
#include "../compat.h"

namespace nvcuda {
namespace wmma {

struct matrix_a {};
struct matrix_b {};
struct accumulator {};
struct row_major {};
struct col_major {};
enum layout_t { mem_row_major, mem_col_major };

typedef _Float16 rocm_wmma_h16 __attribute__( ( ext_vector_type( 16 ) ) );
typedef _Float16 rocm_wmma_h8 __attribute__( ( ext_vector_type( 8 ) ) );
typedef float rocm_wmma_f8 __attribute__( ( ext_vector_type( 8 ) ) );

template <typename Use, int M, int N, int K, typename T, typename Layout = void>
struct fragment;

template <typename Layout>
struct fragment<matrix_a, 16, 16, 16, half, Layout> {
	static constexpr int num_elements = 16;
	rocm_wmma_h16 v;
};
template <typename Layout>
struct fragment<matrix_b, 16, 16, 16, half, Layout> {
	static constexpr int num_elements = 16;
	rocm_wmma_h16 v;
};
template <>
struct fragment<accumulator, 16, 16, 16, float, void> {
	static constexpr int num_elements = 8;
	union {
		rocm_wmma_f8 v;
		float x[8];
	};
};

// A: lane L takes row L % 16 (row_major: contiguous)
__device__ __forceinline__ void load_matrix_sync( fragment<matrix_a, 16, 16, 16, half, row_major>& f, const half* p, unsigned ldm ) {
	const _Float16* s = reinterpret_cast<const _Float16*>( p ) + ( threadIdx.x & 15 ) * ldm;
	#pragma unroll
	for ( int k = 0; k < 16; ++k ) f.v[k] = s[k];
}
__device__ __forceinline__ void load_matrix_sync( fragment<matrix_a, 16, 16, 16, half, col_major>& f, const half* p, unsigned ldm ) {
	const _Float16* s = reinterpret_cast<const _Float16*>( p ) + ( threadIdx.x & 15 );
	#pragma unroll
	for ( int k = 0; k < 16; ++k ) f.v[k] = s[k * ldm];
}
// B: lane L takes column L % 16 (col_major: contiguous)
__device__ __forceinline__ void load_matrix_sync( fragment<matrix_b, 16, 16, 16, half, col_major>& f, const half* p, unsigned ldm ) {
	const _Float16* s = reinterpret_cast<const _Float16*>( p ) + ( threadIdx.x & 15 ) * ldm;
	#pragma unroll
	for ( int k = 0; k < 16; ++k ) f.v[k] = s[k];
}
__device__ __forceinline__ void load_matrix_sync( fragment<matrix_b, 16, 16, 16, half, row_major>& f, const half* p, unsigned ldm ) {
	const _Float16* s = reinterpret_cast<const _Float16*>( p ) + ( threadIdx.x & 15 );
	#pragma unroll
	for ( int k = 0; k < 16; ++k ) f.v[k] = s[k * ldm];
}

// Accumulator: lane L holds row L % 16, columns 2i + L / 16
__device__ __forceinline__ void load_matrix_sync( fragment<accumulator, 16, 16, 16, float>& f, const float* p, unsigned ldm, layout_t layout ) {
	const int r = threadIdx.x & 15, h = ( threadIdx.x >> 4 ) & 1;
	#pragma unroll
	for ( int i = 0; i < 8; ++i ) f.x[i] = layout == mem_row_major ? p[r * ldm + 2 * i + h] : p[( 2 * i + h ) * ldm + r];
}
__device__ __forceinline__ void store_matrix_sync( float* p, const fragment<accumulator, 16, 16, 16, float>& f, unsigned ldm, layout_t layout ) {
	const int r = threadIdx.x & 15, h = ( threadIdx.x >> 4 ) & 1;
	#pragma unroll
	for ( int i = 0; i < 8; ++i ) {
		if ( layout == mem_row_major ) p[r * ldm + 2 * i + h] = f.x[i];
		else p[( 2 * i + h ) * ldm + r] = f.x[i];
	}
}
__device__ __forceinline__ void fill_fragment( fragment<accumulator, 16, 16, 16, float>& f, float value ) {
	#pragma unroll
	for ( int i = 0; i < 8; ++i ) f.x[i] = value;
}

// d = a b + c
template <typename LA, typename LB>
__device__ __forceinline__ void mma_sync( fragment<accumulator, 16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, half, LA>& a,
	const fragment<matrix_b, 16, 16, 16, half, LB>& b, const fragment<accumulator, 16, 16, 16, float>& c ) {
#if defined(__GFX11__)
	d.v = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32( b.v, a.v, c.v );
#elif defined(__GFX12__)
	// 8 halves per lane (k = 8 * (L / 16) ..), accumulator columns i + 8 * (L / 16): four xor-16 swaps each way (upstream's conversion)
	const bool h = ( threadIdx.x & 16 ) != 0;
	const rocm_wmma_h8 a8 = h ? a.v.hi : a.v.lo, b8 = h ? b.v.hi : b.v.lo;
	rocm_wmma_f8 g;
	#pragma unroll
	for ( int t = 0; t < 4; ++t ) {
		const float r = __shfl_xor( h ? c.v[t] : c.v[4 + t], 16 );
		g[2 * t] = h ? r : c.v[t];
		g[2 * t + 1] = h ? c.v[4 + t] : r;
	}
	g = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12( b8, a8, g );
	#pragma unroll
	for ( int t = 0; t < 4; ++t ) {
		const float r = __shfl_xor( h ? g[2 * t] : g[2 * t + 1], 16 );
		d.v[t] = h ? r : g[2 * t];
		d.v[4 + t] = h ? g[2 * t + 1] : r;
	}
#else
	// No WMMA (RDNA1/2, or the host pass): the same layout from shuffles. Lane L computes row L % 16 against the columns 2i + L / 16,
	// whose 16 k it reads from lane (2i + L / 16) two halves at a time
	const int h = ( threadIdx.x >> 4 ) & 1;
	float out[8];
	#pragma unroll
	for ( int i = 0; i < 8; ++i ) {
		const int src = 2 * i + h;
		float s = c.x[i];
		#pragma unroll
		for ( int k = 0; k < 16; k += 2 ) {
			uint32_t pair;
			_Float16 own[2] = { b.v[k], b.v[k + 1] };
			__builtin_memcpy( &pair, own, 4 );
			pair = __shfl( pair, src );
			_Float16 bk[2];
			__builtin_memcpy( bk, &pair, 4 );
			s += static_cast<float>( a.v[k] ) * static_cast<float>( bk[0] ) + static_cast<float>( a.v[k + 1] ) * static_cast<float>( bk[1] );
		}
		out[i] = s;
	}
	#pragma unroll
	for ( int i = 0; i < 8; ++i ) d.x[i] = out[i];
#endif
}

} // namespace wmma
} // namespace nvcuda
