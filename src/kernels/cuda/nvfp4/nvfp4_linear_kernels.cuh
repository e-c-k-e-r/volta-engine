#pragma once
// The NVFP4 W4A16 linear kernels over the prepacked fragment layout (src/loader/nvfp4.cpp): the skinny split-K kernels for decode rows, the
// fused dequantize + tensor-core GEMM for mid-size chunks and the dequantization into FP16 for the GEMM of larger ones. skinny_kernels.cu
// compiles them as kernels; nvfp4_vk.cu, the PTX of the Vulkan backend's CUDA launches, defines NVFP4_KERNEL / NVFP4_KERNEL_BOUNDS as device
// functions under extern "C" entries of its own.
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

#include "nvfp4_common.cuh"

#ifndef NVFP4_KERNEL
	#define NVFP4_KERNEL __global__
	#define NVFP4_KERNEL_BOUNDS( threads ) __global__ __launch_bounds__( threads )
#endif

template <int MT>
NVFP4_KERNEL void skinny_nvfp4_qpn(
	const uint8_t* __restrict__ qcodes,
	const uint8_t* __restrict__ qscales,
	const half* __restrict__ x,
	half* __restrict__ y,
	int N,
	int K,
	int M,
	float gscale
) {
	constexpr int WARPS = 4;
	__shared__ float cs[WARPS][MT * 256];

	const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
	const int tile = blockIdx.x;
	const int qp = ( lane >> 2 ) & 3;
	const int r = ( lane & 3 ) + ( ( lane & 16 ) ? 4 : 0 );
	const int G = K >> 4, Gq = G / WARPS;
	const int g0 = warp * Gq;
	const uint2* cb = reinterpret_cast<const uint2*>( qcodes ) +
					(size_t) tile * G * 32 + lane;
	const uint8_t* sb = qscales + (size_t) tile * G * 32 + lane;

	const half2 gm2 = __float2half2_rn( gscale * 16384.f );
	float c[MT][8];
	#pragma unroll
	for ( int t = 0; t < MT; t++ ) {
		#pragma unroll
		for ( int i = 0; i < 8; i++ ) c[t][i] = 0.f;
	}

#if NVFP4_MMA_SM80
	const int xr0 = nvfp4_x_row( lane );
	#pragma unroll 4
	for ( int g = g0; g < g0 + Gq; g++ ) {
		unsigned aw[2][4];
		nvfp4_wfrag<false>( cb + (size_t) g * 32, sb + (size_t) g * 32, gm2, aw );
		#pragma unroll
		for ( int t = 0; t < MT; t++ ) {
			const int ar = t * 8 + xr0;
			unsigned bx[2];
			nvfp4_xfrag( ar < M ? x + (size_t) ar * K + g * 16 : nullptr, bx );
			nvfp4_step( c[t], aw, bx );
		}
	}
	(void) qp;
	(void) r;

	#pragma unroll
	for ( int t = 0; t < MT; t++ ) {
		#pragma unroll
		for ( int i = 0; i < 8; i++ ) cs[warp][( t * 8 + nvfp4_acc_row( i, lane ) ) * 32 + nvfp4_acc_col( i, lane )] = c[t][i];
	}
#else
	#pragma unroll 4
	for ( int g = g0; g < g0 + Gq; g++ ) {
		const uint2 q2 = __ldcs( cb + (size_t) g * 32 );
		const half2 sc2 = __hmul2( fp8e4m3_to_half2( __ldg( sb + (size_t) g * 32 ) ), gm2 );
		half2 b[8];
		dequant8_tm( q2.x, sc2, b + 0 );
		dequant8_tm( q2.y, sc2, b + 4 );
		const unsigned* B = reinterpret_cast<const unsigned*>( b );

		#pragma unroll
		for ( int t = 0; t < MT; t++ ) {
			const int ar = t * 8 + r;
			uint4 a01 = make_uint4( 0, 0, 0, 0 ), a23 = make_uint4( 0, 0, 0, 0 );
			if ( ar < M ) {
				const half* xrow = x + (size_t) ar * K;
				a01 = *reinterpret_cast<const uint4*>( xrow + g * 16 );
				a23 = *reinterpret_cast<const uint4*>( xrow + g * 16 + 8 );
			}
			const unsigned* A0 = reinterpret_cast<const unsigned*>( &a01 );
			const unsigned* A1 = reinterpret_cast<const unsigned*>( &a23 );
			MMA_8N8K4( c[t], A0[0], A0[1], B[0], B[1] );
			MMA_8N8K4( c[t], A0[2], A0[3], B[2], B[3] );
			MMA_8N8K4( c[t], A1[0], A1[1], B[4], B[5] );
			MMA_8N8K4( c[t], A1[2], A1[3], B[6], B[7] );
		}
	}

	#pragma unroll
	for ( int t = 0; t < MT; t++ ) {
		#pragma unroll
		for ( int i = 0; i < 8; i++ ) {
			const int row = ( i & 2 ) | ( ( lane & 16 ) ? 4 : 0 ) | ( lane & 1 );
			const int col = ( i & 1 ) | ( ( ( lane >> 1 ) & 1 ) << 1 ) | ( ( i >> 2 ) << 2 );
			cs[warp][( t * 8 + row ) * 32 + qp * 8 + col] = c[t][i];
		}
	}
#endif
	__syncthreads();

	for ( int e = threadIdx.x; e < MT * 256; e += blockDim.x ) {
		const float v = cs[0][e] + cs[1][e] + cs[2][e] + cs[3][e];
		const int row = e >> 5, col = e & 31;
		if ( row < M ) {
			y[(size_t) row * N + (size_t) tile * 32 + col] = __float2half( v );
		}
	}
}

// Split-K skinny kernel: SPLITK warps share one 32-column tile, each a slice of K; MT row tiles of 8 (up to 8 * MT rows) reuse every decoded weight
// fragment, and NACC accumulators per row tile keep the 1-tile Volta MMA chains independent
template <int SPLITK, int NACC, int MT = 1>
NVFP4_KERNEL void skinny_nvfp4_qpn2(
	const uint8_t* __restrict__ bcodes,
	const uint8_t* __restrict__ bscales,
	const half* __restrict__ x,
	half* __restrict__ y,
	int N,
	int K,
	int M,
	float gscale
) {
	__shared__ float cs[SPLITK > 1 ? SPLITK : 1][SPLITK > 1 ? MT * 256 : 1];

	const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
	const int tile = blockIdx.x;
	const int qp = ( lane >> 2 ) & 3;
	const int r = ( lane & 3 ) + ( ( lane & 16 ) ? 4 : 0 );
	const int G = K >> 4, Gq = G / SPLITK;
	const int g0 = warp * Gq;
	const uint2* cb = reinterpret_cast<const uint2*>( bcodes ) +
					(size_t) tile * G * 32 + lane;
	const uint8_t* sb = bscales + (size_t) tile * G * 32 + lane;
	const half2 gm2 = __float2half2_rn( gscale * 16384.f );
	float c[MT][NACC][8];
	#pragma unroll
	for ( int t = 0; t < MT; t++ ) {
		#pragma unroll
		for ( int a = 0; a < NACC; a++ ) {
			#pragma unroll
			for ( int i = 0; i < 8; i++ ) c[t][a][i] = 0.f;
		}
	}

#if NVFP4_MMA_SM80
	const int xr0 = nvfp4_x_row( lane );
	const half* xrow[MT];
	#pragma unroll
	for ( int t = 0; t < MT; t++ ) xrow[t] = ( t * 8 + xr0 < M ) ? ( x + (size_t) ( t * 8 + xr0 ) * K ) : nullptr;
	(void) qp;
	(void) r;

	// One step per k16 group into c[t][0] (its two MMAs are already independent chains); the NACC split below then adds zeros
	#pragma unroll 4
	for ( int g = g0; g < g0 + Gq; g++ ) {
		unsigned aw[2][4];
		nvfp4_wfrag<true>( cb + (size_t) g * 32, sb + (size_t) g * 32, gm2, aw );
		#pragma unroll
		for ( int t = 0; t < MT; t++ ) {
			unsigned bx[2];
			nvfp4_xfrag( xrow[t] != nullptr ? xrow[t] + g * 16 : nullptr, bx );
			nvfp4_step( c[t][0], aw, bx );
		}
	}
#else
	const half* xrow[MT];
	#pragma unroll
	for ( int t = 0; t < MT; t++ ) xrow[t] = ( t * 8 + r < M ) ? ( x + (size_t) ( t * 8 + r ) * K ) : nullptr;

	#pragma unroll 4
	for ( int g = g0; g < g0 + Gq; g++ ) {
		const uint2 q2 = __ldcs( cb + (size_t) g * 32 );
		const half2 sc2 = __hmul2( fp8e4m3_to_half2( __ldcs( sb + (size_t) g * 32 ) ), gm2 );
		half2 b[8];
		dequant8_tm( q2.x, sc2, b + 0 );
		dequant8_tm( q2.y, sc2, b + 4 );
		const unsigned* B = reinterpret_cast<const unsigned*>( b );
		#pragma unroll
		for ( int t = 0; t < MT; t++ ) {
			uint4 a01 = make_uint4( 0, 0, 0, 0 ), a23 = make_uint4( 0, 0, 0, 0 );
			if ( xrow[t] != nullptr ) {
				a01 = *reinterpret_cast<const uint4*>( xrow[t] + g * 16 );
				a23 = *reinterpret_cast<const uint4*>( xrow[t] + g * 16 + 8 );
			}
			const unsigned* A0 = reinterpret_cast<const unsigned*>( &a01 );
			const unsigned* A1 = reinterpret_cast<const unsigned*>( &a23 );
			MMA_8N8K4( c[t][0], A0[0], A0[1], B[0], B[1] );
			MMA_8N8K4( c[t][1 % NACC], A0[2], A0[3], B[2], B[3] );
			MMA_8N8K4( c[t][2 % NACC], A1[0], A1[1], B[4], B[5] );
			MMA_8N8K4( c[t][3 % NACC], A1[2], A1[3], B[6], B[7] );
		}
	}
#endif

	#pragma unroll
	for ( int t = 0; t < MT; t++ ) {
		#pragma unroll
		for ( int a = 1; a < NACC; a++ ) {
			#pragma unroll
			for ( int i = 0; i < 8; i++ ) c[t][0][i] += c[t][a][i];
		}
	}

#if NVFP4_MMA_SM80
	if ( SPLITK == 1 ) {
		#pragma unroll
		for ( int t = 0; t < MT; t++ ) {
			#pragma unroll
			for ( int i = 0; i < 8; i++ ) {
				const int row = t * 8 + nvfp4_acc_row( i, lane );
				if ( row < M ) y[(size_t) row * N + (size_t) tile * 32 + nvfp4_acc_col( i, lane )] = __float2half( c[t][0][i] );
			}
		}
		return;
	}

	#pragma unroll
	for ( int t = 0; t < MT; t++ ) {
		#pragma unroll
		for ( int i = 0; i < 8; i++ ) cs[warp][( t * 8 + nvfp4_acc_row( i, lane ) ) * 32 + nvfp4_acc_col( i, lane )] = c[t][0][i];
	}
#else
	if ( SPLITK == 1 ) {
		#pragma unroll
		for ( int t = 0; t < MT; t++ ) {
			#pragma unroll
			for ( int i = 0; i < 8; i++ ) {
				const int row = t * 8 + ( ( i & 2 ) | ( ( lane & 16 ) ? 4 : 0 ) | ( lane & 1 ) );
				const int col = ( i & 1 ) | ( ( ( lane >> 1 ) & 1 ) << 1 ) | ( ( i >> 2 ) << 2 );
				if ( row < M ) {
					y[(size_t) row * N + (size_t) tile * 32 + qp * 8 + col] = __float2half( c[t][0][i] );
				}
			}
		}
		return;
	}

	#pragma unroll
	for ( int t = 0; t < MT; t++ ) {
		#pragma unroll
		for ( int i = 0; i < 8; i++ ) {
			const int row = t * 8 + ( ( i & 2 ) | ( ( lane & 16 ) ? 4 : 0 ) | ( lane & 1 ) );
			const int col = ( i & 1 ) | ( ( ( lane >> 1 ) & 1 ) << 1 ) | ( ( i >> 2 ) << 2 );
			cs[warp][row * 32 + qp * 8 + col] = c[t][0][i];
		}
	}
#endif
	__syncthreads();

	for ( int e = threadIdx.x; e < MT * 256; e += blockDim.x ) {
		float v = 0.f;
		#pragma unroll
		for ( int w = 0; w < SPLITK; w++ ) v += cs[w][e];
		const int row = e >> 5, col = e & 31;
		if ( row < M ) {
			y[(size_t) row * N + (size_t) tile * 32 + col] = __float2half( v );
		}
	}
}


// Dequantize the prepacked weight into dense FP16 W[N, K] (row n = output column n): a block per (32-column tile, DQ_GROUPS groups), a thread
// per (group, lane) decoding into shared memory, then each row's 16 DQ_GROUPS weights stored as one 128-byte run (a lane's 32-byte piece per row
// scatters the stores over 32 rows, at about half the write bandwidth). K is a multiple of 64: whole runs.
constexpr int DQ_GROUPS = 4;

NVFP4_KERNEL_BOUNDS( DQ_GROUPS * 32 )
void nvfp4_dequant_kernel(
	const uint8_t* __restrict__ qcodes,
	const uint8_t* __restrict__ qscales,
	half* __restrict__ w,
	int N,
	int K,
	float gscale
) {
	__shared__ uint4 s[32][DQ_GROUPS * 2 + 1];
	const int G = K >> 4, runs = G / DQ_GROUPS;
	const int tile = blockIdx.x / runs, g0 = ( blockIdx.x - tile * runs ) * DQ_GROUPS;
	const int t = threadIdx.x, lane = t & 31, gl = t >> 5;
	const size_t idx = ( (size_t) tile * G + g0 + gl ) * 32 + lane;
	const uint2 q2 = reinterpret_cast<const uint2*>( qcodes )[idx];
	const half2 gm2 = __float2half2_rn( gscale * 16384.f );
	const half2 sc2 = __hmul2( fp8e4m3_to_half2( qscales[idx] ), gm2 );
	half2 b[8];
	dequant8_tm( q2.x, sc2, b + 0 );
	dequant8_tm( q2.y, sc2, b + 4 );
	const int r = ( ( lane >> 2 ) & 3 ) * 8 + ( lane & 3 ) + ( ( lane & 16 ) ? 4 : 0 );
	s[r][gl * 2] = *reinterpret_cast<const uint4*>( b );
	s[r][gl * 2 + 1] = *reinterpret_cast<const uint4*>( b + 4 );
	__syncthreads();
	for ( int e = t; e < 32 * DQ_GROUPS * 2; e += DQ_GROUPS * 32 ) {
		const int row = e / ( DQ_GROUPS * 2 ), c = e - row * ( DQ_GROUPS * 2 );
		reinterpret_cast<uint4*>( w + (size_t) ( tile * 32 + row ) * K + (size_t) g0 * 16 )[c] = s[row][c];
	}
}


// Fused dequantize + tensor-core GEMM for prefill chunks: Y[M, N] = X[M, K] @ W[N, K]^T with W decoded from the prepacked
// FP4 codes straight into shared memory, so the FP16 weight never goes through HBM. 128 x 128 x 32 tiles, 8 warps (2 x 4),
// each warp 64 x 32 as 4 x 2 wmma 16x16x16 fragments. SM70 has no cp.async: the next tile's raw words are fetched into
// registers before the current tile's MMAs and decoded into the other shared buffer after them.
constexpr int GT_BM = 128, GT_BN = 128, GT_BK = 32, GT_LD = GT_BK + 8;

NVFP4_KERNEL_BOUNDS( 256 )
void nvfp4_gemm_tc_kernel(
	const uint8_t* __restrict__ qcodes,
	const uint8_t* __restrict__ qscales,
	const half* __restrict__ x,
	half* __restrict__ y,
	int M,
	int N,
	int K,
	float gscale
) {
	using namespace nvcuda;
	__shared__ __align__( 16 ) half sm[2][2][GT_BM * GT_LD]; // [buffer][A, B]
	const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
	const int m0 = blockIdx.y * GT_BM;
	const int tile0 = blockIdx.x * 4; // first 32-column weight tile of this block
	const int G = K >> 4;
	const int ntiles32 = N >> 5;

	// B-load role: one (32-column tile, k group of 16) per warp-pair; lanes walk the tile's 32 packed columns
	const int jt = warp & 3, gi = warp >> 2;
	const int qp = ( lane >> 2 ) & 3;
	const int r = ( lane & 3 ) + ( ( lane & 16 ) ? 4 : 0 );
	const int b_row = jt * 32 + qp * 8 + r;
	const bool b_live = tile0 + jt < ntiles32;
	const uint2* cb = reinterpret_cast<const uint2*>( qcodes );
	const half2 gm2 = __float2half2_rn( gscale * 16384.f );

	// A-load role: two 16-byte pieces of the 128 x 32 tile
	const int a_idx0 = t, a_idx1 = t + 256;

	const int KT = K / GT_BK;
	uint4 a_raw[2];
	uint2 b_raw = make_uint2( 0, 0 );
	unsigned char b_sc = 0;

	auto fetch = [&]( int kt ) {
		const int k0 = kt * GT_BK;
		#pragma unroll
		for ( int i = 0; i < 2; ++i ) {
			const int idx = i ? a_idx1 : a_idx0;
			const int row = idx >> 2, c = idx & 3;
			a_raw[i] = ( m0 + row < M ) ? *reinterpret_cast<const uint4*>( x + (size_t) ( m0 + row ) * K + k0 + c * 8 ) : make_uint4( 0, 0, 0, 0 );
		}
		if ( b_live ) {
			const size_t qi = ( (size_t) ( tile0 + jt ) * G + ( kt * 2 + gi ) ) * 32 + lane;
			b_raw = cb[qi];
			b_sc = qscales[qi];
		}
		else {
			b_raw = make_uint2( 0, 0 );
			b_sc = 0;
		}
	};
	auto store = [&]( int buf ) {
		half* As = sm[buf][0];
		half* Bs = sm[buf][1];
		#pragma unroll
		for ( int i = 0; i < 2; ++i ) {
			const int idx = i ? a_idx1 : a_idx0;
			*reinterpret_cast<uint4*>( As + ( idx >> 2 ) * GT_LD + ( idx & 3 ) * 8 ) = a_raw[i];
		}
		const half2 sc2 = __hmul2( fp8e4m3_to_half2( b_sc ), gm2 );
		half2 b[8];
		dequant8_tm( b_raw.x, sc2, b + 0 );
		dequant8_tm( b_raw.y, sc2, b + 4 );
		uint4* dst = reinterpret_cast<uint4*>( Bs + b_row * GT_LD + gi * 16 );
		dst[0] = *reinterpret_cast<const uint4*>( b );
		dst[1] = *reinterpret_cast<const uint4*>( b + 4 );
	};

	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
	#pragma unroll
	for ( int i = 0; i < 4; ++i )
		#pragma unroll
		for ( int j = 0; j < 2; ++j ) wmma::fill_fragment( acc[i][j], 0.0f );

	const int wm = warp & 1, wn = warp >> 1;
	fetch( 0 );
	store( 0 );
	__syncthreads();
	for ( int kt = 0; kt < KT; ++kt ) {
		const bool more = kt + 1 < KT;
		if ( more ) fetch( kt + 1 );
		const half* As = sm[kt & 1][0];
		const half* Bs = sm[kt & 1][1];
		#pragma unroll
		for ( int kk = 0; kk < GT_BK; kk += 16 ) {
			wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[4];
			wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf[2];
			#pragma unroll
			for ( int i = 0; i < 4; ++i ) wmma::load_matrix_sync( af[i], As + ( wm * 64 + i * 16 ) * GT_LD + kk, GT_LD );
			#pragma unroll
			for ( int j = 0; j < 2; ++j ) wmma::load_matrix_sync( bf[j], Bs + ( wn * 32 + j * 16 ) * GT_LD + kk, GT_LD );
			#pragma unroll
			for ( int i = 0; i < 4; ++i )
				#pragma unroll
				for ( int j = 0; j < 2; ++j ) wmma::mma_sync( acc[i][j], af[i], bf[j], acc[i][j] );
		}
		if ( more ) store( ( kt + 1 ) & 1 );
		__syncthreads();
	}

	// Epilogue: stage each fragment through shared memory, convert and store 8 halves per lane
	float* scratch = reinterpret_cast<float*>( &sm[0][0][0] ) + warp * 16 * 16;
	const int n_base = blockIdx.x * GT_BN + wn * 32;
	#pragma unroll
	for ( int i = 0; i < 4; ++i ) {
		#pragma unroll
		for ( int j = 0; j < 2; ++j ) {
			wmma::store_matrix_sync( scratch, acc[i][j], 16, wmma::mem_row_major );
			__syncwarp();
			const int row = m0 + wm * 64 + i * 16 + ( lane >> 1 );
			const int col = n_base + j * 16 + ( lane & 1 ) * 8;
			if ( row < M && col < N ) {
				const float* sp = scratch + ( lane >> 1 ) * 16 + ( lane & 1 ) * 8;
				half2 h[4];
				#pragma unroll
				for ( int q = 0; q < 4; ++q ) h[q] = __floats2half2_rn( sp[2 * q], sp[2 * q + 1] );
				*reinterpret_cast<uint4*>( y + (size_t) row * N + col ) = *reinterpret_cast<const uint4*>( h );
			}
			__syncwarp();
		}
	}
}
