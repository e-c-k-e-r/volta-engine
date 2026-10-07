#include "engine/modules/nvfp4.h"
#include "trellis_context.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>
#include <algorithm>

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

template <int MT>
__global__ void skinny_nvfp4_qpn(
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
	__syncthreads();

	for ( int e = threadIdx.x; e < MT * 256; e += blockDim.x ) {
		const float v = cs[0][e] + cs[1][e] + cs[2][e] + cs[3][e];
		const int row = e >> 5, col = e & 31;
		if ( row < M ) {
			y[(size_t) row * N + (size_t) tile * 32 + col] = __float2half( v );
		}
	}
}

template <int SPLITK, int NACC>
__global__ void skinny_nvfp4_qpn2(
	const uint8_t* __restrict__ bcodes,
	const uint8_t* __restrict__ bscales,
	const half* __restrict__ x,
	half* __restrict__ y,
	int N,
	int K,
	int M,
	float gscale
) {
	__shared__ float cs[SPLITK > 1 ? SPLITK : 1][SPLITK > 1 ? 256 : 1];

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
	float c[NACC][8];
	#pragma unroll
	for ( int a = 0; a < NACC; a++ ) {
		#pragma unroll
		for ( int i = 0; i < 8; i++ ) c[a][i] = 0.f;
	}

	const half* xrow = ( r < M ) ? ( x + (size_t) r * K ) : nullptr;

	#pragma unroll 4
	for ( int g = g0; g < g0 + Gq; g++ ) {
		const uint2 q2 = __ldcs( cb + (size_t) g * 32 );
		const half2 sc2 = __hmul2( fp8e4m3_to_half2( __ldcs( sb + (size_t) g * 32 ) ), gm2 );
		half2 b[8];
		dequant8_tm( q2.x, sc2, b + 0 );
		dequant8_tm( q2.y, sc2, b + 4 );
		const unsigned* B = reinterpret_cast<const unsigned*>( b );
		uint4 a01 = make_uint4( 0, 0, 0, 0 ), a23 = make_uint4( 0, 0, 0, 0 );
		if ( xrow != nullptr ) {
			a01 = *reinterpret_cast<const uint4*>( xrow + g * 16 );
			a23 = *reinterpret_cast<const uint4*>( xrow + g * 16 + 8 );
		}
		const unsigned* A0 = reinterpret_cast<const unsigned*>( &a01 );
		const unsigned* A1 = reinterpret_cast<const unsigned*>( &a23 );
		MMA_8N8K4( c[0], A0[0], A0[1], B[0], B[1] );
		MMA_8N8K4( c[1 % NACC], A0[2], A0[3], B[2], B[3] );
		MMA_8N8K4( c[2 % NACC], A1[0], A1[1], B[4], B[5] );
		MMA_8N8K4( c[3 % NACC], A1[2], A1[3], B[6], B[7] );
	}

	#pragma unroll
	for ( int a = 1; a < NACC; a++ ) {
		#pragma unroll
		for ( int i = 0; i < 8; i++ ) c[0][i] += c[a][i];
	}

	if ( SPLITK == 1 ) {
		#pragma unroll
		for ( int i = 0; i < 8; i++ ) {
			const int row = ( i & 2 ) | ( ( lane & 16 ) ? 4 : 0 ) | ( lane & 1 );
			const int col = ( i & 1 ) | ( ( ( lane >> 1 ) & 1 ) << 1 ) | ( ( i >> 2 ) << 2 );
			if ( row < M ) {
				y[(size_t) row * N + (size_t) tile * 32 + qp * 8 + col] = __float2half( c[0][i] );
			}
		}
		return;
	}

	#pragma unroll
	for ( int i = 0; i < 8; i++ ) {
		const int row = ( i & 2 ) | ( ( lane & 16 ) ? 4 : 0 ) | ( lane & 1 );
		const int col = ( i & 1 ) | ( ( ( lane >> 1 ) & 1 ) << 1 ) | ( ( i >> 2 ) << 2 );
		cs[warp][row * 32 + qp * 8 + col] = c[0][i];
	}
	__syncthreads();

	for ( int e = threadIdx.x; e < 256; e += blockDim.x ) {
		float v = 0.f;
		#pragma unroll
		for ( int w = 0; w < SPLITK; w++ ) v += cs[w][e];
		const int row = e >> 5, col = e & 31;
		if ( row < M ) {
			y[(size_t) row * N + (size_t) tile * 32 + col] = __float2half( v );
		}
	}
}

// Dequantize the prepacked weight into dense FP16 W[N, K] (row n = output column n): one thread per (tile, group, lane).
__global__ void nvfp4_dequant_kernel(
	const uint8_t* __restrict__ qcodes,
	const uint8_t* __restrict__ qscales,
	half* __restrict__ w,
	int N,
	int K,
	float gscale
) {
	const int G = K >> 4;
	const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
	const size_t total = (size_t) ( N >> 5 ) * G * 32;
	if ( idx >= total ) return;
	const int lane = (int) ( idx & 31 );
	const size_t tg = idx >> 5;
	const int g = (int) ( tg % G );
	const int tile = (int) ( tg / G );
	const int qp = ( lane >> 2 ) & 3;
	const int r = ( lane & 3 ) + ( ( lane & 16 ) ? 4 : 0 );
	const int n = tile * 32 + qp * 8 + r;

	const uint2 q2 = reinterpret_cast<const uint2*>( qcodes )[idx];
	const half2 gm2 = __float2half2_rn( gscale * 16384.f );
	const half2 sc2 = __hmul2( fp8e4m3_to_half2( qscales[idx] ), gm2 );
	half2 b[8];
	dequant8_tm( q2.x, sc2, b + 0 );
	dequant8_tm( q2.y, sc2, b + 4 );
	uint4* dst = reinterpret_cast<uint4*>( w + (size_t) n * K + (size_t) g * 16 );
	dst[0] = *reinterpret_cast<const uint4*>( b );
	dst[1] = *reinterpret_cast<const uint4*>( b + 4 );
}

namespace engine::modules {

// Rows up to which the fused skinny kernel beats dequantize + HGEMM; EXL3_NVFP4_SKINNY_MAX overrides (A/B knob).
static int nvfp4_skinny_max_rows() {
	static const int v = std::getenv( "EXL3_NVFP4_SKINNY_MAX" ) ? std::min( 32, std::atoi( std::getenv( "EXL3_NVFP4_SKINNY_MAX" ) ) ) : 32;
	return v;
}

void LinearNVFP4::forward(
	const pod::TensorView& x,
	const pod::TensorView& codes,
	const pod::TensorView& scales,
	float gscale,
	pod::TensorView& out,
	cudaStream_t stream
) {
	const int m = x.shape[0];
	const int k = x.shape[1];
	const int n = out.shape[1];

	ENGINE_ASSERT( k % 64 == 0, "NVFP4: K must be multiple of 64" );
	ENGINE_ASSERT( n % 32 == 0, "NVFP4: N must be multiple of 32" );

	const uint8_t* codes_ptr = codes.ptr<const uint8_t>();
	const uint8_t* scales_ptr = scales.ptr<const uint8_t>();
	const half* x_ptr = x.ptr<const half>();
	half* y_ptr = out.ptr<half>();

	dim3 grid( n / 32 );

	if ( m <= 8 ) {
		if ( ( k / 16 ) % 16 == 0 ) {
			skinny_nvfp4_qpn2<16, 2><<<grid, dim3( 32 * 16 ), 0, stream>>>(
				codes_ptr, scales_ptr, x_ptr, y_ptr, n, k, m, gscale
			);
		}
		else if ( ( k / 16 ) % 8 == 0 ) {
			skinny_nvfp4_qpn2<8, 2><<<grid, dim3( 32 * 8 ), 0, stream>>>(
				codes_ptr, scales_ptr, x_ptr, y_ptr, n, k, m, gscale
			);
		}
		else {
			skinny_nvfp4_qpn<1><<<grid, dim3( 128 ), 0, stream>>>(
				codes_ptr, scales_ptr, x_ptr, y_ptr, n, k, m, gscale
			);
		}
	}
	else if ( m <= 16 ) {
		skinny_nvfp4_qpn<2><<<grid, dim3( 128 ), 0, stream>>>(
			codes_ptr, scales_ptr, x_ptr, y_ptr, n, k, m, gscale
		);
	}
	else if ( m <= nvfp4_skinny_max_rows() ) {
		// Up to 32 rows: the decoded weights are reused across 3 or 4 row tiles of 8
		if ( m <= 24 ) {
			skinny_nvfp4_qpn<3><<<grid, dim3( 128 ), 0, stream>>>(
				codes_ptr, scales_ptr, x_ptr, y_ptr, n, k, m, gscale
			);
		}
		else {
			skinny_nvfp4_qpn<4><<<grid, dim3( 128 ), 0, stream>>>(
				codes_ptr, scales_ptr, x_ptr, y_ptr, n, k, m, gscale
			);
		}
	}
	else {
		// Dequantize once into FP16 and run a tensor-core HGEMM: C[m, n] = X[m, k] @ W[n, k]^T
		int device = 0;
		cudaGetDevice( &device );
		half* w = trellis_get_recon_ws( device, static_cast<size_t>( n ) * k * sizeof( half ) );
		const size_t threads_total = static_cast<size_t>( n / 32 ) * ( k / 16 ) * 32;
		nvfp4_dequant_kernel<<<static_cast<unsigned>( ( threads_total + 255 ) / 256 ), 256, 0, stream>>>(
			codes_ptr, scales_ptr, w, n, k, gscale );
		cublasHandle_t h = trellis_get_cublas( device );
		cublasSetStream( h, stream );
		const float one = 1.0f, zero = 0.0f;
		const cublasStatus_t st = cublasGemmEx( h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
			&one, w, CUDA_R_16F, k, x_ptr, CUDA_R_16F, k,
			&zero, y_ptr, CUDA_R_16F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP );
		ENGINE_ASSERT( st == CUBLAS_STATUS_SUCCESS, "NVFP4: cublasGemmEx failed" );
	}
}

} // namespace engine::modules
