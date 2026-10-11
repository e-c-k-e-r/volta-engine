#include "engine/modules/nvfp4.h"
#include "trellis_context.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <cstdlib>
#include <algorithm>

#include "nvfp4_common.cuh"

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


// Fused dequantize + tensor-core GEMM for prefill chunks: Y[M, N] = X[M, K] @ W[N, K]^T with W decoded from the prepacked
// FP4 codes straight into shared memory, so the FP16 weight never goes through HBM. 128 x 128 x 32 tiles, 8 warps (2 x 4),
// each warp 64 x 32 as 4 x 2 wmma 16x16x16 fragments. SM70 has no cp.async: the next tile's raw words are fetched into
// registers before the current tile's MMAs and decoded into the other shared buffer after them.
constexpr int GT_BM = 128, GT_BN = 128, GT_BK = 32, GT_LD = GT_BK + 8;

__global__ __launch_bounds__( 256 )
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

namespace engine::modules {

// Rows up to which the fused skinny kernel beats dequantize + HGEMM
static constexpr int NVFP4_SKINNY_MAX_ROWS = 32;

void LinearNVFP4::forward(
	const pod::TensorView& x,
	const pod::TensorView& codes,
	const pod::TensorView& scales,
	float gscale,
	pod::TensorView& out,
	cudaStream_t stream
) {
	// N and K come from the packed codes ([N, K/2] bytes) when they carry a shape: the model's shared scratch tensors are as wide as the widest
	// consumer (GLM's expert buffers match the 12288-wide dense MLP), and rows of x and out are packed K and N apart as for every other weight format
	const int m = x.shape[0];
	int k = x.shape[1];
	int n = out.shape[1];
	if ( codes.ndim == 2 ) {
		n = codes.shape[0];
		k = codes.shape[1] * 2;
	}
	ENGINE_ASSERT( x.shape[1] >= k && out.shape[1] >= n, "NVFP4: input or output narrower than the weight" );

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
	else if ( m <= NVFP4_SKINNY_MAX_ROWS ) {
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
	else if ( m < 384 && ( ( n + GT_BN - 1 ) / GT_BN ) * ( ( m + GT_BM - 1 ) / GT_BM ) >= 48 ) {
		// Mid-size chunks with enough tiles to fill the GPU; at 512 rows the cuBLAS GEMM (dequantized weight reused across
		// the whole chunk) is as fast, and narrow outputs leave most SMs idle in the fused kernel
		nvfp4_gemm_tc_kernel<<<dim3( ( n + GT_BN - 1 ) / GT_BN, ( m + GT_BM - 1 ) / GT_BM ), dim3( 256 ), 0, stream>>>(
			codes_ptr, scales_ptr, x_ptr, y_ptr, m, n, k, gscale );
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
