#pragma once
// ROCm only (RDNA3, wave32). Included by src/kernels/cuda/nvfp4/skinny_kernels.cu after nvfp4_gemm_tc_kernel, whose structure it follows:
// the fused dequantize + tensor-core GEMM for prefill chunks, Y[M, N] = X[M, K] @ W[N, K]^T with W decoded from the prepacked FP4 codes
// straight into shared memory (the FP16 weight never goes through HBM, against the dequantize kernel + hipBLAS path's write and re-read of it).
// 128 x 128 x 32 tiles, 8 warps (2 x 4), each warp 64 x 32 as 4 x 2 native 16x16x16 WMMAs (wmma.cuh lane layouts: A row and B column are
// lane & 15 with their 16 k contiguous, so both tiles are stored k-contiguous: X as [m][k], W as [n][k]; accumulator element e of lane l is
// row 2 e + (l >> 4), column l & 15). The next tile's raw words are fetched into registers before the current tile's WMMAs and decoded into
// the other shared buffer after them.
#include "wmma.cuh"

constexpr int RG_BM = 128, RG_BN = 128, RG_BK = 32, RG_LD = RG_BK + 8;

__global__ __launch_bounds__( 256 )
void rocm_nvfp4_gemm_kernel(
	const uint8_t* __restrict__ qcodes,
	const uint8_t* __restrict__ qscales,
	const half* __restrict__ x,
	half* __restrict__ y,
	int M,
	int N,
	int K,
	float gscale
) {
	__shared__ __align__( 16 ) half sm[2][2][RG_BM * RG_LD];   // [buffer][A, B]
	const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
	const int m0 = blockIdx.y * RG_BM;
	const int tile0 = blockIdx.x * 4;   // first 32-column weight tile of this block
	const int G = K >> 4;
	const int ntiles32 = N >> 5;

	// W-load role: one (32-column tile, k group of 16) per warp; lanes walk the tile's 32 packed columns
	const int jt = warp & 3, gi = warp >> 2;
	const int qp = ( lane >> 2 ) & 3;
	const int r = ( lane & 3 ) + ( ( lane & 16 ) ? 4 : 0 );
	const int b_row = jt * 32 + qp * 8 + r;
	const bool b_live = tile0 + jt < ntiles32;
	const uint2* cb = reinterpret_cast<const uint2*>( qcodes );
	const half2 gm2 = __float2half2_rn( gscale * 16384.f );

	const int a_idx0 = t, a_idx1 = t + 256;   // two 16-byte pieces of the 128 x 32 X tile

	const int KT = K / RG_BK;
	uint4 a_raw[2];
	uint2 b_raw = make_uint2( 0, 0 );
	unsigned char b_sc = 0;

	auto fetch = [&]( int kt ) {
		const int k0 = kt * RG_BK;
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
			*reinterpret_cast<uint4*>( As + ( idx >> 2 ) * RG_LD + ( idx & 3 ) * 8 ) = a_raw[i];
		}
		const half2 sc2 = __hmul2( fp8e4m3_to_half2( b_sc ), gm2 );
		half2 b[8];
		dequant8_tm( b_raw.x, sc2, b + 0 );
		dequant8_tm( b_raw.y, sc2, b + 4 );
		uint4* dst = reinterpret_cast<uint4*>( Bs + b_row * RG_LD + gi * 16 );
		dst[0] = *reinterpret_cast<const uint4*>( b );
		dst[1] = *reinterpret_cast<const uint4*>( b + 4 );
	};

	wm_v8f acc[4][2];
	#pragma unroll
	for ( int i = 0; i < 4; ++i ) {
		#pragma unroll
		for ( int j = 0; j < 2; ++j ) acc[i][j] = wm_v8f{ 0, 0, 0, 0, 0, 0, 0, 0 };
	}

	const int wm = warp & 1, wn = warp >> 1;
	const int ql = lane & 15, hi = lane >> 4;
	fetch( 0 );
	store( 0 );
	__syncthreads();
	for ( int kt = 0; kt < KT; ++kt ) {
		const bool more = kt + 1 < KT;
		if ( more ) fetch( kt + 1 );
		const half* As = sm[kt & 1][0];
		const half* Bs = sm[kt & 1][1];
		#pragma unroll
		for ( int kk = 0; kk < RG_BK; kk += 16 ) {
			wm_v16h af[4], bf[2];
			#pragma unroll
			for ( int i = 0; i < 4; ++i ) af[i] = wm_ld16( As + ( wm * 64 + i * 16 + ql ) * RG_LD + kk );
			#pragma unroll
			for ( int j = 0; j < 2; ++j ) bf[j] = wm_ld16( Bs + ( wn * 32 + j * 16 + ql ) * RG_LD + kk );
			#pragma unroll
			for ( int i = 0; i < 4; ++i ) {
				#pragma unroll
				for ( int j = 0; j < 2; ++j ) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32( af[i], bf[j], acc[i][j] );
			}
		}
		if ( more ) store( ( kt + 1 ) & 1 );
		__syncthreads();
	}

	const int n_base = blockIdx.x * RG_BN + wn * 32;
	#pragma unroll
	for ( int i = 0; i < 4; ++i ) {
		#pragma unroll
		for ( int j = 0; j < 2; ++j ) {
			const int col = n_base + j * 16 + ql;
			if ( col >= N ) continue;
			#pragma unroll
			for ( int e = 0; e < 8; ++e ) {
				const int row = m0 + wm * 64 + i * 16 + 2 * e + hi;
				if ( row < M ) y[(size_t) row * N + col] = __float2half( acc[i][j][e] );
			}
		}
	}
}
