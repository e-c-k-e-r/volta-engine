#pragma once
// ROCm only. Included by src/kernels/cuda/nvfp4/skinny_kernels.cu after the NVFP4 dequantize helpers it uses.
// c + the FP32 dot of two half pairs (v_dot2_f32_f16 on RDNA)
DEV_INLINE float rocm_dot2( unsigned a, unsigned b, float c ) {
#if defined(__HIP_DEVICE_COMPILE__)
	typedef _Float16 h2v __attribute__( ( ext_vector_type( 2 ) ) );
	return __builtin_amdgcn_fdot2( __builtin_bit_cast( h2v, a ), __builtin_bit_cast( h2v, b ), c, false );
#else
	const float2 fa = __half22float2( *reinterpret_cast<const half2*>( &a ) ), fb = __half22float2( *reinterpret_cast<const half2*>( &b ) );
	return fmaf( fa.x, fb.x, fmaf( fa.y, fb.y, c ) );
#endif
}

static constexpr int ROCM_GEMV_BATCH = 8;

// ROCm decode GEMV (M <= MR rows) without the tensor-core step, which RDNA has no NVIDIA-layout form of: one block per 32-column tile,
// WARPS warps splitting its k groups. Each lane decodes the 16 codes of its own slot (column n of the prepacked map, the same FP16
// values as nvfp4_dequant_kernel) and dots them with each token row; the packed weight is read once, against the dequantize + GEMM
// path's FP16 write and re-read of the whole matrix
template <int MR, int WARPS>
__global__ __launch_bounds__( 32 * WARPS ) void rocm_nvfp4_gemv(
	const uint8_t* __restrict__ qcodes,
	const uint8_t* __restrict__ qscales,
	const half* __restrict__ x,
	half* __restrict__ y,
	int N,
	int K,
	int M,
	float gscale
) {
	__shared__ float cs[WARPS][MR][32];
	const int lane = threadIdx.x & 31;
	const int warp = __builtin_amdgcn_readfirstlane( threadIdx.x >> 5 );
	const int tile = blockIdx.x;
	const int G = K >> 4, per = ( G + WARPS - 1 ) / WARPS;
	const int g0 = warp * per, g1 = min( G, g0 + per );
	const uint2* cb = reinterpret_cast<const uint2*>( qcodes ) + (size_t) tile * G * 32 + lane;
	const uint8_t* sb = qscales + (size_t) tile * G * 32 + lane;
	const half2 gm2 = __float2half2_rn( gscale * 16384.f );

	float acc[MR];
	#pragma unroll
	for ( int m = 0; m < MR; ++m ) acc[m] = 0.f;

	// Rows past M re-read the last row (branch-free, their sums are dropped): a guard would split the loop body and keep the loads
	// from being issued ahead, leaving one load in flight per wave
	const half* xr[MR];
	#pragma unroll
	for ( int m = 0; m < MR; ++m ) xr[m] = x + (size_t) min( m, M - 1 ) * K;

	auto step = [&]( uint2 q2, uint8_t s8, int g ) {
		const half2 sc2 = __hmul2( fp8e4m3_to_half2( s8 ), gm2 );
		half2 w[8];
		dequant8_tm( q2.x, sc2, w + 0 );
		dequant8_tm( q2.y, sc2, w + 4 );
		const unsigned* W = reinterpret_cast<const unsigned*>( w );
		#pragma unroll
		for ( int m = 0; m < MR; ++m ) {
			const uint4* xp = reinterpret_cast<const uint4*>( xr[m] + g * 16 );
			const uint4 x0 = xp[0], x1 = xp[1];
			float a = acc[m];
			a = rocm_dot2( W[0], x0.x, a );
			a = rocm_dot2( W[1], x0.y, a );
			a = rocm_dot2( W[2], x0.z, a );
			a = rocm_dot2( W[3], x0.w, a );
			a = rocm_dot2( W[4], x1.x, a );
			a = rocm_dot2( W[5], x1.y, a );
			a = rocm_dot2( W[6], x1.z, a );
			a = rocm_dot2( W[7], x1.w, a );
			acc[m] = a;
		}
	};

	// ROCM_GEMV_BATCH groups' loads issued together, then their math
	int g = g0;
	for ( ; g + ROCM_GEMV_BATCH <= g1; g += ROCM_GEMV_BATCH ) {
		uint2 q[ROCM_GEMV_BATCH];
		uint8_t sc[ROCM_GEMV_BATCH];
		#pragma unroll
		for ( int u = 0; u < ROCM_GEMV_BATCH; ++u ) {
			q[u] = __ldcs( cb + (size_t) ( g + u ) * 32 );
			sc[u] = __ldcs( sb + (size_t) ( g + u ) * 32 );
		}
		#pragma unroll
		for ( int u = 0; u < ROCM_GEMV_BATCH; ++u ) step( q[u], sc[u], g + u );
	}
	for ( ; g < g1; ++g ) step( __ldcs( cb + (size_t) g * 32 ), __ldcs( sb + (size_t) g * 32 ), g );

	#pragma unroll
	for ( int m = 0; m < MR; ++m ) cs[warp][m][lane] = acc[m];
	__syncthreads();

	for ( int e = threadIdx.x; e < MR * 32; e += blockDim.x ) {
		const int m = e >> 5, l = e & 31;
		if ( m >= M ) continue;
		float v = 0.f;
		#pragma unroll
		for ( int w = 0; w < WARPS; ++w ) v += cs[w][m][l];
		const int col = ( ( l >> 2 ) & 3 ) * 8 + ( l & 3 ) + ( ( l & 16 ) ? 4 : 0 );
		y[(size_t) m * N + (size_t) tile * 32 + col] = __float2half( v );
	}
}

// Rows the ROCm GEMV covers; above, dequantize + hipBLAS (the weight decoded once serves every row)
static constexpr int ROCM_NVFP4_GEMV_MAX_ROWS = 16;

template <int MR>
static void rocm_nvfp4_gemv_launch( const uint8_t* q, const uint8_t* s, const half* x, half* y, int n, int k, int m, float gscale, cudaStream_t stream ) {
	// 8 warps per tile at every width measured (N 1024 .. 34816 at K 5120 and 17408): 16 warps for the narrow outputs was 15 to 25% slower
	// on the 5120-wide ones (the down, o and qkv projections) and 2% faster only at N = 1024
	rocm_nvfp4_gemv<MR, 8><<<n / 32, 32 * 8, 0, stream>>>( q, s, x, y, n, k, m, gscale );
}
