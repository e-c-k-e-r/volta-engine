// ROCm EXL3 multi-row GEMM (9..EXL3_GEMM_ROCM_MAX_M rows) on RDNA3's native wave32 WMMA, for the rows the GEMV (8 rows) does not take and the
// dequantize + hipBLAS path would serve by writing and re-reading the whole weight as FP16 (8x the packed bytes; 2.4 GB for the Qwen LM head).
// The packed weight is read once. As in the GEMV, the input and output Hadamard transforms are their own launches either side of this one.
//   One block per 16-column tile of the output; its warps split the k tiles. Per k tile a warp decodes the 16 x 16 trellis tile once
//   (dq_dispatch hands lane L the values W[r0][c], W[r0 + 1][c], W[r0 + 8][c], W[r0 + 9][c] of column c = L >> 2 and the same four of column c + 8,
//   r0 = 2 (L & 3)), stages it column-major in shared memory and loads the WMMA B fragment (lane l = column l & 15, its 16 k values) from there.
//   The A fragments (lane l = row l & 15, 16 k values) come straight from the rotated input rows; every 16-row block reuses the decoded tile.
//   The accumulator of lane l holds rows 2 i + (l >> 4) of column l & 15 (src/kernels/rocm/wmma.cuh). The warps' partial tiles are summed in a
//   fixed order through shared memory, so the result does not depend on scheduling.
#if defined(USE_ROCM)
#include <cuda_fp16.h>
#include "exl3_gemv.cuh"
#include "exl3_dq.cuh"
#include "hadamard.cuh"
#include "trellis_context.h"
#include "../wmma.cuh"
#include <cstdint>

namespace {

// Warps per block: the k tiles of the weight are split over them. Up to 32 rows the shared partial tiles are small enough for 16 warps, which hide more of
// the decode latency; at 64 rows the fixed-order reduction would need 64 KB, so 8
template <int MB> constexpr int gemm_warps() { return MB <= 2 ? 16 : 8; }

template <int bits, int cb, int MB>
__global__ __launch_bounds__( 32 * gemm_warps<MB>() ) void exl3_gemm_wmma_kernel(
	const half* __restrict__ A,       // rotated input, M rows of K
	const uint16_t* __restrict__ B,   // trellis, [K / 16][N / 16][16 * bits]
	half* __restrict__ C,             // M rows of N (before the output rotation)
	int M,
	int K,
	int N
) {
	constexpr int W = gemm_warps<MB>();
	__shared__ __align__( 16 ) half bs[W][2][16 * 16];   // two decoded tiles per warp, [column][k]: the next one is decoded while the current one is multiplied
	__shared__ float red[W][MB][8 * 32];                 // a warp's accumulators, [i][lane]
	const int lane = threadIdx.x & 31;
	const int warp = __builtin_amdgcn_readfirstlane( threadIdx.x >> 5 );
	const int tile_n = blockIdx.x, n_tiles = N >> 4, k_tiles = K >> 4;
	const int per = ( k_tiles + W - 1 ) / W;
	const int kb0 = warp * per, kb1 = min( k_tiles, kb0 + per );

	wm_v8f acc[MB];
	#pragma unroll
	for ( int mb = 0; mb < MB; ++mb ) acc[mb] = wm_v8f{ 0, 0, 0, 0, 0, 0, 0, 0 };

	// The row each lane feeds to the A fragment of block mb (rows past M re-read the last row; their results are dropped at the store)
	const half* arow[MB];
	#pragma unroll
	for ( int mb = 0; mb < MB; ++mb ) arow[mb] = A + (size_t) min( mb * 16 + ( lane & 15 ), M - 1 ) * K;

	const int c = lane >> 2, r0 = 2 * ( lane & 3 );
	const size_t bstep = (size_t) n_tiles * 16 * bits / 2;   // uint32 words from one k tile to the next
	const uint32_t* bt0 = reinterpret_cast<const uint32_t*>( B + ( (size_t) kb0 * n_tiles + tile_n ) * ( 16 * bits ) );

	// decodes k tile kt of this warp's range into buffer buf
	auto stage = [&]( int kt, int buf ) {
		FragB f0, f1;
		dq_dispatch<bits, cb>( bt0 + (size_t) ( kt - kb0 ) * bstep, lane << 3, f0, f1 );
		half2* col_lo = reinterpret_cast<half2*>( &bs[warp][buf][c * 16] );
		half2* col_hi = reinterpret_cast<half2*>( &bs[warp][buf][( c + 8 ) * 16] );
		col_lo[r0 >> 1] = __builtin_bit_cast( half2, f0[0] );
		col_lo[( r0 + 8 ) >> 1] = __builtin_bit_cast( half2, f0[1] );
		col_hi[r0 >> 1] = __builtin_bit_cast( half2, f1[0] );
		col_hi[( r0 + 8 ) >> 1] = __builtin_bit_cast( half2, f1[1] );
	};

	if ( kb0 < kb1 ) {
		wm_v16h a_cur[MB], a_nxt[MB];
		stage( kb0, 0 );
		#pragma unroll
		for ( int mb = 0; mb < MB; ++mb ) a_cur[mb] = wm_ld16( arow[mb] + (size_t) kb0 * 16 );
		for ( int kt = kb0; kt < kb1; ++kt ) {
			const int cur = ( kt - kb0 ) & 1;
			const bool more = kt + 1 < kb1;
			if ( more ) {
				stage( kt + 1, cur ^ 1 );
				#pragma unroll
				for ( int mb = 0; mb < MB; ++mb ) a_nxt[mb] = wm_ld16( arow[mb] + (size_t) ( kt + 1 ) * 16 );
			}
			// a tile is written and read inside one wave: LDS accesses of a wave complete in order, only the compiler must keep them apart
			asm volatile( "" ::: "memory" );
			const wm_v16h b = wm_ld16( &bs[warp][cur][( lane & 15 ) * 16] );
			#pragma unroll
			for ( int mb = 0; mb < MB; ++mb ) acc[mb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32( a_cur[mb], b, acc[mb] );
			if ( more ) {
				#pragma unroll
				for ( int mb = 0; mb < MB; ++mb ) a_cur[mb] = a_nxt[mb];
			}
			asm volatile( "" ::: "memory" );
		}
	}

	#pragma unroll
	for ( int mb = 0; mb < MB; ++mb ) {
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) red[warp][mb][i * 32 + lane] = acc[mb][i];
	}
	__syncthreads();

	// element ( row r, column cc ) of a 16 x 16 block sits in accumulator i = r >> 1 of lane ( r & 1 ) * 16 + cc
	for ( int e = threadIdx.x; e < MB * 256; e += blockDim.x ) {
		const int mb = e >> 8, r = ( e >> 4 ) & 15, cc = e & 15;
		const int row = mb * 16 + r;
		if ( row >= M ) continue;
		const int slot = ( r >> 1 ) * 32 + ( r & 1 ) * 16 + cc;
		float v = 0.f;
		#pragma unroll
		for ( int w = 0; w < W; ++w ) v += red[w][mb][slot];
		C[(size_t) row * N + tile_n * 16 + cc] = __float2half( v );
	}
}

template <int bits, int cb, int MB>
void launch( const half* A, const uint16_t* B, half* C, int m, int k, int n, cudaStream_t stream ) {
	exl3_gemm_wmma_kernel<bits, cb, MB><<<n / 16, 32 * gemm_warps<MB>(), 0, stream>>>( A, B, C, m, k, n );
}

template <int bits, int cb>
void launch_rows( const half* A, const uint16_t* B, half* C, int m, int k, int n, cudaStream_t stream ) {
	if ( m <= 16 ) launch<bits, cb, 1>( A, B, C, m, k, n, stream );
	else if ( m <= 32 ) launch<bits, cb, 2>( A, B, C, m, k, n, stream );
	else launch<bits, cb, 4>( A, B, C, m, k, n, stream );
}

template <int bits>
void launch_cb( int cb, const half* A, const uint16_t* B, half* C, int m, int k, int n, cudaStream_t stream ) {
	if ( cb == 0 ) launch_rows<bits, 0>( A, B, C, m, k, n, stream );
	else if ( cb == 1 ) launch_rows<bits, 1>( A, B, C, m, k, n, stream );
	else launch_rows<bits, 2>( A, B, C, m, k, n, stream );
}

} // namespace

bool exl3_gemm_rocm(
	const half* A,
	const uint16_t* B,
	void* C,
	const half* suh,
	half* A_had,
	const half* svh,
	int size_m,
	int size_k,
	int size_n,
	int K,
	bool mcg,
	bool mul1,
	bool c_fp32,
	cudaStream_t stream
) {
	if ( c_fp32 || suh == nullptr || A_had == nullptr || svh == nullptr ) return false;
	if ( size_m < 1 || size_m > EXL3_GEMM_ROCM_MAX_M || K < 1 || K > 8 ) return false;
	if ( size_k % 128 || size_n % 128 ) return false;
	const int cb = mul1 ? 2 : ( mcg ? 1 : 0 );

	// xh = H(x * suh); y = H((xh @ W) * svh), the standalone Hadamard's scale the final one (as forward_recon and the GEMV)
	constexpr float kRScale = 0.088388347648f;
	had_r_128( A, A_had, suh, nullptr, kRScale, size_m, size_k, stream );
	half* y = static_cast<half*>( C );
	switch ( K ) {
		case 1: launch_cb<1>( cb, A_had, B, y, size_m, size_k, size_n, stream ); break;
		case 2: launch_cb<2>( cb, A_had, B, y, size_m, size_k, size_n, stream ); break;
		case 3: launch_cb<3>( cb, A_had, B, y, size_m, size_k, size_n, stream ); break;
		case 4: launch_cb<4>( cb, A_had, B, y, size_m, size_k, size_n, stream ); break;
		case 5: launch_cb<5>( cb, A_had, B, y, size_m, size_k, size_n, stream ); break;
		case 6: launch_cb<6>( cb, A_had, B, y, size_m, size_k, size_n, stream ); break;
		case 7: launch_cb<7>( cb, A_had, B, y, size_m, size_k, size_n, stream ); break;
		default: launch_cb<8>( cb, A_had, B, y, size_m, size_k, size_n, stream ); break;
	}
	had_r_128( y, y, nullptr, svh, kRScale, size_m, size_n, stream );
	return true;
}
#endif
