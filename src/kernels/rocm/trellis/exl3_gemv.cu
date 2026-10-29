// ROCm EXL3 GEMV (M <= 8 rows) without the tensor-core step, after upstream exllamav3's RDNA GEMV (rocm/quant/exl3_gemv_kernel_rdna.cuh,
// its barrier-free "direct" dot core). The CUDA GEMV is one cooperative kernel built on mma.sync, which ROCm runs emulated (<1 tok/s on
// a 27B model); here the input and output Hadamard transforms are their own launches either side of a plain one, and the product
// accumulates with v_dot2_f32_f16 straight from dq_dispatch's fragments:
//   dq_dispatch<bits, cb>( tile, lane << 3, f0, f1 ) hands lane L of the 16 x 16 (k x n) tile
//     f0[0] = ( W[r0][c], W[r0 + 1][c] ), f0[1] = ( W[r0 + 8][c], W[r0 + 9][c] ), f1 the same at column c + 8
//   with r0 = 2 (L & 3) and c = L >> 2 (the placement reconstruct_kernel's unswizzle writes). Each lane quad holds columns q and q + 8
//   over all 16 k between its four lanes: two dot2 per fragment pair, one quad reduction at the end of the k range.
// One block per 16-column tile, its warps splitting the k tiles and reducing through shared memory
#if defined(USE_ROCM)
#include <cuda_fp16.h>
#include "exl3_gemv.cuh"
#include "exl3_dq.cuh"
#include "exl3_tile_rdna.cuh"
#include "hadamard.cuh"
#include "trellis_context.h"
#include <cstdint>

namespace {

constexpr int ROCM_GEMV_MAX_WARPS = 16;

__device__ __forceinline__ float rocm_fdot2( half2 a, half2 b, float c ) {
	typedef _Float16 h2v __attribute__( ( ext_vector_type( 2 ) ) );
	return __builtin_amdgcn_fdot2( __builtin_bit_cast( h2v, a ), __builtin_bit_cast( h2v, b ), c, false );
}

template <int bits, int cb, int MR>
__global__ __launch_bounds__( 32 * ROCM_GEMV_MAX_WARPS ) void exl3_gemv_rocm_kernel(
	const half* __restrict__ A,       // rotated input, M rows of K
	const uint16_t* __restrict__ B,   // trellis, [K / 16][N / 16][16 * bits]
	half* __restrict__ C,             // M rows of N (before the output rotation)
	int M,
	int K,
	int N
) {
	// The warp's slice of the input rows is staged through shared memory a chunk of CH k tiles at a time, laid out [tile][lane & 3][row] with
	// a lane's two half pairs (k = r0, r0 + 1 and r0 + 8, r0 + 9) adjacent: one 8-byte LDS read per row and tile instead of two global loads,
	// which at 8 rows cost more than decoding the weights (86 against 36 ms for a Qwen3.8-27B 8-row block)
	constexpr int CH = 8;
	__shared__ float red[ROCM_GEMV_MAX_WARPS][MR][16];
	__shared__ __align__( 16 ) half as_[ROCM_GEMV_MAX_WARPS][MR][CH * 16];
	const int lane = threadIdx.x & 31;
	const int warp = __builtin_amdgcn_readfirstlane( threadIdx.x >> 5 );
	const int warps = blockDim.x >> 5;
	const int tile_n = blockIdx.x, n_tiles = N >> 4, k_tiles = K >> 4;
	const int per = ( k_tiles + warps - 1 ) / warps;
	const int kb0 = warp * per, kb1 = min( k_tiles, kb0 + per );
	const int nchunks = ( per + CH - 1 ) / CH;

	float acc_a[MR], acc_b[MR];
	#pragma unroll
	for ( int m = 0; m < MR; ++m ) acc_a[m] = acc_b[m] = 0.f;

	// The tile walk: a wave-uniform byte pointer (scalar registers) plus the lane's fixed word offsets (exl3_tile_rdna.cuh)
	const Exl3TilePlan<bits> plan = exl3_tile_plan<bits>( lane );
	const uint64_t bt0 = reinterpret_cast<uint64_t>( B + ( (size_t) kb0 * n_tiles + tile_n ) * ( 16 * bits ) );
	const uint32_t bt_hi = static_cast<uint32_t>( __builtin_amdgcn_readfirstlane( static_cast<uint32_t>( bt0 >> 32 ) ) );
	const uint32_t bt_lo = static_cast<uint32_t>( __builtin_amdgcn_readfirstlane( static_cast<uint32_t>( bt0 ) ) );
	const char* bt = reinterpret_cast<const char*>( ( static_cast<uint64_t>( bt_hi ) << 32 ) | bt_lo );
	const size_t bstep = (size_t) n_tiles * 32 * bits;   // bytes from one k tile to the next
	const uint2* alane = reinterpret_cast<const uint2*>( &as_[warp][0][0] ) + ( lane & 3 ) * MR;
	for ( int c = 0; c < nchunks; ++c ) {
		const int kc = kb0 + c * CH;
		const int n = max( 0, min( CH, kb1 - kc ) );
		// Rows past M re-read the last row (sums dropped at the store)
		for ( int e = lane; e < MR * CH * 2; e += 32 ) {
			const int m = e / ( CH * 2 ), v = e % ( CH * 2 );
			if ( ( v >> 1 ) >= n ) continue;
			const uint4 val = *reinterpret_cast<const uint4*>( A + (size_t) min( m, M - 1 ) * K + (size_t) kc * 16 + v * 8 );
			half2* dst = reinterpret_cast<half2*>( &as_[warp][0][0] ) + ( ( v >> 1 ) * 4 * MR + m ) * 2 + ( v & 1 );
			dst[0 * MR * 2] = __builtin_bit_cast( half2, val.x );
			dst[1 * MR * 2] = __builtin_bit_cast( half2, val.y );
			dst[2 * MR * 2] = __builtin_bit_cast( half2, val.z );
			dst[3 * MR * 2] = __builtin_bit_cast( half2, val.w );
		}
		__syncthreads();
		#pragma unroll 4
		for ( int j = 0; j < n; ++j, bt += bstep ) {
			uint32_t raw[Exl3TilePlan<bits>::R];
			exl3_tile_load<bits>( bt, plan, raw );
			FragB f0, f1;
			exl3_tile_decode<bits, cb>( raw, plan, f0, f1 );
			#pragma unroll
			for ( int m = 0; m < MR; ++m ) {
				const uint2 av = alane[j * 4 * MR + m];
				const half2 a01 = __builtin_bit_cast( half2, av.x ), a89 = __builtin_bit_cast( half2, av.y );
				acc_a[m] = rocm_fdot2( a01, f0[0], acc_a[m] );
				acc_a[m] = rocm_fdot2( a89, f0[1], acc_a[m] );
				acc_b[m] = rocm_fdot2( a01, f1[0], acc_b[m] );
				acc_b[m] = rocm_fdot2( a89, f1[1], acc_b[m] );
			}
		}
		__syncthreads();
	}

	const int q = lane >> 2;
	#pragma unroll
	for ( int m = 0; m < MR; ++m ) {
		float a = acc_a[m], b = acc_b[m];
		a += __shfl_xor( a, 1 );
		a += __shfl_xor( a, 2 );
		b += __shfl_xor( b, 1 );
		b += __shfl_xor( b, 2 );
		if ( ( lane & 3 ) == 0 ) {
			red[warp][m][q] = a;
			red[warp][m][q + 8] = b;
		}
	}
	__syncthreads();

	for ( int e = threadIdx.x; e < MR * 16; e += blockDim.x ) {
		const int m = e >> 4, c = e & 15;
		if ( m >= M ) continue;
		float v = 0.f;
		for ( int w = 0; w < warps; ++w ) v += red[w][m][c];
		C[(size_t) m * N + tile_n * 16 + c] = __float2half( v );
	}
}

template <int bits, int cb, int MR>
void launch( const half* A, const uint16_t* B, half* C, int m, int k, int n, cudaStream_t stream ) {
	// Narrow outputs split k over more waves to keep enough loads in flight
	const int warps = ( n / 16 < 512 ) ? 16 : 8;
	exl3_gemv_rocm_kernel<bits, cb, MR><<<n / 16, 32 * warps, 0, stream>>>( A, B, C, m, k, n );
}

template <int bits, int cb>
void launch_rows( const half* A, const uint16_t* B, half* C, int m, int k, int n, cudaStream_t stream ) {
	if ( m == 1 ) launch<bits, cb, 1>( A, B, C, m, k, n, stream );
	else if ( m == 2 ) launch<bits, cb, 2>( A, B, C, m, k, n, stream );
	else if ( m <= 4 ) launch<bits, cb, 4>( A, B, C, m, k, n, stream );
	else launch<bits, cb, 8>( A, B, C, m, k, n, stream );
}

template <int bits>
void launch_cb( int cb, const half* A, const uint16_t* B, half* C, int m, int k, int n, cudaStream_t stream ) {
	if ( cb == 0 ) launch_rows<bits, 0>( A, B, C, m, k, n, stream );
	else if ( cb == 1 ) launch_rows<bits, 1>( A, B, C, m, k, n, stream );
	else launch_rows<bits, 2>( A, B, C, m, k, n, stream );
}

} // namespace

bool exl3_gemv_rocm(
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
	if ( size_m < 1 || size_m > EXL3_GEMV_ROCM_MAX_M || K < 1 || K > 8 ) return false;
	if ( size_k % 128 || size_n % 128 ) return false;
	const int cb = mul1 ? 2 : ( mcg ? 1 : 0 );

	// xh = H(x * suh); y = H((xh @ W) * svh), the standalone Hadamard's scale the final one (as forward_recon)
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
