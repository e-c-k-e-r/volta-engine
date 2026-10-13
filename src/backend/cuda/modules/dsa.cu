#include "engine/modules/dsa.h"
#include "trellis_context.h"
#include <cublas_v2.h>
#include <cmath>
#include <mma.h>

namespace engine::modules {

namespace wmma = nvcuda::wmma;

namespace {

constexpr int D = DSA::DIM;
constexpr int H = DSA::HEADS;
constexpr int P = DSA::POOL;

// out[r][j] = sum_k w[j][k] x[r][k]: one warp per output, all rows, 128-bit weight loads (rows <= 16)
__global__ __launch_bounds__( 256 )
void project_small_kernel( const half* __restrict__ w, const half* __restrict__ x, half* __restrict__ out, int rows, int n, int k ) {
	const int j = blockIdx.x * 8 + ( threadIdx.x >> 5 );
	const int lane = threadIdx.x & 31;
	if ( j >= n ) return;
	float acc[16];
	#pragma unroll
	for ( int r = 0; r < 16; ++r ) acc[r] = 0.0f;
	for ( int c = lane * 8; c < k; c += 256 ) {
		const uint4 wr = *reinterpret_cast<const uint4*>( w + static_cast<size_t>( j ) * k + c );
		const half2* w2 = reinterpret_cast<const half2*>( &wr );
		float wf[8];
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			const float2 f = __half22float2( w2[i] );
			wf[2 * i] = f.x;
			wf[2 * i + 1] = f.y;
		}
		#pragma unroll
		for ( int r = 0; r < 16; ++r ) {
			if ( r >= rows ) break;
			const uint4 xr = *reinterpret_cast<const uint4*>( x + static_cast<size_t>( r ) * k + c );
			const half2* x2 = reinterpret_cast<const half2*>( &xr );
			#pragma unroll
			for ( int i = 0; i < 4; ++i ) {
				const float2 f = __half22float2( x2[i] );
				acc[r] = fmaf( wf[2 * i], f.x, acc[r] );
				acc[r] = fmaf( wf[2 * i + 1], f.y, acc[r] );
			}
		}
	}
	#pragma unroll
	for ( int r = 0; r < 16; ++r ) {
		if ( r >= rows ) break;
		float v = acc[r];
		#pragma unroll
		for ( int off = 16; off > 0; off >>= 1 ) v += __shfl_xor_sync( 0xffffffff, v, off );
		if ( lane == 0 ) out[static_cast<size_t>( r ) * n + j] = __float2half_rn( v );
	}
}

// One block per row, 128 threads (one per key dimension)
__global__ __launch_bounds__( 128 )
void prep_kernel( const half* __restrict__ proj, const half* __restrict__ ln_w, const half* __restrict__ ln_b, half* __restrict__ plane,
	half* __restrict__ w_out, int layer, int max_seq, const pod::BatchRows* __restrict__ rows, size_t plane_slot_stride ) {
	__shared__ float red[8];
	const int r = blockIdx.x, d = threadIdx.x;
	const half* pr = proj + static_cast<size_t>( r ) * DSA::PROJ;
	const float kv = __half2float( pr[d] );
	// mean and variance over the 128 key dimensions (two-pass, fp32)
	float s = kv;
	for ( int off = 16; off > 0; off >>= 1 ) s += __shfl_xor_sync( 0xffffffff, s, off );
	if ( ( d & 31 ) == 0 ) red[d >> 5] = s;
	__syncthreads();
	const float mean = ( red[0] + red[1] + red[2] + red[3] ) * ( 1.0f / D );
	__syncthreads();
	const float dv = kv - mean;
	float v = dv * dv;
	for ( int off = 16; off > 0; off >>= 1 ) v += __shfl_xor_sync( 0xffffffff, v, off );
	if ( ( d & 31 ) == 0 ) red[4 + ( d >> 5 )] = v;
	__syncthreads();
	const float var = ( red[4] + red[5] + red[6] + red[7] ) * ( 1.0f / D );
	const float kn = dv * rsqrtf( var + 1e-6f ) * __half2float( ln_w[d] ) + __half2float( ln_b[d] );
	const int pos = rows->pos[r];
	half* dst = plane + static_cast<size_t>( rows->slot[r] ) * plane_slot_stride + ( static_cast<size_t>( layer ) * DSA::PLANE_CAP + ( pos & ( DSA::PLANE_CAP - 1 ) ) ) * DSA::PLANE;
	dst[d] = __float2half_rn( kn );
	dst[D + d] = pr[DSA::DIM + H + d];
	if ( d < H ) w_out[static_cast<size_t>( r ) * H + d] = pr[DSA::DIM + d];
}

// One block per row, 128 threads; only the row that completes a pool does work
__global__ __launch_bounds__( 128 )
void pool_kernel( const half* __restrict__ plane, half* __restrict__ pool_plane, const half* __restrict__ ape, int layer, int max_seq,
	const pod::BatchRows* __restrict__ rows, size_t plane_slot_stride, size_t pool_slot_stride ) {
	const int r = blockIdx.x, d = threadIdx.x;
	const int pos = rows->pos[r];
	if ( ( pos & ( P - 1 ) ) != P - 1 ) return;
	const int slot = rows->slot[r];
	const half* base = plane + static_cast<size_t>( slot ) * plane_slot_stride + static_cast<size_t>( layer ) * DSA::PLANE_CAP * DSA::PLANE;
	float k[P], g[P];
	float mx = -1e30f;
	#pragma unroll
	for ( int j = 0; j < P; ++j ) {
		const half* src = base + static_cast<size_t>( ( pos - ( P - 1 ) + j ) & ( DSA::PLANE_CAP - 1 ) ) * DSA::PLANE;
		k[j] = __half2float( src[d] );
		g[j] = __half2float( src[D + d] ) + __half2float( ape[j * D + d] );
		mx = fmaxf( mx, g[j] );
	}
	float sum = 0.0f, acc = 0.0f;
	#pragma unroll
	for ( int j = 0; j < P; ++j ) {
		const float e = __expf( g[j] - mx );
		sum += e;
		acc += e * k[j];
	}
	const int pool = pos / P;
	pool_plane[static_cast<size_t>( slot ) * pool_slot_stride + ( static_cast<size_t>( layer ) * ( max_seq / P ) + pool ) * D + d] = __float2half_rn( acc / sum );
}

// Block = (row, tile of 128 pools), 4 warps of 2 column tiles of 16 pools. logits[h][p] = q[h] . key[p] is a 32 x 128 x 16 GEMM per
// column tile on wmma m16n16k16 (fp16 operands, fp32 accumulate: products are exact as in the scalar form), then ReLU, head weight and
// head sum from shared memory
__global__ __launch_bounds__( 128 )
void score_kernel( const half* __restrict__ q_idx, const half* __restrict__ w, const half* __restrict__ pool_plane, half* __restrict__ scores,
	int score_stride, int layer, int max_seq, const pod::BatchRows* __restrict__ rows, size_t pool_slot_stride ) {
	__shared__ __align__( 32 ) half sq[H * D];
	__shared__ __align__( 32 ) float sl[4][H * 16];
	__shared__ float sw[H];
	const int r = blockIdx.x;
	const int pos = rows->pos[r];
	const int bound = ( pos + 1 ) / P;
	if ( bound <= DSA::SEL_POOLS || blockIdx.y * 128 >= bound ) return; // dense row, or tile past the visible pools
	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	for ( int i = threadIdx.x; i < H * D / 8; i += 128 ) reinterpret_cast<uint4*>( sq )[i] = reinterpret_cast<const uint4*>( q_idx + static_cast<size_t>( r ) * H * D )[i];
	if ( threadIdx.x < H ) sw[threadIdx.x] = __half2float( w[static_cast<size_t>( r ) * H + threadIdx.x] );
	__syncthreads();
	const half* keys = pool_plane + static_cast<size_t>( rows->slot[r] ) * pool_slot_stride + static_cast<size_t>( layer ) * ( max_seq / P ) * D;
	for ( int c = 0; c < 2; ++c ) {
		const int p0 = blockIdx.y * 128 + ( warp * 2 + c ) * 16;
		if ( p0 >= bound ) break;
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
		wmma::fill_fragment( acc0, 0.0f );
		wmma::fill_fragment( acc1, 0.0f );
		#pragma unroll
		for ( int k = 0; k < D; k += 16 ) {
			wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
			wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
			wmma::load_matrix_sync( fb, keys + static_cast<size_t>( p0 ) * D + k, D );
			wmma::load_matrix_sync( fa, sq + k, D );
			wmma::mma_sync( acc0, fa, fb, acc0 );
			wmma::load_matrix_sync( fa, sq + 16 * D + k, D );
			wmma::mma_sync( acc1, fa, fb, acc1 );
		}
		wmma::store_matrix_sync( sl[warp], acc0, 16, wmma::mem_row_major );
		wmma::store_matrix_sync( sl[warp] + 16 * 16, acc1, 16, wmma::mem_row_major );
		__syncwarp();
		const int pl = lane & 15, h0 = ( lane >> 4 ) * 16;
		float acc = 0.0f;
		#pragma unroll
		for ( int h = 0; h < 16; ++h ) acc = fmaf( fmaxf( sl[warp][( h0 + h ) * 16 + pl], 0.0f ), sw[h0 + h], acc );
		acc += __shfl_xor_sync( 0xffffffffu, acc, 16 );
		if ( lane < 16 && p0 + pl < bound ) scores[static_cast<size_t>( r ) * score_stride + p0 + pl] = __float2half_rn( acc * rsqrtf( static_cast<float>( D * H ) ) );
		__syncwarp();
	}
}

__device__ __forceinline__ unsigned score_key( half s ) {
	const unsigned b = __half_as_ushort( s );
	return ( b & 0x8000u ) ? ( ~b & 0xFFFFu ) : ( b | 0x8000u );
}

// Block per row: radix select of the SEL_POOLS best pools by fp16 score (two 8-bit passes), ties at the threshold taken in
// pool order, then expansion to token indices and the raw tail
__global__ __launch_bounds__( 1024 )
void select_kernel( const half* __restrict__ scores, int score_stride, int* __restrict__ idx, int* __restrict__ nkeys,
	const pod::BatchRows* __restrict__ rows ) {
	__shared__ int hist[256];
	__shared__ int s_sel[DSA::SEL_POOLS];
	__shared__ int s_b1, s_above1, s_thr, s_need_eq, s_n_gt, s_n_eq, s_wsum[32], s_carry_gt, s_carry_eq;
	const int r = blockIdx.x, t = threadIdx.x;
	const int pos = rows->pos[r];
	const int vis = pos + 1;
	const int bound = vis / P;
	if ( bound <= DSA::SEL_POOLS ) {
		if ( t == 0 ) nkeys[r] = -1;
		return;
	}
	const half* sc = scores + static_cast<size_t>( r ) * score_stride;
	const int K = DSA::SEL_POOLS;
	if ( t < 256 ) hist[t] = 0;
	__syncthreads();
	for ( int p = t; p < bound; p += 1024 ) atomicAdd( &hist[score_key( sc[p] ) >> 8], 1 );
	__syncthreads();
	if ( t == 0 ) {
		int cum = 0, b = 255;
		for ( ; b > 0; --b ) {
			if ( cum + hist[b] >= K ) break;
			cum += hist[b];
		}
		s_b1 = b;
		s_above1 = cum;
	}
	__syncthreads();
	if ( t < 256 ) hist[t] = 0;
	__syncthreads();
	for ( int p = t; p < bound; p += 1024 ) {
		const unsigned key = score_key( sc[p] );
		if ( ( key >> 8 ) == static_cast<unsigned>( s_b1 ) ) atomicAdd( &hist[key & 255u], 1 );
	}
	__syncthreads();
	if ( t == 0 ) {
		const int rem = K - s_above1;
		int cum = 0, b = 255;
		for ( ; b > 0; --b ) {
			if ( cum + hist[b] >= rem ) break;
			cum += hist[b];
		}
		s_thr = ( s_b1 << 8 ) | b;
		s_need_eq = rem - cum; // pools tied at the threshold that still fit
		s_carry_gt = 0;
		s_carry_eq = 0;
	}
	__syncthreads();
	const unsigned thr = static_cast<unsigned>( s_thr );
	const int need_eq = s_need_eq;
	const int total_gt = K - need_eq;
	// Compaction in pool order: strictly better pools first-come into [0, total_gt), ties into [total_gt, K)
	for ( int base = 0; base < bound; base += 1024 ) {
		const int p = base + t;
		unsigned key = 0;
		if ( p < bound ) key = score_key( sc[p] );
		const bool gt = p < bound && key > thr, eq = p < bound && key == thr;
		const unsigned lane_mask = ( 1u << ( t & 31 ) ) - 1u;
		const unsigned bg = __ballot_sync( 0xffffffff, gt ), be = __ballot_sync( 0xffffffff, eq );
		if ( ( t & 31 ) == 0 ) s_wsum[t >> 5] = __popc( bg ) | ( __popc( be ) << 16 );
		__syncthreads();
		int wg = 0, we = 0;
		for ( int w2 = 0; w2 < ( t >> 5 ); ++w2 ) {
			wg += s_wsum[w2] & 0xFFFF;
			we += s_wsum[w2] >> 16;
		}
		const int pos_gt = s_carry_gt + wg + __popc( bg & lane_mask );
		const int pos_eq = s_carry_eq + we + __popc( be & lane_mask );
		if ( gt && pos_gt < total_gt ) s_sel[pos_gt] = p;
		if ( eq && pos_eq < need_eq ) s_sel[total_gt + pos_eq] = p;
		__syncthreads();
		if ( t == 0 ) {
			int tg = 0, te = 0;
			for ( int w2 = 0; w2 < 32; ++w2 ) {
				tg += s_wsum[w2] & 0xFFFF;
				te += s_wsum[w2] >> 16;
			}
			s_carry_gt += tg;
			s_carry_eq += te;
		}
		__syncthreads();
	}
	int* out = idx + static_cast<size_t>( r ) * DSA::KPAD;
	for ( int i = t; i < K * P; i += 1024 ) out[i] = s_sel[i / P] * P + ( i % P );
	const int tail = vis % P;
	if ( t < P - 1 && t < tail ) out[K * P + t] = vis - tail + t;
	if ( t == 0 ) nkeys[r] = K * P + tail;
}

} // namespace

void DSA::project_small( const half* w, const half* x, half* proj, int rows, int n, int k, cudaStream_t stream ) {
	ENGINE_ASSERT( rows >= 1 && rows <= 16 && k % 256 == 0, "DSA::project_small: up to 16 rows, k a multiple of 256" );
	project_small_kernel<<<dim3( ( n + 7 ) / 8 ), dim3( 256 ), 0, stream>>>( w, x, proj, rows, n, k );
}

void DSA::project_large( const half* w, const half* x, half* proj, int rows, int n, int k, cudaStream_t stream ) {
	int device = 0;
	(void) cudaGetDevice( &device );
	cublasHandle_t h = trellis_get_cublas( device );
	cublasSetStream( h, stream );
	const float one = 1.0f, zero = 0.0f;
	const cublasStatus_t st = cublasGemmEx( h, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k, &one, w, CUDA_R_16F, k, x, CUDA_R_16F, k,
		&zero, proj, CUDA_R_16F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP );
	ENGINE_ASSERT( st == CUBLAS_STATUS_SUCCESS, "DSA::project_large: indexer projection GEMM failed" );
}

void DSA::prep( const half* proj, const half* ln_w, const half* ln_b, half* plane, half* w_out, int layer, int max_seq, int rows,
	const pod::BatchRows* d_rows, size_t plane_slot_stride, cudaStream_t stream ) {
	prep_kernel<<<dim3( rows ), dim3( 128 ), 0, stream>>>( proj, ln_w, ln_b, plane, w_out, layer, max_seq, d_rows, plane_slot_stride );
}

void DSA::pool_update( const half* plane, half* pool_plane, const half* ape, int layer, int max_seq, int rows,
	const pod::BatchRows* d_rows, size_t plane_slot_stride, size_t pool_slot_stride, cudaStream_t stream ) {
	pool_kernel<<<dim3( rows ), dim3( 128 ), 0, stream>>>( plane, pool_plane, ape, layer, max_seq, d_rows, plane_slot_stride, pool_slot_stride );
}

void DSA::score( const half* q_idx, const half* w, const half* pool_plane, half* scores, int score_stride, int layer, int max_seq, int rows,
	const pod::BatchRows* d_rows, size_t pool_slot_stride, cudaStream_t stream ) {
	ENGINE_ASSERT( max_seq % 64 == 0, "DSA score: max_seq must be a multiple of 64" );
	const int tiles = ( max_seq / P + 127 ) / 128;
	score_kernel<<<dim3( rows, tiles ), dim3( 128 ), 0, stream>>>( q_idx, w, pool_plane, scores, score_stride, layer, max_seq, d_rows, pool_slot_stride );
}

void DSA::select( const half* scores, int score_stride, int* idx, int* nkeys, int rows, const pod::BatchRows* d_rows, cudaStream_t stream ) {
	select_kernel<<<dim3( rows ), dim3( 1024 ), 0, stream>>>( scores, score_stride, idx, nkeys, d_rows );
}

} // namespace engine::modules
