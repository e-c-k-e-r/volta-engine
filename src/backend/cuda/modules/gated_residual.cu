#include "engine/modules/gated_residual.h"
#include "engine/config.h"
#include "trellis_context.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cmath>

namespace engine::modules {

namespace {

__device__ __forceinline__ float warp_sum( float v ) {
	#pragma unroll
	for ( int o = 16; o > 0; o >>= 1 ) v += __shfl_xor_sync( 0xffffffff, v, o );
	return v;
}

__global__ void gr_init_kernel( const half* __restrict__ in, float* __restrict__ x, int hidden, int hc ) {
	const int r = blockIdx.y;
	for ( int d = blockIdx.x * blockDim.x + threadIdx.x; d < hidden; d += gridDim.x * blockDim.x ) {
		const float v = __half2float( in[static_cast<size_t>( r ) * hidden + d] );
		for ( int s = 0; s < hc; ++s ) x[( static_cast<size_t>( r ) * hc + s ) * hidden + d] = v;
	}
}

// One block per (row, stream): normed[r, s, :] = x * rsqrt(mean(x^2) + eps) * (1 + w[s, :]). With prev_y the previous site's gated add
// x += post[r, s] * y[r, :] is applied (and stored) in the same pass, so the streams are read and written once between two sites.
// Each thread keeps its values (hidden <= GR_NORM_CACHE * blockDim) in registers between the two passes: one dependent global load round trip less,
// same summation order as the plain loop.
constexpr int GR_NORM_CACHE = 12;
__global__ void gr_norm_kernel( float* __restrict__ x, const half* __restrict__ w, float* __restrict__ normed, int hidden, int hc, float eps,
	const float* __restrict__ prev_post, const half* __restrict__ prev_y, half* __restrict__ normed_hi, half* __restrict__ normed_lo ) {
	const int r = blockIdx.x, s = blockIdx.y;
	const size_t base = ( static_cast<size_t>( r ) * hc + s ) * hidden;
	const float p = prev_y != nullptr ? prev_post[static_cast<size_t>( r ) * hc + s] : 0.0f;
	float cache[GR_NORM_CACHE];
	float sq = 0.0f;
	#pragma unroll
	for ( int i = 0; i < GR_NORM_CACHE; ++i ) {
		const int d = threadIdx.x + i * blockDim.x;
		if ( d < hidden ) {
			float v = x[base + d];
			if ( prev_y != nullptr ) {
				v += p * __half2float( prev_y[static_cast<size_t>( r ) * hidden + d] );
				x[base + d] = v;
			}
			cache[i] = v;
			sq += v * v;
		}
	}
	for ( int d = threadIdx.x + GR_NORM_CACHE * blockDim.x; d < hidden; d += blockDim.x ) {   // hidden beyond the cache: plain loop
		float v = x[base + d];
		if ( prev_y != nullptr ) {
			v += p * __half2float( prev_y[static_cast<size_t>( r ) * hidden + d] );
			x[base + d] = v;
		}
		sq += v * v;
	}
	sq = warp_sum( sq );
	__shared__ float ws[32];
	if ( ( threadIdx.x & 31 ) == 0 ) ws[threadIdx.x >> 5] = sq;
	__syncthreads();
	float total = 0.0f;
	for ( int i = 0; i < ( blockDim.x >> 5 ); ++i ) total += ws[i];
	const float inv = rsqrtf( total / hidden + eps );
	#pragma unroll
	for ( int i = 0; i < GR_NORM_CACHE; ++i ) {
		const int d = threadIdx.x + i * blockDim.x;
		if ( d < hidden ) {
			const float nv = cache[i] * inv * ( 1.0f + __half2float( w[static_cast<size_t>( s ) * hidden + d] ) );
			normed[base + d] = nv;
			if ( normed_hi != nullptr ) {
				const half hi = __float2half_rn( nv );
				normed_hi[base + d] = hi;
				normed_lo[base + d] = __float2half_rn( nv - __half2float( hi ) );
			}
		}
	}
	for ( int d = threadIdx.x + GR_NORM_CACHE * blockDim.x; d < hidden; d += blockDim.x ) {
		const float nv = x[base + d] * inv * ( 1.0f + __half2float( w[static_cast<size_t>( s ) * hidden + d] ) );
		normed[base + d] = nv;
		if ( normed_hi != nullptr ) {
			const half hi = __float2half_rn( nv );
			normed_hi[base + d] = hi;
			normed_lo[base + d] = __float2half_rn( nv - __half2float( hi ) );
		}
	}
}

constexpr int GR_RT = 8;   // rows per block tile: a weight row read from memory serves 8 rows
constexpr int GR_JT = 1;   // outputs per block of the gate kernel (one output per block: 324 blocks hide the weight-read latency)

// t[r][j] = silu( dot(normed[r], down[j]) / hc ) for j < rank and post[r][s] = 2 * sigmoid( dot(normed[r], inject[s]) / hc ) (the inject rows follow
// the down rows). A block owns GR_JT consecutive outputs and a tile of GR_RT rows; its 4 warps split the K range, so every normed value is loaded
// once for all the outputs and every weight row is streamed once for all the rows. Requires flat % 1024 == 0.
__global__ __launch_bounds__( 128 )
void gr_gate_kernel( const float* __restrict__ normed, const half* __restrict__ down, const half* __restrict__ inject, float* __restrict__ t, float* __restrict__ post,
	int flat, int rank, int hc, int rows, int has_inject ) {
	const int j0 = blockIdx.x * GR_JT;
	const int r0 = blockIdx.y * GR_RT;
	const int nr = min( GR_RT, rows - r0 );
	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	const int total_out = rank + ( has_inject ? hc : 0 );
	const int kq = flat / 4;
	float acc[GR_JT][GR_RT];
	#pragma unroll
	for ( int a = 0; a < GR_JT; ++a ) {
		#pragma unroll
		for ( int r = 0; r < GR_RT; ++r ) acc[a][r] = 0.0f;
	}
	for ( int k = lane * 8; k < kq; k += 256 ) {
		const int kk = warp * kq + k;
		float xv[GR_RT][8];
		#pragma unroll
		for ( int r = 0; r < GR_RT; ++r ) {
			if ( r < nr ) {
				const float4 a0 = *reinterpret_cast<const float4*>( normed + static_cast<size_t>( r0 + r ) * flat + kk );
				const float4 a1 = *reinterpret_cast<const float4*>( normed + static_cast<size_t>( r0 + r ) * flat + kk + 4 );
				xv[r][0] = a0.x; xv[r][1] = a0.y; xv[r][2] = a0.z; xv[r][3] = a0.w; xv[r][4] = a1.x; xv[r][5] = a1.y; xv[r][6] = a1.z; xv[r][7] = a1.w;
			}
			else {
				#pragma unroll
				for ( int i = 0; i < 8; ++i ) xv[r][i] = 0.0f;
			}
		}
		#pragma unroll
		for ( int a = 0; a < GR_JT; ++a ) {
			const int j = j0 + a;
			if ( j >= total_out ) continue;
			const half* wrow = j < rank ? down + static_cast<size_t>( j ) * flat : inject + static_cast<size_t>( j - rank ) * flat;
			const uint4 raw = *reinterpret_cast<const uint4*>( wrow + kk );
			const half2* w2 = reinterpret_cast<const half2*>( &raw );
			float wf[8];
			#pragma unroll
			for ( int i = 0; i < 4; ++i ) { const float2 f = __half22float2( w2[i] ); wf[2 * i] = f.x; wf[2 * i + 1] = f.y; }
			#pragma unroll
			for ( int r = 0; r < GR_RT; ++r ) {
				#pragma unroll
				for ( int i = 0; i < 8; ++i ) acc[a][r] = fmaf( xv[r][i], wf[i], acc[a][r] );
			}
		}
	}
	#pragma unroll
	for ( int a = 0; a < GR_JT; ++a ) {
		#pragma unroll
		for ( int r = 0; r < GR_RT; ++r ) acc[a][r] = warp_sum( acc[a][r] );
	}
	__shared__ float sh[4][GR_JT][GR_RT];
	if ( lane == 0 ) {
		#pragma unroll
		for ( int a = 0; a < GR_JT; ++a ) {
			#pragma unroll
			for ( int r = 0; r < GR_RT; ++r ) sh[warp][a][r] = acc[a][r];
		}
	}
	__syncthreads();
	if ( threadIdx.x < GR_JT * GR_RT ) {
		const int a = threadIdx.x / GR_RT, r = threadIdx.x % GR_RT;
		const int j = j0 + a;
		if ( r < nr && j < total_out ) {
			const float z = ( sh[0][a][r] + sh[1][a][r] + sh[2][a][r] + sh[3][a][r] ) / hc;
			if ( j < rank ) t[static_cast<size_t>( r0 + r ) * rank + j] = z / ( 1.0f + __expf( -z ) );
			else post[static_cast<size_t>( r0 + r ) * hc + ( j - rank )] = 2.0f / ( 1.0f + __expf( -z ) );
		}
	}
}

// mixed[r, d] = mean_s( sigmoid( up[s * hidden + d] . t[r] ) * normed[r, s, d] ). A block owns 8 channels d and a tile of GR_RT rows; warp s walks the
// 8 up rows of its stream (coalesced 640 B reads shared by every row of the tile).
constexpr int GR_DT = 2;   // channels per block of the up-mix kernel (1280 blocks)
__global__ __launch_bounds__( 128 )
void gr_up_mix_kernel( const float* __restrict__ normed, const half* __restrict__ up, const float* __restrict__ t, half* __restrict__ mixed,
	int hidden, int hc, int rank, int rows ) {
	extern __shared__ float s_t[];   // [GR_RT][rank]
	const int d0 = blockIdx.x * GR_DT;
	const int r0 = blockIdx.y * GR_RT;
	const int nr = min( GR_RT, rows - r0 );
	for ( int i = threadIdx.x; i < nr * rank; i += blockDim.x ) s_t[i] = t[static_cast<size_t>( r0 ) * rank + i];
	__syncthreads();
	const int s = threadIdx.x >> 5, lane = threadIdx.x & 31;
	__shared__ float sg[4][GR_DT][GR_RT];   // sigmoid gate per stream, channel, row
	for ( int dl = 0; dl < GR_DT; ++dl ) {
		const half* u = up + static_cast<size_t>( s * hidden + d0 + dl ) * rank;
		float acc[GR_RT];
		#pragma unroll
		for ( int r = 0; r < GR_RT; ++r ) acc[r] = 0.0f;
		for ( int k = lane * 2; k < rank; k += 64 ) {
			const half2 uv = *reinterpret_cast<const half2*>( u + k );
			const float u0 = __low2float( uv ), u1 = __high2float( uv );
			#pragma unroll
			for ( int r = 0; r < GR_RT; ++r ) {
				if ( r < nr ) acc[r] = fmaf( s_t[r * rank + k], u0, fmaf( s_t[r * rank + k + 1], u1, acc[r] ) );
			}
		}
		#pragma unroll
		for ( int r = 0; r < GR_RT; ++r ) {
			const float v = warp_sum( acc[r] );
			if ( lane == 0 ) sg[s][dl][r] = 1.0f / ( 1.0f + __expf( -v ) );
		}
	}
	__syncthreads();
	if ( threadIdx.x < GR_DT * GR_RT ) {
		const int dl = threadIdx.x / GR_RT, r = threadIdx.x % GR_RT;
		if ( r < nr ) {
			float out = 0.0f;
			for ( int ss = 0; ss < hc; ++ss ) out += sg[ss][dl][r] * normed[( static_cast<size_t>( r0 + r ) * hc + ss ) * hidden + d0 + dl];
			mixed[static_cast<size_t>( r0 + r ) * hidden + d0 + dl] = __float2half_rn( out / hc );
		}
	}
}

// ---- decode-sized calls (rows <= 8) ----
// gate / up weights are streamed at near HBM rate: each thread keeps all its 16-byte weight loads in flight before touching the activations.

// z_j = dot( normed[r], w_j ) for the rank down rows and the hc inject rows: a block of 256 threads owns NJ consecutive outputs, each thread flat / 8 / 256
// 16-byte weight chunks per output (5 at flat = 10240), so the activations are read once for NJ outputs. t[r][j] = silu( z / hc ),
// post[r][j - rank] = 2 sigmoid( z / hc ).
constexpr int GRD_THREADS = 256;
constexpr int GRD_CHUNKS = 5;   // 16-byte chunks per thread: flat <= 8 * GRD_CHUNKS * GRD_THREADS = 10240
template<int NR, int NJ>
__global__ __launch_bounds__( GRD_THREADS )
void gr_gate_decode_kernel( const float* __restrict__ normed, const half* __restrict__ down, const half* __restrict__ inject, float* __restrict__ t, float* __restrict__ post,
	int flat, int rank, int hc, int rows ) {
	const int j0 = blockIdx.x * NJ;
	uint4 raw[NJ][GRD_CHUNKS];
	#pragma unroll
	for ( int a = 0; a < NJ; ++a ) {
		const int j = j0 + a;
		const half* wrow = j < rank ? down + static_cast<size_t>( j ) * flat : inject + static_cast<size_t>( j - rank ) * flat;
		#pragma unroll
		for ( int c = 0; c < GRD_CHUNKS; ++c ) {
			const int k = ( threadIdx.x + c * GRD_THREADS ) * 8;
			if ( k < flat ) raw[a][c] = __ldg( reinterpret_cast<const uint4*>( wrow + k ) );
		}
	}
	float acc[NJ][NR];
	#pragma unroll
	for ( int a = 0; a < NJ; ++a ) {
		#pragma unroll
		for ( int r = 0; r < NR; ++r ) acc[a][r] = 0.0f;
	}
	#pragma unroll
	for ( int c = 0; c < GRD_CHUNKS; ++c ) {
		const int k = ( threadIdx.x + c * GRD_THREADS ) * 8;
		if ( k >= flat ) continue;
		float wf[NJ][8];
		#pragma unroll
		for ( int a = 0; a < NJ; ++a ) {
			const half2* w2 = reinterpret_cast<const half2*>( &raw[a][c] );
			#pragma unroll
			for ( int i = 0; i < 4; ++i ) { const float2 f = __half22float2( w2[i] ); wf[a][2 * i] = f.x; wf[a][2 * i + 1] = f.y; }
		}
		#pragma unroll
		for ( int r = 0; r < NR; ++r ) {
			if ( r < rows ) {
				const float4 a0 = *reinterpret_cast<const float4*>( normed + static_cast<size_t>( r ) * flat + k );
				const float4 a1 = *reinterpret_cast<const float4*>( normed + static_cast<size_t>( r ) * flat + k + 4 );
				#pragma unroll
				for ( int a = 0; a < NJ; ++a ) {
					float v = acc[a][r];
					v = fmaf( a0.x, wf[a][0], v ); v = fmaf( a0.y, wf[a][1], v ); v = fmaf( a0.z, wf[a][2], v ); v = fmaf( a0.w, wf[a][3], v );
					v = fmaf( a1.x, wf[a][4], v ); v = fmaf( a1.y, wf[a][5], v ); v = fmaf( a1.z, wf[a][6], v ); v = fmaf( a1.w, wf[a][7], v );
					acc[a][r] = v;
				}
			}
		}
	}
	__shared__ float sh[GRD_THREADS / 32][NJ][NR];
	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	#pragma unroll
	for ( int a = 0; a < NJ; ++a ) {
		#pragma unroll
		for ( int r = 0; r < NR; ++r ) {
			const float v = warp_sum( acc[a][r] );
			if ( lane == 0 ) sh[warp][a][r] = v;
		}
	}
	__syncthreads();
	if ( threadIdx.x < NJ * NR ) {
		const int a = threadIdx.x / NR, r = threadIdx.x % NR;
		const int j = j0 + a;
		if ( r < rows ) {
			float sum = 0.0f;
			#pragma unroll
			for ( int wi = 0; wi < GRD_THREADS / 32; ++wi ) sum += sh[wi][a][r];
			const float z = sum / hc;
			if ( j < rank ) t[static_cast<size_t>( r ) * rank + j] = z / ( 1.0f + __expf( -z ) );
			else post[static_cast<size_t>( r ) * hc + ( j - rank )] = 2.0f / ( 1.0f + __expf( -z ) );
		}
	}
}

// Several rows (3..8): the activations (rows x flat fp32) would be re-read by every block, so the work is split along K instead. Block b owns the K slice
// [b * 128, ...) of every output: a half-warp (16 lanes) reads one output's 256-byte slice (lane i holds the 8 activation values of every row at 8 * i),
// so a warp covers two outputs per step and the activations are read once per block; the per-row dot products are reduced over the 16 lanes with a
// reduce-scatter. Block b writes its partial sums; gr_gate_reduce_kernel adds the GRS_SK partials in a fixed order.
constexpr int GRS_SK = 80;
constexpr int GRS_SLICE = 128;
constexpr int GRS_WARPS = 12;
constexpr int GRS_MAX_OUT = 384;
__device__ float g_gr_partials[GRS_SK * 8 * GRS_MAX_OUT];

__global__ __launch_bounds__( GRS_WARPS * 32 )
void gr_gate_splitk_kernel( const float* __restrict__ normed, const half* __restrict__ down, const half* __restrict__ inject, int flat, int rank, int outputs, int rows ) {
	const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	const int l16 = lane & 15, half_id = lane >> 4;
	const int k = blockIdx.x * GRS_SLICE + l16 * 8;
	float xv[8][8];
	#pragma unroll
	for ( int r = 0; r < 8; ++r ) {
		if ( r < rows ) {
			const float4 a0 = *reinterpret_cast<const float4*>( normed + static_cast<size_t>( r ) * flat + k );
			const float4 a1 = *reinterpret_cast<const float4*>( normed + static_cast<size_t>( r ) * flat + k + 4 );
			xv[r][0] = a0.x; xv[r][1] = a0.y; xv[r][2] = a0.z; xv[r][3] = a0.w; xv[r][4] = a1.x; xv[r][5] = a1.y; xv[r][6] = a1.z; xv[r][7] = a1.w;
		}
		else {
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) xv[r][i] = 0.0f;
		}
	}
	constexpr int UNROLL = 3;
	constexpr int STRIDE = GRS_WARPS * 2;   // outputs covered per step by the block
	for ( int jb = warp * 2; jb < outputs; jb += STRIDE * UNROLL ) {   // warp-uniform: the shuffles below need every lane
		uint4 raw[UNROLL];
		#pragma unroll
		for ( int u = 0; u < UNROLL; ++u ) {
			const int j = jb + half_id + u * STRIDE;
			if ( j < outputs ) {
				const half* wrow = j < rank ? down + static_cast<size_t>( j ) * flat : inject + static_cast<size_t>( j - rank ) * flat;
				raw[u] = __ldg( reinterpret_cast<const uint4*>( wrow + k ) );
			}
		}
		#pragma unroll
		for ( int u = 0; u < UNROLL; ++u ) {
			const int j = jb + half_id + u * STRIDE;
			// every lane takes part in the shuffles; lanes of a half-warp past the last output work on zeros
			float wf[8];
			if ( j < outputs ) {
				const half2* w2 = reinterpret_cast<const half2*>( &raw[u] );
				#pragma unroll
				for ( int i = 0; i < 4; ++i ) { const float2 f = __half22float2( w2[i] ); wf[2 * i] = f.x; wf[2 * i + 1] = f.y; }
			}
			else {
				#pragma unroll
				for ( int i = 0; i < 8; ++i ) wf[i] = 0.0f;
			}
			float v[8];
			#pragma unroll
			for ( int r = 0; r < 8; ++r ) {
				float a = 0.0f;
				#pragma unroll
				for ( int i = 0; i < 8; ++i ) a = fmaf( xv[r][i], wf[i], a );
				v[r] = a;
			}
			// reduce-scatter over lane bits 3, 2, 1 (within the half-warp): the lane keeps row ( b3 b2 b1 ), then one more add over bit 0
			#pragma unroll
			for ( int step = 0; step < 3; ++step ) {
				const int off = 8 >> step, half_n = 4 >> step;
				const bool hi = ( lane & off ) != 0;
				#pragma unroll
				for ( int i = 0; i < half_n; ++i ) {
					const float send = hi ? v[i] : v[i + half_n];
					const float recv = __shfl_xor_sync( 0xffffffff, send, off );
					v[i] = ( hi ? v[i + half_n] : v[i] ) + recv;
				}
			}
			const float sum = v[0] + __shfl_xor_sync( 0xffffffff, v[0], 1 );
			const int r = ( ( lane >> 3 ) & 1 ) * 4 + ( ( lane >> 2 ) & 1 ) * 2 + ( ( lane >> 1 ) & 1 );
			if ( ( lane & 1 ) == 0 && r < rows && j < outputs ) g_gr_partials[( static_cast<size_t>( blockIdx.x ) * 8 + r ) * GRS_MAX_OUT + j] = sum;
		}
	}
}

__global__ void gr_gate_reduce_kernel( float* __restrict__ t, float* __restrict__ post, int rank, int hc, int outputs, int rows ) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx >= rows * outputs ) return;
	const int r = idx / outputs, j = idx % outputs;
	float sum = 0.0f;
	#pragma unroll 16
	for ( int b = 0; b < GRS_SK; ++b ) sum += g_gr_partials[( static_cast<size_t>( b ) * 8 + r ) * GRS_MAX_OUT + j];
	const float z = sum / hc;
	if ( j < rank ) t[static_cast<size_t>( r ) * rank + j] = z / ( 1.0f + __expf( -z ) );
	else post[static_cast<size_t>( r ) * hc + ( j - rank )] = 2.0f / ( 1.0f + __expf( -z ) );
}

// mixed[r, d] = mean_s( sigmoid( up[s * hidden + d] . t[r] ) * normed[r, s, d] ). A block owns 8 channels d of all 4 streams = 32 up rows; 8 lanes
// stream one up row (16-byte chunks, all in flight), the 8 lanes reduce with shuffles.
constexpr int GRD_CH = 8;
constexpr int GRD_M = 2;   // up rows per thread of the several-rows up-mix
template<int NR>
__global__ __launch_bounds__( 256 )
void gr_up_mix_decode_kernel( const float* __restrict__ normed, const half* __restrict__ up, const float* __restrict__ t, half* __restrict__ mixed,
	int hidden, int hc, int rank, int rows ) {
	extern __shared__ float s_t[];   // [NR][rank]
	const int d0 = blockIdx.x * GRD_CH;
	const int o = threadIdx.x >> 3;           // output within the block: stream s = o / 8, channel d0 + o % 8
	const int l8 = threadIdx.x & 7;
	const int s = o >> 3, dl = o & 7;
	const int chunks = rank / 8;
	const half* u = up + static_cast<size_t>( s * hidden + d0 + dl ) * rank;
	uint4 raw[6];   // rank <= 384
	#pragma unroll
	for ( int i = 0; i < 6; ++i ) {
		const int c = l8 + i * 8;
		if ( c < chunks ) raw[i] = __ldg( reinterpret_cast<const uint4*>( u ) + c );
	}
	{
		// t -> shared memory as float4 (rows * rank is a multiple of 4), every load of a thread issued before the first store
		const int n4 = rows * rank / 4;
		float4 tv[( NR * 384 / 4 + 255 ) / 256];
		#pragma unroll
		for ( int q = 0; q < ( NR * 384 / 4 + 255 ) / 256; ++q ) {
			const int i = threadIdx.x + q * 256;
			if ( i < n4 ) tv[q] = __ldg( reinterpret_cast<const float4*>( t ) + i );
		}
		#pragma unroll
		for ( int q = 0; q < ( NR * 384 / 4 + 255 ) / 256; ++q ) {
			const int i = threadIdx.x + q * 256;
			if ( i < n4 ) reinterpret_cast<float4*>( s_t )[i] = tv[q];
		}
	}
	__syncthreads();
	float acc[NR];
	#pragma unroll
	for ( int r = 0; r < NR; ++r ) acc[r] = 0.0f;
	#pragma unroll
	for ( int i = 0; i < 6; ++i ) {
		const int c = l8 + i * 8;
		if ( c >= chunks ) continue;
		const half2* u2 = reinterpret_cast<const half2*>( &raw[i] );
		float uf[8];
		#pragma unroll
		for ( int q = 0; q < 4; ++q ) { const float2 f = __half22float2( u2[q] ); uf[2 * q] = f.x; uf[2 * q + 1] = f.y; }
		#pragma unroll
		for ( int r = 0; r < NR; ++r ) {
			if ( r < rows ) {
				const float4 t0 = *reinterpret_cast<const float4*>( s_t + r * rank + c * 8 );
				const float4 t1 = *reinterpret_cast<const float4*>( s_t + r * rank + c * 8 + 4 );
				float v = acc[r];
				v = fmaf( t0.x, uf[0], v ); v = fmaf( t0.y, uf[1], v ); v = fmaf( t0.z, uf[2], v ); v = fmaf( t0.w, uf[3], v );
				v = fmaf( t1.x, uf[4], v ); v = fmaf( t1.y, uf[5], v ); v = fmaf( t1.z, uf[6], v ); v = fmaf( t1.w, uf[7], v );
				acc[r] = v;
			}
		}
	}
	__shared__ float sg[4][GRD_CH][NR];
	#pragma unroll
	for ( int r = 0; r < NR; ++r ) {
		float v = acc[r];
		v += __shfl_xor_sync( 0xffffffff, v, 1 );
		v += __shfl_xor_sync( 0xffffffff, v, 2 );
		v += __shfl_xor_sync( 0xffffffff, v, 4 );
		if ( l8 == 0 ) sg[s][dl][r] = 1.0f / ( 1.0f + __expf( -v ) );
	}
	__syncthreads();
	if ( threadIdx.x < GRD_CH * NR ) {
		const int dl2 = threadIdx.x / NR, r = threadIdx.x % NR;
		if ( r < rows ) {
			float out = 0.0f;
			for ( int ss = 0; ss < hc; ++ss ) out += sg[ss][dl2][r] * normed[( static_cast<size_t>( r ) * hc + ss ) * hidden + d0 + dl2];
			mixed[static_cast<size_t>( r ) * hidden + d0 + dl2] = __float2half_rn( out / hc );
		}
	}
}

// Several rows (3..8): a thread streams M up rows (channels d .. d + M - 1 of one stream) against the same chunks of t, so each t value read from shared
// memory serves M outputs. A block owns 8 * M channels of all 4 streams; the 8 lanes of a group reduce-scatter their M outputs x 8 rows, each lane
// ending with row l8 of the M outputs.
template<int RANK_CHUNKS, int M>
__global__ __launch_bounds__( 256 )
void gr_up_mix_rows_kernel( const float* __restrict__ normed, const half* __restrict__ up, const float* __restrict__ t, half* __restrict__ mixed,
	int hidden, int hc, int rank, int rows ) {
	constexpr int CB = 8 * M;   // channels per block
	extern __shared__ float s_t[];   // [8][rank]
	const int d0 = blockIdx.x * CB;
	const int l8 = threadIdx.x & 7, g = threadIdx.x >> 3;
	const int s = g >> 3, cq = g & 7;
	uint4 raw[M][RANK_CHUNKS];
	#pragma unroll
	for ( int m = 0; m < M; ++m ) {
		const half* u = up + static_cast<size_t>( s * hidden + d0 + cq * M + m ) * rank;
		#pragma unroll
		for ( int i = 0; i < RANK_CHUNKS; ++i ) raw[m][i] = __ldg( reinterpret_cast<const uint4*>( u ) + l8 + 8 * i );
	}
	{
		const int n4 = rows * rank / 4;
		for ( int i = threadIdx.x; i < n4; i += 256 ) reinterpret_cast<float4*>( s_t )[i] = __ldg( reinterpret_cast<const float4*>( t ) + i );
	}
	__syncthreads();
	float acc[M][8];
	#pragma unroll
	for ( int m = 0; m < M; ++m ) {
		#pragma unroll
		for ( int r = 0; r < 8; ++r ) acc[m][r] = 0.0f;
	}
	#pragma unroll
	for ( int i = 0; i < RANK_CHUNKS; ++i ) {
		const int c = l8 + 8 * i;
		float uf[M][8];
		#pragma unroll
		for ( int m = 0; m < M; ++m ) {
			const half2* u2 = reinterpret_cast<const half2*>( &raw[m][i] );
			#pragma unroll
			for ( int q = 0; q < 4; ++q ) { const float2 f = __half22float2( u2[q] ); uf[m][2 * q] = f.x; uf[m][2 * q + 1] = f.y; }
		}
		#pragma unroll
		for ( int r = 0; r < 8; ++r ) {
			if ( r < rows ) {
				const float4 t0 = *reinterpret_cast<const float4*>( s_t + r * rank + c * 8 );
				const float4 t1 = *reinterpret_cast<const float4*>( s_t + r * rank + c * 8 + 4 );
				#pragma unroll
				for ( int m = 0; m < M; ++m ) {
					float v = acc[m][r];
					v = fmaf( t0.x, uf[m][0], v ); v = fmaf( t0.y, uf[m][1], v ); v = fmaf( t0.z, uf[m][2], v ); v = fmaf( t0.w, uf[m][3], v );
					v = fmaf( t1.x, uf[m][4], v ); v = fmaf( t1.y, uf[m][5], v ); v = fmaf( t1.z, uf[m][6], v ); v = fmaf( t1.w, uf[m][7], v );
					acc[m][r] = v;
				}
			}
		}
	}
	// 8 * M values (r * M + m) over the 8 lanes: reduce-scatter on lane bits 2, 1, 0 leaves lane l8 with row l8 of the M outputs
	float v[8 * M];
	#pragma unroll
	for ( int r = 0; r < 8; ++r ) {
		#pragma unroll
		for ( int m = 0; m < M; ++m ) v[r * M + m] = acc[m][r];
	}
	#pragma unroll
	for ( int step = 0; step < 3; ++step ) {
		const int off = 4 >> step, half_n = ( 4 * M ) >> step;
		const bool hi = ( l8 & off ) != 0;
		#pragma unroll
		for ( int i = 0; i < half_n; ++i ) {
			const float send = hi ? v[i] : v[i + half_n];
			const float recv = __shfl_xor_sync( 0xffffffff, send, off );
			v[i] = ( hi ? v[i + half_n] : v[i] ) + recv;
		}
	}
	__shared__ float sg[4][CB][8];   // sigmoid gate per stream, channel, row
	#pragma unroll
	for ( int m = 0; m < M; ++m ) sg[s][cq * M + m][l8] = 1.0f / ( 1.0f + __expf( -v[m] ) );
	__syncthreads();
	if ( threadIdx.x < 8 * CB ) {
		const int r = threadIdx.x / CB, dl = threadIdx.x % CB;
		if ( r < rows ) {
			float out = 0.0f;
			#pragma unroll
			for ( int ss = 0; ss < 4; ++ss ) out += sg[ss][dl][r] * normed[( static_cast<size_t>( r ) * hc + ss ) * hidden + d0 + dl];
			mixed[static_cast<size_t>( r ) * hidden + d0 + dl] = __float2half_rn( out / hc );
		}
	}
}

__global__ void gr_apply_kernel( float* __restrict__ x, const float* __restrict__ post, const half* __restrict__ y, int hidden, int hc ) {
	const int r = blockIdx.y, s = blockIdx.z;
	const int d = blockIdx.x * blockDim.x + threadIdx.x;
	if ( d >= hidden ) return;
	const float p = post[static_cast<size_t>( r ) * hc + s];
	x[( static_cast<size_t>( r ) * hc + s ) * hidden + d] += p * __half2float( y[static_cast<size_t>( r ) * hidden + d] );
}

// One block per row: g = sigmoid( x . w ), then y += g * s
__global__ void gr_sigmoid_gated_kernel( half* __restrict__ y, const half* __restrict__ sh, const half* __restrict__ x, const half* __restrict__ w, int hidden ) {
	const int r = blockIdx.x;
	const half* xr = x + static_cast<size_t>( r ) * hidden;
	float acc = 0.0f;
	for ( int i = threadIdx.x; i < hidden; i += blockDim.x ) acc = fmaf( __half2float( xr[i] ), __half2float( w[i] ), acc );
	acc = warp_sum( acc );
	__shared__ float ws[32];
	__shared__ float gate;
	if ( ( threadIdx.x & 31 ) == 0 ) ws[threadIdx.x >> 5] = acc;
	__syncthreads();
	if ( threadIdx.x == 0 ) {
		float total = 0.0f;
		for ( int i = 0; i < ( blockDim.x >> 5 ); ++i ) total += ws[i];
		gate = 1.0f / ( 1.0f + __expf( -total ) );
	}
	__syncthreads();
	for ( int i = threadIdx.x; i < hidden; i += blockDim.x ) {
		const size_t o = static_cast<size_t>( r ) * hidden + i;
		y[o] = __float2half_rn( __half2float( y[o] ) + gate * __half2float( sh[o] ) );
	}
}

// One block per (row, stream)
__global__ void gr_stream_norm_kernel( const float* __restrict__ x, const half* __restrict__ w, half* __restrict__ out, int hidden, int hc, float eps ) {
	const int r = blockIdx.x, s = blockIdx.y;
	const size_t base = ( static_cast<size_t>( r ) * hc + s ) * hidden;
	float sq = 0.0f;
	for ( int d = threadIdx.x; d < hidden; d += blockDim.x ) {
		const float v = x[base + d];
		sq += v * v;
	}
	sq = warp_sum( sq );
	__shared__ float ws[32];
	if ( ( threadIdx.x & 31 ) == 0 ) ws[threadIdx.x >> 5] = sq;
	__syncthreads();
	float total = 0.0f;
	for ( int i = 0; i < ( blockDim.x >> 5 ); ++i ) total += ws[i];
	const float inv = rsqrtf( total / hidden + eps );
	for ( int d = threadIdx.x; d < hidden; d += blockDim.x ) {
		out[base + d] = __float2half_rn( x[base + d] * inv * ( 1.0f + __half2float( w[static_cast<size_t>( s ) * hidden + d] ) ) );
	}
}

__global__ void gr_add_broadcast_kernel( const half* __restrict__ h, const half* __restrict__ e, float* __restrict__ streams, int hidden, int hc ) {
	const int r = blockIdx.y, s = blockIdx.z;
	const int d = blockIdx.x * blockDim.x + threadIdx.x;
	if ( d >= hidden ) return;
	const size_t o = ( static_cast<size_t>( r ) * hc + s ) * hidden + d;
	streams[o] = __half2float( h[o] ) + __half2float( e[static_cast<size_t>( r ) * hidden + d] );
}

// Tensor-core path (more than a few rows). The fp32 operands are split into an fp16 hi part and an fp16 lo part (x = hi + lo to ~2^-22) and each product is two
// fp16 GEMMs accumulated in fp32, so the result keeps the fp32 kernels' accuracy: z = fp32 [rows, ld] = normed @ [down; inject]^T. t = silu( z / hc ) over the
// rank columns (stored as hi / lo), post = 2 * sigmoid( z / hc ) over the hc inject columns that follow
__global__ void gr_gate_act_kernel( const float* __restrict__ z, int ld, half* __restrict__ t_hi, half* __restrict__ t_lo, float* __restrict__ post, int rank, int hc, int has_inject ) {
	const int r = blockIdx.x;
	const float* zr = z + static_cast<size_t>( r ) * ld;
	for ( int j = threadIdx.x; j < rank + ( has_inject ? hc : 0 ); j += blockDim.x ) {
		const float v = zr[j] / hc;
		if ( j < rank ) {
			const float tv = v / ( 1.0f + __expf( -v ) );
			const half hi = __float2half_rn( tv );
			t_hi[static_cast<size_t>( r ) * rank + j] = hi;
			t_lo[static_cast<size_t>( r ) * rank + j] = __float2half_rn( tv - __half2float( hi ) );
		}
		else post[static_cast<size_t>( r ) * hc + ( j - rank )] = 2.0f / ( 1.0f + __expf( -v ) );
	}
}

// mixed[r, d] = mean_s( sigmoid( u[r, s * hidden + d] ) * normed[r, s, d] )
__global__ void gr_up_epilogue_kernel( const float* __restrict__ u, const float* __restrict__ normed, half* __restrict__ mixed, int hidden, int hc ) {
	const int r = blockIdx.y;
	const int d = blockIdx.x * blockDim.x + threadIdx.x;
	if ( d >= hidden ) return;
	float out = 0.0f;
	for ( int s = 0; s < hc; ++s ) {
		const float g = 1.0f / ( 1.0f + __expf( -u[static_cast<size_t>( r ) * hc * hidden + static_cast<size_t>( s ) * hidden + d] ) );
		out += g * normed[( static_cast<size_t>( r ) * hc + s ) * hidden + d];
	}
	mixed[static_cast<size_t>( r ) * hidden + d] = __float2half_rn( out / hc );
}

} // namespace

void GatedResidual::norm( float* x, const backend::half_t* norm_w, float* normed, backend::half_t* normed_hi, backend::half_t* normed_lo, int rows, int hidden, int hc, float eps,
	const float* prev_post, const backend::half_t* prev_y, backend::stream_t stream ) {
	gr_norm_kernel<<<dim3( rows, hc ), 256, 0, stream>>>( x, reinterpret_cast<const half*>( norm_w ), normed, hidden, hc, eps, prev_post, reinterpret_cast<const half*>( prev_y ),
		reinterpret_cast<half*>( normed_hi ), reinterpret_cast<half*>( normed_lo ) );
}

void GatedResidual::gemm_f32( const backend::half_t* a, const backend::half_t* w, float* out, int m, int n, int k, bool accumulate, backend::stream_t stream ) {
	int device = 0;
	cudaGetDevice( &device );
	cublasHandle_t h = trellis_get_cublas( device );
	cublasSetStream( h, stream );
	const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;
	// C^T[n, m] = W[n, k] A^T with W and A row-major (k fastest)
	const cublasStatus_t st = cublasGemmEx( h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, w, CUDA_R_16F, k, a, CUDA_R_16F, k, &beta, out, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
		CUBLAS_GEMM_DEFAULT_TENSOR_OP );
	ENGINE_ASSERT( st == CUBLAS_STATUS_SUCCESS, "GatedResidual::gemm_f32: cublasGemmEx failed" );
}

void GatedResidual::gate_act( const float* z, int ld, backend::half_t* t_hi, backend::half_t* t_lo, float* post, int rows, int rank, int hc, bool has_inject, backend::stream_t stream ) {
	gr_gate_act_kernel<<<rows, 128, 0, stream>>>( z, ld, reinterpret_cast<half*>( t_hi ), reinterpret_cast<half*>( t_lo ), post, rank, hc, has_inject ? 1 : 0 );
}

void GatedResidual::up_epilogue( const float* u, const float* normed, backend::half_t* mixed, int rows, int hidden, int hc, backend::stream_t stream ) {
	gr_up_epilogue_kernel<<<dim3( ( hidden + 255 ) / 256, rows ), 256, 0, stream>>>( u, normed, reinterpret_cast<half*>( mixed ), hidden, hc );
}

void GatedResidual::stream_norm( const float* x, const backend::half_t* w, backend::half_t* out, int rows, int hidden, int hc, float eps, backend::stream_t stream ) {
	gr_stream_norm_kernel<<<dim3( rows, hc ), 256, 0, stream>>>( x, reinterpret_cast<const half*>( w ), reinterpret_cast<half*>( out ), hidden, hc, eps );
}

void GatedResidual::add_broadcast( const backend::half_t* h, const backend::half_t* e, float* streams, int rows, int hidden, int hc, backend::stream_t stream ) {
	gr_add_broadcast_kernel<<<dim3( ( hidden + 255 ) / 256, rows, hc ), 256, 0, stream>>>( reinterpret_cast<const half*>( h ), reinterpret_cast<const half*>( e ), streams, hidden, hc );
}

void GatedResidual::add_sigmoid_gated( backend::half_t* y, const backend::half_t* s, const backend::half_t* x, const backend::half_t* w, int rows, int hidden, backend::stream_t stream ) {
	gr_sigmoid_gated_kernel<<<rows, 256, 0, stream>>>( reinterpret_cast<half*>( y ), reinterpret_cast<const half*>( s ), reinterpret_cast<const half*>( x ), reinterpret_cast<const half*>( w ), hidden );
}

void GatedResidual::init_streams( const backend::half_t* in, float* x, int rows, int hidden, int hc, backend::stream_t stream ) {
	gr_init_kernel<<<dim3( ( hidden + 255 ) / 256, rows ), 256, 0, stream>>>( reinterpret_cast<const half*>( in ), x, hidden, hc );
}

namespace {

void mix_fast( const float* normed, const half* down, const half* up, const half* inject, half* mixed, float* post, float* t, int rows, int hidden, int hc, int rank,
	cudaStream_t stream ) {
	const int flat = hc * hidden;
	const int outputs = rank + ( inject != nullptr ? hc : 0 );
	// 3..8 rows: split-K gate + the 4-outputs-per-thread up-mix (hc 4, flat 40 * 256, rank a multiple of 64 up to 384)
	const bool several = rows >= 3 && hc == 4 && flat == GRS_SK * GRS_SLICE && outputs <= GRS_MAX_OUT && hidden % ( 8 * GRD_M ) == 0 && rank % 64 == 0 && rank <= 384;
	#define GR_UPMIX( NR ) \
		gr_up_mix_decode_kernel<NR><<<hidden / GRD_CH, 256, static_cast<size_t>( NR ) * rank * sizeof( float ), stream>>>( normed, up, t, mixed, hidden, hc, rank, rows )
	#define GR_FAST( NR ) \
		gr_gate_decode_kernel<NR, 1><<<outputs, GRD_THREADS, 0, stream>>>( normed, down, inject, t, post, flat, rank, hc, rows ); \
		GR_UPMIX( NR )
	if ( rows == 1 ) { GR_FAST( 1 ); }
	else if ( rows == 2 ) { GR_FAST( 2 ); }
	else if ( several ) {
		gr_gate_splitk_kernel<<<GRS_SK, GRS_WARPS * 32, 0, stream>>>( normed, down, inject, flat, rank, outputs, rows );
		gr_gate_reduce_kernel<<<( rows * outputs + 127 ) / 128, 128, 0, stream>>>( t, post, rank, hc, outputs, rows );
		const size_t smem = static_cast<size_t>( 8 ) * rank * sizeof( float );
		switch ( rank / 64 ) {
			case 5: gr_up_mix_rows_kernel<5, GRD_M><<<hidden / ( 8 * GRD_M ), 256, smem, stream>>>( normed, up, t, mixed, hidden, hc, rank, rows ); break;
			case 4: gr_up_mix_rows_kernel<4, GRD_M><<<hidden / ( 8 * GRD_M ), 256, smem, stream>>>( normed, up, t, mixed, hidden, hc, rank, rows ); break;
			case 6: gr_up_mix_rows_kernel<6, GRD_M><<<hidden / ( 8 * GRD_M ), 256, smem, stream>>>( normed, up, t, mixed, hidden, hc, rank, rows ); break;
			default: GR_UPMIX( 8 ); break;
		}
	}
	else if ( rows <= 4 ) { GR_FAST( 4 ); }
	else { GR_FAST( 8 ); }
	#undef GR_FAST
	#undef GR_UPMIX
}

void mix_rows( const float* x, const half* norm_w, const half* down, const half* up, const half* inject, half* mixed, float* post, float* scratch, int rows, int hidden,
	int hc, int rank, float eps, cudaStream_t stream, const float* prev_post, const half* prev_y, bool fast ) {
	const int flat = hc * hidden;
	float* normed = scratch;
	float* t = scratch + static_cast<size_t>( rows ) * flat;
	gr_norm_kernel<<<dim3( rows, hc ), 256, 0, stream>>>( const_cast<float*>( x ), norm_w, normed, hidden, hc, eps, prev_post, prev_y, nullptr, nullptr );
	if ( fast ) {
		mix_fast( normed, down, up, inject, mixed, post, t, rows, hidden, hc, rank, stream );
		return;
	}
	const int has_inject = inject != nullptr ? 1 : 0;
	gr_gate_kernel<<<dim3( ( rank + ( has_inject ? hc : 0 ) + GR_JT - 1 ) / GR_JT, ( rows + GR_RT - 1 ) / GR_RT ), 128, 0, stream>>>(
		normed, down, inject, t, post, flat, rank, hc, rows, has_inject );
	gr_up_mix_kernel<<<dim3( hidden / GR_DT, ( rows + GR_RT - 1 ) / GR_RT ), 128, static_cast<size_t>( GR_RT ) * rank * sizeof( float ), stream>>>(
		normed, up, t, mixed, hidden, hc, rank, rows );
}

} // namespace

void GatedResidual::mix(
	const float* x, const backend::half_t* norm_w, const backend::half_t* down, const backend::half_t* up, const backend::half_t* inject,
	backend::half_t* mixed, float* post, float* scratch, int rows, int hidden, int hc, int rank, float eps, backend::stream_t stream,
	const float* prev_post, const backend::half_t* prev_y
) {
	ENGINE_ASSERT( hc == 4 && ( hc * hidden ) % 1024 == 0 && hidden % GR_DT == 0 && rank % 64 == 0, "GatedResidual::mix: shape" );
	// Decode-sized calls take the streaming kernels (hc 4, flat <= 10240, hidden % 8, rank <= 384); anything else the row-tile ones
	const bool fast = rows <= 8 && hc * hidden <= 8 * GRD_CHUNKS * GRD_THREADS && hidden % GRD_CH == 0 && rank <= 384;
	mix_rows( x, reinterpret_cast<const half*>( norm_w ), reinterpret_cast<const half*>( down ), reinterpret_cast<const half*>( up ), reinterpret_cast<const half*>( inject ),
		reinterpret_cast<half*>( mixed ), post, scratch, rows, hidden, hc, rank, eps, stream, prev_post, reinterpret_cast<const half*>( prev_y ), fast );
}

void GatedResidual::mix_reference(
	const float* x, const backend::half_t* norm_w, const backend::half_t* down, const backend::half_t* up, const backend::half_t* inject,
	backend::half_t* mixed, float* post, float* scratch, int rows, int hidden, int hc, int rank, float eps, backend::stream_t stream,
	const float* prev_post, const backend::half_t* prev_y
) {
	ENGINE_ASSERT( hc == 4 && ( hc * hidden ) % 1024 == 0 && hidden % GR_DT == 0 && rank % 64 == 0, "GatedResidual::mix_reference: shape" );
	mix_rows( x, reinterpret_cast<const half*>( norm_w ), reinterpret_cast<const half*>( down ), reinterpret_cast<const half*>( up ), reinterpret_cast<const half*>( inject ),
		reinterpret_cast<half*>( mixed ), post, scratch, rows, hidden, hc, rank, eps, stream, prev_post, reinterpret_cast<const half*>( prev_y ), false );
}

void GatedResidual::apply( float* x, const float* post, const backend::half_t* y, int rows, int hidden, int hc, backend::stream_t stream ) {
	gr_apply_kernel<<<dim3( ( hidden + 255 ) / 256, rows, hc ), 256, 0, stream>>>( x, post, reinterpret_cast<const half*>( y ), hidden, hc );
}

} // namespace engine::modules
