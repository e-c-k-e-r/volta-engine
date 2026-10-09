#include "engine/modules/gated_residual.h"
#include "engine/config.h"
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

// One block per (row, stream): normed[r, s, :] = x * rsqrt(mean(x^2) + eps) * (1 + w[s, :])
__global__ void gr_norm_kernel( const float* __restrict__ x, const half* __restrict__ w, float* __restrict__ normed, int hidden, int hc, float eps ) {
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
		normed[base + d] = x[base + d] * inv * ( 1.0f + __half2float( w[static_cast<size_t>( s ) * hidden + d] ) );
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

} // namespace

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

void GatedResidual::mix(
	const float* x, const backend::half_t* norm_w, const backend::half_t* down, const backend::half_t* up, const backend::half_t* inject,
	backend::half_t* mixed, float* post, float* scratch, int rows, int hidden, int hc, int rank, float eps, backend::stream_t stream
) {
	const int flat = hc * hidden;
	ENGINE_ASSERT( hc == 4 && flat % 1024 == 0 && hidden % GR_DT == 0 && rank % 64 == 0, "GatedResidual::mix: shape" );
	float* normed = scratch;
	float* t = scratch + static_cast<size_t>( rows ) * flat;
	gr_norm_kernel<<<dim3( rows, hc ), 256, 0, stream>>>( x, reinterpret_cast<const half*>( norm_w ), normed, hidden, hc, eps );
	const int has_inject = inject != nullptr ? 1 : 0;
	gr_gate_kernel<<<dim3( ( rank + ( has_inject ? hc : 0 ) + GR_JT - 1 ) / GR_JT, ( rows + GR_RT - 1 ) / GR_RT ), 128, 0, stream>>>(
		normed, reinterpret_cast<const half*>( down ), reinterpret_cast<const half*>( inject ), t, post, flat, rank, hc, rows, has_inject );
	gr_up_mix_kernel<<<dim3( hidden / GR_DT, ( rows + GR_RT - 1 ) / GR_RT ), 128, static_cast<size_t>( GR_RT ) * rank * sizeof( float ), stream>>>(
		normed, reinterpret_cast<const half*>( up ), t, reinterpret_cast<half*>( mixed ), hidden, hc, rank, rows );
}

void GatedResidual::apply( float* x, const float* post, const backend::half_t* y, int rows, int hidden, int hc, backend::stream_t stream ) {
	gr_apply_kernel<<<dim3( ( hidden + 255 ) / 256, rows, hc ), 256, 0, stream>>>( x, post, reinterpret_cast<const half*>( y ), hidden, hc );
}

} // namespace engine::modules
