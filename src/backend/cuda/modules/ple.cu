#include "engine/modules/ple.h"
#include <cuda_fp16.h>
#include <cmath>

namespace engine::modules {

namespace {

__device__ __forceinline__ float warp_sum_p( float v ) {
	#pragma unroll
	for ( int o = 16; o > 0; o >>= 1 ) v += __shfl_xor_sync( 0xffffffff, v, o );
	return v;
}

// Sum over the block of up to 3 values at once (blockDim.x <= 1024)
__device__ __forceinline__ void block_sum3( float& a, float& b, float& c ) {
	__shared__ float sh[3][32];
	a = warp_sum_p( a ); b = warp_sum_p( b ); c = warp_sum_p( c );
	const int w = threadIdx.x >> 5;
	__syncthreads();
	if ( ( threadIdx.x & 31 ) == 0 ) { sh[0][w] = a; sh[1][w] = b; sh[2][w] = c; }
	__syncthreads();
	float ta = 0.0f, tb = 0.0f, tc = 0.0f;
	for ( int i = 0; i < ( blockDim.x >> 5 ); ++i ) { ta += sh[0][i]; tb += sh[1][i]; tc += sh[2][i]; }
	a = ta; b = tb; c = tc;
}

// One block per (row, stream)
__global__ void ple_gate_kernel(
	const half* __restrict__ key, const half* __restrict__ value, const float* __restrict__ streams,
	const half* __restrict__ wk, const half* __restrict__ wq, const half* __restrict__ wc,
	float* __restrict__ gated, half* __restrict__ normed, int hidden, int hc, float eps, float gate_scale
) {
	const int r = blockIdx.x, s = blockIdx.y;
	const size_t base = ( static_cast<size_t>( r ) * hc + s ) * hidden;
	const half* k = key + base;
	const float* x = streams + base;
	const half* v = value + static_cast<size_t>( r ) * hidden;
	const half* wks = wk + static_cast<size_t>( s ) * hidden;
	const half* wqs = wq + static_cast<size_t>( s ) * hidden;
	const half* wcs = wc + static_cast<size_t>( s ) * hidden;

	float sk = 0.0f, sq = 0.0f, sv = 0.0f;
	for ( int d = threadIdx.x; d < hidden; d += blockDim.x ) {
		const float kv = __half2float( k[d] ), xv = x[d], vv = __half2float( v[d] );
		sk += kv * kv; sq += xv * xv; sv += vv * vv;
	}
	block_sum3( sk, sq, sv );
	const float ik = rsqrtf( sk / hidden + eps ), iq = rsqrtf( sq / hidden + eps );
	float dot = 0.0f, z0 = 0.0f, z1 = 0.0f;
	for ( int d = threadIdx.x; d < hidden; d += blockDim.x ) {
		const float kn = __half2float( k[d] ) * ik * ( 1.0f + __half2float( wks[d] ) );
		const float qn = x[d] * iq * ( 1.0f + __half2float( wqs[d] ) );
		dot += kn * qn;
	}
	block_sum3( dot, z0, z1 );
	const float g = dot * gate_scale;
	const float sign = ( g > 0.0f ) ? 1.0f : ( g < 0.0f ? -1.0f : 0.0f );
	const float ss = sign * sqrtf( fmaxf( fabsf( g ), 1e-6f ) );
	const float sg = 1.0f / ( 1.0f + __expf( -ss ) );
	// rms of gated = |sg| * rms(value)
	const float ig = rsqrtf( sg * sg * sv / hidden + eps );
	for ( int d = threadIdx.x; d < hidden; d += blockDim.x ) {
		const float gv = sg * __half2float( v[d] );
		gated[base + d] = gv;
		normed[base + d] = __float2half_rn( gv * ig * ( 1.0f + __half2float( wcs[d] ) ) );
	}
}

__global__ void ple_conv_kernel(
	const half* __restrict__ normed, const half* __restrict__ conv_w, const half* __restrict__ ring, const pod::BatchRows* __restrict__ rows,
	const float* __restrict__ gated, float* __restrict__ streams, int channels, int dilation
) {
	const int r = blockIdx.y;
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	if ( c >= channels ) return;
	const int slot = rows->slot[r], pos = rows->pos[r];
	float acc = 0.0f;
	#pragma unroll
	for ( int j = 0; j < 4; ++j ) {
		const int k = ( 3 - j ) * dilation;
		const int sp = pos - k;
		float xv = 0.0f;
		if ( sp >= 0 ) {
			const int rr = r - k;
			if ( k == 0 || ( rr >= 0 && rows->slot[rr] == slot && rows->pos[rr] == sp ) ) {
				xv = __half2float( normed[static_cast<size_t>( k == 0 ? r : rr ) * channels + c] );
			}
			else {
				xv = __half2float( ring[( static_cast<size_t>( slot ) * PLE::RING + ( sp % PLE::RING ) ) * channels + c] );
			}
		}
		acc = fmaf( __half2float( conv_w[static_cast<size_t>( c ) * 4 + j] ), xv, acc );
	}
	const float act = acc / ( 1.0f + __expf( -acc ) );
	const size_t o = static_cast<size_t>( r ) * channels + c;
	streams[o] += gated[o] + act;
}

__global__ void ple_ring_kernel( const half* __restrict__ normed, half* __restrict__ ring, const pod::BatchRows* __restrict__ rows, int channels ) {
	const int r = blockIdx.y;
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	if ( c >= channels ) return;
	ring[( static_cast<size_t>( rows->slot[r] ) * PLE::RING + ( rows->pos[r] % PLE::RING ) ) * channels + c] = normed[static_cast<size_t>( r ) * channels + c];
}

} // namespace

void PLE::gate_and_norm( const backend::half_t* key, const backend::half_t* value, const float* streams,
	const backend::half_t* nk, const backend::half_t* nq, const backend::half_t* nc, float* gated, backend::half_t* normed,
	int rows, int hidden, int hc, float eps, float gate_scale, backend::stream_t stream ) {
	ple_gate_kernel<<<dim3( rows, hc ), 256, 0, stream>>>( reinterpret_cast<const half*>( key ), reinterpret_cast<const half*>( value ), streams,
		reinterpret_cast<const half*>( nk ), reinterpret_cast<const half*>( nq ), reinterpret_cast<const half*>( nc ), gated, reinterpret_cast<half*>( normed ),
		hidden, hc, eps, gate_scale );
}

void PLE::conv_apply( const backend::half_t* normed, const backend::half_t* conv_w, const backend::half_t* ring, const pod::BatchRows* d_rows,
	const float* gated, float* streams, int rows, int channels, int dilation, backend::stream_t stream ) {
	ple_conv_kernel<<<dim3( ( channels + 255 ) / 256, rows ), 256, 0, stream>>>( reinterpret_cast<const half*>( normed ), reinterpret_cast<const half*>( conv_w ),
		reinterpret_cast<const half*>( ring ), d_rows, gated, streams, channels, dilation );
}

void PLE::ring_update( const backend::half_t* normed, backend::half_t* ring, const pod::BatchRows* d_rows, int rows, int channels, backend::stream_t stream ) {
	ple_ring_kernel<<<dim3( ( channels + 255 ) / 256, rows ), 256, 0, stream>>>( reinterpret_cast<const half*>( normed ), reinterpret_cast<half*>( ring ), d_rows, channels );
}

} // namespace engine::modules
