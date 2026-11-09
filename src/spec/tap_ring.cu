#include "engine/spec/tap_ring.h"

namespace engine::spec {

namespace {

__device__ __forceinline__ half tap_value( half v ) { return v; }
__device__ __forceinline__ half tap_value( float v ) { return __float2half_rn( v ); }

// T = half, or float for a model whose residual stream is FP32 (Gemma 4); a position row holds taps_n taps of H
template <typename T>
__global__ void dflash_store_taps_kernel(
	half* __restrict__ taps,
	const T* __restrict__ hidden,
	const pod::BatchRows* __restrict__ rows,
	const int* __restrict__ owner,
	int m,
	int tap,
	int H,
	int cap,
	int taps_n
) {
	const int r = blockIdx.y;
	if ( r >= m ) return;
	if ( rows->slot[r] != *owner ) return;
	const int pos = rows->pos[r];
	if ( pos < 0 ) return;
	half* dst = taps + static_cast<size_t>( pos % cap ) * taps_n * H + static_cast<size_t>( tap ) * H;
	const T* src = hidden + static_cast<size_t>( r ) * H;
	for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < H; i += blockDim.x ) dst[i] = tap_value( src[i] );
}

template <typename T>
__global__ void dflash_store_taps_slots_kernel(
	half* __restrict__ taps,
	const T* __restrict__ hidden,
	const pod::BatchRows* __restrict__ rows,
	int m,
	int tap,
	int H,
	int cap,
	size_t slot_stride,
	int taps_n
) {
	const int r = blockIdx.y;
	if ( r >= m ) return;
	const int pos = rows->pos[r];
	if ( pos < 0 ) return;
	half* dst = taps + static_cast<size_t>( rows->slot[r] ) * slot_stride + static_cast<size_t>( pos % cap ) * taps_n * H + static_cast<size_t>( tap ) * H;
	const T* src = hidden + static_cast<size_t>( r ) * H;
	for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < H; i += blockDim.x ) dst[i] = tap_value( src[i] );
}

__global__ void reduce_streams_kernel( const float* __restrict__ streams, half* __restrict__ out, int H, int mode ) {
	const int r = blockIdx.y;
	const float* in = streams + static_cast<size_t>( r ) * 4 * H;
	for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < H; i += gridDim.x * blockDim.x ) {
		float v;
		if ( mode == 0 ) v = ( in[i] + in[H + i] + in[2 * H + i] + in[3 * H + i] ) * 0.25f;
		else v = in[( mode - 1 ) * H + i];
		out[static_cast<size_t>( r ) * H + i] = __float2half_rn( v );
	}
}

} // namespace

void dflash_publish_taps(
	half* taps,
	const half* hidden,
	const pod::BatchRows* d_rows,
	const int* d_owner,
	int m,
	int tap,
	int H,
	int cap,
	cudaStream_t stream,
	int taps_n
) {
	dim3 g( ( H + 255 ) / 256, m );
	dflash_store_taps_kernel<half><<<g, 256, 0, stream>>>( taps, hidden, d_rows, d_owner, m, tap, H, cap, taps_n );
}

void dflash_publish_taps_slots(
	half* taps,
	const half* hidden,
	const pod::BatchRows* d_rows,
	int m,
	int tap,
	int H,
	int cap,
	size_t slot_stride,
	cudaStream_t stream,
	int taps_n
) {
	dim3 g( ( H + 255 ) / 256, m );
	dflash_store_taps_slots_kernel<half><<<g, 256, 0, stream>>>( taps, hidden, d_rows, m, tap, H, cap, slot_stride, taps_n );
}

void dflash_publish_taps_slots_f32( half* taps, const float* hidden, const pod::BatchRows* d_rows, int m, int tap, int H, int cap, size_t slot_stride, int taps_n,
	cudaStream_t stream ) {
	dim3 g( ( H + 255 ) / 256, m );
	dflash_store_taps_slots_kernel<float><<<g, 256, 0, stream>>>( taps, hidden, d_rows, m, tap, H, cap, slot_stride, taps_n );
}

void dflash_reduce_streams( const float* streams, half* out, int m, int H, int mode, cudaStream_t stream ) {
	dim3 g( ( H + 255 ) / 256, m );
	reduce_streams_kernel<<<g, 256, 0, stream>>>( streams, out, H, mode );
}

} // namespace engine::spec
