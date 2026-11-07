#include "engine/modules/hessian.h"
#include "engine/modules/gemm.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <mutex>

namespace engine::modules {

namespace {

// xt[k, rows8] = x[rows, k]^T, the columns past rows zero
__global__ void transpose_pad_kernel( const half* __restrict__ x, half* __restrict__ xt, int rows, int rows8, int k ) {
	__shared__ half tile[32][33];
	const int r0 = blockIdx.y * 32, c0 = blockIdx.x * 32;
	for ( int i = threadIdx.y; i < 32; i += blockDim.y ) {
		const int r = r0 + i, c = c0 + threadIdx.x;
		tile[i][threadIdx.x] = r < rows && c < k ? x[static_cast<size_t>( r ) * k + c] : __float2half( 0.0f );
	}
	__syncthreads();
	for ( int i = threadIdx.y; i < 32; i += blockDim.y ) {
		const int c = c0 + i, r = r0 + threadIdx.x;
		if ( c < k && r < rows8 ) xt[static_cast<size_t>( c ) * rows8 + r] = tile[threadIdx.x][i];
	}
}

} // namespace

void Hessian::accumulate( float* h, const half* x, int rows, int k, cudaStream_t stream ) {
	// H [k, k] += x^T x: with xt = x^T ([k, rows], rows padded to 16 bytes), H = xt xt^T, the GEMM's weights-as-rows form
	static std::mutex mu;
	static half* buf[64] = {};
	static size_t cap[64] = {};
	int device = 0;
	CUDA_CHECK( cudaGetDevice( &device ) );
	const int rows8 = ( rows + 7 ) / 8 * 8;
	const size_t need = static_cast<size_t>( rows8 ) * k;
	half* xt = nullptr;
	{
		std::lock_guard<std::mutex> lk( mu );
		if ( need > cap[device] ) {
			CUDA_CHECK( cudaStreamSynchronize( stream ) );   // calibration runs eagerly: the old buffer's readers are done
			if ( buf[device] ) CUDA_CHECK( cudaFree( buf[device] ) );
			CUDA_CHECK( cudaMalloc( &buf[device], need * sizeof( half ) ) );
			cap[device] = need;
		}
		xt = buf[device];
	}
	transpose_pad_kernel<<<dim3( ( k + 31 ) / 32, ( rows8 + 31 ) / 32 ), dim3( 32, 8 ), 0, stream>>>( x, xt, rows, rows8, k );
	Gemm::tn( xt, rows8, xt, rows8, h, k, true, k, k, rows8, 1.0f, stream );
}

} // namespace engine::modules
