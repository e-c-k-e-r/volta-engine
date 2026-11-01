#include "engine/modules/sparse_head.h"
#include "engine/config.h"
#include "engine/modules/linear.h"
#include "engine/spec/dflash.h"
#include <cuda_fp16.h>

namespace engine::modules {

namespace {

__global__ void identity_rows_kernel( half* ident, int hidden, int k0 ) {
	const int r = blockIdx.x;
	for ( int c = threadIdx.x; c < hidden; c += blockDim.x ) ident[static_cast<size_t>( r ) * hidden + c] = __float2half( c == k0 + r ? 1.0f : 0.0f );
}

// grid (count), block rows threads (rows <= 1024)
__global__ void gather_kernel( const half* __restrict__ chunk, int chunk_ld, int rows, int k0, const int* __restrict__ cols, half* __restrict__ w_sub, int hidden ) {
	const int j = blockIdx.x;
	const int c = cols[j];
	for ( int i = threadIdx.x; i < rows; i += blockDim.x ) w_sub[static_cast<size_t>( j ) * hidden + k0 + i] = chunk[static_cast<size_t>( i ) * chunk_ld + c];
}

__global__ void fill_kernel( half* logits, int ld, int rows ) {
	const size_t i = static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x;
	if ( i < static_cast<size_t>( ld ) * rows ) logits[i] = __float2half( -65504.0f );
}

__global__ void scatter_kernel( const half* __restrict__ sub, int count, const int* __restrict__ cols, half* __restrict__ logits, int ld ) {
	const int r = blockIdx.y;
	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	if ( j < count ) logits[static_cast<size_t>( r ) * ld + cols[j]] = sub[static_cast<size_t>( r ) * count + j];
}

} // namespace

void SparseHead::identity_rows( half* ident, int rows, int hidden, int k0, cudaStream_t stream ) {
	identity_rows_kernel<<<rows, 256, 0, stream>>>( ident, hidden, k0 );
}

void SparseHead::gather( const half* chunk, int chunk_ld, int rows, int k0, const int* cols, int count, half* w_sub, int hidden, cudaStream_t stream ) {
	if ( count > 0 ) gather_kernel<<<count, 128, 0, stream>>>( chunk, chunk_ld, rows, k0, cols, w_sub, hidden );
}

void SparseHead::extract( const pod::LinearWeight& head, int hidden, int shard_cols, const int* cols, int count, half* ident, half* chunk, int ident_rows,
	half* w_sub, cudaStream_t stream ) {
	for ( int k0 = 0; k0 < hidden && count > 0; k0 += ident_rows ) {
		const int rows = k0 + ident_rows <= hidden ? ident_rows : hidden - k0;
		identity_rows( ident, rows, hidden, k0, stream );
		pod::TensorView x = pod::make_tensor_2d( ident, pod::DType::FP16, rows, hidden );
		pod::TensorView y = pod::make_tensor_2d( chunk, pod::DType::FP16, rows, shard_cols );
		Linear::forward( x, head, y, stream );
		gather( chunk, shard_cols, rows, k0, cols, count, w_sub, hidden, stream );
	}
	CUDA_CHECK( cudaStreamSynchronize( stream ) );
}

void SparseHead::logits( const half* w_sub, int count, const int* cols, const half* x, int rows, int hidden, half* sub, half* logits, int logits_ld, cudaStream_t stream ) {
	engine::spec::dflash_f16_gemv_nt( w_sub, x, sub, rows, count, hidden, stream );
	scatter( sub, count, cols, logits, logits_ld, rows, stream );
}

void SparseHead::scatter( const half* sub, int count, const int* cols, half* logits, int logits_ld, int rows, cudaStream_t stream ) {
	const size_t n = static_cast<size_t>( logits_ld ) * rows;
	fill_kernel<<<static_cast<unsigned>( ( n + 255 ) / 256 ), 256, 0, stream>>>( logits, logits_ld, rows );
	if ( count > 0 ) scatter_kernel<<<dim3( ( count + 255 ) / 256, rows ), 256, 0, stream>>>( sub, count, cols, logits, logits_ld );
}

} // namespace engine::modules
