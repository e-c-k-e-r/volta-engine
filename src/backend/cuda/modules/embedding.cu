#include "engine/modules/embedding.h"
#include <cuda_bf16.h>

namespace engine::modules {

namespace impl {

__global__ void embed_tokens_kernel(
	const __nv_bfloat16* __restrict__ table,
	const int* __restrict__ token_ids,
	half* __restrict__ out,
	int hidden_size
) {
	const int row = blockIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	if ( col >= hidden_size ) return;
	const size_t src = static_cast<size_t>( token_ids[row] ) * hidden_size + col;
	out[static_cast<size_t>( row ) * hidden_size + col] = __float2half_rn( __bfloat162float( table[src] ) );
}

} // namespace impl

void Embedding::forward(
	const pod::TensorView& table,
	const int* token_ids,
	int count,
	pod::TensorView& out,
	cudaStream_t stream
) {
	ENGINE_ASSERT( table.dtype == pod::DType::BF16, "Embedding::forward expects a BF16 table" );
	const int hidden = table.shape[1];
	const int threads = 256;
	dim3 grid( ( hidden + threads - 1 ) / threads, count );
	impl::embed_tokens_kernel<<<grid, threads, 0, stream>>>(
		table.ptr<const __nv_bfloat16>(), token_ids, out.ptr<half>(), hidden
	);
}

} // namespace engine::modules
