// Launcher for the Slang-generated CUDA kernel 'hc_mix_partials' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers would otherwise collide across the kernels
namespace {
#include "hc_mix_partials.cuda.cu"
}
#include "shaders_cuda.h"

void slang_hc_mix_partials( void* streams, void* fn, void* partials, int row_len, int chunk_cols, int rows, cudaStream_t s ) {
	const int num_chunks = ( row_len + chunk_cols - 1 ) / chunk_cols;
	PartialsParams_0 p{ (float*) streams, (float*) fn, (float*) partials, row_len, chunk_cols, num_chunks };
	hc_mix_partials<<<dim3( num_chunks, rows ), 64, 0, s>>>( p );
}
