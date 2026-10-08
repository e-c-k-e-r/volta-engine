// Launcher for the Slang-generated CUDA kernel 'hyperhead_mean' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers would otherwise collide across the kernels
namespace {
#include "hyperhead_mean.cuda.cu"
}
#include "shaders_cuda.h"

void slang_hyperhead_mean( void* streams, void* final_hidden, int hidden_size, int rows, cudaStream_t s ) {
	HeadMeanParams_0 p{ (float*) streams, (__half*) final_hidden, hidden_size };
	hyperhead_mean<<<dim3( ( hidden_size + 255 ) / 256, rows ), 256, 0, s>>>( p );
}
