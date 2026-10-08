// Launcher for the Slang-generated CUDA kernel 'init_streams' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers would otherwise collide across the kernels
namespace {
#include "init_streams.cuda.cu"
}
#include "shaders_cuda.h"

void slang_init_streams( void* in, void* streams, int hidden_size, int rows, cudaStream_t s ) {
	InitStreamsParams_0 p{ (__half*) in, (float*) streams, hidden_size };
	init_streams<<<dim3( ( hidden_size + 255 ) / 256, rows ), 256, 0, s>>>( p );
}
