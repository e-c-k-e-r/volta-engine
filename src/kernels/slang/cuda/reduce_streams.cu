// Launcher for the Slang-generated CUDA kernel 'reduce_streams' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers would otherwise collide across the kernels
namespace {
#include "reduce_streams.cuda.cu"
}
#include "shaders_cuda.h"

void slang_reduce_streams( void* streams, void* out, int m, int H, int mode, cudaStream_t s ) {
	ReduceStreamsParams_0 p{ (float*) streams, (__half*) out, H, mode };
	reduce_streams<<<dim3( ( H + 255 ) / 256, m ), 256, 0, s>>>( p );
}
