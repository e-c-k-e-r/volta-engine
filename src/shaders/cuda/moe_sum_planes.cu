// Launcher for the Slang-generated CUDA kernel 'moe_sum_planes' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers would otherwise collide across the kernels
namespace {
#include "moe_sum_planes.cuda.cu"
}
#include "shaders_cuda.h"

void slang_moe_sum_planes( void* planes, void* out, int n, int groups, cudaStream_t s ) {
	SumPlanesParams_0 p{ (__half*) planes, (__half*) out, n, groups };
	moe_sum_planes<<<( n + 255 ) / 256, 256, 0, s>>>( p );
}
