// Launcher for the Slang-generated CUDA kernel 'moe_accumulate' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers would otherwise collide across the kernels
namespace {
#include "moe_accumulate.cuda.cu"
}
#include "shaders_cuda.h"

void slang_moe_accumulate( void* accum, void* expert_out, float weight, int n, cudaStream_t s ) {
	AccumParams_0 p{ (__half*) accum, (__half*) expert_out, weight, n };
	moe_accumulate<<<( n + 255 ) / 256, 256, 0, s>>>( p );
}
