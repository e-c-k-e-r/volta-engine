// Launcher for the Slang-generated CUDA kernel 'hc_apply' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers would otherwise collide across the kernels
namespace {
#include "hc_apply.cuda.cu"
}
#include "shaders_cuda.h"

void slang_hc_apply( void* x_in, void* x_out, void* y, void* post, void* comb, int hidden_size, int rows, cudaStream_t s ) {
	ApplyParams_0 p{ (float*) x_in, (float*) x_out, (__half*) y, (float*) post, (float*) comb, hidden_size, comb != nullptr ? 1 : 0 };
	hc_apply<<<dim3( ( hidden_size + 255 ) / 256, rows ), 256, 0, s>>>( p );
}
