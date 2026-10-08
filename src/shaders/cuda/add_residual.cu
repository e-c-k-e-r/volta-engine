// Launcher for the Slang-generated CUDA kernel 'add_residual' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers (inv_rms_from_0, swiglu_value_0, ...) would otherwise collide across the kernels
namespace {
#include "add_residual.cuda.cu"
}
#include "shaders_cuda.h"

void slang_add_residual( void* hidden, void* residual, int n, cudaStream_t s ) {
	AddParams_0 p{ (__half*) hidden, (__half*) residual, n };
	add_residual<<<( n + 255 ) / 256, 256, 0, s>>>( p );
}
