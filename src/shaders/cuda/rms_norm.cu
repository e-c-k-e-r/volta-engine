// Launcher for the Slang-generated CUDA kernel 'rms_norm' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers (inv_rms_from_0, swiglu_value_0, ...) would otherwise collide across the kernels
namespace {
#include "rms_norm.cuda.cu"
}
#include "shaders_cuda.h"

void slang_rms_norm( void* src, void* dst, void* weight, int rows, int hidden, float eps, float bias, cudaStream_t s ) {
	NormParams_0 p{ (__half*) src, (__half*) dst, (__half*) weight, hidden, weight != nullptr ? 1 : 0, eps, bias };
	rms_norm<<<rows, 256, 0, s>>>( p );
}
