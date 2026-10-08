// Launcher for the Slang-generated CUDA kernel 'rms_norm_heads' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers (inv_rms_from_0, swiglu_value_0, ...) would otherwise collide across the kernels
namespace {
#include "rms_norm_heads.cuda.cu"
}
#include "shaders_cuda.h"

void slang_rms_norm_heads( void* qk, void* weight, int rows, int heads, int head_dim, float eps, float bias, cudaStream_t s ) {
	HeadNormParams_0 p{ (__half*) qk, (__half*) weight, heads, head_dim, weight != nullptr ? 1 : 0, eps, bias };
	rms_norm_heads<<<dim3( heads, rows ), 256, 0, s>>>( p );
}
