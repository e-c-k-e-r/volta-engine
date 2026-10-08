// Launcher for the Slang-generated CUDA kernel 'swiglu' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers (inv_rms_from_0, swiglu_value_0, ...) would otherwise collide across the kernels
namespace {
#include "swiglu.cuda.cu"
}
#include "shaders_cuda.h"

void slang_swiglu( void* gate, void* up, void* out, int n, float limit, cudaStream_t s ) {
	SwigluParams_0 p{ (__half*) gate, (__half*) up, (__half*) out, n, limit };
	swiglu<<<( n + 255 ) / 256, 256, 0, s>>>( p );
}
