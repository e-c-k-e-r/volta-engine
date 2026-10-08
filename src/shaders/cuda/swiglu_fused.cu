// Launcher for the Slang-generated CUDA kernel 'swiglu_fused' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers (inv_rms_from_0, swiglu_value_0, ...) would otherwise collide across the kernels
namespace {
#include "swiglu_fused.cuda.cu"
}
#include "shaders_cuda.h"

void slang_swiglu_fused( void* gate_up, void* out, int inter_dim, int total, float limit, cudaStream_t s ) {
	SwigluFusedParams_0 p{ (__half*) gate_up, (__half*) out, inter_dim, total, limit };
	swiglu_fused<<<( total + 255 ) / 256, 256, 0, s>>>( p );
}
