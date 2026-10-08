// Launcher for the Slang-generated CUDA kernel 'gated_rmsnorm' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers would otherwise collide across the kernels
namespace {
#include "gated_rmsnorm.cuda.cu"
}
#include "shaders_cuda.h"

void slang_gated_rmsnorm( void* attn_out, void* z, void* norm_weight, int rows, int num_v_heads, int head_dim, int use_sigmoid, cudaStream_t s ) {
	GatedNormParams_0 p{ (__half*) attn_out, (__half*) z, (__half*) norm_weight, num_v_heads, head_dim, use_sigmoid, z != nullptr ? 1 : 0, norm_weight != nullptr ? 1 : 0 };
	gated_rmsnorm<<<dim3( num_v_heads, rows ), 256, 0, s>>>( p );
}
