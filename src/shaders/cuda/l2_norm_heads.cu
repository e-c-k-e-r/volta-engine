// Launcher for the Slang-generated CUDA kernel 'l2_norm_heads' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers would otherwise collide across the kernels
namespace {
#include "l2_norm_heads.cuda.cu"
}
#include "shaders_cuda.h"

void slang_l2_norm_heads( void* qkv, int rows, int num_k_heads, int head_dim, int scale_q, long long row_stride, cudaStream_t s ) {
	L2Params_0 p{ (__half*) qkv, num_k_heads, head_dim, scale_q, row_stride };
	l2_norm_heads<<<dim3( 2 * num_k_heads, rows ), 256, 0, s>>>( p );
}
