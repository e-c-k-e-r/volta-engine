// Launcher for the Slang-generated CUDA kernel 'rope' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers (inv_rms_from_0, swiglu_value_0, ...) would otherwise collide across the kernels
namespace {
#include "rope.cuda.cu"
}
#include "shaders_cuda.h"

void slang_rope( void* q, void* k, void* pos, int rows, int q_heads, int kv_heads, int head_dim, int rot_dim, int interleaved, float base, cudaStream_t s ) {
	RopeParams_0 p{ (__half*) q, (__half*) k, (int*) pos, q_heads, kv_heads, head_dim, rot_dim, interleaved, base };
	rope<<<rows, 256, 0, s>>>( p );
}
