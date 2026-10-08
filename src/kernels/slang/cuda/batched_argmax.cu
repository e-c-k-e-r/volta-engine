// Launcher for the Slang-generated CUDA kernel 'batched_argmax' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers would otherwise collide across the kernels
namespace {
#include "batched_argmax.cuda.cu"
}
#include "shaders_cuda.h"

void slang_batched_argmax( void* logits, void* out_idx, void* out_val, int numel, int batch, int index_offset, int row_stride, cudaStream_t s ) {
	ArgmaxParams_0 p{ (__half*) logits, (int*) out_idx, (float*) out_val, numel, batch, index_offset, row_stride, out_val != nullptr ? 1 : 0 };
	batched_argmax<<<batch, 256, 0, s>>>( p );
}
