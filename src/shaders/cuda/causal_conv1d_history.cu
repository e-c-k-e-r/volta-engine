// Launcher for the Slang-generated CUDA kernel 'causal_conv1d_history' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers would otherwise collide across the kernels
namespace {
#include "causal_conv1d_history.cuda.cu"
}
#include "shaders_cuda.h"

void slang_causal_conv1d_history( void* src, void* history, void* slots, int channels, int rows, long long slot_stride, cudaStream_t s ) {
	ConvTiledParams_0 p{ (__half*) src, nullptr, (float*) history, nullptr, (int*) slots, channels, rows, 0, 0, slot_stride };
	causal_conv1d_history<<<( channels + 127 ) / 128, 128, 0, s>>>( p );
}
