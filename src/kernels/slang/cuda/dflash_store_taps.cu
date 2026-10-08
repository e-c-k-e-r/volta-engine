// Launcher for the Slang-generated CUDA kernel 'dflash_store_taps' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers would otherwise collide across the kernels
namespace {
#include "dflash_store_taps.cuda.cu"
}
#include "shaders_cuda.h"

void slang_dflash_store_taps( void* taps, void* hidden, void* slots, void* positions, void* owner, int m, int tap, int H, int cap, cudaStream_t s ) {
	TapStoreParams_0 p{ (__half*) taps, (__half*) hidden, (int*) slots, (int*) positions, (int*) owner, m, tap, H, cap };
	dflash_store_taps<<<dim3( ( H + 255 ) / 256, m ), 256, 0, s>>>( p );
}
