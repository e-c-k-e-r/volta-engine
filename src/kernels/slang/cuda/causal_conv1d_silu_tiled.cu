// Launcher for the Slang-generated CUDA kernel 'causal_conv1d_silu_tiled' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers would otherwise collide across the kernels
namespace {
#include "causal_conv1d_silu_tiled.cuda.cu"
}
#include "shaders_cuda.h"

void slang_causal_conv1d_silu_tiled( void* src, void* dst, void* history, void* weights, void* slots, int channels, int rows, int tile_rows, long long slot_stride, cudaStream_t s ) {
	ConvTiledParams_0 p{ (__half*) src, (__half*) dst, (float*) history, (__half*) weights, (int*) slots, channels, rows, tile_rows, weights != nullptr ? 1 : 0, slot_stride };
	causal_conv1d_silu_tiled<<<dim3( ( channels + 127 ) / 128, ( rows + tile_rows - 1 ) / tile_rows ), 128, 0, s>>>( p );
}
