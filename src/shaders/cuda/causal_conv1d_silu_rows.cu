// Launcher for the Slang-generated CUDA kernel 'causal_conv1d_silu_rows' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers would otherwise collide across the kernels
namespace {
#include "causal_conv1d_silu_rows.cuda.cu"
}
#include "shaders_cuda.h"

void slang_causal_conv1d_silu_rows( void* qkv, void* history, void* weights, void* slots, void* snap, void* snap_on, int channels, int rows, int snap_rows, long long slot_stride, long long snap_stride, cudaStream_t s ) {
	ConvRowsParams_0 p{ (__half*) qkv, (float*) history, (__half*) weights, (int*) slots, (float*) snap, (int*) snap_on, channels, rows, snap_rows,
		weights != nullptr ? 1 : 0, slot_stride, snap_stride };
	causal_conv1d_silu_rows<<<( channels + 255 ) / 256, 256, 0, s>>>( p );
}
