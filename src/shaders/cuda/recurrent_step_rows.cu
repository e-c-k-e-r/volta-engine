// Launcher for the Slang-generated CUDA kernel 'recurrent_step_rows' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers would otherwise collide across the kernels
namespace {
#include "recurrent_step_rows.cuda.cu"
}
#include "shaders_cuda.h"
#include <cmath>

// head_dim is fixed at 128 (state layout [v_head][col][row]); 32 column groups of 4 per head
void slang_recurrent_step_rows( void* state, void* qkv, void* gate, void* beta, void* output, void* slots, void* snap, void* snap_on, int h_k, int h_v, int rows, int snap_rows, long long slot_stride, long long snap_stride, cudaStream_t s ) {
	RecurrentParams_0 p{ (float*) state, (__half*) qkv, (float*) gate, (float*) beta, (__half*) output, (int*) slots, (float*) snap, (int*) snap_on,
		h_k, h_v, rows, snap_rows, 1.0f / sqrtf( 128.0f ), slot_stride, snap_stride };
	recurrent_step_rows<<<dim3( h_v, 1, 32 ), dim3( 32, 4 ), 0, s>>>( p );
}
