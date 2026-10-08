// Launcher for the Slang-generated CUDA kernel 'project_gates' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers would otherwise collide across the kernels
namespace {
#include "project_gates.cuda.cu"
}
#include "shaders_cuda.h"

void slang_project_gates( void* norm_hidden, void* in_proj_a, void* in_proj_b, void* A_log, void* dt_bias, void* gate_out, void* beta_out, int rows, int num_v_heads, int hidden_size, cudaStream_t s ) {
	GatesParams_0 p{ (__half*) norm_hidden, (__half*) in_proj_a, (__half*) in_proj_b, (__half*) A_log, (__half*) dt_bias, (float*) gate_out, (float*) beta_out,
		num_v_heads, hidden_size, A_log != nullptr ? 1 : 0, dt_bias != nullptr ? 1 : 0 };
	project_gates<<<dim3( num_v_heads, rows ), 256, 0, s>>>( p );
}
