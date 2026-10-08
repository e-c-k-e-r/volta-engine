// Launcher for the Slang-generated CUDA kernel 'dsa_prep' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers would otherwise collide across the kernels
namespace {
#include "dsa_prep.cuda.cu"
}
#include "shaders_cuda.h"

void slang_dsa_prep( void* proj, void* ln_w, void* ln_b, void* plane, void* w_out, void* slots, void* positions, int layer, long long plane_slot_stride, int rows, cudaStream_t s ) {
	PrepParams_0 p{ (__half*) proj, (__half*) ln_w, (__half*) ln_b, (__half*) plane, (__half*) w_out, (int*) slots, (int*) positions, layer, plane_slot_stride };
	dsa_prep<<<rows, 128, 0, s>>>( p );
}
