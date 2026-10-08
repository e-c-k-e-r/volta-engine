// Launcher for the Slang-generated CUDA kernel 'router_select' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers would otherwise collide across the kernels
namespace {
#include "router_select.cuda.cu"
}
#include "shaders_cuda.h"

void slang_router_select( void* partials, void* bias, void* topk_idx, void* topk_w, int rows, int num_experts, int top_k, int k_slices, int max_slices, float scale, cudaStream_t s ) {
	RouterParams_0 p{ (float*) partials, (__half*) bias, (int*) topk_idx, (__half*) topk_w, num_experts, top_k, k_slices, max_slices, scale, bias != nullptr ? 1 : 0 };
	router_select<<<rows, 288, 0, s>>>( p );
}
