// Launcher for the Slang-generated CUDA kernel 'dsa_pool_update' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers would otherwise collide across the kernels
namespace {
#include "dsa_pool_update.cuda.cu"
}
#include "shaders_cuda.h"

void slang_dsa_pool_update( void* plane, void* pool_plane, void* ape, void* slots, void* positions, int layer, int max_seq, long long plane_slot_stride, long long pool_slot_stride, int rows, cudaStream_t s ) {
	PoolParams_0 p{ (__half*) plane, (__half*) pool_plane, (__half*) ape, (int*) slots, (int*) positions, layer, max_seq, plane_slot_stride, pool_slot_stride };
	dsa_pool_update<<<rows, 128, 0, s>>>( p );
}
