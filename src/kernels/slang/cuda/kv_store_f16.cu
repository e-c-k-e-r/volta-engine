// Launcher for the Slang-generated CUDA kernel 'kv_store_f16' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers would otherwise collide across the kernels
namespace {
#include "kv_store_f16.cuda.cu"
}
#include "shaders_cuda.h"

void slang_kv_store_f16( void* data, void* src, void* slots, void* positions, long long slot_stride, int layer_idx, int count, int total_elems, int max_seq, cudaStream_t s ) {
	KvStoreParams_0 p{ (__half*) data, (__half*) src, (int*) slots, (int*) positions, slot_stride, layer_idx, count, total_elems, max_seq };
	kv_store_f16<<<( count * total_elems + 127 ) / 128, 128, 0, s>>>( p );
}
