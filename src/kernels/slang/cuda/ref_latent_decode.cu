// Launcher for the Slang-generated CUDA kernel 'ref_latent_decode' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
namespace {
#include "ref_latent_decode.cuda.cu"
}
#include "ref_attn_cuda.h"

void ref_latent_launch_decode( void* q_lat, void* cache, void* out, void* slots, void* pos, long long slot_stride,
	int layer, int rows, int heads, int max_seq, cudaStream_t s ) {
	RefLatentParams_0 p{ (__half*) q_lat, (__half*) cache, (__half*) out, (int*) slots, (int*) pos, slot_stride, layer, heads, max_seq, 1.0f / 16.0f };
	ref_latent_decode<<<dim3( heads, rows ), 256, 0, s>>>( p );
}
