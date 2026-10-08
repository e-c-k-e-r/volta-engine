// Launcher for the Slang-generated CUDA kernel 'embed_tokens' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers (inv_rms_from_0, swiglu_value_0, ...) would otherwise collide across the kernels
namespace {
#include "embed_tokens.cuda.cu"
}
#include "shaders_cuda.h"

void slang_embed_tokens( void* table_bf16, void* token_ids, void* out, int hidden, int rows, cudaStream_t s ) {
	EmbedParams_0 p{ (uint16_t*) table_bf16, (int*) token_ids, (__half*) out, hidden };
	embed_tokens<<<dim3( ( hidden + 255 ) / 256, rows ), 256, 0, s>>>( p );
}
