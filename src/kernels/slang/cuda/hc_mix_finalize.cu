// Launcher for the Slang-generated CUDA kernel 'hc_mix_finalize' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// Internal linkage: the generated helpers would otherwise collide across the kernels
namespace {
#include "hc_mix_finalize.cuda.cu"
}
#include "shaders_cuda.h"

void slang_hc_mix_finalize( void* streams, void* partials, void* base, void* scale, void* post, void* comb, void* collapsed, void* inv_rms_buf, void* mix_logits_buf, void* pre_buf,
	int hidden_size, int num_chunks, int chunk_cols_c, int sinkhorn_iters, float rms_eps, float hc_eps, int rows, cudaStream_t s ) {
	FinalizeParams_0 p{ (float*) streams, (float*) partials, (float*) base, (float*) scale, (float*) post, (float*) comb, (__half*) collapsed,
		(float*) inv_rms_buf, (float*) mix_logits_buf, (float*) pre_buf, hidden_size, num_chunks, chunk_cols_c, sinkhorn_iters, rms_eps, hc_eps };
	hc_mix_finalize<<<dim3( ( hidden_size + chunk_cols_c - 1 ) / chunk_cols_c, rows ), 256, 0, s>>>( p );
}
