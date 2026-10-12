// Launcher for the Slang-generated CUDA kernel 'ref_attn_decode' (its own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
namespace {
#include "ref_attn_decode.cuda.cu"
}
#include "ref_attn_cuda.h"
#include <cmath>

void ref_attn_launch_decode( void* q, void* k_cache, void* v_cache, void* gate, void* out, void* slots, void* pos, long long slot_stride,
	int layer_idx, int rows, int num_q_heads, int num_kv_heads, int head_dim, int max_seq, cudaStream_t s ) {
	RefAttnParams_0 p{ (__half*) q, (__half*) k_cache, (__half*) v_cache, (__half*) gate, (__half*) out, (int*) slots, (int*) pos, slot_stride,
		layer_idx, num_q_heads, num_kv_heads, head_dim, max_seq, gate != nullptr ? 1 : 0, 1.0f / sqrtf( (float) head_dim ), nullptr, nullptr, 0, 0 };
	ref_attn_decode<<<dim3( num_q_heads, rows ), 256, 0, s>>>( p );
}
