#pragma once
#include <cuda_runtime_api.h>

// Host entry points of the Slang-generated CUDA kernels (src/shaders/*.slang). Pointers are device pointers to FP16 data
// (BF16 bits for the embedding table); each call is one launch on the stream. Used by the parity test and as the reference
// for new backends; the CUDA build's own kernels remain the default.
void slang_rms_norm( void* src, void* dst, void* weight, int rows, int hidden, float eps, float bias, cudaStream_t s );
void slang_rms_norm_heads( void* qk, void* weight, int rows, int heads, int head_dim, float eps, float bias, cudaStream_t s );
void slang_add_residual( void* hidden, void* residual, int n, cudaStream_t s );
void slang_swiglu( void* gate, void* up, void* out, int n, float limit, cudaStream_t s );
void slang_swiglu_fused( void* gate_up, void* out, int inter_dim, int total, float limit, cudaStream_t s );
void slang_embed_tokens( void* table_bf16, void* token_ids, void* out, int hidden, int rows, cudaStream_t s );
void slang_rope( void* q, void* k, void* pos, int rows, int q_heads, int kv_heads, int head_dim, int rot_dim, int interleaved, float base, cudaStream_t s );
