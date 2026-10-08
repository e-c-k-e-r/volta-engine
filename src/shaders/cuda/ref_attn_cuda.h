#pragma once
#include <cuda_runtime_api.h>

// Launchers of the Slang-generated CUDA reference attention kernels (src/shaders/ref_attention.slang, ref_latent.slang).
// Pointers are device pointers to FP16 data; slots / pos are the int arrays of BatchRows (one entry per row).
// Caches are [slot][layer][max_seq][kv_heads * head_dim] (attention) or [slot][layer][max_seq][512] (latent); slot_stride in elements.
// head_dim <= 256; one group per (head, row). gate may be null.
void ref_attn_launch_decode( void* q, void* k_cache, void* v_cache, void* gate, void* out, void* slots, void* pos, long long slot_stride,
	int layer_idx, int rows, int num_q_heads, int num_kv_heads, int head_dim, int max_seq, cudaStream_t s );
// Any head count over the single 512-wide latent cache, scale 1/sqrt(256)
void ref_latent_launch_decode( void* q_lat, void* cache, void* out, void* slots, void* pos, long long slot_stride,
	int layer, int rows, int heads, int max_seq, cudaStream_t s );
