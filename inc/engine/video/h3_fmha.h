#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace engine::video {

// Non-causal multi-head attention, head_dim 128, FP16 q/k/v/out in [rows, heads, 128] (contiguous), FP32 accumulation, one
// document. SM70 CUTLASS tensor-op kernel (derived from CUTLASS example 41 via 1Cat's FlashAttention-V100). Returns false when the
// build has no CUTLASS (ENGINE_CUTLASS_DIR) and the caller must use another path.
bool h3_fmha_available();
bool h3_flash_attention( const half* q, const half* k, const half* v, half* out, int rows, int heads, float scale, int key_tile, int query_tile, cudaStream_t stream );

} // namespace engine::video
