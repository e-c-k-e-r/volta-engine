#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// H3's own attention kernel for V100 (no CUTLASS): multi-head attention with head_dim 128, FP16 q / k / v / out in [rows, heads, 128] (contiguous), FP32 accumulation,
// full or causal. Q has q_rows rows, K / V have k_rows rows (sequence-parallel shards attend their own queries to every key). Always available.
namespace engine::video {

bool h3_attention( const half* q, const half* k, const half* v, half* out, int q_rows, int k_rows, int heads, float scale, bool causal, cudaStream_t stream );

} // namespace engine::video
