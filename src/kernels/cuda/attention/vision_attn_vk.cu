// The vision towers' bidirectional attention (the CUDA backend's attention_mma_kernel, vision_attn_mma.cuh: register-resident flash
// attention on the Volta mma.m8n8k4) as the PTX of the Vulkan backend's launches through VK_NV_cuda_kernel_launch (mod_cuda.cpp), for
// head_dim 64 and 72: 4 warps of 16 queries each, a block per 64 queries of one head.
#include <cuda_fp16.h>
#include <cstdint>
#include "../mma_sm80.cuh"

#define VISION_ATTN_KERNEL __device__ __forceinline__ void
#include "../../../backend/cuda/modules/vision_attn_mma.cuh"

extern "C" __global__ void __launch_bounds__( 128, 2 ) vk_vision_attn_64( const half* q, const half* k, const half* v, half* out, int m, int heads, float scale ) {
	attention_mma_kernel<64>( q, k, v, out, m, heads, scale );
}

extern "C" __global__ void __launch_bounds__( 128, 2 ) vk_vision_attn_72( const half* q, const half* k, const half* v, half* out, int m, int heads, float scale ) {
	attention_mma_kernel<72>( q, k, v, out, m, heads, scale );
}
