// The NVFP4 linear kernels (nvfp4_linear_kernels.cuh) as the PTX of the Vulkan backend's launches through VK_NV_cuda_kernel_launch
// (src/backend/vulkan/mod_cuda.cpp): an extern "C" entry per kernel form skinny_kernels.cu launches, with the same launch bounds
#define NVFP4_KERNEL __device__ __forceinline__
#define NVFP4_KERNEL_BOUNDS( threads ) __device__ __forceinline__
#include "nvfp4_linear_kernels.cuh"

#define VK_NVFP4_ARGS const uint8_t* codes, const uint8_t* scales, const half* x, half* y, int n, int k, int m, float gscale

extern "C" __global__ void vk_nvfp4_qpn_1( VK_NVFP4_ARGS ) { skinny_nvfp4_qpn<1>( codes, scales, x, y, n, k, m, gscale ); }
extern "C" __global__ void vk_nvfp4_qpn_2( VK_NVFP4_ARGS ) { skinny_nvfp4_qpn<2>( codes, scales, x, y, n, k, m, gscale ); }
extern "C" __global__ void vk_nvfp4_qpn_3( VK_NVFP4_ARGS ) { skinny_nvfp4_qpn<3>( codes, scales, x, y, n, k, m, gscale ); }
extern "C" __global__ void vk_nvfp4_qpn_4( VK_NVFP4_ARGS ) { skinny_nvfp4_qpn<4>( codes, scales, x, y, n, k, m, gscale ); }
extern "C" __global__ void vk_nvfp4_qpn2_16_2_1( VK_NVFP4_ARGS ) { skinny_nvfp4_qpn2<16, 2>( codes, scales, x, y, n, k, m, gscale ); }
extern "C" __global__ void vk_nvfp4_qpn2_8_2_1( VK_NVFP4_ARGS ) { skinny_nvfp4_qpn2<8, 2>( codes, scales, x, y, n, k, m, gscale ); }
extern "C" __global__ void vk_nvfp4_qpn2_16_2_2( VK_NVFP4_ARGS ) { skinny_nvfp4_qpn2<16, 2, 2>( codes, scales, x, y, n, k, m, gscale ); }
extern "C" __global__ void vk_nvfp4_qpn2_8_2_2( VK_NVFP4_ARGS ) { skinny_nvfp4_qpn2<8, 2, 2>( codes, scales, x, y, n, k, m, gscale ); }
extern "C" __global__ __launch_bounds__( 256 ) void vk_nvfp4_gemm_tc( VK_NVFP4_ARGS ) {
	nvfp4_gemm_tc_kernel( codes, scales, x, y, m, n, k, gscale );
}
