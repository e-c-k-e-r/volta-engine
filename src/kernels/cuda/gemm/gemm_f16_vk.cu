// The FP16 tensor-core GEMM of gemm_f16_kernel.cuh as PTX entry points with C names, for the Vulkan backend's launches through
// VK_NV_cuda_kernel_launch on NVIDIA GPUs without cooperative matrices (src/backend/vulkan/mod_cuda.cpp): out[m, n] = a[m, k] * w[n, k]^T with
// FP16 or FP32 output, the tile configurations gemm_f16.cu dispatches (each within the 48 KiB of shared memory such a launch has), and the
// split-k reduce. Built by cmake/vulkan_backend.cmake into the kernel directory as cuda_gemm.ptx.
#include "gemm_f16_kernel.cuh"

using namespace engine::modules::gemm_f16;

#define VK_GEMM_TN( BM, BN, BK, WM, WN )                                                                                                           \
	extern "C" __global__ __launch_bounds__( WM * WN * 32 ) void vk_gemm_tn_##BM##x##BN##x##BK##_h( const half* a, int lda, const half* w, int ldw, \
		void* out, int ldc, int m, int n, int k, float beta, float* part, int k_per_split ) {                                                       \
		gemm_tn_body<BM, BN, BK, WM, WN, false, false>( a, lda, w, ldw, out, ldc, m, n, k, beta, part, k_per_split );                              \
	}                                                                                                                                              \
	extern "C" __global__ __launch_bounds__( WM * WN * 32 ) void vk_gemm_tn_##BM##x##BN##x##BK##_f( const half* a, int lda, const half* w, int ldw, \
		void* out, int ldc, int m, int n, int k, float beta, float* part, int k_per_split ) {                                                       \
		gemm_tn_body<BM, BN, BK, WM, WN, true, false>( a, lda, w, ldw, out, ldc, m, n, k, beta, part, k_per_split );                               \
	}

VK_GEMM_TN( 32, 64, 64, 1, 2 )
VK_GEMM_TN( 64, 64, 64, 2, 2 )
VK_GEMM_TN( 64, 128, 64, 2, 2 )
VK_GEMM_TN( 128, 128, 32, 2, 2 )

extern "C" __global__ void vk_split_reduce_h( const float* part, void* out, int ldc, int m, int n, int splits, float beta ) {
	split_reduce_body<false>( part, out, ldc, m, n, splits, beta );
}
extern "C" __global__ void vk_split_reduce_f( const float* part, void* out, int ldc, int m, int n, int splits, float beta ) {
	split_reduce_body<true>( part, out, ldc, m, n, splits, beta );
}
