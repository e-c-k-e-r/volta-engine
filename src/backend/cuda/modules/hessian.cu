#include "engine/modules/hessian.h"
#include "trellis_context.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>

namespace engine::modules {

void Hessian::accumulate( float* h, const half* x, int rows, int k, cudaStream_t stream ) {
	int device = 0;
	CUDA_CHECK( cudaGetDevice( &device ) );
	cublasHandle_t handle = trellis_get_cublas( device );
	cublasSetStream( handle, stream );
	// Row-major x [rows, k] is column-major [k, rows] (ld k): C [k, k] += X X^T
	const float one = 1.0f;
	const cublasStatus_t st = cublasGemmEx( handle, CUBLAS_OP_N, CUBLAS_OP_T, k, k, rows, &one, x, CUDA_R_16F, k, x, CUDA_R_16F, k,
		&one, h, CUDA_R_32F, k, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP );
	ENGINE_ASSERT( st == CUBLAS_STATUS_SUCCESS, "Hessian::accumulate: cublasGemmEx failed" );
}

} // namespace engine::modules
