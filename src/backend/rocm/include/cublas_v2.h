#pragma once
// ROCM configuration: cuBLAS names onto hipBLAS (the large-M prefill HGEMM). hipBLAS's v2 API takes hipDataType and
// hipblasComputeType_t, the counterparts of cudaDataType and cublasComputeType_t
#include "../compat.h"
#ifndef HIPBLAS_V2
	#define HIPBLAS_V2
#endif
#include <hipblas/hipblas.h>

using cublasHandle_t = hipblasHandle_t;
using cublasStatus_t = hipblasStatus_t;
using cublasGemmAlgo_t = hipblasGemmAlgo_t;
using cudaDataType = hipDataType;

#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_ALGO0_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#define CUDA_R_16F HIP_R_16F
#define CUDA_R_32F HIP_R_32F
#define CUDA_R_16BF HIP_R_16BF

#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetStream hipblasSetStream
#define cublasGemmEx hipblasGemmEx
#define cublasSgemm hipblasSgemm

// Tensor-op math is cuBLAS's switch; rocBLAS picks its kernels itself
enum cublasMath_t { CUBLAS_DEFAULT_MATH = 0, CUBLAS_TENSOR_OP_MATH = 1 };
inline hipblasStatus_t cublasSetMathMode( hipblasHandle_t, cublasMath_t ) { return HIPBLAS_STATUS_SUCCESS; }
