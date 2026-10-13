#include "engine/modules/trellis.h"
#include "exl3_gemm.cuh"
#include "exl3_gemv.cuh"
#include "trellis_context.h"
#include "reconstruct.cuh"
#include "hadamard.cuh"
#include <cstdlib>

namespace engine::modules {

namespace {

// Rows above which dequantizing the weight once and running a tensor-core HGEMM beats the fused GEMM kernel, whose
// cost is linear in M (~4.3 us per row on the Qwen MLP shapes, ~10 TFLOP/s: the trellis is re-decoded per 16-row tile).
int recon_threshold() {
	return 36;
}

} // namespace

void LinearTrellis::forward_recon(
	const pod::TensorView& x,
	const pod::TensorView& trellis,
	const pod::TensorView& suh,
	const pod::TensorView& svh,
	pod::TensorView& out,
	int bits,
	bool mcg,
	bool mul1,
	cudaStream_t stream
) {
	const int m = x.shape[0];
	const int k = trellis.shape[0] * 16;
	const int n = trellis.shape[1] * 16;
	ENGINE_ASSERT( out.dtype == pod::DType::FP16, "LinearTrellis::forward_recon: FP16 output only" );
	ENGINE_ASSERT( k % 128 == 0 && n % 128 == 0, "LinearTrellis::forward_recon: dims must be multiples of 128" );
	int device = 0;
	(void) cudaGetDevice( &device );

	// xh = H(x * suh); W = dequantized weight in the rotated basis; y = H((xh @ W) * svh). The standalone
	// Hadamard's scale is the final one, so 1/sqrt(128) as in the fused kernels.
	constexpr float kRScale = 0.088388347648f;
	half* xh = trellis_get_workspace( device, static_cast<size_t>( m ) * k * sizeof( half ) );
	half* w = trellis_get_recon_ws( device, static_cast<size_t>( k ) * n * sizeof( half ) );
	had_r_128( x.ptr<const half>(), xh, suh.ptr<const half>(), nullptr, kRScale, m, k, stream );
	reconstruct_slice( w, trellis.ptr<const uint16_t>(), bits, mcg, mul1, k / 16, n, 0, stream );

	cublasHandle_t h = trellis_get_cublas( device );
	cublasSetStream( h, stream );
	const float alpha = 1.0f, beta = 0.0f;
	// Row-major C[m,n] = A[m,k] @ W[k,n] as column-major C^T = W^T A^T.
	const cublasStatus_t st = cublasGemmEx( h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
		&alpha, w, CUDA_R_16F, n, xh, CUDA_R_16F, k,
		&beta, out.data, CUDA_R_16F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP );
	ENGINE_ASSERT( st == CUBLAS_STATUS_SUCCESS, "LinearTrellis::forward_recon: cublasGemmEx failed" );
	had_r_128( out.ptr<const half>(), out.ptr<half>(), nullptr, svh.ptr<const half>(), kRScale, m, n, stream );
}

void LinearTrellis::forward(
	const pod::TensorView& x,
	const pod::TensorView& trellis,
	const pod::TensorView& suh,
	const pod::TensorView& svh,
	pod::TensorView& out,
	int bits,
	bool mcg,
	bool mul1,
	cudaStream_t stream
) {
	const int m = x.shape[0];
	const int k = trellis.shape[0] * 16;
	const int n = trellis.shape[1] * 16;

	int device = 0;
	(void) cudaGetDevice( &device );
	half* A_had = trellis_get_workspace( device, static_cast<size_t>( m ) * k * sizeof( half ) );

	bool c_fp32 = ( out.dtype == pod::DType::FP32 );
	const half* x_ptr = x.ptr<const half>();
	const uint16_t* trellis_ptr = trellis.ptr<const uint16_t>();
	void* out_ptr = out.data;
	const half* suh_ptr = suh.ptr<const half>();
	const half* svh_ptr = svh.ptr<const half>();

	// GLM 5-bit n=6144 projection (the KDA input) at 4..8 rows: the emulated-MMA GEMV falls to ~120 GB/s while the 16x512 tensor-core
	// GEMM tile keeps ~190 (bench_shapes cold sweep over 24 weight copies: 129 -> 82 us). Every other shape measures level or
	// better on the GEMV with cold weights (the L2-resident loop of a single matrix says otherwise and is misleading).
	int gemm_shape = 0;
	if ( bits == 5 && m >= 4 && m <= 8 && mul1 && !c_fp32 && n == 6144 && ( k % 16 == 0 ) ) gemm_shape = 4;
#if defined(USE_ROCM)
	// ROCm: the fused GEMM runs on emulated mma, 15-125x slower than reconstruct + hipBLAS at M = 16..64 (test_prefill_linear on a
	// 7900 XTX). Up to EXL3_GEMV_ROCM_MAX_M rows the v_dot2 GEMV (exl3_gemv_rocm.cu), above that reconstruct
	const bool recon = m > EXL3_GEMV_ROCM_MAX_M;
	gemm_shape = 0;
#else
	const bool recon = recon_threshold() > 0 && m > recon_threshold();
#endif
	if ( recon && !c_fp32 && ( k % 128 == 0 ) && ( n % 128 == 0 ) ) {
		forward_recon( x, trellis, suh, svh, out, bits, mcg, mul1, stream );
	}
	else if ( gemm_shape > 0 ) {
		exl3_gemm( x_ptr, trellis_ptr, out_ptr, suh_ptr, A_had, svh_ptr, m, k, n, bits, mcg, mul1, c_fp32, stream, gemm_shape );
	}
#if defined(USE_ROCM)
	else if ( m <= EXL3_GEMV_ROCM_MAX_M && ( k % 128 == 0 ) && ( n % 128 == 0 ) ) {
#else
	else if ( m <= 8 && ( m == 1 || m == 2 || m == 4 || m == 8 ) && ( k % 128 == 0 ) && ( n % 128 == 0 ) ) {
#endif
		exl3_gemv(
			x_ptr,
			trellis_ptr,
			out_ptr,
			suh_ptr,
			A_had,
			svh_ptr,
			m,
			k,
			n,
			bits,
			mcg,
			mul1,
			c_fp32,
			stream
		);
	}
	else {
		exl3_gemm(
			x_ptr,
			trellis_ptr,
			out_ptr,
			suh_ptr,
			A_had,
			svh_ptr,
			m,
			k,
			n,
			bits,
			mcg,
			mul1,
			c_fp32,
			stream
		);
	}
}

} // namespace engine::modules
