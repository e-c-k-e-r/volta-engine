#include "engine/modules/nvfp4.h"
#include "engine/runtime/options.h"
#include "trellis_context.h"
#include "engine/modules/gemm.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <cstdlib>
#include <algorithm>

#include "nvfp4_linear_kernels.cuh"

#if defined(USE_ROCM)
#include "../../rocm/nvfp4_gemv.cuh"
#include "../../rocm/nvfp4_gemm.cuh"
#endif

namespace engine::modules {

// Rows up to which the fused skinny kernel beats dequantize + HGEMM
static constexpr int NVFP4_SKINNY_MAX_ROWS = 32;

void LinearNVFP4::forward(
	const pod::TensorView& x,
	const pod::TensorView& codes,
	const pod::TensorView& scales,
	float gscale,
	pod::TensorView& out,
	cudaStream_t stream
) {
	// N and K come from the packed codes ([N, K/2] bytes) when they carry a shape: the model's shared scratch tensors are as wide as the widest
	// consumer (GLM's expert buffers match the 12288-wide dense MLP), and rows of x and out are packed K and N apart as for every other weight format
	const int m = x.shape[0];
	int k = x.shape[1];
	int n = out.shape[1];
	if ( codes.ndim == 2 ) {
		n = codes.shape[0];
		k = codes.shape[1] * 2;
	}
	ENGINE_ASSERT( x.shape[1] >= k && out.shape[1] >= n, "NVFP4: input or output narrower than the weight" );

	ENGINE_ASSERT( k % 64 == 0, "NVFP4: K must be multiple of 64" );
	ENGINE_ASSERT( n % 32 == 0, "NVFP4: N must be multiple of 32" );

	const uint8_t* codes_ptr = codes.ptr<const uint8_t>();
	const uint8_t* scales_ptr = scales.ptr<const uint8_t>();
	const half* x_ptr = x.ptr<const half>();
	half* y_ptr = out.ptr<half>();

	dim3 grid( n / 32 );

#if defined(USE_ROCM)
	// ROCm: the skinny and fused kernels run on emulated mma (1-5 ms per call at M <= 32 on a 7900 XTX, gate_proj shape). Decode rows
	// take the SIMT GEMV, which reads the packed weight once; the rest dequantize once into FP16 for one hipBLAS GEMM (the last branch)
	if ( m <= ROCM_NVFP4_GEMV_MAX_ROWS ) {
		if ( m == 1 ) rocm_nvfp4_gemv_launch<1>( codes_ptr, scales_ptr, x_ptr, y_ptr, n, k, m, gscale, stream );
		else if ( m == 2 ) rocm_nvfp4_gemv_launch<2>( codes_ptr, scales_ptr, x_ptr, y_ptr, n, k, m, gscale, stream );
		else if ( m <= 4 ) rocm_nvfp4_gemv_launch<4>( codes_ptr, scales_ptr, x_ptr, y_ptr, n, k, m, gscale, stream );
		else if ( m <= 8 ) rocm_nvfp4_gemv_launch<8>( codes_ptr, scales_ptr, x_ptr, y_ptr, n, k, m, gscale, stream );
		else rocm_nvfp4_gemv_launch<16>( codes_ptr, scales_ptr, x_ptr, y_ptr, n, k, m, gscale, stream );
		return;
	}
	// More rows: the fused dequantize + WMMA GEMM (the option nvfp4_hipblas takes dequantize + hipBLAS instead)
	static const bool rocm_gemm = !engine::runtime::options().nvfp4_hipblas;
	if ( rocm_gemm ) {
		rocm_nvfp4_gemm_kernel<<<dim3( ( n + RG_BN - 1 ) / RG_BN, ( m + RG_BM - 1 ) / RG_BM ), dim3( 256 ), 0, stream>>>(
			codes_ptr, scales_ptr, x_ptr, y_ptr, m, n, k, gscale );
		return;
	}
	constexpr bool fused = false;
#else
	constexpr bool fused = true;
#endif
	if ( fused && m <= 8 ) {
		if ( ( k / 16 ) % 16 == 0 ) {
			skinny_nvfp4_qpn2<16, 2><<<grid, dim3( 32 * 16 ), 0, stream>>>(
				codes_ptr, scales_ptr, x_ptr, y_ptr, n, k, m, gscale
			);
		}
		else if ( ( k / 16 ) % 8 == 0 ) {
			skinny_nvfp4_qpn2<8, 2><<<grid, dim3( 32 * 8 ), 0, stream>>>(
				codes_ptr, scales_ptr, x_ptr, y_ptr, n, k, m, gscale
			);
		}
		else {
			skinny_nvfp4_qpn<1><<<grid, dim3( 128 ), 0, stream>>>(
				codes_ptr, scales_ptr, x_ptr, y_ptr, n, k, m, gscale
			);
		}
	}
	else if ( fused && m <= 16 ) {
		// 9 to 16 rows: the split-K kernel with two row tiles keeps the 8-row kernel's K split, which a few hundred column tiles need to fill the GPU;
		// wider outputs (lm_head, the drafter's gate/up) have enough tiles for 4 warps each, and the split's reduction costs more than it hides there
		const bool split = n <= 8192;
		if ( split && ( k / 16 ) % 16 == 0 ) {
			skinny_nvfp4_qpn2<16, 2, 2><<<grid, dim3( 32 * 16 ), 0, stream>>>(
				codes_ptr, scales_ptr, x_ptr, y_ptr, n, k, m, gscale
			);
		}
		else if ( split && ( k / 16 ) % 8 == 0 ) {
			skinny_nvfp4_qpn2<8, 2, 2><<<grid, dim3( 32 * 8 ), 0, stream>>>(
				codes_ptr, scales_ptr, x_ptr, y_ptr, n, k, m, gscale
			);
		}
		else {
			skinny_nvfp4_qpn<2><<<grid, dim3( 128 ), 0, stream>>>(
				codes_ptr, scales_ptr, x_ptr, y_ptr, n, k, m, gscale
			);
		}
	}
	else if ( fused && m <= NVFP4_SKINNY_MAX_ROWS ) {
		// Up to 32 rows: the decoded weights are reused across 3 or 4 row tiles of 8
		if ( m <= 24 ) {
			skinny_nvfp4_qpn<3><<<grid, dim3( 128 ), 0, stream>>>(
				codes_ptr, scales_ptr, x_ptr, y_ptr, n, k, m, gscale
			);
		}
		else {
			skinny_nvfp4_qpn<4><<<grid, dim3( 128 ), 0, stream>>>(
				codes_ptr, scales_ptr, x_ptr, y_ptr, n, k, m, gscale
			);
		}
	}
	else if ( fused && m < 384 && ( ( n + GT_BN - 1 ) / GT_BN ) * ( ( m + GT_BM - 1 ) / GT_BM ) >= 48 ) {
		// Mid-size chunks with enough tiles to fill the GPU; at 512 rows the cuBLAS GEMM (dequantized weight reused across
		// the whole chunk) is as fast, and narrow outputs leave most SMs idle in the fused kernel
		nvfp4_gemm_tc_kernel<<<dim3( ( n + GT_BN - 1 ) / GT_BN, ( m + GT_BM - 1 ) / GT_BM ), dim3( 256 ), 0, stream>>>(
			codes_ptr, scales_ptr, x_ptr, y_ptr, m, n, k, gscale );
	}
	else {
		// Dequantize once into FP16 and run a tensor-core HGEMM: C[m, n] = X[m, k] @ W[n, k]^T
		int device = 0;
		(void) cudaGetDevice( &device );
		half* w = trellis_get_recon_ws( device, static_cast<size_t>( n ) * k * sizeof( half ) );
		nvfp4_dequant_kernel<<<static_cast<unsigned>( static_cast<size_t>( n / 32 ) * ( k / 16 / DQ_GROUPS ) ), DQ_GROUPS * 32, 0, stream>>>(
			codes_ptr, scales_ptr, w, n, k, gscale );
		Gemm::tn( x_ptr, k, w, k, y_ptr, n, false, m, n, k, 0.0f, stream );
	}
}

} // namespace engine::modules
