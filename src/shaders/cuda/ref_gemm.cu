// Launcher for the Slang-generated CUDA kernel 'ref_gemm_f16' (own translation unit: the Slang CUDA prelude redefines the half2 operators)
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
// extern "C" kernel symbols ignore namespaces and would clash with the host-target entry of the same name in the test binary
#define ref_gemm_f16 ref_gemm_f16_cu
namespace {
#include "ref_gemm_f16.cuda.cu"
}
#include "ref_cuda.h"

void ref_gemm_f16_cuda( void* x, void* w, void* out, int m, int k, int n, cudaStream_t s ) {
	RefGemmParams_0 p{ (__half*) x, (__half*) w, (__half*) out, m, k, n };
	dim3 grid( ( n + 15 ) / 16, ( m + 15 ) / 16 );
	ref_gemm_f16<<<grid, dim3( 16, 16 ), 0, s>>>( p );
}
