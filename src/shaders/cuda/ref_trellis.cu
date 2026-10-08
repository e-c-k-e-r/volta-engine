// Launchers for the Slang-generated CUDA kernels 'ref_trellis_reconstruct' and 'ref_had128'
#define SLANG_CUDA_ENABLE_HALF 1
#include <slang-cuda-prelude.h>
#define ref_trellis_reconstruct ref_trellis_reconstruct_cu
#define ref_had128 ref_had128_cu
namespace recon {
#include "ref_trellis_reconstruct.cuda.cu"
}
namespace had {
#include "ref_had128.cuda.cu"
}
#include "ref_cuda.h"

void ref_trellis_reconstruct_cuda( void* codes, void* w, int k, int n, int bits, int cb, cudaStream_t s ) {
	recon::RefReconParams_0 p{ (ushort*) codes, (__half*) w, k, n, bits, cb };
	const long long total = (long long) k * n;
	recon::ref_trellis_reconstruct<<<(unsigned) ( ( total + 255 ) / 256 ), 256, 0, s>>>( p );
}

void ref_had128_cuda( void* src, void* dst, void* pre, void* post, int rows, int cols, float r_scale, cudaStream_t s ) {
	had::RefHadParams_0 p{ (__half*) src, (__half*) dst, (__half*) pre, (__half*) post, cols, r_scale };
	had::ref_had128<<<dim3( cols / 128, rows ), 128, 0, s>>>( p );
}

void ref_trellis_linear_cuda( void* x, void* codes, void* suh, void* svh, void* out, void* ws_xh, void* ws_w, void* ws_y,
	int m, int k, int n, int bits, int cb, cudaStream_t s ) {
	const float r_scale = 0.08838834764831845f; // 1/sqrt(128)
	ref_had128_cuda( x, ws_xh, suh, nullptr, m, k, r_scale, s );
	ref_trellis_reconstruct_cuda( codes, ws_w, k, n, bits, cb, s );
	ref_gemm_f16_cuda( ws_xh, ws_w, ws_y, m, k, n, s );
	ref_had128_cuda( ws_y, out, nullptr, svh, m, n, r_scale, s );
}
