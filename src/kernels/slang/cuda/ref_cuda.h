#pragma once
// Stream handle spelled without the CUDA headers so host-only translation units (Slang C++ prelude) can include this file
struct CUstream_st;
typedef CUstream_st* cudaStream_t;

// Host entry points of the portable reference kernels (src/kernels/slang/ref_gemm.slang, ref_trellis.slang) built for CUDA, plus a
// bridge to the engine's own trellis Linear for comparison. All pointers are device pointers; FP16 data except codes (uint16).
// Codebook id cb: 0 = 3INST, 1 = MCG, 2 = MUL1. Dimensions k and n must be multiples of 128.

// out[m,n] = x[m,k] @ w[k,n], FP32 accumulate
void ref_gemm_f16_cuda( void* x, void* w, void* out, int m, int k, int n, cudaStream_t s );
// codes [k/16, n/16, 16*bits] -> w [k,n] in the rotated basis (no suh/svh)
void ref_trellis_reconstruct_cuda( void* codes, void* w, int k, int n, int bits, int cb, cudaStream_t s );
// Blockwise 128-point Hadamard on [rows, cols]: dst = post * ( r_scale * H( pre * src ) ); pre/post may be null
void ref_had128_cuda( void* src, void* dst, void* pre, void* post, int rows, int cols, float r_scale, cudaStream_t s );
// Composition: out = had128( gemm( had128( x * suh ), reconstruct( codes ) ) ) * svh. Workspaces: xh [m*k], w [k*n], y [m*n] halves
void ref_trellis_linear_cuda( void* x, void* codes, void* suh, void* svh, void* out, void* ws_xh, void* ws_w, void* ws_y,
	int m, int k, int n, int bits, int cb, cudaStream_t s );

// Engine side (LinearTrellis): recon = 0 -> forward (fused GEMV/GEMM kernels), recon = 1 -> forward_recon (cuBLAS path)
void ref_engine_trellis_linear( void* x, void* codes, void* suh, void* svh, void* out, int m, int k, int n, int bits, int cb, int recon, cudaStream_t s );
// Engine reconstruct_slice (dequantized rotated-basis weight, [k,n])
void ref_engine_reconstruct( void* codes, void* w, int k, int n, int bits, int cb, cudaStream_t s );
