// cuBLAS algorithm sweep for the DiT's four GEMM shapes at one rank's rows (FP16 operands, FP16 or FP32 output, FP32 accumulation).   bench_h3_gemm [rows=8618]
#include "engine/config.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main( int argc, char** argv ) {
	const int M = argc > 1 ? std::atoi( argv[1] ) : 8618;
	struct Shape { const char* name; int N, K; bool c32; };
	const Shape shapes[] = { { "qkv", 16128, 5376, false }, { "out_proj", 5376, 7168, true }, { "fc1", 28672, 5376, false }, { "fc2", 5376, 14336, true } };
	cublasHandle_t h;
	cublasCreate( &h );
	void *A, *W, *C;
	CUDA_CHECK( cudaMalloc( &A, static_cast<size_t>( M ) * 14336 * 2 ) );
	CUDA_CHECK( cudaMalloc( &W, static_cast<size_t>( 28672 ) * 5376 * 2 ) );
	CUDA_CHECK( cudaMalloc( &C, static_cast<size_t>( M ) * 28672 * 4 ) );
	CUDA_CHECK( cudaMemset( A, 0x2a, static_cast<size_t>( M ) * 14336 * 2 ) );
	CUDA_CHECK( cudaMemset( W, 0x2a, static_cast<size_t>( 28672 ) * 5376 * 2 ) );
	const float alpha = 1.0f, beta = 0.0f;
	for ( const Shape& s : shapes ) {
		const double flops = 2.0 * M * s.N * s.K;
		double best = 0;
		int best_algo = -99;
		for ( int algo = -1; algo <= 23; ++algo ) {
			const cublasGemmAlgo_t a = algo < 0 ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : static_cast<cublasGemmAlgo_t>( CUBLAS_GEMM_ALGO0_TENSOR_OP + algo );
			auto run = [&]() { return cublasGemmEx( h, CUBLAS_OP_T, CUBLAS_OP_N, s.N, M, s.K, &alpha, W, CUDA_R_16F, s.K, A, CUDA_R_16F, s.K, &beta, C, s.c32 ? CUDA_R_32F : CUDA_R_16F, s.N, CUBLAS_COMPUTE_32F, a ); };
			if ( run() != CUBLAS_STATUS_SUCCESS ) { cudaGetLastError(); continue; }
			CUDA_CHECK( cudaDeviceSynchronize() );
			const auto t0 = std::chrono::steady_clock::now();
			const int reps = 6;
			for ( int r = 0; r < reps; ++r ) { run(); }
			CUDA_CHECK( cudaDeviceSynchronize() );
			const double sec = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count() / reps;
			const double tf = flops / sec / 1e12;
			if ( algo < 0 ) { std::printf( "%-9s M%d N%d K%d default: %.1f TFLOP/s\n", s.name, M, s.N, s.K, tf ); }
			if ( tf > best ) { best = tf; best_algo = algo; }
		}
		std::printf( "%-9s best algo %d: %.1f TFLOP/s\n", s.name, best_algo, best );
	}
	return 0;
}
