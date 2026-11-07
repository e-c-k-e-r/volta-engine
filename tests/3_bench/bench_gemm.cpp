// Gemm::tn / Gemm::nn against cuBLAS (cublasGemmEx, FP16 operands, FP32 accumulation, FP16 out) on the language models' prefill shapes: time per call in
// graph replay over distinct weights (cold L2), TFLOPS. The only place the engine still links cuBLAS outside the video pipeline.
//   bench_gemm [--nn] [M ...] [-- N K ...]   default rows 20 64 128 512 and the 27B / Flash-Next per-rank projections at TP=4; --nn: weights
//   stored [k, n] (Gemm::nn, the trellis prefill product)
#include "engine/modules/gemm.h"
#include "engine/config.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <functional>
#include <vector>

int main( int argc, char** argv ) {
	CUDA_CHECK( cudaSetDevice( 0 ) );
	std::vector<int> rows;
	std::vector<std::pair<int, int>> shapes;
	int i = 1;
	const bool nn = argc > 1 && std::strcmp( argv[1], "--nn" ) == 0;
	if ( nn ) ++i;
	for ( ; i < argc && std::strcmp( argv[i], "--" ) != 0; ++i ) rows.push_back( std::atoi( argv[i] ) );
	for ( ++i; i + 1 < argc; i += 2 ) shapes.push_back( { std::atoi( argv[i] ), std::atoi( argv[i + 1] ) } );
	if ( rows.empty() ) rows = { 20, 64, 128, 512 };
	// (n, k): 27B gate / up, down, in_proj_qkv, out_proj; Flash-Next attention q, linear-attention qkv; the GLM indexer projection
	if ( shapes.empty() ) shapes = { { 4352, 5120 }, { 5120, 4352 }, { 2560, 5120 }, { 5120, 1536 }, { 3072, 2560 }, { 2560, 2560 }, { 288, 4096 } };
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );
	cublasHandle_t h;
	ENGINE_ASSERT( cublasCreate( &h ) == CUBLAS_STATUS_SUCCESS, "cublasCreate" );
	cublasSetStream( h, stream );
	cublasSetMathMode( h, CUBLAS_TENSOR_OP_MATH );
	cudaEvent_t e0, e1;
	CUDA_CHECK( cudaEventCreate( &e0 ) );
	CUDA_CHECK( cudaEventCreate( &e1 ) );
	std::printf( "%6s %6s %6s | %10s %7s | %10s %7s | %s\n", "m", "n", "k", "ours us", "TFLOPS", "cuBLAS us", "TFLOPS", "ours / cuBLAS" );
	for ( auto [n, k] : shapes ) {
		const size_t wbytes = static_cast<size_t>( n ) * k * sizeof( half );
		const int sets = std::max( 2, std::min( 32, static_cast<int>( 1.5e9 / wbytes ) ) );
		std::vector<half*> w( sets );
		for ( auto& p : w ) { CUDA_CHECK( cudaMalloc( &p, wbytes ) ); CUDA_CHECK( cudaMemset( p, 0x11, wbytes ) ); }
		for ( int m : rows ) {
			half *x, *out;
			CUDA_CHECK( cudaMalloc( &x, static_cast<size_t>( m ) * k * sizeof( half ) ) );
			CUDA_CHECK( cudaMalloc( &out, static_cast<size_t>( m ) * n * sizeof( half ) ) );
			CUDA_CHECK( cudaMemset( x, 0x22, static_cast<size_t>( m ) * k * sizeof( half ) ) );
			auto time = [&]( const std::function<void( const half* )>& call ) {
				for ( auto* p : w ) call( p );
				CUDA_CHECK( cudaStreamSynchronize( stream ) );
				cudaGraph_t g;
				cudaGraphExec_t ex;
				CUDA_CHECK( cudaStreamBeginCapture( stream, cudaStreamCaptureModeGlobal ) );
				for ( auto* p : w ) call( p );
				CUDA_CHECK( cudaStreamEndCapture( stream, &g ) );
				CUDA_CHECK( cudaGraphInstantiate( &ex, g, nullptr, nullptr, 0 ) );
				CUDA_CHECK( cudaGraphLaunch( ex, stream ) );
				CUDA_CHECK( cudaStreamSynchronize( stream ) );
				const int iters = 10;
				CUDA_CHECK( cudaEventRecord( e0, stream ) );
				for ( int it = 0; it < iters; ++it ) CUDA_CHECK( cudaGraphLaunch( ex, stream ) );
				CUDA_CHECK( cudaEventRecord( e1, stream ) );
				CUDA_CHECK( cudaEventSynchronize( e1 ) );
				float ms = 0.0f;
				CUDA_CHECK( cudaEventElapsedTime( &ms, e0, e1 ) );
				CUDA_CHECK( cudaGraphExecDestroy( ex ) );
				CUDA_CHECK( cudaGraphDestroy( g ) );
				return 1000.0 * ms / iters / sets;
			};
			const double ours = time( [&]( const half* wp ) {
				if ( nn ) engine::modules::Gemm::nn( x, k, wp, n, out, n, false, m, n, k, 0.0f, stream );
				else engine::modules::Gemm::tn( x, k, wp, k, out, n, false, m, n, k, 0.0f, stream );
			} );
			const float one = 1.0f, zero = 0.0f;
			const double theirs = time( [&]( const half* wp ) {
				cublasGemmEx( h, nn ? CUBLAS_OP_N : CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one, wp, CUDA_R_16F, nn ? n : k, x, CUDA_R_16F, k, &zero, out, CUDA_R_16F, n,
					CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP );
			} );
			const double flop = 2.0 * m * n * k;
			std::printf( "%6d %6d %6d | %10.2f %7.2f | %10.2f %7.2f | %.2f\n", m, n, k, ours, flop / ours / 1e6, theirs, flop / theirs / 1e6, ours / theirs );
			CUDA_CHECK( cudaFree( x ) );
			CUDA_CHECK( cudaFree( out ) );
		}
		for ( auto* p : w ) CUDA_CHECK( cudaFree( p ) );
	}
	return 0;
}
