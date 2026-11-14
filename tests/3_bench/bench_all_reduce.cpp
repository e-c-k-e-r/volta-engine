#include "engine/config.h"
#include "engine/core/comm.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>

int main( int argc, char** argv ) {
	(void) argc;
	(void) argv;

	std::printf( "========================================================\n" );
	std::printf( "   volta-engine Phase 1: 4-GPU NVLink AllReduce Benchmark\n" );
	std::printf( "========================================================\n" );

	int dev_count = 0;
	CUDA_CHECK( cudaGetDeviceCount( &dev_count ) );
	ENGINE_ASSERT( dev_count >= ENGINE_NUM_GPUS, "Benchmark requires 4 GPUs" );

	engine::comm::CollectiveRing rings[ENGINE_NUM_GPUS];
	void* staging_ptrs[ENGINE_NUM_GPUS] = { nullptr };
	void* flag_ptrs[ENGINE_NUM_GPUS] = { nullptr };

	engine::comm::init_4gpu_mesh( rings, staging_ptrs, flag_ptrs, engine::comm::kMeshStagingBytes );

	// BENCH_AR: rms (the default: all_reduce_rms_norm, hidden 5120), plain (all_reduce, 5120), hc (all_reduce_hc_apply, 4096 x 4 streams),
	// gated (all_reduce_gated_norm, 2560 x 4 streams), normed (all_reduce_normed_add, 5376); BENCH_ROWS rows (1)
	const std::string mode = std::getenv( "BENCH_AR" ) ? std::getenv( "BENCH_AR" ) : "rms";
	const int rows = std::getenv( "BENCH_ROWS" ) ? std::atoi( std::getenv( "BENCH_ROWS" ) ) : 1;
	const int hidden_size = mode == "hc" ? 4096 : mode == "gated" ? 2560 : mode == "normed" ? 5376 : 5120;
	const size_t numel = static_cast<size_t>( rows ) * hidden_size;
	const size_t bytes = numel * sizeof( half );

	pod::TensorView tv_x[ENGINE_NUM_GPUS];
	pod::TensorView tv_res[ENGINE_NUM_GPUS];
	pod::TensorView tv_w[ENGINE_NUM_GPUS];

	void* d_x[ENGINE_NUM_GPUS];
	void* d_res[ENGINE_NUM_GPUS];
	void* d_w[ENGINE_NUM_GPUS];
	float* d_streams[ENGINE_NUM_GPUS];   // hc / gated: rows x 4 x hidden FP32 (gated: and the normed streams after them); normed: the FP32 residual
	float* d_mix[ENGINE_NUM_GPUS];       // hc: post (rows x 4) and comb (rows x 16); gated: post
	pod::TensorView tv_r32[ENGINE_NUM_GPUS], tv_out[ENGINE_NUM_GPUS];
	cudaStream_t streams[ENGINE_NUM_GPUS];

	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		CUDA_CHECK( cudaSetDevice( d ) );
		CUDA_CHECK( cudaStreamCreate( &streams[d] ) );

		CUDA_CHECK( cudaMalloc( &d_x[d], bytes ) );
		CUDA_CHECK( cudaMalloc( &d_res[d], bytes ) );
		CUDA_CHECK( cudaMalloc( &d_w[d], hidden_size * sizeof( half ) ) );

		CUDA_CHECK( cudaMemset( d_x[d], 0x3c, bytes ) );
		CUDA_CHECK( cudaMemset( d_res[d], 0x3c, bytes ) );
		CUDA_CHECK( cudaMemset( d_w[d], 0x3c, hidden_size * sizeof( half ) ) );

		const size_t nstreams = static_cast<size_t>( rows ) * 4 * hidden_size;
		CUDA_CHECK( cudaMalloc( &d_streams[d], 2 * nstreams * sizeof( float ) ) );
		CUDA_CHECK( cudaMemset( d_streams[d], 0, 2 * nstreams * sizeof( float ) ) );
		CUDA_CHECK( cudaMalloc( &d_mix[d], static_cast<size_t>( rows ) * 20 * sizeof( float ) ) );
		CUDA_CHECK( cudaMemset( d_mix[d], 0, static_cast<size_t>( rows ) * 20 * sizeof( float ) ) );
		if ( mode == "gated" ) {   // the per-stream norm weights: 4 x hidden
			CUDA_CHECK( cudaFree( d_w[d] ) );
			CUDA_CHECK( cudaMalloc( &d_w[d], 4 * hidden_size * sizeof( half ) ) );
			CUDA_CHECK( cudaMemset( d_w[d], 0, 4 * hidden_size * sizeof( half ) ) );
		}
		tv_r32[d] = pod::make_tensor_2d( d_streams[d], pod::DType::FP32, rows, hidden_size );
		tv_out[d] = pod::make_tensor_2d( d_res[d], pod::DType::FP16, rows, hidden_size );
		tv_x[d] = pod::make_tensor_2d( d_x[d], pod::DType::FP16, rows, hidden_size );
		tv_res[d] = pod::make_tensor_2d( d_res[d], pod::DType::FP16, rows, hidden_size );
		tv_w[d] = pod::make_tensor_1d( d_w[d], pod::DType::FP16, hidden_size );
	}

	std::atomic<int> start_gate{ 0 };
	std::atomic<int> done_gate{ 0 };
	double latencies[ENGINE_NUM_GPUS] = { 0.0 };

	const int warmups = 50;
	const int iters = 500;
	auto op = [&]( int d, cudaStream_t s ) {
		if ( mode == "plain" ) rings[d].all_reduce( tv_x[d], s );
		else if ( mode == "hc" ) rings[d].all_reduce_hc_apply( tv_x[d], d_streams[d], d_mix[d], d_mix[d] + 4 * rows, s );
		else if ( mode == "gated" ) rings[d].all_reduce_gated_norm( tv_x[d], d_streams[d], d_mix[d], static_cast<const engine::backend::half_t*>( d_w[d] ),
			d_streams[d] + static_cast<size_t>( rows ) * 4 * hidden_size, 4, 1e-6f, s );
		else if ( mode == "normed" ) rings[d].all_reduce_normed_add( tv_x[d], tv_r32[d], tv_w[d], 1.0f, &tv_w[d], &tv_out[d], 1e-6f, 1.0f, s );
		else rings[d].all_reduce_rms_norm( tv_x[d], tv_res[d], tv_w[d], 1e-5f, s );
	};

	std::thread workers[ENGINE_NUM_GPUS];
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		workers[d] = std::thread( [&, d]() {
			CUDA_CHECK( cudaSetDevice( d ) );
			cudaStream_t s = streams[d];

			for ( int i = 0; i < warmups; ++i ) {
				op( d, s );
				CUDA_CHECK( cudaStreamSynchronize( s ) );
			}

			start_gate.fetch_add( 1 );
			while ( start_gate.load() < ENGINE_NUM_GPUS ) {
				std::this_thread::yield();
			}

			cudaEvent_t ev0, ev1;
			CUDA_CHECK( cudaEventCreate( &ev0 ) );
			CUDA_CHECK( cudaEventCreate( &ev1 ) );

			CUDA_CHECK( cudaEventRecord( ev0, s ) );
			for ( int i = 0; i < iters; ++i ) {
				op( d, s );
			}
			CUDA_CHECK( cudaEventRecord( ev1, s ) );
			CUDA_CHECK( cudaEventSynchronize( ev1 ) );

			float ms = 0.0f;
			CUDA_CHECK( cudaEventElapsedTime( &ms, ev0, ev1 ) );
			latencies[d] = ( ms * 1000.0 ) / iters;

			CUDA_CHECK( cudaEventDestroy( ev0 ) );
			CUDA_CHECK( cudaEventDestroy( ev1 ) );

			done_gate.fetch_add( 1 );
		} );
	}

	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		workers[d].join();
	}

	double max_latency = 0.0;
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		if ( latencies[d] > max_latency ) {
			max_latency = latencies[d];
		}
	}

	std::printf( "[+] Latencies by GPU: [0]=%.2f us, [1]=%.2f us, [2]=%.2f us, [3]=%.2f us\n",
		latencies[0], latencies[1], latencies[2], latencies[3] );
	std::printf( "[+] 4-GPU %s (M=%d, H=%d): %.2f us per call (target <= 12.00 us)\n", mode == "rms" ? "AllReduce+RMSNorm" : mode.c_str(),
		rows, hidden_size, max_latency );
	std::fflush( stdout );

	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		CUDA_CHECK( cudaSetDevice( d ) );
		CUDA_CHECK( cudaFree( d_x[d] ) );
		CUDA_CHECK( cudaFree( d_res[d] ) );
		CUDA_CHECK( cudaFree( d_w[d] ) );
		CUDA_CHECK( cudaFree( d_streams[d] ) );
		CUDA_CHECK( cudaFree( d_mix[d] ) );
		CUDA_CHECK( cudaStreamDestroy( streams[d] ) );
	}

	engine::comm::destroy_4gpu_mesh( staging_ptrs, flag_ptrs );

	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		(void) cudaSetDevice( d );
		(void) cudaDeviceReset();
	}

	if ( mode != "rms" ) return 0;
	ENGINE_ASSERT( max_latency <= 12.0, "Latency exceeds 12.0 us limit" );
	std::printf( "[+] 4-GPU AllReduce+RMSNorm latency gate PASS (%.2f us <= 12.00 us)\n", max_latency );

	return 0;
}
