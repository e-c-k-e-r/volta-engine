#include "engine/config.h"
#include "engine/core/comm.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
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

	const int rows = 1;
	const int hidden_size = 5120;
	const size_t numel = static_cast<size_t>( rows ) * hidden_size;
	const size_t bytes = numel * sizeof( half );

	pod::TensorView tv_x[ENGINE_NUM_GPUS];
	pod::TensorView tv_res[ENGINE_NUM_GPUS];
	pod::TensorView tv_w[ENGINE_NUM_GPUS];

	void* d_x[ENGINE_NUM_GPUS];
	void* d_res[ENGINE_NUM_GPUS];
	void* d_w[ENGINE_NUM_GPUS];
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

		tv_x[d] = pod::make_tensor_2d( d_x[d], pod::DType::FP16, rows, hidden_size );
		tv_res[d] = pod::make_tensor_2d( d_res[d], pod::DType::FP16, rows, hidden_size );
		tv_w[d] = pod::make_tensor_1d( d_w[d], pod::DType::FP16, hidden_size );
	}

	std::atomic<int> start_gate{ 0 };
	std::atomic<int> done_gate{ 0 };
	double latencies[ENGINE_NUM_GPUS] = { 0.0 };

	const int warmups = 50;
	const int iters = 500;

	std::thread workers[ENGINE_NUM_GPUS];
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		workers[d] = std::thread( [&, d]() {
			CUDA_CHECK( cudaSetDevice( d ) );
			cudaStream_t s = streams[d];

			for ( int i = 0; i < warmups; ++i ) {
				rings[d].all_reduce_rms_norm( tv_x[d], tv_res[d], tv_w[d], 1e-5f, s );
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
				rings[d].all_reduce_rms_norm( tv_x[d], tv_res[d], tv_w[d], 1e-5f, s );
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
	std::printf( "[+] 4-GPU AllReduce+RMSNorm (M=%d, H=%d): %.2f us per call (target <= 12.00 us)\n",
		rows, hidden_size, max_latency );
	std::fflush( stdout );

	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		CUDA_CHECK( cudaSetDevice( d ) );
		CUDA_CHECK( cudaFree( d_x[d] ) );
		CUDA_CHECK( cudaFree( d_res[d] ) );
		CUDA_CHECK( cudaFree( d_w[d] ) );
		CUDA_CHECK( cudaStreamDestroy( streams[d] ) );
	}

	engine::comm::destroy_4gpu_mesh( staging_ptrs, flag_ptrs );

	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		(void) cudaSetDevice( d );
		(void) cudaDeviceReset();
	}

	ENGINE_ASSERT( max_latency <= 12.0, "Latency exceeds 12.0 us limit" );
	std::printf( "[+] 4-GPU AllReduce+RMSNorm latency gate PASS (%.2f us <= 12.00 us)\n", max_latency );

	return 0;
}
