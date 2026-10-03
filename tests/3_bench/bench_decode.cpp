#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/models/qwen38_27b.h"
#include "engine/models/glm53_flash.h"
#include "trellis_context.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>

int main( int argc, char** argv ) {
	(void) argc;
	(void) argv;

	std::printf( "========================================================\n" );
	std::printf( "   volta-engine Phase 2: Autoregressive Decode Benchmark\n" );
	std::printf( "========================================================\n" );

	int dev = 0;
	CUDA_CHECK( cudaSetDevice( dev ) );

	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );

	pod::MemoryArena arena;
	size_t arena_bytes = 1024 * 1024 * 1024; // 1 GB arena
	bool ok = engine::core::arena_create( arena, dev, arena_bytes );
	ENGINE_ASSERT( ok, "Failed to create static arena" );

	engine::comm::CollectiveRing rings[ENGINE_NUM_GPUS];
	void* staging_ptrs[ENGINE_NUM_GPUS] = { nullptr };
	void* flag_ptrs[ENGINE_NUM_GPUS] = { nullptr };
	engine::comm::init_4gpu_mesh( rings, staging_ptrs, flag_ptrs, 4 * 1024 * 1024 );

	engine::modules::trellis_context_init_all();
	CUDA_CHECK( cudaSetDevice( dev ) );

	// -----------------------------------------------------------------------
	// 1. Qwen3.8-27B Benchmark
	// -----------------------------------------------------------------------
	std::printf( "[*] Initializing Qwen3.8-27B Pipeline (64 layers, 48 GDN + 16 SWA)...\n" );
	engine::models::Qwen38Config qwen_cfg;
	engine::models::Qwen38Pipeline qwen_pipe;
	ok = qwen_pipe.init( qwen_cfg, dev, arena, nullptr );
	ENGINE_ASSERT( ok, "Failed to init Qwen38Pipeline" );

	// Capture static CUDA Graph for M=1 decode
	std::printf( "[*] Capturing static CUDA Graph for M=1 decode...\n" );
	ok = qwen_pipe.capture_graph( 1, stream );
	ENGINE_ASSERT( ok, "Failed to capture Qwen38 M=1 graph" );

	// Warm-up
	for ( int i = 0; i < 20; ++i ) {
		qwen_pipe.launch_graph( 1, stream );
	}
	CUDA_CHECK( cudaStreamSynchronize( stream ) );

	const int qwen_steps = 100;
	cudaEvent_t ev0, ev1;
	CUDA_CHECK( cudaEventCreate( &ev0 ) );
	CUDA_CHECK( cudaEventCreate( &ev1 ) );

	CUDA_CHECK( cudaEventRecord( ev0, stream ) );
	for ( int i = 0; i < qwen_steps; ++i ) {
		qwen_pipe.launch_graph( 1, stream );
	}
	CUDA_CHECK( cudaEventRecord( ev1, stream ) );
	CUDA_CHECK( cudaEventSynchronize( ev1 ) );

	float qwen_ms = 0.0f;
	CUDA_CHECK( cudaEventElapsedTime( &qwen_ms, ev0, ev1 ) );
	double qwen_step_lat = qwen_ms / qwen_steps;
	double qwen_tok_s = 1000.0 / qwen_step_lat;

	std::printf( "[+] Qwen3.8-27B Decode (M=1): %.2f ms/tok | %.1f tok/s (target >= 58.0 tok/s)\n",
		qwen_step_lat, qwen_tok_s );
	std::fflush( stdout );

	ENGINE_ASSERT( qwen_tok_s >= 58.0, "Qwen3.8-27B throughput below 58 tok/s gate" );
	std::printf( "[+] Qwen3.8-27B Exit Gate PASS.\n" );

	qwen_pipe.destroy();

	// -----------------------------------------------------------------------
	// 2. GLM-5.3-Flash Benchmark
	// -----------------------------------------------------------------------
	std::printf( "\n[*] Initializing GLM-5.3-Flash Pipeline (40 layers, 288-expert MoE + MLA/DSA)...\n" );
	engine::models::GLM53Config glm_cfg;
	engine::models::GLM53Pipeline glm_pipe;
	ok = glm_pipe.init( glm_cfg, dev, arena, nullptr );
	ENGINE_ASSERT( ok, "Failed to init GLM53Pipeline" );

	std::printf( "[*] Capturing static CUDA Graph for M=1 decode...\n" );
	ok = glm_pipe.capture_graph( 1, stream );
	ENGINE_ASSERT( ok, "Failed to capture GLM53 M=1 graph" );

	for ( int i = 0; i < 20; ++i ) {
		glm_pipe.launch_graph( 1, stream );
	}
	CUDA_CHECK( cudaStreamSynchronize( stream ) );

	const int glm_steps = 100;
	CUDA_CHECK( cudaEventRecord( ev0, stream ) );
	for ( int i = 0; i < glm_steps; ++i ) {
		glm_pipe.launch_graph( 1, stream );
	}
	CUDA_CHECK( cudaEventRecord( ev1, stream ) );
	CUDA_CHECK( cudaEventSynchronize( ev1 ) );

	float glm_ms = 0.0f;
	CUDA_CHECK( cudaEventElapsedTime( &glm_ms, ev0, ev1 ) );
	double glm_step_lat = glm_ms / glm_steps;
	double glm_tok_s = 1000.0 / glm_step_lat;

	std::printf( "[+] GLM-5.3-Flash Decode (M=1): %.2f ms/tok | %.1f tok/s (target >= 35.0 tok/s)\n",
		glm_step_lat, glm_tok_s );
	std::fflush( stdout );

	ENGINE_ASSERT( glm_tok_s >= 35.0, "GLM-5.3-Flash throughput below 35 tok/s gate" );
	std::printf( "[+] GLM-5.3-Flash Exit Gate PASS.\n" );

	glm_pipe.destroy();

	// Teardown
	CUDA_CHECK( cudaEventDestroy( ev0 ) );
	CUDA_CHECK( cudaEventDestroy( ev1 ) );
	CUDA_CHECK( cudaStreamDestroy( stream ) );

	engine::core::arena_destroy( arena );
	engine::comm::destroy_4gpu_mesh( staging_ptrs, flag_ptrs );

	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		cudaSetDevice( d );
		cudaDeviceReset();
	}

	std::printf( "========================================================\n" );
	std::printf( "[+] ALL AUTOREGRESSIVE DECODE BENCHMARKS PASSED!\n" );
	std::printf( "========================================================\n" );

	return 0;
}
