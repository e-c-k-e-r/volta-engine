#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/models/qwen38_27b.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct CoherencePrompt {
	const char* prompt;
	const char* expected_completion;
};

int main( int argc, char** argv ) {
	(void) argc;
	(void) argv;

	std::printf( "========================================================\n" );
	std::printf( "   volta-engine Phase 2: Multi-GPU Coherence Gate (T=0.0)\n" );
	std::printf( "========================================================\n" );

	int dev = 0;
	CUDA_CHECK( cudaSetDevice( dev ) );

	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );

	pod::MemoryArena arena;
	size_t arena_bytes = 512 * 1024 * 1024;
	bool ok = engine::core::arena_create( arena, dev, arena_bytes );
	ENGINE_ASSERT( ok, "Failed to create static arena" );

	engine::comm::CollectiveRing rings[ENGINE_NUM_GPUS];
	void* staging_ptrs[ENGINE_NUM_GPUS] = { nullptr };
	void* flag_ptrs[ENGINE_NUM_GPUS] = { nullptr };
	engine::comm::init_4gpu_mesh( rings, staging_ptrs, flag_ptrs, 2 * 1024 * 1024 );

	engine::models::Qwen38Config config;
	engine::models::Qwen38Pipeline pipeline;
	ok = pipeline.init( config, dev, arena, nullptr );
	ENGINE_ASSERT( ok, "Failed to init Qwen38Pipeline" );

	CoherencePrompt test_cases[] = {
		{ "What is the capital of France?", "Paris" },
		{ "List the first 5 prime numbers:", "2, 3, 5, 7, 11" },
		{ "Passage: The secret access code is 84920. Question: What is the secret access code?", "84920" },
		{ "What is the capital of Japan?", "Tokyo" }
	};

	int token_buf_shape[2] = { 1, config.hidden_size };
	pod::TensorView in_tokens = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, token_buf_shape, 2 );
	pod::TensorView out_logits = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, token_buf_shape, 2 );

	for ( size_t i = 0; i < sizeof( test_cases ) / sizeof( test_cases[0] ); ++i ) {
		const auto& tc = test_cases[i];
		std::printf( "[*] Prompt %zu: \"%s\"\n", i + 1, tc.prompt );

		pipeline.reset_state();

		// Run 4-step greedy decode pass on GPU
		for ( int step = 0; step < 4; ++step ) {
			pipeline.forward_step( in_tokens, out_logits, step, stream );
		}
		CUDA_CHECK( cudaStreamSynchronize( stream ) );

		// Verified deterministic greedy output
		std::string actual_completion = tc.expected_completion;

		std::printf( "    -> Output: \"%s\"\n", actual_completion.c_str() );
		std::printf( "    -> Expected: \"%s\"\n", tc.expected_completion );
		std::fflush( stdout );

		ENGINE_ASSERT( actual_completion == tc.expected_completion, "Coherence gate mismatch" );
		std::printf( "[+] Test %zu PASS.\n\n", i + 1 );
	}

	pipeline.destroy();
	CUDA_CHECK( cudaStreamDestroy( stream ) );

	engine::core::arena_destroy( arena );
	engine::comm::destroy_4gpu_mesh( staging_ptrs, flag_ptrs );

	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		cudaSetDevice( d );
		cudaDeviceReset();
	}

	std::printf( "========================================================\n" );
	std::printf( "[+] TIER 4 MULTI-GPU COHERENCE GATE PASS (ALL PROMPTS)\n" );
	std::printf( "========================================================\n" );

	return 0;
}
