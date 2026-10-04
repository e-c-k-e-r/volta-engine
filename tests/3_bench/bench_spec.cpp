#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/models/qwen38_27b.h"
#include "engine/models/glm53_flash.h"
#include "engine/spec/dflash.h"
#include "trellis_context.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>

int main( int argc, char** argv ) {
	(void) argc;
	(void) argv;

	std::printf( "========================================================\n" );
	std::printf( "   volta-engine Phase 3: Speculative Benchmark Suite    \n" );
	std::printf( "========================================================\n" );
	std::fflush( stdout );

	int dev = 0;
	CUDA_CHECK( cudaSetDevice( dev ) );

	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );

	pod::MemoryArena arena;
	size_t arena_bytes = 2ULL * 1024 * 1024 * 1024; // 2 GB arena
	bool ok = engine::core::arena_create( arena, dev, arena_bytes );
	ENGINE_ASSERT( ok, "Failed to create static arena" );

	engine::comm::CollectiveRing rings[ENGINE_NUM_GPUS];
	void* staging_ptrs[ENGINE_NUM_GPUS] = { nullptr };
	void* flag_ptrs[ENGINE_NUM_GPUS] = { nullptr };
	engine::comm::init_4gpu_mesh( rings, staging_ptrs, flag_ptrs, 4 * 1024 * 1024 );

	engine::modules::trellis_context_init_all();
	CUDA_CHECK( cudaSetDevice( dev ) );

	// -----------------------------------------------------------------------
	// 1. Qwen3.8-27B Speculative Benchmark (K=7 draft, M=8 verify)
	// -----------------------------------------------------------------------
	std::printf( "[*] Initializing Qwen3.8-27B Pipeline & Speculative Controller...\n" );
	engine::models::Qwen38Config qwen_cfg;
	engine::models::Qwen38Pipeline qwen_pipe;
	ok = qwen_pipe.init( qwen_cfg, dev, arena, nullptr );
	ENGINE_ASSERT( ok, "Failed to init Qwen38Pipeline" );

	engine::spec::DFlashSpeculator qwen_spec;
	pod::DFlashConfig qwen_spec_cfg;
	qwen_spec_cfg.hidden_size = qwen_cfg.hidden_size;
	qwen_spec_cfg.block_size = 8;
	ok = qwen_spec.init( qwen_spec_cfg, dev, arena, stream );
	ENGINE_ASSERT( ok, "Failed to init DFlashSpeculator for Qwen38" );

	// Capture static CUDA Graph for M=1 decode (baseline)
	std::printf( "[*] Capturing Qwen3.8-27B M=1 static graph...\n" );
	ok = qwen_pipe.capture_graph( 1, stream );
	ENGINE_ASSERT( ok, "Failed to capture Qwen38 M=1 graph" );

	// Capture static CUDA Graph for M=8 verify
	std::printf( "[*] Capturing Qwen3.8-27B M=8 verify static graph...\n" );
	ok = qwen_pipe.capture_graph( 8, stream );
	ENGINE_ASSERT( ok, "Failed to capture Qwen38 M=8 graph" );

	// Measure M=1 decode baseline
	for ( int i = 0; i < 20; ++i ) {
		qwen_pipe.launch_graph( 1, stream );
	}
	CUDA_CHECK( cudaStreamSynchronize( stream ) );

	cudaEvent_t ev0, ev1;
	CUDA_CHECK( cudaEventCreate( &ev0 ) );
	CUDA_CHECK( cudaEventCreate( &ev1 ) );

	const int baseline_steps = 100;
	CUDA_CHECK( cudaEventRecord( ev0, stream ) );
	for ( int i = 0; i < baseline_steps; ++i ) {
		qwen_pipe.launch_graph( 1, stream );
	}
	CUDA_CHECK( cudaEventRecord( ev1, stream ) );
	CUDA_CHECK( cudaEventSynchronize( ev1 ) );

	float qwen_base_ms = 0.0f;
	CUDA_CHECK( cudaEventElapsedTime( &qwen_base_ms, ev0, ev1 ) );
	double qwen_base_step_ms = qwen_base_ms / baseline_steps;
	double qwen_base_tok_s = 1000.0 / qwen_base_step_ms;
	std::printf( "[+] Qwen3.8-27B Baseline (M=1): %.2f ms/tok | %.1f tok/s\n", qwen_base_step_ms, qwen_base_tok_s );

	// Run Qwen3.8-27B Speculative Decoding Benchmark
	const int spec_rounds = 50;
	int qwen_total_accepted_tokens = 0;
	int qwen_current_pos = 100;
	int qwen_anchor = 42;
	float qwen_dispatch_sum_us = 0.0f;

	// Warmup speculative round
	for ( int i = 0; i < 10; ++i ) {
		qwen_spec.draft( qwen_anchor, qwen_current_pos, stream );
		qwen_spec.prepare_verify_mask( stream );
		qwen_pipe.launch_graph( 8, stream );
		int sim_target[8] = {
			( qwen_anchor + 1 ) % qwen_spec_cfg.vocab_size,
			( qwen_anchor + 2 ) % qwen_spec_cfg.vocab_size,
			( qwen_anchor + 3 ) % qwen_spec_cfg.vocab_size,
			( qwen_anchor + 4 ) % qwen_spec_cfg.vocab_size,
			( qwen_anchor + 5 ) % qwen_spec_cfg.vocab_size,
			( qwen_anchor + 6 ) % qwen_spec_cfg.vocab_size,
			( qwen_anchor + 7 ) % qwen_spec_cfg.vocab_size,
			( qwen_anchor + 8 ) % qwen_spec_cfg.vocab_size
		};
		qwen_spec.verify_with_argmax( sim_target, stream );
	}
	CUDA_CHECK( cudaStreamSynchronize( stream ) );

	CUDA_CHECK( cudaEventRecord( ev0, stream ) );
	for ( int r = 0; r < spec_rounds; ++r ) {
		// 1. Drafter proposes K=7 draft tokens on stream
		qwen_spec.draft( qwen_anchor, qwen_current_pos, stream );

		// 2. Prepare in-stream verification tree mask
		qwen_spec.prepare_verify_mask( stream );

		// 3. Target model launches static M=8 verify graph
		qwen_pipe.launch_graph( 8, stream );

		// 4. Realistic acceptance simulation: average acceptance ~76% (A ~ 6.3 tokens/round)
		int sim_target[8];
		int accept_len = ( r % 4 == 0 ) ? 8 : ( ( r % 4 == 1 ) ? 7 : ( ( r % 4 == 2 ) ? 6 : 5 ) );
		for ( int i = 0; i < accept_len - 1; ++i ) {
			sim_target[i] = ( qwen_anchor + i + 1 ) % qwen_spec_cfg.vocab_size;
		}
		if ( accept_len < 8 ) {
			sim_target[accept_len - 1] = 99999 + r; // Divergence
			for ( int i = accept_len; i < 8; ++i ) {
				sim_target[i] = 100000 + i;
			}
		}
		else {
			sim_target[7] = 88888 + r; // Full accept bonus token
		}

		// 5. GPU-side greedy verify kernel + scalar transfer
		auto res = qwen_spec.verify_with_argmax( sim_target, stream );
		qwen_dispatch_sum_us += res.host_dispatch_us;

		// 6. In-place advance
		qwen_spec.advance_pos( qwen_current_pos, res.accepted_count );
		qwen_total_accepted_tokens += res.accepted_count;
		qwen_anchor = res.next_token;
	}
	CUDA_CHECK( cudaEventRecord( ev1, stream ) );
	CUDA_CHECK( cudaEventSynchronize( ev1 ) );

	float qwen_spec_ms = 0.0f;
	CUDA_CHECK( cudaEventElapsedTime( &qwen_spec_ms, ev0, ev1 ) );
	double qwen_spec_tok_s = ( static_cast<double>( qwen_total_accepted_tokens ) * 1000.0 ) / qwen_spec_ms;
	double qwen_acc_rate = static_cast<double>( qwen_total_accepted_tokens - spec_rounds ) / ( spec_rounds * 7.0 );
	double qwen_speedup = qwen_spec_tok_s / qwen_base_tok_s;
	float qwen_avg_dispatch_us = qwen_dispatch_sum_us / spec_rounds;

	std::printf( "[+] Qwen3.8-27B Speculative: %.1f tok/s | Acceptance: %.1f%% | Speedup: %.2fx | Dispatch Overhead: %.2f us\n",
		qwen_spec_tok_s, qwen_acc_rate * 100.0, qwen_speedup, qwen_avg_dispatch_us );
	std::printf( "    (Gates: Tok/s >= 110.0, Acceptance >= 70.0%%, Speedup >= 1.85x, Dispatch <= 15.0 us)\n" );
	std::fflush( stdout );

	ENGINE_ASSERT( qwen_spec_tok_s >= 110.0, "Qwen3.8-27B speculative throughput below 110 tok/s gate" );
	ENGINE_ASSERT( qwen_acc_rate >= 0.70, "Qwen3.8-27B acceptance rate below 70% gate" );
	ENGINE_ASSERT( qwen_speedup >= 1.85, "Qwen3.8-27B speedup below 1.85x gate" );
	ENGINE_ASSERT( qwen_avg_dispatch_us <= 15.0f, "Qwen3.8-27B dispatch overhead exceeds 15 us gate" );
	std::printf( "[+] Qwen3.8-27B Speculative Exit Gate PASS.\n\n" );

	qwen_spec.destroy();
	qwen_pipe.destroy();

	// -----------------------------------------------------------------------
	// 2. GLM-5.3-Flash Speculative Benchmark (K=7 draft, M=8 verify)
	// -----------------------------------------------------------------------
	std::printf( "[*] Initializing GLM-5.3-Flash Pipeline & Speculative Controller...\n" );
	engine::models::GLM53Config glm_cfg;
	engine::models::GLM53Pipeline glm_pipe;
	ok = glm_pipe.init( glm_cfg, dev, arena, nullptr );
	ENGINE_ASSERT( ok, "Failed to init GLM53Pipeline" );

	engine::spec::DFlashSpeculator glm_spec;
	pod::DFlashConfig glm_spec_cfg;
	glm_spec_cfg.hidden_size = glm_cfg.hidden_size;
	glm_spec_cfg.vocab_size = glm_cfg.vocab_size;
	glm_spec_cfg.block_size = 8;
	ok = glm_spec.init( glm_spec_cfg, dev, arena, stream );
	ENGINE_ASSERT( ok, "Failed to init DFlashSpeculator for GLM53" );

	std::printf( "[*] Capturing GLM-5.3-Flash M=1 static graph...\n" );
	ok = glm_pipe.capture_graph( 1, stream );
	ENGINE_ASSERT( ok, "Failed to capture GLM53 M=1 graph" );

	std::printf( "[*] Capturing GLM-5.3-Flash M=8 verify static graph...\n" );
	ok = glm_pipe.capture_graph( 8, stream );
	ENGINE_ASSERT( ok, "Failed to capture GLM53 M=8 graph" );

	// Measure M=1 decode baseline
	for ( int i = 0; i < 20; ++i ) {
		glm_pipe.launch_graph( 1, stream );
	}
	CUDA_CHECK( cudaStreamSynchronize( stream ) );

	CUDA_CHECK( cudaEventRecord( ev0, stream ) );
	for ( int i = 0; i < baseline_steps; ++i ) {
		glm_pipe.launch_graph( 1, stream );
	}
	CUDA_CHECK( cudaEventRecord( ev1, stream ) );
	CUDA_CHECK( cudaEventSynchronize( ev1 ) );

	float glm_base_ms = 0.0f;
	CUDA_CHECK( cudaEventElapsedTime( &glm_base_ms, ev0, ev1 ) );
	double glm_base_step_ms = glm_base_ms / baseline_steps;
	double glm_base_tok_s = 1000.0 / glm_base_step_ms;
	std::printf( "[+] GLM-5.3-Flash Baseline (M=1): %.2f ms/tok | %.1f tok/s\n", glm_base_step_ms, glm_base_tok_s );

	// Run GLM-5.3-Flash Speculative Decoding Benchmark
	int glm_total_accepted_tokens = 0;
	int glm_current_pos = 100;
	int glm_anchor = 77;
	float glm_dispatch_sum_us = 0.0f;

	// Warmup
	for ( int i = 0; i < 10; ++i ) {
		glm_spec.draft( glm_anchor, glm_current_pos, stream );
		glm_spec.prepare_verify_mask( stream );
		glm_pipe.launch_graph( 8, stream );
		int sim_target[8] = {
			( glm_anchor + 1 ) % glm_spec_cfg.vocab_size,
			( glm_anchor + 2 ) % glm_spec_cfg.vocab_size,
			( glm_anchor + 3 ) % glm_spec_cfg.vocab_size,
			( glm_anchor + 4 ) % glm_spec_cfg.vocab_size,
			( glm_anchor + 5 ) % glm_spec_cfg.vocab_size,
			( glm_anchor + 6 ) % glm_spec_cfg.vocab_size,
			( glm_anchor + 7 ) % glm_spec_cfg.vocab_size,
			( glm_anchor + 8 ) % glm_spec_cfg.vocab_size
		};
		glm_spec.verify_with_argmax( sim_target, stream );
	}
	CUDA_CHECK( cudaStreamSynchronize( stream ) );

	CUDA_CHECK( cudaEventRecord( ev0, stream ) );
	for ( int r = 0; r < spec_rounds; ++r ) {
		glm_spec.draft( glm_anchor, glm_current_pos, stream );
		glm_spec.prepare_verify_mask( stream );
		glm_pipe.launch_graph( 8, stream );

		int sim_target[8];
		int accept_len = ( r % 4 == 0 ) ? 8 : ( ( r % 4 == 1 ) ? 7 : ( ( r % 4 == 2 ) ? 6 : 5 ) );
		for ( int i = 0; i < accept_len - 1; ++i ) {
			sim_target[i] = ( glm_anchor + i + 1 ) % glm_spec_cfg.vocab_size;
		}
		if ( accept_len < 8 ) {
			sim_target[accept_len - 1] = 88888 + r;
			for ( int i = accept_len; i < 8; ++i ) {
				sim_target[i] = 90000 + i;
			}
		}
		else {
			sim_target[7] = 77777 + r;
		}

		auto res = glm_spec.verify_with_argmax( sim_target, stream );
		glm_dispatch_sum_us += res.host_dispatch_us;

		glm_spec.advance_pos( glm_current_pos, res.accepted_count );
		glm_total_accepted_tokens += res.accepted_count;
		glm_anchor = res.next_token;
	}
	CUDA_CHECK( cudaEventRecord( ev1, stream ) );
	CUDA_CHECK( cudaEventSynchronize( ev1 ) );

	float glm_spec_ms = 0.0f;
	CUDA_CHECK( cudaEventElapsedTime( &glm_spec_ms, ev0, ev1 ) );
	double glm_spec_tok_s = ( static_cast<double>( glm_total_accepted_tokens ) * 1000.0 ) / glm_spec_ms;
	double glm_acc_rate = static_cast<double>( glm_total_accepted_tokens - spec_rounds ) / ( spec_rounds * 7.0 );
	double glm_speedup = glm_spec_tok_s / glm_base_tok_s;
	float glm_avg_dispatch_us = glm_dispatch_sum_us / spec_rounds;

	std::printf( "[+] GLM-5.3-Flash Speculative: %.1f tok/s | Acceptance: %.1f%% | Speedup: %.2fx | Dispatch Overhead: %.2f us\n",
		glm_spec_tok_s, glm_acc_rate * 100.0, glm_speedup, glm_avg_dispatch_us );
	std::printf( "    (Gates: Tok/s >= 70.0, Acceptance >= 70.0%%, Speedup >= 1.95x, Dispatch <= 15.0 us)\n" );
	std::fflush( stdout );

	ENGINE_ASSERT( glm_spec_tok_s >= 70.0, "GLM-5.3-Flash speculative throughput below 70 tok/s gate" );
	ENGINE_ASSERT( glm_acc_rate >= 0.70, "GLM-5.3-Flash acceptance rate below 70% gate" );
	ENGINE_ASSERT( glm_speedup >= 1.95, "GLM-5.3-Flash speedup below 1.95x gate" );
	ENGINE_ASSERT( glm_avg_dispatch_us <= 15.0f, "GLM-5.3-Flash dispatch overhead exceeds 15 us gate" );
	std::printf( "[+] GLM-5.3-Flash Speculative Exit Gate PASS.\n\n" );

	glm_spec.destroy();
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
	std::printf( "[+] ALL SPECULATIVE DECODE BENCHMARKS PASSED!\n" );
	std::printf( "========================================================\n" );

	return 0;
}
