#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/models/qwen38_27b.h"
#include "engine/spec/dflash.h"
#include "trellis_context.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <thread>
#include <memory>
#include <chrono>
#include <atomic>
#include <pthread.h>

int main( int argc, char** argv ) {
	(void) argc;
	(void) argv;

	std::printf( "========================================================\n" );
	std::printf( "   volta-engine Phase 3: Speculative Benchmark          \n" );
	std::printf( "   Mode: Qwen3.8-27B + DFlash2 (TP=4)                   \n" );
	std::printf( "========================================================\n" );
	std::fflush( stdout );

	engine::comm::CollectiveRing rings[ENGINE_NUM_GPUS];
	void* staging_ptrs[ENGINE_NUM_GPUS] = { nullptr };
	void* flag_ptrs[ENGINE_NUM_GPUS] = { nullptr };
	engine::comm::init_4gpu_mesh( rings, staging_ptrs, flag_ptrs, 4 * 1024 * 1024 );

	cudaStream_t streams[ENGINE_NUM_GPUS];
	pod::MemoryArena arenas[ENGINE_NUM_GPUS];
	auto pipes = std::make_unique<engine::models::Qwen38Pipeline[]>( ENGINE_NUM_GPUS );
	engine::models::Qwen38Config config;

	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		CUDA_CHECK( cudaSetDevice( d ) );
		CUDA_CHECK( cudaStreamCreate( &streams[d] ) );
		bool arena_ok = engine::core::arena_create( arenas[d], d, 10ULL * 1024 * 1024 * 1024 );
		ENGINE_ASSERT( arena_ok, "Failed to create Qwen arena" );
		bool pipe_ok = pipes[d].init( config, d, arenas[d], &rings[d] );
		ENGINE_ASSERT( pipe_ok, "Failed to init Qwen pipeline" );
	}
	engine::modules::trellis_context_init_all();

	// 1. Initialize DFlash2 Speculator on GPU 0
	engine::spec::DFlashSpeculator spec;
	pod::DFlashConfig spec_cfg;
	spec_cfg.hidden_size = config.hidden_size;
	spec_cfg.vocab_size = 248320;
	spec_cfg.block_size = 8;
	bool spec_init = spec.init( spec_cfg, 0, arenas[0], streams[0] );
	ENGINE_ASSERT( spec_init, "Failed to init DFlashSpeculator" );

	const char* draft_dir = "/home/mrq/Documents/models/Qwen3.8-27B-DFlash2-EXL3-5.0bpw";
	std::printf( "[*] Loading DFlash2 weights from %s...\n", draft_dir );
	bool spec_loaded = spec.load_weights( draft_dir, streams[0] );
	ENGINE_ASSERT( spec_loaded, "Failed to load DFlash2 weights" );
	std::printf( "[+] DFlash2 Speculator loaded successfully.\n" );

	// 2. Load Target Model Weights
	const char* model_dir = "/home/mrq/Documents/models/Qwen3.8-27B-exl3";
	std::printf( "[*] Loading Qwen3.8-27B weights across 4 GPUs (TP=4) from %s...\n", model_dir );
	std::fflush( stdout );
	{
		std::thread loaders[ENGINE_NUM_GPUS];
		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			loaders[d] = std::thread( [&, d]() {
				CUDA_CHECK( cudaSetDevice( d ) );
				bool load_ok = pipes[d].load_weights( model_dir, streams[d] );
				ENGINE_ASSERT( load_ok, "Failed to load Qwen weights" );
				CUDA_CHECK( cudaStreamSynchronize( streams[d] ) );
			} );
		}
		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			loaders[d].join();
		}
	}
	std::printf( "[+] Qwen3.8-27B weights loaded successfully on all 4 GPUs.\n" );
	std::fflush( stdout );

	pthread_barrier_t step_barrier;
	pthread_barrier_init( &step_barrier, nullptr, 5 ); // 4 workers + 1 master

	pthread_barrier_t capture_barrier;
	pthread_barrier_init( &capture_barrier, nullptr, 4 );

	int* d_candidate_tokens[ENGINE_NUM_GPUS];
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		CUDA_CHECK( cudaSetDevice( d ) );
		CUDA_CHECK( cudaMalloc( &d_candidate_tokens[d], 8 * sizeof( int ) ) );
	}

	std::atomic<bool> worker_running{ true };
	std::atomic<bool> use_graph{ false };
	std::atomic<int> current_batch_size{ 1 };
	std::atomic<int> current_tok{ 15 };
	std::atomic<int> current_pos{ 0 };

	std::thread workers[ENGINE_NUM_GPUS];
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		workers[d] = std::thread( [&, d]() {
			cpu_set_t cpuset;
			CPU_ZERO( &cpuset );
			CPU_SET( d, &cpuset );
			pthread_setaffinity_np( pthread_self(), sizeof( cpu_set_t ), &cpuset );

			CUDA_CHECK( cudaSetDevice( d ) );

			cudaGraph_t graph_m1;
			cudaGraphExec_t instance_m1;
			bool graph_m1_captured = false;

			cudaGraph_t graph_m8;
			cudaGraphExec_t instance_m8;
			bool graph_m8_captured = false;

			while ( true ) {
				pthread_barrier_wait( &step_barrier );
				if ( !worker_running.load() ) break;

				int bsz = current_batch_size.load();
				int tok = current_tok.load();
				int pos = current_pos.load();
				if ( use_graph.load() ) {
					if ( bsz == 1 ) {
						if ( !graph_m1_captured ) {
							CUDA_CHECK( cudaStreamBeginCapture( streams[d], cudaStreamCaptureModeGlobal ) );
							pipes[d].forward_step( pipes[d].hidden_states(), pipes[d].logits(), pos, 0, streams[d] );
							CUDA_CHECK( cudaStreamEndCapture( streams[d], &graph_m1 ) );
							pthread_barrier_wait( &capture_barrier );
							CUDA_CHECK( cudaGraphInstantiate( &instance_m1, graph_m1, nullptr, nullptr, 0 ) );
							graph_m1_captured = true;
						}
						if ( tok >= 0 ) {
							pipes[d].embed_token( tok, pos, streams[d] );
						}
						CUDA_CHECK( cudaGraphLaunch( instance_m1, streams[d] ) );
					}
					else {
						pipes[d].embed_tokens( d_candidate_tokens[d], bsz, streams[d] );
						if ( !graph_m8_captured ) {
							CUDA_CHECK( cudaStreamBeginCapture( streams[d], cudaStreamCaptureModeGlobal ) );
							pipes[d].forward_step( pipes[d].hidden_states(), pipes[d].logits(), pos, 0, streams[d] );
							CUDA_CHECK( cudaStreamEndCapture( streams[d], &graph_m8 ) );
							pthread_barrier_wait( &capture_barrier );
							CUDA_CHECK( cudaGraphInstantiate( &instance_m8, graph_m8, nullptr, nullptr, 0 ) );
							graph_m8_captured = true;
						}
						CUDA_CHECK( cudaGraphLaunch( instance_m8, streams[d] ) );
					}
				}
				else {
					if ( bsz == 1 ) {
						if ( tok >= 0 ) {
							pipes[d].embed_token( tok, pos, streams[d] );
						}
						pipes[d].forward_step( pipes[d].hidden_states(), pipes[d].logits(), pos, 0, streams[d] );
					}
					else {
						pipes[d].embed_tokens( d_candidate_tokens[d], bsz, streams[d] );
						pipes[d].forward_step( pipes[d].hidden_states(), pipes[d].logits(), pos, 0, streams[d] );
					}
				}
				CUDA_CHECK( cudaStreamSynchronize( streams[d] ) );

				pthread_barrier_wait( &step_barrier );
			}

			if ( graph_m1_captured ) {
				CUDA_CHECK( cudaGraphExecDestroy( instance_m1 ) );
				CUDA_CHECK( cudaGraphDestroy( graph_m1 ) );
			}
			if ( graph_m8_captured ) {
				CUDA_CHECK( cudaGraphExecDestroy( instance_m8 ) );
				CUDA_CHECK( cudaGraphDestroy( graph_m8 ) );
			}
		} );
	}

	auto run_step_4gpu = [&]( int tok, int pos ) {
		current_batch_size.store( 1 );
		current_tok.store( tok );
		current_pos.store( pos );
		pthread_barrier_wait( &step_barrier );
		pthread_barrier_wait( &step_barrier );
	};

	auto run_step_tp4_batched = [&]( const int* h_tokens, int batch_size, int pos ) {
		current_batch_size.store( batch_size );
		current_pos.store( pos );
		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			CUDA_CHECK( cudaMemcpyAsync( d_candidate_tokens[d], h_tokens, batch_size * sizeof( int ), cudaMemcpyHostToDevice, streams[d] ) );
		}
		pthread_barrier_wait( &step_barrier );
		pthread_barrier_wait( &step_barrier );
	};

	// 3. Warm-up
	std::printf( "[*] Warming up speculative execution...\n" );
	use_graph.store( true );
	int dummy_tokens[8] = { 15, 16, 17, 18, 19, 20, 21, 22 };
	run_step_4gpu( 15, 0 );
	run_step_tp4_batched( dummy_tokens, 8, 0 );
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		pipes[d].reset_kv_cache();
	}
	std::printf( "[+] Warm-up complete.\n" );

	// 4. Run Speculative Benchmark over 100 generated tokens
	std::printf( "[*] Executing speculative generation (target: 100 tokens)...\n" );
	std::fflush( stdout );

	// Coherence Prompt: "List the first 5 prime numbers:"
	std::vector<int> prompt = { 248045, 846, 198, 826, 279, 1118, 220, 20, 9944, 4947, 25, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 };
	int pos = 0;
	use_graph.store( false );
	for ( int tok : prompt ) {
		run_step_4gpu( tok, pos++ );
	}
	CUDA_CHECK( cudaSetDevice( 0 ) );
	int cur_token = pipes[0].sample_next_token( streams[0] );
	use_graph.store( true );

	std::vector<int> context = prompt;
	context.push_back( cur_token );

	int total_generated = 1;
	int total_draft_proposed = 0;
	int total_draft_accepted = 0;
	int rounds = 0;
	int prev_a = 0;

	auto t_start = std::chrono::high_resolution_clock::now();

	while ( total_generated < 100 ) {
		rounds++;

		// 1. Draft proposes K=7 candidate tokens [T_1..T_7] from anchor T_0 at pos conditioned on tapped states from row A of previous round
		const half* anchor_logits = pipes[0].logits().ptr<const half>() + static_cast<size_t>( prev_a ) * 248320;
		spec.draft( cur_token, pos, streams[0], anchor_logits,
			context.data(), static_cast<int>( context.size() ), pipes[0].tapped_states( prev_a ) );

		int h_draft[8];
		CUDA_CHECK( cudaMemcpyAsync( h_draft, spec.d_draft_tokens(), 8 * sizeof( int ), cudaMemcpyDeviceToHost, streams[0] ) );
		CUDA_CHECK( cudaStreamSynchronize( streams[0] ) );

		// 2. Candidate block [T_0, T_1..T_7] assembled: T_0 is current anchor
		h_draft[0] = cur_token;

		// 3. Launch single batched target forward pass (M=8) across all 4 GPUs via cudaGraphLaunch
		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			pipes[d].checkpoint_state( streams[d] );
		}
		run_step_tp4_batched( h_draft, 8, pos );

		// 4. Extract the 8 target predictions [P_0..P_7] via GPU-side argmax on GPU 0
		CUDA_CHECK( cudaSetDevice( 0 ) );
		int target_preds[8];
		pipes[0].sample_next_tokens( target_preds, 8, streams[0] );

		// 5. Find first mismatch index A in [0, 7] where T_{i+1} != P_i
		int A = 7;
		for ( int i = 0; i < 7; ++i ) {
			if ( h_draft[i + 1] != target_preds[i] ) {
				A = i;
				break;
			}
		}

		// 6. Emitted tokens this round: A + 1 (the A accepted draft tokens plus the target model's correction/bonus token P_A)
		total_draft_proposed += 7;
		total_draft_accepted += A;
		for ( int i = 0; i < A; ++i ) {
			context.push_back( h_draft[i + 1] );
			total_generated++;
			if ( total_generated >= 100 ) break;
		}
		if ( total_generated < 100 ) {
			context.push_back( target_preds[A] );
			total_generated++;
		}

		// 7. Next anchor token is T_next = P_A. Advance position: pos += (A + 1)
		cur_token = target_preds[A];
		pos += ( A + 1 );
		prev_a = A;

		// 8. Rollback GDN/KV state to pos
		if ( A < 7 ) {
			for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
				pipes[d].rollback_state( streams[d] );
			}
		}
	}

	auto t_end = std::chrono::high_resolution_clock::now();
	double elapsed_ms = std::chrono::duration<double, std::milli>( t_end - t_start ).count();
	double tok_s = ( total_generated * 1000.0 ) / elapsed_ms;
	double acc_rate = ( total_draft_proposed > 0 ) ? ( static_cast<double>( total_draft_accepted ) / total_draft_proposed ) : 0.0;
	double avg_accepted_per_round = static_cast<double>( total_draft_accepted ) / rounds;

	std::printf( "\n========================================================\n" );
	std::printf( "        SPECULATIVE BENCHMARK RESULTS (Qwen3.8-27B)     \n" );
	std::printf( "========================================================\n" );
	std::printf( "   Total Generated Tokens : %d tokens\n", total_generated );
	std::printf( "   Elapsed Wall Time      : %.2f ms\n", elapsed_ms );
	std::printf( "   Throughput             : %.2f tok/s\n", tok_s );
	std::printf( "   Rounds Executed        : %d rounds\n", rounds );
	std::printf( "   Draft Acceptance Rate  : %.1f%% (%d / %d)\n", acc_rate * 100.0, total_draft_accepted, total_draft_proposed );
	std::printf( "   Mean Accepted / Round  : %.2f tokens/round\n", avg_accepted_per_round );
	std::printf( "   Mean Wall Round Time   : %.2f ms/round\n", elapsed_ms / rounds );
	std::printf( "========================================================\n" );

	worker_running.store( false );
	pthread_barrier_wait( &step_barrier );
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		workers[d].join();
	}
	pthread_barrier_destroy( &step_barrier );
	pthread_barrier_destroy( &capture_barrier );

	spec.destroy();
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		CUDA_CHECK( cudaSetDevice( d ) );
		CUDA_CHECK( cudaFree( d_candidate_tokens[d] ) );
		CUDA_CHECK( cudaStreamDestroy( streams[d] ) );
		pipes[d].destroy();
		engine::core::arena_destroy( arenas[d] );
	}
	engine::comm::destroy_4gpu_mesh( staging_ptrs, flag_ptrs );
	engine::modules::trellis_context_destroy();

	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		cudaSetDevice( d );
		cudaDeviceReset();
	}
	return 0;
}
