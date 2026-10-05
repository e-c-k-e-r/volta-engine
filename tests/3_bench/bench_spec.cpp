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

		// 1. Evaluate target model on anchor token
		run_step_4gpu( cur_token, pos++ );
		CUDA_CHECK( cudaSetDevice( 0 ) );
		int target_pred = pipes[0].sample_next_token( streams[0] );

		// 2. Draft proposes K=7 candidate tokens conditioned on anchor logits & tapped states
		spec.draft( cur_token, pos, streams[0], pipes[0].logits().ptr<const half>(),
			context.data(), static_cast<int>( context.size() ), pipes[0].tapped_states() );

		int h_draft[8];
		float h_conf[8];
		CUDA_CHECK( cudaMemcpyAsync( h_draft, spec.d_draft_tokens(), 8 * sizeof( int ), cudaMemcpyDeviceToHost, streams[0] ) );
		CUDA_CHECK( cudaMemcpyAsync( h_conf, spec.d_conf(), 8 * sizeof( float ), cudaMemcpyDeviceToHost, streams[0] ) );
		CUDA_CHECK( cudaStreamSynchronize( streams[0] ) );

		int active_draft = 7;
		for ( int k = 1; k <= 7; ++k ) {
			if ( h_conf[k] < 0.25f ) {
				active_draft = k - 1;
				break;
			}
		}
		if ( active_draft < 1 ) active_draft = 1;
		total_draft_proposed += active_draft;

		// 3. Batched Target Verification: evaluate active draft tokens in a single forward pass
		int h_cands[8];
		for ( int k = 1; k <= active_draft; ++k ) {
			h_cands[k - 1] = h_draft[k];
		}

		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			pipes[d].checkpoint_state( streams[d] );
		}

		run_step_tp4_batched( h_cands, active_draft, pos );
		CUDA_CHECK( cudaSetDevice( 0 ) );
		int target_preds[8];
		pipes[0].sample_next_tokens( target_preds, active_draft, streams[0] );

		int accepted_count = 0;
		int bonus_or_diverged = target_pred;

		for ( int k = 1; k <= active_draft; ++k ) {
			int draft_tok = h_draft[k];
			int expected_tok = ( k == 1 ) ? target_pred : target_preds[k - 2];

			if ( draft_tok == expected_tok ) {
				accepted_count++;
				total_draft_accepted++;
				total_generated++;
				context.push_back( draft_tok );

				if ( total_generated >= 100 ) {
					bonus_or_diverged = draft_tok;
					break;
				}

				if ( k == active_draft ) {
					// All active draft tokens matched! Bonus token from target prediction
					bonus_or_diverged = target_preds[active_draft - 1];
					total_generated++;
					context.push_back( bonus_or_diverged );
				}
			}
			else {
				bonus_or_diverged = expected_tok;
				total_generated++;
				context.push_back( expected_tok );
				break;
			}
		}

		// Update position and state: if partial accept, rollback and step the accepted prefix
		if ( accepted_count < active_draft ) {
			for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
				pipes[d].rollback_state( streams[d] );
			}
			if ( accepted_count > 0 ) {
				run_step_tp4_batched( h_cands, accepted_count, pos );
			}
		}
		pos += ( accepted_count + 1 );
		cur_token = bonus_or_diverged;
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
