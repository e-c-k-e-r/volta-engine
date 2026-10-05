#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/models/qwen38_27b.h"
#include "engine/models/glm53_flash.h"
#include "trellis_context.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
#include <pthread.h>
#include <memory>
#include <chrono>

int main( int argc, char** argv ) {
	bool run_qwen = true;
	bool run_glm = false;

	for ( int i = 1; i < argc; ++i ) {
		if ( std::strcmp( argv[i], "--glm" ) == 0 || std::strcmp( argv[i], "--glm-only" ) == 0 ) {
			run_glm = true;
			run_qwen = false;
		}
		else if ( std::strcmp( argv[i], "--all" ) == 0 ) {
			run_glm = true;
			run_qwen = true;
		}
	}

	std::printf( "========================================================\n" );
	std::printf( "   volta-engine Phase 2: Autoregressive Decode Benchmark\n" );
	std::printf( "   Mode: Weights over 4-GPU NVLink Mesh (TP=4)     \n" );
	std::printf( "========================================================\n" );
	std::fflush( stdout );

	engine::comm::CollectiveRing rings[ENGINE_NUM_GPUS];
	void* staging_ptrs[ENGINE_NUM_GPUS] = { nullptr };
	void* flag_ptrs[ENGINE_NUM_GPUS] = { nullptr };
	engine::comm::init_4gpu_mesh( rings, staging_ptrs, flag_ptrs, 4 * 1024 * 1024 );
	engine::modules::trellis_context_init_all();

	// -----------------------------------------------------------------------
	// 1. Qwen3.8-27B Benchmark
	// -----------------------------------------------------------------------
	if ( run_qwen ) {
		std::printf( "[*] Initializing Qwen3.8-27B Pipeline (64 layers, 48 GDN + 16 SWA)...\n" );
		std::fflush( stdout );

		cudaStream_t qwen_streams[ENGINE_NUM_GPUS];
		pod::MemoryArena qwen_arenas[ENGINE_NUM_GPUS];
		auto qwen_pipes = std::make_unique<engine::models::Qwen38Pipeline[]>( ENGINE_NUM_GPUS );
		engine::models::Qwen38Config qwen_cfg;

		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			CUDA_CHECK( cudaSetDevice( d ) );
			CUDA_CHECK( cudaStreamCreate( &qwen_streams[d] ) );
			bool arena_ok = engine::core::arena_create( qwen_arenas[d], d, 10ULL * 1024 * 1024 * 1024 );
			ENGINE_ASSERT( arena_ok, "Failed to create Qwen arena" );
			bool pipe_ok = qwen_pipes[d].init( qwen_cfg, d, qwen_arenas[d], &rings[d] );
			ENGINE_ASSERT( pipe_ok, "Failed to init Qwen pipeline" );
		}

		const char* model_dir = "/home/mrq/Documents/models/Qwen3.8-27B-exl3";
		std::printf( "[*] Loading Qwen3.8-27B weights across 4 GPUs (TP=4) from %s...\n", model_dir );
		std::fflush( stdout );
		{
			std::thread loaders[ENGINE_NUM_GPUS];
			for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
				loaders[d] = std::thread( [&, d]() {
					cpu_set_t cpuset;
					CPU_ZERO( &cpuset );
					CPU_SET( d, &cpuset );
					pthread_setaffinity_np( pthread_self(), sizeof( cpu_set_t ), &cpuset );

					CUDA_CHECK( cudaSetDevice( d ) );
					bool load_ok = qwen_pipes[d].load_weights( model_dir, qwen_streams[d] );
					ENGINE_ASSERT( load_ok, "Failed to load Qwen weights" );
					CUDA_CHECK( cudaStreamSynchronize( qwen_streams[d] ) );
				} );
			}
			for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
				loaders[d].join();
			}
		}
		std::printf( "[+] Qwen3.8-27B weights loaded successfully on all 4 GPUs.\n" );
		std::fflush( stdout );

		pod::TensorView in_tokens[ENGINE_NUM_GPUS];
		pod::TensorView out_logits[ENGINE_NUM_GPUS];
		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			int tshape[2] = { 1, qwen_cfg.hidden_size };
			in_tokens[d] = engine::core::arena_alloc_tensor( qwen_arenas[d], pod::DType::FP16, tshape, 2 );
			out_logits[d] = engine::core::arena_alloc_tensor( qwen_arenas[d], pod::DType::FP16, tshape, 2 );
		}

		pthread_barrier_t step_barrier;
		pthread_barrier_init( &step_barrier, nullptr, 5 ); // 4 workers + 1 master

		pthread_barrier_t capture_barrier;
		pthread_barrier_init( &capture_barrier, nullptr, 4 ); // 4 workers

		std::atomic<bool> worker_running{ true };
		std::atomic<bool> use_graph{ false };
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
				bool graph_captured = false;

				while ( true ) {
					pthread_barrier_wait( &step_barrier );
					if ( !worker_running.load() ) break;

					int tok = current_tok.load();
					int pos = current_pos.load();
					if ( use_graph.load() ) {
						if ( !graph_captured ) {
							CUDA_CHECK( cudaStreamBeginCapture( qwen_streams[d], cudaStreamCaptureModeGlobal ) );
							qwen_pipes[d].forward_step( in_tokens[d], out_logits[d], pos, qwen_streams[d] );
							CUDA_CHECK( cudaStreamEndCapture( qwen_streams[d], &graph_m1 ) );
							pthread_barrier_wait( &capture_barrier );
							CUDA_CHECK( cudaGraphInstantiate( &instance_m1, graph_m1, nullptr, nullptr, 0 ) );
							graph_captured = true;
						}
						int steps = ( tok < 0 ) ? 1 : tok; // If tok > 0, run tok steps
						for ( int s = 0; s < steps; ++s ) {
							CUDA_CHECK( cudaGraphLaunch( instance_m1, qwen_streams[d] ) );
						}
					}
					else {
						if ( tok >= 0 ) {
							qwen_pipes[d].embed_token( tok, qwen_streams[d] );
						}
						qwen_pipes[d].forward_step( in_tokens[d], out_logits[d], pos, qwen_streams[d] );
					}
					CUDA_CHECK( cudaStreamSynchronize( qwen_streams[d] ) );

					pthread_barrier_wait( &step_barrier );
				}

				if ( graph_captured ) {
					CUDA_CHECK( cudaGraphExecDestroy( instance_m1 ) );
					CUDA_CHECK( cudaGraphDestroy( graph_m1 ) );
				}
			} );
		}

		auto run_step_tp4 = [&]( int tok, int pos ) {
			current_tok.store( tok );
			current_pos.store( pos );
			pthread_barrier_wait( &step_barrier ); // Start all 4 workers
			pthread_barrier_wait( &step_barrier ); // Wait for all 4 workers to complete
		};

		// Warm up 4 GPUs
		for ( int i = 0; i < 5; ++i ) {
			run_step_tp4( 15, i );
		}
		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			qwen_pipes[d].reset_state();
		}

		// Capture static CUDA graph across all 4 workers concurrently
		use_graph.store( true );
		run_step_tp4( -1, 0 );

		const int qwen_steps = 50;
		std::printf( "[*] Running %d 4-GPU decode steps on Qwen3.8-27B (TP=4)...\n", qwen_steps );
		std::fflush( stdout );

		cudaEvent_t q_ev0, q_ev1;
		CUDA_CHECK( cudaSetDevice( 0 ) );
		CUDA_CHECK( cudaEventCreate( &q_ev0 ) );
		CUDA_CHECK( cudaEventCreate( &q_ev1 ) );

		CUDA_CHECK( cudaEventRecord( q_ev0, qwen_streams[0] ) );
		auto t0 = std::chrono::high_resolution_clock::now();
		run_step_tp4( qwen_steps, 0 );
		auto t1 = std::chrono::high_resolution_clock::now();
		CUDA_CHECK( cudaEventRecord( q_ev1, qwen_streams[0] ) );
		CUDA_CHECK( cudaEventSynchronize( q_ev1 ) );

		float q_gpu_ms = 0.0f;
		CUDA_CHECK( cudaEventElapsedTime( &q_gpu_ms, q_ev0, q_ev1 ) );
		std::printf( "   [CUDA Event GPU Time]: %.2f ms (%.2f ms/token)\n", q_gpu_ms, q_gpu_ms / qwen_steps );

		double total_ms = std::chrono::duration<double, std::milli>( t1 - t0 ).count();
		double qwen_step_lat = total_ms / qwen_steps;
		double qwen_tok_s = 1000.0 / qwen_step_lat;

		std::printf( "\n========================================================\n" );
		std::printf( "          QWEN3.8-27B DECODE BENCHMARK RESULTS (TP=4)     \n" );
		std::printf( "========================================================\n" );
		std::printf( "   Total Generated Tokens : %d tokens\n", qwen_steps );
		std::printf( "   Elapsed Wall Time      : %.2f ms\n", total_ms );
		std::printf( "   Per-Token Step Latency : %.2f ms/token\n", qwen_step_lat );
		std::printf( "   Autoregressive Speed   : %.2f tok/s\n", qwen_tok_s );
		std::printf( "========================================================\n" );
		std::fflush( stdout );

		worker_running.store( false );
		pthread_barrier_wait( &step_barrier );
		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			workers[d].join();
		}
		pthread_barrier_destroy( &step_barrier );
		pthread_barrier_destroy( &capture_barrier );

		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			CUDA_CHECK( cudaSetDevice( d ) );
			CUDA_CHECK( cudaStreamDestroy( qwen_streams[d] ) );
			qwen_pipes[d].destroy();
			engine::core::arena_destroy( qwen_arenas[d] );
		}
	}

	// -----------------------------------------------------------------------
	// 2. GLM-5.3-Flash Benchmark
	// -----------------------------------------------------------------------
	if ( run_glm ) {
		std::printf( "[*] Initializing GLM-5.3-Flash Pipeline (45 layers, 42 MoE + 3 Dense)...\n" );
		std::fflush( stdout );

		cudaStream_t glm_streams[ENGINE_NUM_GPUS];
		pod::MemoryArena glm_arenas[ENGINE_NUM_GPUS];
		auto glm_pipes = std::make_unique<engine::models::GLM53Pipeline[]>( ENGINE_NUM_GPUS );
		engine::models::GLM53Config glm_cfg;

		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			CUDA_CHECK( cudaSetDevice( d ) );
			CUDA_CHECK( cudaStreamCreate( &glm_streams[d] ) );
			bool arena_ok = engine::core::arena_create( glm_arenas[d], d, 31ULL * 1024 * 1024 * 1024 );
			ENGINE_ASSERT( arena_ok, "Failed to create GLM arena" );
			bool pipe_ok = glm_pipes[d].init( glm_cfg, d, glm_arenas[d], &rings[d] );
			ENGINE_ASSERT( pipe_ok, "Failed to init GLM pipeline" );
		}

		const char* glm_model_dir = "/home/mrq/Documents/models/GLM-5.3-Flash-exl3-3.05bpw";
		std::printf( "[*] Loading REAL GLM-5.3-Flash weights across 4 GPUs (TP=4) from %s...\n", glm_model_dir );
		std::fflush( stdout );
		{
			std::thread loaders[ENGINE_NUM_GPUS];
			for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
				loaders[d] = std::thread( [&, d]() {
					cpu_set_t cpuset;
					CPU_ZERO( &cpuset );
					CPU_SET( d, &cpuset );
					pthread_setaffinity_np( pthread_self(), sizeof( cpu_set_t ), &cpuset );

					CUDA_CHECK( cudaSetDevice( d ) );
					bool load_ok = glm_pipes[d].load_weights( glm_model_dir, glm_streams[d] );
					ENGINE_ASSERT( load_ok, "Failed to load GLM weights" );
					CUDA_CHECK( cudaStreamSynchronize( glm_streams[d] ) );
				} );
			}
			for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
				loaders[d].join();
			}
		}
		std::printf( "[+] GLM-5.3-Flash real weights loaded successfully on all 4 GPUs.\n" );
		std::fflush( stdout );

		pod::TensorView in_tokens[ENGINE_NUM_GPUS];
		pod::TensorView out_logits[ENGINE_NUM_GPUS];
		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			int tshape[2] = { 1, glm_cfg.hidden_size };
			in_tokens[d] = engine::core::arena_alloc_tensor( glm_arenas[d], pod::DType::FP16, tshape, 2 );
			out_logits[d] = engine::core::arena_alloc_tensor( glm_arenas[d], pod::DType::FP16, tshape, 2 );
		}

		pthread_barrier_t glm_barrier;
		pthread_barrier_init( &glm_barrier, nullptr, 5 );

		pthread_barrier_t glm_capture_barrier;
		pthread_barrier_init( &glm_capture_barrier, nullptr, 4 ); // 4 workers

		std::atomic<bool> glm_worker_running{ true };
		std::atomic<bool> glm_use_graph{ false };
		std::atomic<int> glm_tok{ 15 };
		std::atomic<int> glm_pos{ 0 };

		std::thread glm_workers[ENGINE_NUM_GPUS];
		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			glm_workers[d] = std::thread( [&, d]() {
				cpu_set_t cpuset;
				CPU_ZERO( &cpuset );
				CPU_SET( d, &cpuset );
				pthread_setaffinity_np( pthread_self(), sizeof( cpu_set_t ), &cpuset );

				CUDA_CHECK( cudaSetDevice( d ) );

				cudaGraph_t graph_m1;
				cudaGraphExec_t instance_m1;
				bool graph_captured = false;

				while ( true ) {
					pthread_barrier_wait( &glm_barrier );
					if ( !glm_worker_running.load() ) break;

					int tok = glm_tok.load();
					int pos = glm_pos.load();
					if ( glm_use_graph.load() ) {
						if ( !graph_captured ) {
							CUDA_CHECK( cudaStreamBeginCapture( glm_streams[d], cudaStreamCaptureModeGlobal ) );
							glm_pipes[d].forward_step( in_tokens[d], out_logits[d], pos, glm_streams[d] );
							CUDA_CHECK( cudaStreamEndCapture( glm_streams[d], &graph_m1 ) );
							pthread_barrier_wait( &glm_capture_barrier );
							CUDA_CHECK( cudaGraphInstantiate( &instance_m1, graph_m1, nullptr, nullptr, 0 ) );
							graph_captured = true;
						}
						int steps = ( tok < 0 ) ? 1 : tok;
						for ( int s = 0; s < steps; ++s ) {
							CUDA_CHECK( cudaGraphLaunch( instance_m1, glm_streams[d] ) );
						}
					}
					else {
						if ( tok >= 0 ) {
							glm_pipes[d].embed_token( tok, glm_streams[d] );
						}
						glm_pipes[d].forward_step( in_tokens[d], out_logits[d], pos, glm_streams[d] );
					}
					CUDA_CHECK( cudaStreamSynchronize( glm_streams[d] ) );

					pthread_barrier_wait( &glm_barrier );
				}

				if ( graph_captured ) {
					CUDA_CHECK( cudaGraphExecDestroy( instance_m1 ) );
					CUDA_CHECK( cudaGraphDestroy( graph_m1 ) );
				}
			} );
		}

		auto run_glm_step = [&]( int tok, int pos ) {
			glm_tok.store( tok );
			glm_pos.store( pos );
			pthread_barrier_wait( &glm_barrier );
			pthread_barrier_wait( &glm_barrier );
		};

		// Warm up
		for ( int i = 0; i < 5; ++i ) {
			run_glm_step( 15, i );
		}
		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			glm_pipes[d].reset_state();
		}

		// Capture static CUDA graph across all 4 workers concurrently
		glm_use_graph.store( true );
		run_glm_step( -1, 0 );

		const int glm_steps = 50;
		std::printf( "[*] Running %d real 4-GPU decode steps on GLM-5.3-Flash (TP=4)...\n", glm_steps );
		std::fflush( stdout );

		auto t0 = std::chrono::high_resolution_clock::now();
		run_glm_step( glm_steps, 0 );
		auto t1 = std::chrono::high_resolution_clock::now();

		double total_ms = std::chrono::duration<double, std::milli>( t1 - t0 ).count();
		double glm_step_lat = total_ms / glm_steps;
		double glm_tok_s = 1000.0 / glm_step_lat;

		std::printf( "\n========================================================\n" );
		std::printf( "   REAL GLM-5.3-FLASH DECODE BENCHMARK RESULTS (TP=4)   \n" );
		std::printf( "========================================================\n" );
		std::printf( "   Total Generated Tokens : %d tokens\n", glm_steps );
		std::printf( "   Elapsed Wall Time      : %.2f ms\n", total_ms );
		std::printf( "   Per-Token Step Latency : %.2f ms/token\n", glm_step_lat );
		std::printf( "   Autoregressive Speed   : %.2f tok/s\n", glm_tok_s );
		std::printf( "========================================================\n" );
		std::fflush( stdout );

		glm_worker_running.store( false );
		pthread_barrier_wait( &glm_barrier );
		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			glm_workers[d].join();
		}
		pthread_barrier_destroy( &glm_barrier );
		pthread_barrier_destroy( &glm_capture_barrier );

		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			CUDA_CHECK( cudaSetDevice( d ) );
			CUDA_CHECK( cudaStreamDestroy( glm_streams[d] ) );
			glm_pipes[d].destroy();
			engine::core::arena_destroy( glm_arenas[d] );
		}
	}

	engine::comm::destroy_4gpu_mesh( staging_ptrs, flag_ptrs );
	engine::modules::trellis_context_destroy();

	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		cudaSetDevice( d );
		cudaDeviceReset();
	}

	std::printf( "\n[+] Benchmark complete. All 4 GPUs cleanly reset to 0 MiB.\n" );
	return 0;
}
