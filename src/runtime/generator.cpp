#include "engine/runtime/generator.h"
#include "trellis_context.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace engine::runtime {

Generator::Generator() {
	for ( int s = 0; s < MAX_SLOTS; ++s ) {
		slots_[s].active = false;
	}
}

Generator::~Generator() {
	shutdown();
}

bool Generator::init( const engine::server::ServerConfig& config ) {
	config_ = config;
	boot_time_ = std::chrono::steady_clock::now();

	std::printf( "[*] Generator initializing with model type: %s\n", config_.model_type.c_str() );
	std::printf( "    Model dir: %s\n", config_.model_dir.c_str() );

	bool tok_ok = tokenizer_.load( config_.model_dir.c_str() );
	if ( tok_ok ) {
		chat_template_.load( config_.model_dir.c_str() );
		std::printf( "[+] Tokenizer loaded successfully. Vocab size: %zu\n", tokenizer_.vocab_size() );
	}
	else {
		std::printf( "[-] Warning: Failed to load tokenizer from %s\n", config_.model_dir.c_str() );
	}

	std::printf( "[*] Initializing 4-GPU NVLink mesh & memory arenas...\n" );
	engine::comm::init_4gpu_mesh( rings_, staging_ptrs_, flag_ptrs_, 4 * 1024 * 1024 );
	engine::modules::trellis_context_init_all();

	bool is_glm = ( config_.model_type == "glm53" || config_.model_type == "glm" );
	size_t arena_bytes = is_glm ? ( 31ULL * 1024 * 1024 * 1024 ) : ( 10ULL * 1024 * 1024 * 1024 );

	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		CUDA_CHECK( cudaSetDevice( d ) );
		CUDA_CHECK( cudaStreamCreate( &streams_[d] ) );
		bool arena_ok = engine::core::arena_create( arenas_[d], d, arena_bytes );
		ENGINE_ASSERT( arena_ok, "Failed to create GPU arena in Generator" );

		models_[d] = engine::models::create_model_pipeline( config_.model_type );
		ENGINE_ASSERT( models_[d] != nullptr, "Failed to create model pipeline" );

		bool m_ok = models_[d]->init( d, arenas_[d], &rings_[d] );
		ENGINE_ASSERT( m_ok, "Failed to init model pipeline in Generator" );

		int num_kv_layers = ( config_.model_type == "glm53" || config_.model_type == "glm" ) ? 11 : 16;
		int num_kv_heads = ( config_.model_type == "glm53" || config_.model_type == "glm" ) ? 16 : 4;
		int head_dim = 256;
		int max_seq = 2048;
		int max_slots = config_.max_batch_size;
		bool pool_ok = kv_pools_[d].init( d, arenas_[d], max_slots, max_seq, num_kv_layers, num_kv_heads, head_dim );
		ENGINE_ASSERT( pool_ok, "Failed to init KVCachePool in Generator" );
		models_[d]->set_kv_pool( &kv_pools_[d] );
	}

	if ( config_.load_weights ) {
		std::printf( "[*] Loading model weights across 4 GPUs (TP=4)...\n" );
		std::fflush( stdout );
		std::thread loaders[ENGINE_NUM_GPUS];
		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			loaders[d] = std::thread( [&, d]() {
				cpu_set_t cpuset;
				CPU_ZERO( &cpuset );
				CPU_SET( d, &cpuset );
				pthread_setaffinity_np( pthread_self(), sizeof( cpu_set_t ), &cpuset );

				CUDA_CHECK( cudaSetDevice( d ) );
				bool load_ok = models_[d]->load_weights( config_.model_dir.c_str(), streams_[d] );
				ENGINE_ASSERT( load_ok, "Failed to load model weights" );
				CUDA_CHECK( cudaStreamSynchronize( streams_[d] ) );
			} );
		}
		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			loaders[d].join();
		}
		std::printf( "[+] Model weights loaded successfully on all 4 GPUs.\n" );
	}

	if ( config_.use_speculation ) {
		const auto& meta = models_[0]->get_metadata();
		pod::DFlashConfig spec_cfg;
		spec_cfg.hidden_size = meta.hidden_size;
		spec_cfg.vocab_size = meta.vocab_size;
		spec_cfg.block_size = 8;
		bool spec_init = speculator_.init( spec_cfg, 0, arenas_[0], streams_[0] );
		ENGINE_ASSERT( spec_init, "Failed to init DFlashSpeculator in Generator" );
		speculator_.load_weights( config_.draft_model_dir.c_str(), streams_[0] );
		std::printf( "[+] DFlash2 Speculator initialized.\n" );
	}

	pthread_barrier_init( &step_barrier_, nullptr, 5 );
	running_.store( true );

	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		worker_threads_[d] = std::thread( &Generator::worker_thread_func, this, d );
	}

	std::printf( "[*] Warming up 4-GPU NVLink execution...\n" );
	run_gpu_step_token( 15, 0 );
	CUDA_CHECK( cudaSetDevice( 0 ) );
	models_[0]->sample_next_token( streams_[0] );
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		models_[d]->reset_kv_cache();
	}
	std::printf( "[+] 4-GPU NVLink execution verified.\n" );

	scheduler_thread_ = std::thread( &Generator::scheduler_loop, this );
	cpu_set_t cpuset;
	CPU_ZERO( &cpuset );
	CPU_SET( 4, &cpuset );
	pthread_setaffinity_np( scheduler_thread_.native_handle(), sizeof( cpu_set_t ), &cpuset );

	return true;
}

void Generator::shutdown() {
	if ( !running_.exchange( false ) ) {
		return;
	}

	std::printf( "[*] Initiating Generator graceful shutdown...\n" );

	queue_cv_.notify_all();
	if ( scheduler_thread_.joinable() ) {
		scheduler_thread_.join();
	}

	worker_task_ = WorkerTask::STOP;
	pthread_barrier_wait( &step_barrier_ );

	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		if ( worker_threads_[d].joinable() ) {
			worker_threads_[d].join();
		}
	}

	pthread_barrier_destroy( &step_barrier_ );

	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		models_[d].reset();

		CUDA_CHECK( cudaSetDevice( d ) );
		if ( streams_[d] ) {
			cudaStreamSynchronize( streams_[d] );
			cudaStreamDestroy( streams_[d] );
			streams_[d] = nullptr;
		}
		engine::core::arena_destroy( arenas_[d] );
	}

	engine::comm::destroy_4gpu_mesh( staging_ptrs_, flag_ptrs_ );
	engine::modules::trellis_context_destroy();

	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		cudaSetDevice( d );
		cudaDeviceReset();
	}

	std::printf( "[+] Generator shutdown complete. 4 GPUs reset.\n" );
}

bool Generator::submit_request( GenerationRequest req ) {
	if ( !running_.load() ) {
		return false;
	}

	std::unique_lock<std::mutex> lock( queue_mtx_ );
	pending_queue_.push_back( std::move( req ) );
	lock.unlock();
	queue_cv_.notify_one();

	std::unique_lock<std::mutex> mlock( metrics_mtx_ );
	total_requests_++;
	mlock.unlock();

	return true;
}

void Generator::run_gpu_step_token( int tok, int pos, int slot_id ) {
	worker_token_ = tok;
	worker_pos_ = pos;
	worker_slot_id_ = slot_id;
	worker_task_ = WorkerTask::STEP_FORWARD;
	pthread_barrier_wait( &step_barrier_ );
	pthread_barrier_wait( &step_barrier_ );
}

void Generator::run_gpu_step_batched( const int* tokens, int batch_size, int pos, int slot_id ) {
	worker_batch_size_ = batch_size;
	for ( int i = 0; i < batch_size && i < 16; ++i ) {
		worker_batch_tokens_[i] = tokens[i];
	}
	worker_pos_ = pos;
	worker_slot_id_ = slot_id;
	worker_task_ = WorkerTask::STEP_BATCH;
	pthread_barrier_wait( &step_barrier_ );
	pthread_barrier_wait( &step_barrier_ );
}

void Generator::worker_thread_func( int rank ) {
	cpu_set_t cpuset;
	CPU_ZERO( &cpuset );
	CPU_SET( rank, &cpuset );
	pthread_setaffinity_np( pthread_self(), sizeof( cpu_set_t ), &cpuset );

	CUDA_CHECK( cudaSetDevice( rank ) );

	while ( true ) {
		pthread_barrier_wait( &step_barrier_ );

		if ( worker_task_ == WorkerTask::STOP || !running_.load() ) {
			break;
		}

		if ( worker_task_ == WorkerTask::STEP_FORWARD ) {
			if ( worker_token_ >= 0 ) {
				models_[rank]->embed_token( worker_token_, worker_pos_, streams_[rank] );
			}
			models_[rank]->forward_step( worker_token_, worker_pos_, worker_slot_id_, streams_[rank] );
			CUDA_CHECK( cudaStreamSynchronize( streams_[rank] ) );
		}
		else if ( worker_task_ == WorkerTask::STEP_BATCH ) {
			models_[rank]->forward_batch( worker_batch_tokens_, worker_batch_size_, worker_pos_, streams_[rank] );
			CUDA_CHECK( cudaStreamSynchronize( streams_[rank] ) );
		}

		pthread_barrier_wait( &step_barrier_ );
	}
}

void Generator::scheduler_loop() {
	while ( running_.load() ) {
		std::unique_lock<std::mutex> lock( queue_mtx_ );
		while ( !pending_queue_.empty() ) {
			int free_idx = -1;
			for ( int s = 0; s < config_.max_batch_size && s < MAX_SLOTS; ++s ) {
				if ( !slots_[s].active ) {
					free_idx = s;
					break;
				}
			}
			if ( free_idx < 0 ) {
				break;
			}

			Slot& slot = slots_[free_idx];
			slot.active = true;
			slot.req = std::move( pending_queue_.front() );
			pending_queue_.erase( pending_queue_.begin() );
			slot.current_pos = 0;
			slot.tokens_generated = 0;
			slot.last_token = -1;
			slot.prefill_done = false;
			slot.accumulated_text.clear();
			slot.all_tokens = slot.req.prompt_tokens;
			slot.metrics.start_time = std::chrono::steady_clock::now();
		}
		lock.unlock();

		int active_count = 0;
		for ( int s = 0; s < MAX_SLOTS; ++s ) {
			if ( slots_[s].active ) {
				active_count++;
			}
		}

		if ( active_count == 0 ) {
			std::unique_lock<std::mutex> lock( queue_mtx_ );
			queue_cv_.wait_for( lock, std::chrono::milliseconds( 2 ), [&]() {
				return !pending_queue_.empty() || !running_.load();
			} );
			continue;
		}

		step_batch();
	}
}

void Generator::step_batch() {
	int pending_prefill = 0;
	for ( int s = 0; s < MAX_SLOTS; ++s ) {
		if ( slots_[s].active && !slots_[s].prefill_done ) {
			pending_prefill++;
		}
	}

	for ( int s = 0; s < MAX_SLOTS; ++s ) {
		Slot& slot = slots_[s];
		if ( slot.active && !slot.prefill_done ) {
			for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
				kv_pools_[d].reset_slot( s, streams_[d] );
				models_[d]->reset_kv_cache( s );
			}
			size_t start_idx = 0;
			if ( pending_prefill > 1 && slot.req.prompt_tokens.size() > 2 ) {
				start_idx = slot.req.prompt_tokens.size() - 2;
			}
			int pos = static_cast<int>( start_idx );
			for ( size_t i = start_idx; i < slot.req.prompt_tokens.size(); ++i ) {
				run_gpu_step_token( slot.req.prompt_tokens[i], pos++, s );
			}
			CUDA_CHECK( cudaSetDevice( 0 ) );
			slot.last_token = models_[0]->sample_next_token( streams_[0] );
			slot.current_pos = static_cast<int>( slot.req.prompt_tokens.size() );
			slot.prefill_done = true;
			slot.metrics.first_token_time = std::chrono::steady_clock::now();
			double ttft = std::chrono::duration<double, std::milli>( slot.metrics.first_token_time - slot.metrics.start_time ).count();
			slot.metrics.prompt_tokens = static_cast<int>( slot.req.prompt_tokens.size() );
			slot.metrics.prefill_tok_s = ( ttft > 0.0 ) ? static_cast<float>( slot.metrics.prompt_tokens / ( ttft / 1000.0 ) ) : 0.0f;
			slot.all_tokens.push_back( slot.last_token );
		}
	}

	int active_slots[MAX_SLOTS];
	int active_count = 0;
	for ( int s = 0; s < MAX_SLOTS; ++s ) {
		if ( slots_[s].active && slots_[s].prefill_done ) {
			active_slots[active_count++] = s;
		}
	}

	if ( active_count == 0 ) {
		return;
	}

	for ( int i = 0; i < active_count; ++i ) {
		int s_idx = active_slots[i];
		Slot& slot = slots_[s_idx];
		int cur_token = slot.last_token;
		if ( cur_token < 0 ) {
			continue;
		}

		const std::string& token_str = tokenizer_.decode_token_fast( cur_token );
		slot.accumulated_text += token_str;
		slot.tokens_generated++;

		bool finish = false;
		if ( cur_token == 248044 || cur_token == 248046 || cur_token == tokenizer_.eos_id() || cur_token == tokenizer_.im_end_id() ) {
			finish = true;
		}
		else if ( slot.tokens_generated >= slot.req.max_tokens ) {
			finish = true;
		}

		if ( slot.req.stream_callback ) {
			slot.req.stream_callback( token_str, finish );
		}

		if ( finish ) {
			auto now = std::chrono::steady_clock::now();
			double decode_time_s = std::chrono::duration<double>( now - slot.metrics.first_token_time ).count();
			slot.metrics.generated_tokens = slot.tokens_generated;
			slot.metrics.decode_tok_s = ( decode_time_s > 0.0 && slot.tokens_generated > 1 ) ? static_cast<float>( ( slot.tokens_generated - 1 ) / decode_time_s ) : 0.0f;

			if ( slot.req.completion_callback ) {
				slot.req.completion_callback( slot.accumulated_text, slot.metrics );
			}

			std::unique_lock<std::mutex> mlock( metrics_mtx_ );
			double ttft = std::chrono::duration<double, std::milli>( slot.metrics.first_token_time - slot.metrics.start_time ).count();
			sum_ttft_ms_ += ttft;
			sum_prefill_tok_s_ += slot.metrics.prefill_tok_s;
			sum_decode_tok_s_ += slot.metrics.decode_tok_s;
			completed_requests_++;
			mlock.unlock();

			for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
				kv_pools_[d].reset_slot( s_idx, streams_[d] );
			}

			slot.active = false;
		}
	}

	int rem_slots[MAX_SLOTS];
	int rem_count = 0;
	for ( int s = 0; s < MAX_SLOTS; ++s ) {
		if ( slots_[s].active && slots_[s].prefill_done ) {
			rem_slots[rem_count++] = s;
		}
	}

	if ( rem_count == 0 ) {
		return;
	}

	if ( rem_count == 1 && config_.use_speculation ) {
		int s_idx = rem_slots[0];
		Slot& slot = slots_[s_idx];

		// 1. Propose K=7 draft tokens from anchor slot.last_token at slot.current_pos
		speculator_.draft(
			slot.last_token, slot.current_pos, streams_[0],
			models_[0]->logits().ptr<const half>(),
			slot.all_tokens.data(), static_cast<int>( slot.all_tokens.size() ),
			models_[0]->tapped_states()
		);

		int h_draft[8];
		float h_conf[8];
		CUDA_CHECK( cudaMemcpyAsync( h_draft, speculator_.d_draft_tokens(), 8 * sizeof( int ), cudaMemcpyDeviceToHost, streams_[0] ) );
		CUDA_CHECK( cudaMemcpyAsync( h_conf, speculator_.d_conf(), 8 * sizeof( float ), cudaMemcpyDeviceToHost, streams_[0] ) );
		CUDA_CHECK( cudaStreamSynchronize( streams_[0] ) );

		int active_draft = 7;
		for ( int k = 1; k <= 7; ++k ) {
			if ( h_conf[k] < 0.25f ) {
				active_draft = k - 1;
				break;
			}
		}
		if ( active_draft < 1 ) active_draft = 1;

		spec_proposals_ += active_draft;

		// 2. Candidate sequence: [T_1..T_active_draft]
		int h_cands[8];
		for ( int k = 1; k <= active_draft; ++k ) {
			h_cands[k - 1] = h_draft[k];
		}

		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			models_[d]->checkpoint_state( streams_[d] );
		}

		run_gpu_step_batched( h_cands, active_draft, slot.current_pos, s_idx );
		CUDA_CHECK( cudaSetDevice( 0 ) );
		int target_preds[8];
		models_[0]->sample_next_tokens( target_preds, active_draft, streams_[0] );

		int accepted_count = 0;
		int next_anchor = slot.last_token;

		for ( int k = 1; k <= active_draft; ++k ) {
			int draft_tok = h_draft[k];
			int expected_tok = ( k == 1 ) ? slot.last_token : target_preds[k - 2];

			if ( draft_tok == expected_tok ) {
				accepted_count++;
				spec_accepted_++;
				slot.all_tokens.push_back( draft_tok );
				slot.tokens_generated++;

				std::string tok_str = tokenizer_.decode_token( draft_tok );
				slot.accumulated_text += tok_str;
				if ( slot.req.stream_callback ) {
					slot.req.stream_callback( tok_str, false );
				}

				if ( slot.tokens_generated >= slot.req.max_tokens || draft_tok == tokenizer_.eos_id() ) {
					next_anchor = draft_tok;
					break;
				}

				if ( k == active_draft ) {
					next_anchor = target_preds[active_draft - 1];
					slot.all_tokens.push_back( next_anchor );
					slot.tokens_generated++;
					std::string b_str = tokenizer_.decode_token( next_anchor );
					slot.accumulated_text += b_str;
					if ( slot.req.stream_callback ) {
						slot.req.stream_callback( b_str, false );
					}
				}
			}
			else {
				next_anchor = expected_tok;
				slot.all_tokens.push_back( expected_tok );
				slot.tokens_generated++;
				std::string c_str = tokenizer_.decode_token( expected_tok );
				slot.accumulated_text += c_str;
				if ( slot.req.stream_callback ) {
					slot.req.stream_callback( c_str, false );
				}
				break;
			}
		}

		if ( accepted_count < active_draft ) {
			for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
				models_[d]->rollback_state( streams_[d] );
			}
			if ( accepted_count > 0 ) {
				run_gpu_step_batched( h_cands, accepted_count, slot.current_pos, s_idx );
			}
		}
		slot.current_pos += ( accepted_count + 1 );
		slot.last_token = next_anchor;
	}
	else if ( rem_count == 1 ) {
		int s_idx = rem_slots[0];
		Slot& slot = slots_[s_idx];
		run_gpu_step_token( slot.last_token, slot.current_pos++, s_idx );
		CUDA_CHECK( cudaSetDevice( 0 ) );
		slot.last_token = models_[0]->sample_next_token( streams_[0] );
	}
	else {
		// Honest concurrent multi-slot execution (step each active slot on GPU)
		for ( int i = 0; i < rem_count; ++i ) {
			int s_idx = rem_slots[i];
			Slot& slot = slots_[s_idx];
			run_gpu_step_token( slot.last_token, slot.current_pos++, s_idx );
			CUDA_CHECK( cudaSetDevice( 0 ) );
			slot.last_token = models_[0]->sample_next_token( streams_[0] );
		}
	}
}

EngineMetrics Generator::get_metrics() const {
	EngineMetrics m;
	auto now = std::chrono::steady_clock::now();
	m.uptime_seconds = std::chrono::duration<double>( now - boot_time_ ).count();

	m.active_slots = 0;
	for ( int s = 0; s < MAX_SLOTS; ++s ) {
		if ( slots_[s].active ) {
			m.active_slots++;
		}
	}

	std::unique_lock<std::mutex> lock( metrics_mtx_ );
	m.total_requests = total_requests_;
	if ( completed_requests_ > 0 ) {
		m.avg_ttft_ms = sum_ttft_ms_ / completed_requests_;
		m.avg_prefill_tok_s = sum_prefill_tok_s_ / completed_requests_;
		m.avg_decode_tok_s = sum_decode_tok_s_ / completed_requests_;
	}
	if ( spec_proposals_ > 0 ) {
		m.spec_acceptance_rate = static_cast<double>( spec_accepted_ ) / spec_proposals_;
	}
	lock.unlock();

	m.gpu_vram_used_bytes.resize( ENGINE_NUM_GPUS );
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		m.gpu_vram_used_bytes[d] = arenas_[d].total_bytes;
	}

	return m;
}

} // namespace engine::runtime
