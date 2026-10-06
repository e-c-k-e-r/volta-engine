#include "engine/runtime/generator.h"
#include <cstdio>
#include <cstring>
#include <algorithm>

namespace engine::runtime {

Generator::Generator() = default;

Generator::~Generator() {
	shutdown();
}

bool Generator::init( const engine::server::ServerConfig& config ) {
	config_ = config;
	boot_time_ = std::chrono::steady_clock::now();
	std::printf( "[*] Generator initializing: model=%s dir=%s\n", config_.model_type.c_str(), config_.model_dir.c_str() );

	if ( !config_.load_weights ) {
		std::printf( "[-] Generator requires real weights (load_weights=false is no longer supported)\n" );
		return false;
	}
	if ( tokenizer_.load( config_.model_dir.c_str() ) ) {
		chat_template_.load( config_.model_dir.c_str() );
		std::printf( "[+] Tokenizer loaded. Vocab size: %zu\n", tokenizer_.vocab_size() );
	}
	else {
		std::printf( "[-] Warning: failed to load tokenizer from %s\n", config_.model_dir.c_str() );
	}

	ClusterConfig cc;
	cc.model_type = config_.model_type;
	cc.model_dir = config_.model_dir;
	if ( config_.use_speculation ) cc.draft_model_dir = config_.draft_model_dir;
	if ( !cluster_.init( cc ) ) {
		std::printf( "[-] Generator: model cluster init failed\n" );
		return false;
	}
	num_slots_ = MIN( MIN( config_.max_batch_size, MAX_SLOTS ), cluster_.metadata().max_slots );
	num_slots_ = MAX( num_slots_, 1 );
	// Mode B uses m <= num_slots_; speculative verify uses m <= 7 draft rows
	cluster_.warm_graphs( MAX( num_slots_, cluster_.has_speculator() ? 7 : 1 ) );
	std::printf( "[+] Generator ready: %d slots, speculation %s\n", num_slots_, cluster_.has_speculator() ? "on" : "off" );

	running_.store( true );
	scheduler_thread_ = std::thread( &Generator::scheduler_loop, this );
	cpu_set_t cpuset;
	CPU_ZERO( &cpuset );
	CPU_SET( 4, &cpuset );
	pthread_setaffinity_np( scheduler_thread_.native_handle(), sizeof( cpu_set_t ), &cpuset );
	return true;
}

void Generator::shutdown() {
	const bool was_running = running_.exchange( false );
	queue_cv_.notify_all();
	if ( scheduler_thread_.joinable() ) {
		scheduler_thread_.join();
	}
	cluster_.shutdown();
	if ( was_running ) {
		std::printf( "[+] Generator shutdown complete. 4 GPUs reset.\n" );
	}
}

bool Generator::submit_request( GenerationRequest req ) {
	if ( !running_.load() ) return false;
	if ( req.prompt_tokens.empty() || static_cast<int>( req.prompt_tokens.size() ) >= cluster_.metadata().max_seq_len ) return false;
	{
		std::unique_lock<std::mutex> lock( queue_mtx_ );
		pending_queue_.push_back( std::move( req ) );
	}
	queue_cv_.notify_one();
	std::unique_lock<std::mutex> mlock( metrics_mtx_ );
	total_requests_++;
	return true;
}

void Generator::admit_pending() {
	std::unique_lock<std::mutex> lock( queue_mtx_ );
	for ( int s = 0; s < num_slots_ && !pending_queue_.empty(); ++s ) {
		Slot& slot = slots_[s];
		if ( slot.active ) continue;
		slot = Slot{};
		slot.active = true;
		slot.req = std::move( pending_queue_.front() );
		pending_queue_.erase( pending_queue_.begin() );
		slot.all_tokens = slot.req.prompt_tokens;
		slot.metrics.start_time = std::chrono::steady_clock::now();
	}
}

void Generator::scheduler_loop() {
	while ( running_.load() ) {
		admit_pending();
		bool any_active = false;
		for ( int s = 0; s < num_slots_; ++s ) {
			any_active = any_active || slots_[s].active;
		}
		if ( !any_active ) {
			std::unique_lock<std::mutex> lock( queue_mtx_ );
			queue_cv_.wait_for( lock, std::chrono::milliseconds( 2 ), [&]() {
				return !pending_queue_.empty() || !running_.load();
			} );
			continue;
		}
		step_batch();
	}
}

bool Generator::emit_token( int s, int tok ) {
	Slot& slot = slots_[s];
	slot.all_tokens.push_back( tok );
	slot.tokens_generated++;
	const std::string& tok_str = tokenizer_.decode_token_fast( tok );
	slot.accumulated_text += tok_str;

	const bool eos = cluster_.is_eos( tok ) || tok == tokenizer_.eos_id() || tok == tokenizer_.im_end_id();
	const bool finish = eos
		|| slot.tokens_generated >= slot.req.max_tokens
		|| slot.current_pos + pod::MAX_BATCH >= cluster_.metadata().max_seq_len;
	if ( slot.req.stream_callback ) {
		slot.req.stream_callback( tok_str, finish );
	}
	if ( finish ) {
		finish_slot( s );
	}
	return finish;
}

void Generator::finish_slot( int s ) {
	Slot& slot = slots_[s];
	auto now = std::chrono::steady_clock::now();
	double decode_s = std::chrono::duration<double>( now - slot.metrics.first_token_time ).count();
	slot.metrics.generated_tokens = slot.tokens_generated;
	slot.metrics.decode_tok_s = ( decode_s > 0.0 && slot.tokens_generated > 1 ) ? static_cast<float>( ( slot.tokens_generated - 1 ) / decode_s ) : 0.0f;
	if ( slot.req.completion_callback ) {
		slot.req.completion_callback( slot.accumulated_text, slot.metrics );
	}
	{
		std::unique_lock<std::mutex> mlock( metrics_mtx_ );
		sum_ttft_ms_ += std::chrono::duration<double, std::milli>( slot.metrics.first_token_time - slot.metrics.start_time ).count();
		sum_prefill_tok_s_ += slot.metrics.prefill_tok_s;
		sum_decode_tok_s_ += slot.metrics.decode_tok_s;
		completed_requests_++;
	}
	slot.active = false;
}

void Generator::prefill_slot( int s ) {
	Slot& slot = slots_[s];
	const int n = static_cast<int>( slot.req.prompt_tokens.size() );
	cluster_.reset_slot( s );
	const int first = cluster_.prefill( slot.req.prompt_tokens.data(), n, s );
	slot.current_pos = n;
	slot.prefill_done = true;
	slot.metrics.first_token_time = std::chrono::steady_clock::now();
	const double ttft_ms = std::chrono::duration<double, std::milli>( slot.metrics.first_token_time - slot.metrics.start_time ).count();
	slot.metrics.prompt_tokens = n;
	slot.metrics.prefill_tok_s = ( ttft_ms > 0.0 ) ? static_cast<float>( n / ( ttft_ms / 1000.0 ) ) : 0.0f;
	slot.last_token = first;
	emit_token( s, first );
}

void Generator::step_batch() {
	for ( int s = 0; s < num_slots_; ++s ) {
		if ( slots_[s].active && !slots_[s].prefill_done ) {
			prefill_slot( s );
		}
	}

	int ids[MAX_SLOTS];
	int n = 0;
	for ( int s = 0; s < num_slots_; ++s ) {
		if ( slots_[s].active && slots_[s].prefill_done ) ids[n++] = s;
	}
	if ( n == 0 ) return;

	if ( n == 1 && cluster_.has_speculator() ) {
		// Case A: single request, DFlash2 speculative round (Mode A causal verify block)
		const int s = ids[0];
		Slot& slot = slots_[s];
		const SpecResult r = cluster_.speculate( s, slot.last_token, slot.current_pos, slot.all_tokens );
		slot.current_pos = r.next_pos;
		slot.last_token = r.emitted[r.emitted_count - 1];
		{
			std::unique_lock<std::mutex> mlock( metrics_mtx_ );
			spec_proposals_ += r.proposed;
			spec_accepted_ += r.accepted;
		}
		for ( int i = 0; i < r.emitted_count; ++i ) {
			if ( emit_token( s, r.emitted[i] ) ) break;
		}
		return;
	}

	// Case B (n == 1) / Case C (n > 1): one token per active slot in a single Mode B forward
	int tokens[MAX_SLOTS];
	int positions[MAX_SLOTS];
	for ( int i = 0; i < n; ++i ) {
		tokens[i] = slots_[ids[i]].last_token;
		positions[i] = slots_[ids[i]].current_pos;
	}
	cluster_.forward( tokens, pod::BatchRows::batched( n, ids, positions ) );
	int preds[MAX_SLOTS];
	cluster_.sample( preds, n );
	for ( int i = 0; i < n; ++i ) {
		Slot& slot = slots_[ids[i]];
		slot.current_pos++;
		slot.last_token = preds[i];
		emit_token( ids[i], preds[i] );
	}
}

EngineMetrics Generator::get_metrics() const {
	EngineMetrics m;
	auto now = std::chrono::steady_clock::now();
	m.uptime_seconds = std::chrono::duration<double>( now - boot_time_ ).count();
	for ( int s = 0; s < MAX_SLOTS; ++s ) {
		if ( slots_[s].active ) m.active_slots++;
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
		m.gpu_vram_used_bytes[d] = cluster_.arena( d ).total_bytes;
	}
	return m;
}

} // namespace engine::runtime
