#include "engine/core/utf8.h"
#include "engine/runtime/generator.h"
#include "engine/core/log.h"
#include "engine/runtime/options.h"
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
	hist_last_ = heat_last_ = boot_time_;
	LOG_INFO( "Generator initializing: model=%s dir=%s\n", config_.model_type.c_str(), config_.model_dir.c_str() );

	if ( !config_.load_weights ) {
		LOG_ERROR( "Generator requires real weights (load_weights=false is no longer supported)\n" );
		return false;
	}
	if ( tokenizer_.load( config_.model_dir.c_str() ) ) {
		chat_template_.load( config_.model_dir.c_str() );
		LOG_INFO( "Tokenizer loaded. Vocab size: %zu\n", tokenizer_.vocab_size() );
	}
	else {
		LOG_ERROR( "Warning: failed to load tokenizer from %s\n", config_.model_dir.c_str() );
	}

	ClusterConfig cc;
	cc.model_type = config_.model_type;
	cc.model_dir = config_.model_dir;
	cc.tp = config_.tp_size;
	cc.speculate = config_.use_speculation;
	if ( config_.use_speculation ) cc.draft_model_dir = config_.draft_model_dir;
	if ( !cluster_.init( cc ) ) {
		LOG_ERROR( "Generator: model cluster init failed\n" );
		return false;
	}
	num_slots_ = MIN( MIN( config_.max_batch_size, MAX_SLOTS ), cluster_.metadata().max_slots );
	num_slots_ = MAX( num_slots_, 1 );
	// Mode B uses m <= num_slots_; speculative verify uses m <= 7 draft rows
	cluster_.warm_graphs( MAX( num_slots_, cluster_.has_speculator() ? ( cluster_.speculator().joint() ? pod::MAX_BATCH : 7 ) : 1 ) );   // joint rounds verify the blocks of several slots at once
	{
		// the warm-up forwards route dummy rows through every expert: drop what they counted
		std::vector<std::vector<uint32_t>> discard;
		int layers = 0, experts = 0;
		if ( runtime::options().expert_stats ) cluster_.expert_heat( discard, layers, experts );
	}
	LOG_INFO( "Generator ready: %d slots, speculation %s\n", num_slots_, cluster_.has_speculator() ? "on" : "off" );

	const size_t cache_budget = static_cast<size_t>( std::max( 0, runtime::options().prefix_cache_gib ) ) << 30;
	if ( cache_budget > 0 && cluster_.snapshot_bytes( 0, 1 ) > 0 ) {
		prefix_cache_ = std::make_unique<PrefixCache>( cluster_, cache_budget );
		LOG_INFO( "Prefix cache: %.0f GiB pinned host budget\n", cache_budget / 1073741824.0 );
	}
	else {
		LOG_INFO( "Prefix cache off\n" );
	}

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
	prefix_cache_.reset();
	cluster_.shutdown();
	if ( was_running ) {
		LOG_INFO( "Generator shutdown complete.\n" );
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
		slot = Slot();
		slot.active = true;
		slot.req = std::move( pending_queue_.front() );
		pending_queue_.erase( pending_queue_.begin() );
		slot.all_tokens = slot.req.prompt_tokens;
		if ( !slot.req.images.empty() ) {
			slot.key_tokens = slot.req.prompt_tokens;
			for ( const GenerationRequest::Image& im : slot.req.images ) {
				for ( int i = 0; i < im.count; ++i ) slot.key_tokens[im.start + i] = mm_key_token( im.hash, i, cluster_.metadata().vocab_size );
			}
		}
		slot.sp.temperature = slot.req.temperature;
		slot.sp.top_p = slot.req.top_p;
		slot.sp.top_k = MAX( 1, MIN( slot.req.top_k, kSampleTopK ) );
		slot.sp.seed = slot.req.seed != 0 ? slot.req.seed : ( 0x51ED270B0F5A8C3DULL ^ slot.req.req_id );
		slot.rng.reseed( slot.sp.seed );
		slot.in_reasoning = slot.req.starts_in_reasoning;
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
			const bool settle = published_active_ || heat_pending_;
			if ( published_active_ ) publish_slots();
			if ( settle ) fold_expert_heat( true );   // the last second of a short request would otherwise wait for the next one
			std::unique_lock<std::mutex> lock( queue_mtx_ );
			queue_cv_.wait_for( lock, std::chrono::milliseconds( 2 ), [&]() {
				return !pending_queue_.empty() || !running_.load();
			} );
			continue;
		}
		step_batch();
		heat_pending_ = true;
		for ( int s = 0; s < num_slots_; ++s ) apply_forced_close( s );
		publish_slots();
		fold_expert_heat();
	}
}

bool Generator::emit_token( int s, int tok ) {
	Slot& slot = slots_[s];
	slot.all_tokens.push_back( tok );
	slot.tokens_generated++;
	tokens_out_.fetch_add( 1, std::memory_order_relaxed );
	{
		// the gap a client sees between bursts of tokens (the tokens of one speculation round arrive together)
		const auto now = std::chrono::steady_clock::now();
		const double gap_ms = std::chrono::duration<double, std::milli>( now - slot.last_emit ).count();
		if ( slot.tokens_generated > 1 && gap_ms > 1.0 ) {
			std::unique_lock<std::mutex> mlock( metrics_mtx_ );
			gap_ring_[gap_n_++ % gap_ring_.size()] = static_cast<float>( gap_ms );
		}
		slot.last_emit = now;
	}

	const bool eos = cluster_.is_eos( tok ) || tok == tokenizer_.eos_id() || tok == tokenizer_.im_end_id();
	FinishReason reason = FinishReason::Stop;
	bool finish = eos;
	if ( eos ) LOG_DEBUG( "[gen] slot %d stops on token %d after %d tokens%s\n", s, tok, slot.tokens_generated, slot.in_reasoning ? " (inside the reasoning)" : "" );
	if ( !finish && ( slot.tokens_generated >= slot.req.max_tokens || slot.current_pos + pod::MAX_BATCH >= cluster_.metadata().max_seq_len ) ) {
		finish = true;
		reason = FinishReason::Length;
	}

	if ( slot.in_reasoning && !eos ) {
		if ( tok == slot.req.think_close_id ) slot.in_reasoning = false;
		else if ( slot.req.reasoning_budget > 0 && ++slot.reasoning_tokens >= slot.req.reasoning_budget ) slot.force_close = true;
	}
	std::string out;
	if ( !eos ) slot.held_text += tokenizer_.decode_token_fast( tok );
	if ( !slot.req.stop_strings.empty() ) {
		// earliest stop string wins; otherwise hold back a tail that could still grow into one
		size_t hit = std::string::npos;
		for ( const std::string& stop : slot.req.stop_strings ) {
			const size_t p = stop.empty() ? std::string::npos : slot.held_text.find( stop );
			if ( p != std::string::npos && p < hit ) {
				hit = p;
				slot.req.matched_stop = stop;
			}
		}
		if ( hit != std::string::npos ) {
			out = slot.held_text.substr( 0, hit );
			slot.held_text.clear();
			finish = true;
			reason = FinishReason::StopString;
		}
		else {
			size_t hold = 0;
			if ( !finish ) {
				for ( const std::string& stop : slot.req.stop_strings ) {
					const size_t max = std::min( slot.held_text.size(), stop.size() - 1 );
					for ( size_t n = max; n > hold; --n ) {
						if ( slot.held_text.compare( slot.held_text.size() - n, n, stop, 0, n ) == 0 ) {
							hold = n;
							break;
						}
					}
				}
			}
			out = slot.held_text.substr( 0, slot.held_text.size() - hold );
			slot.held_text.erase( 0, out.size() );
		}
	}
	else {
		out = std::move( slot.held_text );
		slot.held_text.clear();
	}
	if ( !finish ) {
		// A token can end inside a multi-byte character: send up to the boundary and keep the rest for the next token
		const size_t cut = engine::core::utf8_incomplete_tail( out );
		if ( cut > 0 ) {
			slot.held_text.insert( 0, out, out.size() - cut, cut );
			out.erase( out.size() - cut );
		}
	}
	if ( finish ) {
		out += slot.held_text;
		slot.held_text.clear();
	}
	if ( !out.empty() && slot.req.on_text && !slot.req.on_text( out ) ) {
		finish = true;
		reason = FinishReason::Cancelled;
	}
	if ( finish ) {
		finish_slot( s, reason );
	}
	return finish;
}

void Generator::finish_slot( int s, FinishReason reason ) {
	Slot& slot = slots_[s];
	auto now = std::chrono::steady_clock::now();
	double decode_s = std::chrono::duration<double>( now - slot.metrics.first_token_time ).count();
	slot.metrics.generated_tokens = slot.tokens_generated;
	slot.metrics.decode_tok_s = ( decode_s > 0.0 && slot.tokens_generated > 1 ) ? static_cast<float>( ( slot.tokens_generated - 1 ) / decode_s ) : 0.0f;
	if ( slot.req.on_finish ) {
		slot.req.on_finish( reason, slot.req.matched_stop, slot.metrics );
	}
	{
		std::unique_lock<std::mutex> mlock( metrics_mtx_ );
		sum_ttft_ms_ += std::chrono::duration<double, std::milli>( slot.metrics.first_token_time - slot.metrics.start_time ).count();
		sum_prefill_tok_s_ += slot.metrics.prefill_tok_s;
		sum_decode_tok_s_ += slot.metrics.decode_tok_s;
		completed_requests_++;
		const double ttft = std::chrono::duration<double, std::milli>( slot.metrics.first_token_time - slot.metrics.start_time ).count();
		finished_[static_cast<int>( reason )]++;
		prompt_tokens_total_ += static_cast<uint64_t>( slot.metrics.prompt_tokens );
		cached_tokens_total_ += static_cast<uint64_t>( slot.metrics.cached_tokens );
		ttft_ring_[ttft_n_++ % ttft_ring_.size()] = static_cast<float>( ttft );
		RequestRecord rec;
		rec.req_id = slot.req.req_id;
		rec.end_s = std::chrono::duration<double>( now - boot_time_ ).count();
		rec.prompt_tokens = slot.metrics.prompt_tokens;
		rec.cached_tokens = slot.metrics.cached_tokens;
		rec.generated = slot.tokens_generated;
		rec.ttft_ms = ttft;
		rec.total_ms = std::chrono::duration<double, std::milli>( now - slot.metrics.start_time ).count();
		rec.decode_tok_s = slot.metrics.decode_tok_s;
		rec.reason = reason;
		recent_.push_back( rec );
		while ( recent_.size() > 40 ) recent_.pop_front();
	}
	slot.active = false;
	if ( !slot.req.images.empty() ) cluster_.set_slot_mm( s, MmLayout() );
	slot.key_tokens.clear();
	slot.req.images.clear();
}

std::shared_ptr<const std::vector<backend::half_t>> Generator::image_embeddings( const GenerationRequest::Image& im ) {
	auto it = image_cache_.find( im.hash );
	if ( it != image_cache_.end() ) {
		image_cache_lru_.remove( im.hash );
		image_cache_lru_.push_front( im.hash );
		return it->second;
	}
	if ( !im.prepared ) return nullptr;
	auto out = std::make_shared<std::vector<backend::half_t>>();
	const auto t0 = std::chrono::steady_clock::now();
	if ( !cluster_.encode_image( *im.prepared, out.get() ) ) return nullptr;
	LOG_INFO( "[vision] encoded %dx%d patches -> %d tokens in %.0f ms\n", im.prepared->grid_h, im.prepared->grid_w, im.count,
		std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count() );
	constexpr size_t kCacheBytes = 1024ULL << 20;
	image_cache_bytes_ += out->size() * sizeof( backend::half_t );
	image_cache_[im.hash] = out;
	image_cache_lru_.push_front( im.hash );
	while ( image_cache_bytes_ > kCacheBytes && image_cache_lru_.size() > 1 ) {
		const uint64_t victim = image_cache_lru_.back();
		image_cache_lru_.pop_back();
		image_cache_bytes_ -= image_cache_[victim]->size() * sizeof( backend::half_t );
		image_cache_.erase( victim );
	}
	return out;
}

bool Generator::prefill_slot( int s, int budget ) {
	Slot& slot = slots_[s];
	heat_pending_ = true;
	const std::vector<int>& prompt = slot.req.prompt_tokens;
	const int n = static_cast<int>( prompt.size() );
	if ( !slot.pf_started ) {
		slot.pf_started = true;
		// Prefix-cache keys: image placeholders are keyed by the image content, so the same prompt with another picture never matches
		const std::vector<int>& key_prompt = slot.key_tokens.empty() ? prompt : slot.key_tokens;
		// Resume from the deepest cached prefix of the prompt, else start cold (which also hands the drafter to this slot)
		int pos = 0;
		slot.pf_node = nullptr;
		if ( prefix_cache_ && n >= kPrefixMinTokens ) {
			if ( PrefixCache::Entry* hit = prefix_cache_->find( key_prompt ) ) {
				if ( prefix_cache_->restore( s, hit ) ) {
					slot.pf_node = hit;
					pos = static_cast<int>( hit->tokens.size() );
					LOG_INFO( "[prefix] slot %d resumes at token %d of %d\n", s, pos, n );
				}
			}
		}
		slot.pf_resumed = pos;
		if ( pos == 0 ) cluster_.reset_slot( s );
		if ( !slot.req.images.empty() ) {
			MmLayout layout;
			for ( const GenerationRequest::Image& im : slot.req.images ) {
				MmImage mi;
				mi.start = im.start;
				mi.count = im.count;
				mi.grid_h = im.grid_h;
				mi.grid_w = im.grid_w;
				mi.hash = im.hash;
				// Rows before the resume point are already in the restored KV; only images that reach past it need embeddings
				if ( im.start + im.count > pos ) {
					mi.embeds = image_embeddings( im );
					if ( !mi.embeds ) {
						LOG_ERROR( "[vision] slot %d: image encoding failed; finishing the request\n", s );
						slot.metrics.prompt_tokens = n;
						slot.metrics.first_token_time = std::chrono::steady_clock::now();
						finish_slot( s, FinishReason::Cancelled );
						return true;
					}
				}
				layout.images.push_back( std::move( mi ) );
			}
			cluster_.set_slot_mm( s, std::move( layout ) );
		}
		// Checkpoints on the way through: the requested ones (end of the system prompt, turn boundary) plus one every
		// prefix_interval tokens of a long prompt. Each stores only the KV rows since the previous one.
		slot.pf_cps.clear();
		if ( prefix_cache_ && n >= kPrefixMinTokens ) {
			slot.pf_cps = slot.req.checkpoint_at;
			const int align = cluster_.snapshot_align();
			for ( auto& cp : slot.pf_cps ) cp.pos -= cp.pos % align;
			const int interval = runtime::options().prefix_interval;
			for ( int p = interval; interval > 0 && p < n - kPrefixMinTokens; p += interval ) slot.pf_cps.push_back( { p, false, false } );
			std::sort( slot.pf_cps.begin(), slot.pf_cps.end(), []( const GenerationRequest::Checkpoint& a, const GenerationRequest::Checkpoint& b ) { return a.pos < b.pos || ( a.pos == b.pos && a.pinned > b.pinned ); } );
		}
		if ( prefix_cache_ && !slot.pf_cps.empty() ) {
			std::vector<ModelCluster::SnapshotPlan> plan;
			int from = pos;
			for ( const auto& c : slot.pf_cps ) {
				if ( c.pos <= from || c.pos >= n ) continue;
				plan.push_back( { from, c.pos, c.taps } );
				from = c.pos;
			}
			cluster_.prewarm_snapshots( plan );
		}
		slot.pf_cp_next = 0;
		slot.pf_last_cp = -1;
		slot.pf_chain_ok = true;
		slot.pf_pos = pos;
	}

	int pos = slot.pf_pos;
	int fed = 0;
	for ( ;; ) {
		// The next stop: a checkpoint to store, else the end of the prompt
		int end = n;
		bool is_cp = false;
		const GenerationRequest::Checkpoint* cp = nullptr;
		while ( slot.pf_cp_next < slot.pf_cps.size() ) {
			const GenerationRequest::Checkpoint& c = slot.pf_cps[slot.pf_cp_next];
			if ( c.pos <= pos || c.pos >= n || c.pos == slot.pf_last_cp || !slot.pf_chain_ok ) {
				++slot.pf_cp_next;
				continue;
			}
			end = c.pos;
			is_cp = true;
			cp = &c;
			break;
		}
		int seg_end = end;
		if ( budget > 0 ) seg_end = MIN( end, pos + MAX( 1, budget - fed ) );
		if ( seg_end >= n ) {
			Candidates first_cands;
			int first = cluster_.prefill( prompt.data() + pos, n - pos, s, 0, slot.sp.greedy() ? nullptr : &first_cands, pos );
			tokens_prefilled_.fetch_add( static_cast<uint64_t>( n - pos ), std::memory_order_relaxed );
			if ( !slot.sp.greedy() ) first = sample_token( first_cands, slot.sp, slot.rng );
			slot.current_pos = n;
			slot.prefill_done = true;
			slot.metrics.first_token_time = std::chrono::steady_clock::now();
			const double ttft_ms = std::chrono::duration<double, std::milli>( slot.metrics.first_token_time - slot.metrics.start_time ).count();
			slot.metrics.prompt_tokens = n;
			slot.metrics.cached_tokens = slot.pf_resumed;
			slot.metrics.prefill_tok_s = ( ttft_ms > 0.0 ) ? static_cast<float>( n / ( ttft_ms / 1000.0 ) ) : 0.0f;
			slot.last_token = first;
			emit_token( s, first );
			return true;
		}
		cluster_.prefill( prompt.data() + pos, seg_end - pos, s, 0, nullptr, pos );
		tokens_prefilled_.fetch_add( static_cast<uint64_t>( seg_end - pos ), std::memory_order_relaxed );
		fed += seg_end - pos;
		pos = seg_end;
		if ( is_cp && seg_end == end ) {
			slot.pf_last_cp = end;
			++slot.pf_cp_next;
			const auto t_cp = std::chrono::steady_clock::now();
			slot.pf_node = store_prefix( s, slot.pf_node, pos, cp->pinned, cp->taps );
			LOG_DEBUG( "[gen] slot %d checkpoint at %d stored in %.0f ms\n", s, pos, std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t_cp ).count() );
			slot.pf_chain_ok = slot.pf_node != nullptr;
		}
		slot.pf_pos = pos;
		if ( budget > 0 && fed >= budget ) return false;
	}
}

// Snapshots slot s's prompt up to `tokens` as a child of `parent` (KV rows since the parent only). Returns the cache node for
// that prefix, an existing one if present, or nullptr if it could not be stored.
PrefixCache::Entry* Generator::store_prefix( int s, PrefixCache::Entry* parent, int tokens, bool pinned, bool with_taps ) {
	const std::vector<int>& full = slots_[s].key_tokens.empty() ? slots_[s].req.prompt_tokens : slots_[s].key_tokens;
	const std::vector<int> key( full.begin(), full.begin() + tokens );
	if ( PrefixCache::Entry* e = prefix_cache_->exact( key ) ) return e;
	ModelCluster::SeqSnapshot snap;
	if ( !cluster_.save_snapshot( s, parent ? static_cast<int>( parent->tokens.size() ) : 0, tokens, &snap, with_taps ) ) return nullptr;
	return prefix_cache_->insert( parent, key, snap, pinned );
}

// The reasoning budget ran out: feed the closing tokens as if the model had written them (anchor first, then all but the
// last close token as a short prefill), emit them, and let the model continue with the answer.
void Generator::apply_forced_close( int s ) {
	Slot& slot = slots_[s];
	if ( !slot.active || !slot.force_close ) return;
	slot.force_close = false;
	slot.in_reasoning = false;
	const std::vector<int>& close = slot.req.reasoning_close;
	if ( close.empty() || slot.current_pos + static_cast<int>( close.size() ) + pod::MAX_BATCH >= cluster_.metadata().max_seq_len ) return;
	std::vector<int> feed = { slot.last_token };
	feed.insert( feed.end(), close.begin(), close.end() - 1 );
	cluster_.prefill( feed.data(), static_cast<int>( feed.size() ), s, 0, nullptr, slot.current_pos );
	slot.current_pos += static_cast<int>( feed.size() );
	slot.last_token = close.back();
	for ( int t : close ) {
		if ( emit_token( s, t ) ) return;
	}
}

void Generator::step_batch() {
	// Tiered experts: fold the routing since the last pass into the residency every so often (a device sync per rank)
	const int adapt_every = runtime::options().expert_adapt_every;
	if ( adapt_every > 0 && ++adapt_steps_ >= adapt_every ) {
		adapt_steps_ = 0;
		const auto t_ad = std::chrono::steady_clock::now();
		const bool moved = cluster_.rebalance_experts();
		const double ad_ms = std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t_ad ).count();
		if ( ad_ms > 50.0 ) LOG_DEBUG( "[gen] expert rebalance took %.0f ms%s\n", ad_ms, moved ? "" : " (nothing moved)" );
	}

	// New prompts: while other requests are decoding (and the speculator has no single shared context that a decode step could disturb),
	// one prompt is fed a piece per step and the decode steps run in between; otherwise every pending prompt is fed in full
	bool decoding = false;
	for ( int s = 0; s < num_slots_; ++s ) decoding = decoding || ( slots_[s].active && slots_[s].prefill_done );
	const int chunk = ( decoding && ( !cluster_.has_speculator() || cluster_.speculator().per_slot() ) ) ? runtime::options().prefill_chunk : 0;
	if ( chunk > 0 ) {
		for ( int i = 1; i <= num_slots_; ++i ) {
			const int s = ( pf_rr_ + i ) % num_slots_;
			if ( slots_[s].active && !slots_[s].prefill_done ) {
				pf_rr_ = s;
				const auto t0 = std::chrono::steady_clock::now();
				const int pos0 = slots_[s].pf_pos;
				prefill_slot( s, chunk );
				const double ms = std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count();
				if ( ms > 300.0 ) LOG_DEBUG( "[gen] slot %d prefill piece at %d (%d rows) took %.0f ms\n", s, pos0, slots_[s].pf_pos - pos0, ms );
				break;
			}
		}
	}
	else {
		for ( int s = 0; s < num_slots_; ++s ) {
			while ( slots_[s].active && !slots_[s].prefill_done ) prefill_slot( s, 0 );
		}
	}

	int ids[MAX_SLOTS];
	int n = 0;
	for ( int s = 0; s < num_slots_; ++s ) {
		if ( slots_[s].active && slots_[s].prefill_done ) ids[n++] = s;
	}
	if ( n == 0 ) return;

	// Per-slot speculators (MTP): any number of slots keep a context, and one slot's round runs per step (a round costs about a plain
	// step, so interleaving rounds returns more tokens per GPU-ms than a batched plain step). Slots whose controller found plain
	// decoding faster take a plain step together, alternating with the rounds.
	const bool per_slot = cluster_.has_speculator() && cluster_.speculator().per_slot() && ( n == 1 || runtime::options().spec_interleave );
	int spec_slot = -1;
	int e_ids[MAX_SLOTS], d_ids[MAX_SLOTS], ne = 0, nd = 0;
	int group[MAX_SLOTS], ng = 0;   // slots of a joint round
	if ( per_slot ) {
		for ( int i = 0; i < n; ++i ) {
			const int s = ids[i];
			if ( cluster_.spec_owns( s ) && !slots_[s].ctrl.dense_mode() ) e_ids[ne++] = s;
			else d_ids[nd++] = s;
		}
	}
	const bool joint_rounds = per_slot && ne >= 2 && cluster_.speculator().joint() && runtime::options().spec_joint;
	// With several slots decoding, a plain step over all of them returns n tokens for little more than the cost of one, while a round returns
	// a few tokens of one slot (or of several, as a joint round): compare the measured rates (and re-measure the unused side now and then)
	bool agg_dense = false;
	if ( per_slot && n >= 2 && n <= MAX_SLOTS ) {
		const bool probe = ( ++agg_steps_ % 64 ) == 0;
		const double rate = spec_tok_ms_n_[joint_rounds ? ne : 1];
		agg_dense = dense_ms_n_[n] <= 0.0 || ( rate > 0.0 && n / dense_ms_n_[n] > rate * 1.1 );
		if ( probe ) agg_dense = !agg_dense;
	}
	if ( per_slot && !agg_dense ) {
		// A slot in plain mode gets its probe round when its counter says so
		for ( int i = 0; i < nd && spec_slot < 0; ++i ) {
			if ( cluster_.spec_owns( d_ids[i] ) && slots_[d_ids[i]].ctrl.use_spec() ) spec_slot = d_ids[i];
		}
		if ( spec_slot < 0 && ne > 0 && ( nd == 0 || spec_turn_ ) ) {
			if ( joint_rounds ) {
				for ( int i = 0; i < ne; ++i ) group[ng++] = e_ids[i];
				spec_slot = group[0];
			}
			else {
				spec_slot = e_ids[0];
				for ( int i = 0; i < ne; ++i ) {
					if ( e_ids[i] > spec_rr_ ) { spec_slot = e_ids[i]; break; }
				}
				spec_rr_ = spec_slot;
			}
			spec_turn_ = ( nd == 0 );
		}
		else if ( spec_slot < 0 ) {
			spec_turn_ = true;
		}
		if ( spec_slot >= 0 ) {
			ids[0] = spec_slot;
			n = 1;
		}
		else {
			for ( int i = 0; i < nd; ++i ) ids[i] = d_ids[i];
			n = nd;
		}
	}

	if ( ng >= 2 ) {
		// Joint round: one draft chain and one verify forward for the blocks of every eligible slot
		int anchors[MAX_SLOTS], poss[MAX_SLOTS];
		const SamplingParams* sps[MAX_SLOTS];
		Rng* rngs[MAX_SLOTS];
		SpecResult res[MAX_SLOTS];
		for ( int i = 0; i < ng; ++i ) {
			Slot& sl = slots_[group[i]];
			anchors[i] = sl.last_token;
			poss[i] = sl.current_pos;
			sps[i] = &sl.sp;
			rngs[i] = &sl.rng;
		}
		const auto t0 = std::chrono::steady_clock::now();
		cluster_.speculator().speculate_joint( ng, group, anchors, poss, sps, rngs, res );
		const double round_ms = std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count();
		int total_tokens = 0;
		for ( int i = 0; i < ng; ++i ) total_tokens += res[i].emitted_count;
		if ( round_ms > 0.0 ) spec_tok_ms_n_[ng] = spec_tok_ms_n_[ng] > 0.0 ? 0.8 * spec_tok_ms_n_[ng] + 0.2 * ( total_tokens / round_ms ) : total_tokens / round_ms;
		for ( int i = 0; i < ng; ++i ) {
			const int s = group[i];
			Slot& slot = slots_[s];
			slot.ctrl.observe_spec( res[i].emitted_count, round_ms / ng );
			slot.current_pos = res[i].next_pos;
			slot.last_token = res[i].emitted[res[i].emitted_count - 1];
			{
				std::unique_lock<std::mutex> mlock( metrics_mtx_ );
				spec_rounds_++;
				spec_proposals_ += res[i].proposed;
				spec_accepted_ += res[i].accepted;
			}
			for ( int t = 0; t < res[i].emitted_count; ++t ) {
				if ( emit_token( s, res[i].emitted[t] ) ) break;
			}
		}
		return;
	}

	const bool single_spec = ( n == 1 && cluster_.spec_owns( ids[0] ) );
	if ( per_slot ? ( spec_slot >= 0 ) : ( single_spec && slots_[ids[0]].ctrl.use_spec() ) ) {
		// Case A: single request that owns the drafter context, DFlash2 speculative round (Mode A
		// causal verify block). A slot that lost ownership to a newer request decodes normally.
		const int s = ids[0];
		Slot& slot = slots_[s];
		const auto t0 = std::chrono::steady_clock::now();
		const SpecResult r = cluster_.speculate( s, slot.last_token, slot.current_pos, -1.0f, &slot.sp, &slot.rng );
		const double round_ms = std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count();
		slot.ctrl.observe_spec( r.emitted_count, round_ms );
		if ( per_slot && round_ms > 0.0 ) spec_tok_ms_n_[1] = spec_tok_ms_n_[1] > 0.0 ? 0.8 * spec_tok_ms_n_[1] + 0.2 * ( r.emitted_count / round_ms ) : r.emitted_count / round_ms;
		slot.current_pos = r.next_pos;
		slot.last_token = r.emitted[r.emitted_count - 1];
		{
			std::unique_lock<std::mutex> mlock( metrics_mtx_ );
			spec_rounds_++;
			spec_proposals_ += r.proposed;
			spec_accepted_ += r.accepted;
		}
		for ( int i = 0; i < r.emitted_count; ++i ) {
			if ( emit_token( s, r.emitted[i] ) ) break;
		}
		return;
	}

	// Case B (n == 1) / Case C (n > 1): one token per active slot in a single Mode B forward
	const auto dense_t0 = std::chrono::steady_clock::now();
	int tokens[MAX_SLOTS];
	int positions[MAX_SLOTS];
	for ( int i = 0; i < n; ++i ) {
		tokens[i] = slots_[ids[i]].last_token;
		positions[i] = slots_[ids[i]].current_pos;
	}
	cluster_.forward( tokens, pod::BatchRows::batched( n, ids, positions ) );
	int preds[MAX_SLOTS];
	cluster_.sample( preds, n );
	bool any_sampled = false;
	for ( int i = 0; i < n; ++i ) any_sampled = any_sampled || !slots_[ids[i]].sp.greedy();
	if ( any_sampled ) {
		Candidates cs[MAX_SLOTS];
		cluster_.sample_topk( n, cs );
		for ( int i = 0; i < n; ++i ) {
			Slot& sl = slots_[ids[i]];
			if ( !sl.sp.greedy() ) preds[i] = sample_token( cs[i], sl.sp, sl.rng );
		}
	}
	if ( per_slot ) {
		// Plain-step cost per token, for the slots that keep a speculator context (compared with their rounds' tokens per ms)
		const double total_ms = std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - dense_t0 ).count();
		const double ms = total_ms / n;
		for ( int i = 0; i < n; ++i ) {
			if ( cluster_.spec_owns( ids[i] ) ) slots_[ids[i]].ctrl.observe_dense( ms );
		}
		if ( n <= MAX_SLOTS ) dense_ms_n_[n] = dense_ms_n_[n] > 0.0 ? 0.7 * dense_ms_n_[n] + 0.3 * total_ms : total_ms;
	}
	else if ( single_spec ) {
		slots_[ids[0]].ctrl.observe_dense( std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - dense_t0 ).count() );
	}
	if ( cluster_.has_speculator() ) {
		for ( int i = 0; i < n; ++i ) cluster_.speculator().on_dense_row( ids[i], tokens[i], positions[i], i );
	}
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

	if ( prefix_cache_ ) {
		m.prefix_enabled = true;
		m.prefix = prefix_cache_->stats();
	}
	m.gpu_vram_used_bytes.resize( cluster_.tp() );
	for ( int d = 0; d < cluster_.tp(); ++d ) {
		m.gpu_vram_used_bytes[d] = cluster_.arena( d ).total_bytes;
	}
	return m;
}

// ---- monitor page data ----

void Generator::publish_slots() {
	const auto now = std::chrono::steady_clock::now();
	std::unique_lock<std::mutex> lock( metrics_mtx_ );
	bool any = false;
	for ( int s = 0; s < MAX_SLOTS; ++s ) {
		SlotView& v = slot_view_[s];
		const Slot& slot = slots_[s];
		v = SlotView();
		if ( s >= num_slots_ || !slot.active ) continue;
		any = true;
		v.active = true;
		v.prefilling = !slot.prefill_done;
		v.reasoning = slot.in_reasoning;
		v.req_id = slot.req.req_id;
		v.pos = slot.prefill_done ? slot.current_pos : slot.pf_pos;
		v.prompt_tokens = static_cast<int>( slot.req.prompt_tokens.size() );
		v.cached_tokens = slot.pf_resumed;
		v.generated = slot.tokens_generated;
		v.max_tokens = slot.req.max_tokens;
		v.age_s = std::chrono::duration<double>( now - slot.metrics.start_time ).count();
		if ( slot.prefill_done && slot.tokens_generated > 1 ) {
			const double d = std::chrono::duration<double>( now - slot.metrics.first_token_time ).count();
			if ( d > 0.0 ) v.decode_tok_s = ( slot.tokens_generated - 1 ) / d;
		}
	}
	published_active_ = any;
	tick_history_locked( now );
}

void Generator::tick_history_locked( std::chrono::steady_clock::time_point now ) const {
	int active = 0;
	for ( int s = 0; s < MAX_SLOTS; ++s ) active += slot_view_[s].active ? 1 : 0;
	const double behind = std::chrono::duration<double>( now - hist_last_ ).count();
	if ( behind < 1.0 ) return;
	const uint64_t out = tokens_out_.load( std::memory_order_relaxed ), pf = tokens_prefilled_.load( std::memory_order_relaxed );
	// the counters moved over `behind` seconds; seconds with nothing in them are zero rate
	const int secs = static_cast<int>( MIN( behind, 3600.0 ) );
	for ( int i = 0; i < secs; ++i ) {
		HistorySample h;
		h.t = std::chrono::duration<double>( hist_last_ - boot_time_ ).count() + i + 1;
		const bool last = i == secs - 1;
		h.decode_tok_s = last ? static_cast<float>( ( out - hist_out_ ) / behind ) : 0.0f;
		h.prefill_tok_s = last ? static_cast<float>( ( pf - hist_pf_ ) / behind ) : 0.0f;
		h.active = last ? active : 0;
		history_.push_back( h );
	}
	while ( history_.size() > 900 ) history_.pop_front();
	hist_last_ += std::chrono::seconds( secs );
	hist_out_ = out;
	hist_pf_ = pf;
}

void Generator::fold_expert_heat( bool force ) {
	if ( heat_off_ || !runtime::options().expert_stats ) return;
	const auto now = std::chrono::steady_clock::now();
	if ( !force && std::chrono::duration<double>( now - heat_last_ ).count() < 1.0 ) return;
	heat_last_ = now;
	heat_pending_ = false;
	std::vector<std::vector<uint32_t>> per_rank;
	int layers = 0, experts = 0;
	if ( !cluster_.expert_heat( per_rank, layers, experts ) ) {
		heat_off_ = true;
		return;
	}
	std::unique_lock<std::mutex> lock( metrics_mtx_ );
	const size_t per = static_cast<size_t>( layers ) * experts;
	if ( experts_.layers != layers || experts_.experts != experts || experts_.ranks != static_cast<int>( per_rank.size() ) ) {
		experts_.ranks = static_cast<int>( per_rank.size() );
		experts_.layers = layers;
		experts_.experts = experts;
		experts_.total.assign( per * per_rank.size(), 0 );
		experts_.recent.assign( per * per_rank.size(), 0.0f );
	}
	for ( size_t r = 0; r < per_rank.size(); ++r ) {
		if ( per_rank[r].size() != per ) continue;
		for ( size_t i = 0; i < per; ++i ) {
			experts_.total[r * per + i] += per_rank[r][i];
			experts_.recent[r * per + i] = experts_.recent[r * per + i] * 0.95f + static_cast<float>( per_rank[r][i] );
		}
	}
}

ExpertStats Generator::get_expert_stats() const {
	std::unique_lock<std::mutex> lock( metrics_mtx_ );
	return experts_;
}

static void percentiles( std::vector<float> v, const double* q, double* out, int n ) {
	if ( v.empty() ) return;
	std::sort( v.begin(), v.end() );
	for ( int i = 0; i < n; ++i ) out[i] = v[MIN( v.size() - 1, static_cast<size_t>( q[i] * ( v.size() - 1 ) + 0.5 ) )];
}

MonitorSnapshot Generator::get_monitor() const {
	MonitorSnapshot m;
	const auto now = std::chrono::steady_clock::now();
	m.uptime_seconds = std::chrono::duration<double>( now - boot_time_ ).count();
	m.num_slots = num_slots_;
	m.max_context = cluster_.metadata().max_seq_len;
	m.tp = cluster_.tp();
	{
		std::unique_lock<std::mutex> qlock( queue_mtx_ );
		m.queued = pending_queue_.size();
	}
	std::unique_lock<std::mutex> lock( metrics_mtx_ );
	tick_history_locked( now );
	m.requests = total_requests_;
	m.completed = completed_requests_;
	m.tokens_out = tokens_out_.load( std::memory_order_relaxed );
	m.tokens_prefilled = tokens_prefilled_.load( std::memory_order_relaxed );
	m.prompt_tokens = prompt_tokens_total_;
	m.cached_tokens = cached_tokens_total_;
	for ( int i = 0; i < 4; ++i ) m.finished[i] = finished_[i];
	m.spec_rounds = spec_rounds_;
	m.spec_proposed = spec_proposals_;
	m.spec_accepted = spec_accepted_;
	for ( int s = 0; s < num_slots_; ++s ) m.slots.push_back( slot_view_[s] );
	m.recent.assign( recent_.begin(), recent_.end() );
	m.history.assign( history_.begin(), history_.end() );
	const size_t ng = MIN( gap_n_, gap_ring_.size() ), nt = MIN( ttft_n_, ttft_ring_.size() );
	std::vector<float> gaps( gap_ring_.begin(), gap_ring_.begin() + ng ), ttfts( ttft_ring_.begin(), ttft_ring_.begin() + nt );
	lock.unlock();
	m.gap_samples = ng;
	const double gq[4] = { 0.5, 0.9, 0.99, 0.999 }, tq[3] = { 0.5, 0.9, 0.99 };
	percentiles( gaps, gq, m.gap_ms, 4 );
	if ( !gaps.empty() ) m.gap_ms[4] = *std::max_element( gaps.begin(), gaps.end() );
	percentiles( ttfts, tq, m.ttft_ms, 3 );
	m.gap_hist.assign( 12, 0.0f );
	for ( float g : gaps ) {
		int b = 0;
		for ( float t = 2.0f; g >= t && b < 11; t *= 2.0f ) ++b;
		m.gap_hist[b] += 1.0f;
	}
	return m;
}

} // namespace engine::runtime
