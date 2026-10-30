#pragma once

#include "engine/runtime/spec_controller.h"
#include "engine/runtime/suffix_index.h"
#include "engine/runtime/sampling.h"
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/runtime/cluster.h"
#include "engine/runtime/prefix_cache.h"
#include "engine/runtime/steer_file.h"
#include "engine/loader/tokenizer.h"
#include "engine/loader/image.h"
#include "engine/loader/chat_template.h"
#include "engine/server/config.h"

#include "engine/backend/types.h"
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <functional>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <pthread.h>
#include <list>
#include <deque>
#include <unordered_map>

namespace engine::runtime {

struct SlotMetrics {
	std::chrono::steady_clock::time_point start_time;
	std::chrono::steady_clock::time_point first_token_time;
	int prompt_tokens = 0;
	int cached_tokens = 0;   // prompt tokens restored from the prefix cache
	int generated_tokens = 0;
	float prefill_tok_s = 0.0f;
	float decode_tok_s = 0.0f;
	// Where the time before the first token went (ms): waiting for a slot, restoring a cached prefix, the prefill kernels, storing
	// checkpoints; what is left of the time to first token (start_time to first_token_time) went to other slots' decode steps
	double queue_ms = 0.0, restore_ms = 0.0, prefill_ms = 0.0, store_ms = 0.0;
	float gate_score = -1.0f;   // steering probe: the look-ahead score (-1 = no probe ran)
	bool gate_fired = false;
};

enum class FinishReason { Stop, Length, StopString, Cancelled };

struct GenerationRequest {
	uint64_t req_id = 0;
	std::chrono::steady_clock::time_point submit_time;
	std::string prompt_text;
	std::vector<int> prompt_tokens;
	int max_tokens = 64;
	float temperature = 0.0f;
	float top_p = 1.0f;
	int top_k = 16;          // capped at 16 (kSampleTopK)
	uint64_t seed = 0;       // 0 = derive from req_id
	struct Checkpoint {
		int pos = 0;
		bool pinned = false;
		bool taps = true;        // snapshot the drafter taps too (a resume there keeps speculating)
	};
	// Images in the prompt: the placeholder run [start, start + count) of prompt_tokens belongs to one image. The generator encodes
	// it (or takes the embeddings from its cache) when the prefill reaches it.
	struct Image {
		int start = 0;
		int count = 0;
		int grid_h = 0;   // merged-patch grid, count == grid_h * grid_w
		int grid_w = 0;
		uint64_t hash = 0;
		std::shared_ptr<const engine::loader::PreparedImage> prepared;
	};
	std::vector<Image> images;
	std::vector<Checkpoint> checkpoint_at;   // ascending prompt positions to snapshot into the prefix cache (system-prompt end, turn boundary)
	// Reasoning budget: after this many reasoning tokens the generator appends reasoning_close (e.g. "\n</think>\n\n") itself
	// and the model continues with the answer. 0 = unlimited. think_close_id ends the reasoning when the model emits it.
	int reasoning_budget = 0;
	bool starts_in_reasoning = false;
	std::vector<int> reasoning_close;
	int think_close_id = -1;
	// Runtime steering (needs the steer option, ignored without it). steer_vector: "" = the option's choice, "file" = the steer_file option,
	// else a name in steer_dir; scale and scope replace the options' (<0 / empty = the options'); think_penalty is the logit taken off the
	// reflection markers inside the reasoning block (a prompt that ends in the template's <think> opens one; the tokenizer's </think> closes it).
	std::string steer_vector;
	float steer_scale = -1.0f;
	std::string steer_scope;
	// Overrides of the steer_gate_threshold / steer_edit_strength / steer_hedge_strength options for research (NaN = the option)
	float probe_threshold = NAN, probe_edit_strength = NAN, probe_hedge_strength = NAN, probe_push = NAN, probe_directions = NAN, probe_keep_norm = NAN;
	std::string probe_edit_layers;   // per-request steer_edit_layers (empty = the option)
	bool steer_probe = false;   // read the prompt with the steering probe first (Generator::probe_slot); replaces steer_vector when the gate fires
	float think_penalty = -1.0f;
	std::vector<std::string> stop_strings;   // generation ends before the first match
	std::string matched_stop;                // set when FinishReason::StopString
	// Called from the scheduler thread with detokenized text (never the end-of-sequence token). Returning false cancels.
	std::function<bool( const std::string& delta )> on_text;
	// Called once, from the scheduler thread, when the request leaves its slot.
	std::function<void( FinishReason reason, const std::string& matched_stop, const SlotMetrics& metrics )> on_finish;
};

struct EngineMetrics {
	double uptime_seconds = 0.0;
	int active_slots = 0;
	uint64_t total_requests = 0;
	double avg_ttft_ms = 0.0;
	double avg_prefill_tok_s = 0.0;
	double avg_decode_tok_s = 0.0;
	double spec_acceptance_rate = 0.0;
	uint64_t spec_proposals = 0, spec_accepted = 0;   // drafted tokens offered to the verifier and taken, since boot
	std::vector<size_t> gpu_vram_used_bytes;
	PrefixCache::Stats prefix;
	bool prefix_enabled = false;
};

// What the monitor page shows: live slots, the last finished requests, per-second rates, latency samples and expert routing
struct SlotView {
	bool active = false;
	bool prefilling = false;
	bool reasoning = false;
	uint64_t req_id = 0;
	int pos = 0;                 // tokens in the slot's context
	int prompt_tokens = 0;
	int cached_tokens = 0;
	int generated = 0;
	int max_tokens = 0;
	double age_s = 0.0;
	double decode_tok_s = 0.0;   // since the first token
};

struct RequestRecord {
	uint64_t req_id = 0;
	double end_s = 0.0;          // engine uptime at the end
	int prompt_tokens = 0;
	int cached_tokens = 0;
	int generated = 0;
	double ttft_ms = 0.0;
	double queue_ms = 0.0, restore_ms = 0.0, prefill_ms = 0.0, store_ms = 0.0;   // see SlotMetrics
	double total_ms = 0.0;
	double decode_tok_s = 0.0;
	FinishReason reason = FinishReason::Stop;
};

struct HistorySample {
	double t = 0.0;              // engine uptime of the end of the second
	float decode_tok_s = 0.0f;
	float prefill_tok_s = 0.0f;
	int active = 0;
};

struct ExpertStats {
	int ranks = 0, layers = 0, experts = 0;     // experts: per rank
	std::vector<uint64_t> total;                // [rank][layer][expert] routes since boot
	std::vector<float> recent;                  // the same, exponentially decayed (about the last minute)
};

struct MonitorSnapshot {
	double uptime_seconds = 0.0;
	int num_slots = 0;
	int max_context = 0;
	int tp = 0;
	size_t queued = 0;
	uint64_t requests = 0, completed = 0;
	uint64_t tokens_out = 0, tokens_prefilled = 0, prompt_tokens = 0, cached_tokens = 0;
	uint64_t finished[4] = {};                  // by FinishReason
	uint64_t spec_rounds = 0, spec_proposed = 0, spec_accepted = 0;
	uint64_t suffix_rounds = 0, suffix_proposed = 0, suffix_accepted = 0;   // the rounds whose block came from a suffix match (also counted above)
	double gap_ms[5] = {};                      // token-gap percentiles: p50 p90 p99 p999 max, over the last samples
	double ttft_ms[3] = {};                     // p50 p90 p99 over the last requests
	size_t gap_samples = 0;
	std::vector<SlotView> slots;
	std::vector<RequestRecord> recent;          // newest last
	std::vector<HistorySample> history;         // one per second, oldest first
	std::vector<float> gap_hist;                // token-gap histogram, log2 buckets from 1 ms (counts)
};

class Generator {
public:
	Generator();
	~Generator();

	bool init( const engine::server::ServerConfig& config );
	void shutdown();

	bool submit_request( GenerationRequest req );

	EngineMetrics get_metrics() const;
	MonitorSnapshot get_monitor() const;
	// Route counts of the routed experts (empty when the model keeps none); a copy, taken under the metrics lock
	ExpertStats get_expert_stats() const;

	const engine::loader::Tokenizer& tokenizer() const {
		return tokenizer_;
	}

	const engine::loader::ChatTemplate& chat_template() const {
		return chat_template_;
	}

	int max_context() const { return cluster_.metadata().max_seq_len; }
	// Image preprocessing parameters of the loaded vision tower; null when the model takes no images
	const engine::loader::VisionParams* vision_params() const { return cluster_.vision_params(); }
	int vocab_size() const { return cluster_.metadata().vocab_size; }
	int num_slots() const { return num_slots_; }
	bool steer_enabled() const { return cluster_.steer_enabled(); }
	// Steering request fields: false with the reason when the named control-vector file cannot be used (server: 400)
	bool check_steer( const std::string& vector, const std::string& scope, std::string* err );

private:
	struct Slot {
		bool active = false;
		GenerationRequest req;
		int current_pos = 0;   // next position to write (== tokens committed to KV / recurrent state)
		int tokens_generated = 0;
		int last_token = -1;   // sampled, not yet forwarded (next anchor)
		SamplingParams sp;
		Rng rng;
		bool prefill_done = false;
		// A prompt fed in pieces between decode steps of the other slots
		bool pf_started = false;
		int pf_pos = 0;                  // next prompt position to prefill
		int pf_resumed = 0;              // prompt tokens restored from the prefix cache
		PrefixCache::Entry* pf_node = nullptr;   // deepest cache node of the prompt so far
		std::vector<GenerationRequest::Checkpoint> pf_cps;
		size_t pf_cp_next = 0;
		int pf_last_cp = -1;
		bool pf_chain_ok = true;
		bool in_reasoning = false;
		int reasoning_tokens = 0;
		bool steer_think_open = false;   // steering: the reasoning block has not closed yet (the slot's think_end is still open)
		int steer_scope = 0;             // steering: SteerScope bits of the slot's residual edit (0 = none)
		bool steer_prompt = false;       // the prompt rows are edited: its KV is not the cache's (no restore, no checkpoints)
		bool force_close = false;   // budget reached: close the reasoning block before the next step
		std::string held_text;   // detokenized text withheld while it could still begin a stop string
		std::vector<int> all_tokens;
		SuffixIndex suffix;            // all_tokens indexed for spec_suffix
		bool suffix_offered = false;   // this round's block was offered from the index
		std::vector<int> key_tokens;   // prompt_tokens with image placeholders replaced by content keys (empty without images)
		SlotMetrics metrics;
		SpecController ctrl;
		std::chrono::steady_clock::time_point last_emit;
	};

	void scheduler_loop();
	void admit_pending();
	void step_batch();
	// Offers the slot's next speculation round the tokens that followed the last earlier occurrence of its current suffix (spec_suffix)
	void offer_suffix( int s );
	void note_round( int s, const SpecResult& r );
	int adapt_steps_ = 0;
	// Feeds up to `budget` prompt tokens of slot s (0 = all of them); true once the prompt is done and the first token is out
	bool prefill_slot( int s, int budget );
	double pf_rows_ = 0.0;   // prefill_budget_ms: the piece size (rows) the next piece starts with
	int pf_cooldown_left_ = 0;   // prefill_cooldown: the steps still to run without a prompt piece
	int pf_rr_ = 0;   // the slot after which the next prefilling slot is looked for
	PrefixCache::Entry* store_prefix( int s, PrefixCache::Entry* parent, int tokens, bool pinned, bool with_taps );
	static constexpr int kPrefixMinTokens = 512;
	// Commits `tok` to slot s's output; returns true when the request finished.
	bool emit_token( int s, int tok );
	void finish_slot( int s, FinishReason reason );
	void apply_forced_close( int s );
	// Steering (option steer): the slot's edits, scope, prompt length and reflection-marker penalty on every rank, set when the request is admitted
	void steer_admit( int s );
	// The edits of the request's steer_vector (scaled, with its scope in *scope), or none
	std::vector<engine::modules::SteerEntry> request_steer_entries( int s, int* scope );
	// Steering probe (request field steer_probe): reads the prompt unedited (openings forced after it, then a short greedy look-ahead), builds the
	// direction clamps from the states it sees and, when the look-ahead score passes the gate, leaves them on the slot for the real prefill
	void probe_slot( int s );
	const SteerFile* steer_file( const std::string& vector, std::string* err );
	void build_penalty_ids();
	std::mutex steer_mtx_;
	std::map<std::string, SteerFile> steer_files_;
	std::vector<int> penalty_ids_;
	// Embeddings of an image: from the cache, else the vision tower (null on failure)
	std::shared_ptr<const std::vector<backend::half_t>> image_embeddings( const GenerationRequest::Image& im );
	std::unordered_map<uint64_t, std::shared_ptr<const std::vector<backend::half_t>>> image_cache_;
	std::list<uint64_t> image_cache_lru_;
	size_t image_cache_bytes_ = 0;

	engine::server::ServerConfig config_;
	ModelCluster cluster_;
	engine::loader::Tokenizer tokenizer_;
	engine::loader::ChatTemplate chat_template_;
	std::unique_ptr<PrefixCache> prefix_cache_;

	std::thread scheduler_thread_;
	std::atomic<bool> running_{ false };

	static constexpr int MAX_SLOTS = 8;
	int spec_rr_ = 0;          // per-slot speculators: the slot after which the next round is looked for
	bool spec_turn_ = true;
	double spec_tok_ms_n_[9] = {};             // per-slot speculators: EMA of tokens per ms of a round over n slots (index 1 = a single slot; 0 = unmeasured)
	double dense_ms_n_[9] = {};                // per-slot speculators: EMA of a plain step over n slots (0 = unmeasured)
	unsigned agg_steps_ = 0;    // per-slot speculators: alternate a round with a plain step of the slots that do not speculate
	int num_slots_ = 1;
	Slot slots_[MAX_SLOTS];

	mutable std::mutex queue_mtx_;
	std::condition_variable queue_cv_;
	std::vector<GenerationRequest> pending_queue_;

	std::chrono::steady_clock::time_point boot_time_;
	mutable std::mutex metrics_mtx_;
	uint64_t total_requests_ = 0;
	double sum_ttft_ms_ = 0.0;
	double sum_prefill_tok_s_ = 0.0;
	double sum_decode_tok_s_ = 0.0;
	uint64_t completed_requests_ = 0;
	uint64_t spec_proposals_ = 0;
	uint64_t suffix_rounds_ = 0, suffix_proposed_ = 0, suffix_accepted_ = 0;
	uint64_t spec_accepted_ = 0;

	// Monitor state (metrics_mtx_ guards everything but the two atomics)
	void publish_slots();
	void tick_history_locked( std::chrono::steady_clock::time_point now ) const;
	void fold_expert_heat( bool force = false );
	std::atomic<uint64_t> tokens_out_{ 0 };
	std::atomic<uint64_t> tokens_prefilled_{ 0 };
	uint64_t prompt_tokens_total_ = 0, cached_tokens_total_ = 0, finished_[4] = {}, spec_rounds_ = 0;
	SlotView slot_view_[MAX_SLOTS];
	std::deque<RequestRecord> recent_;
	std::vector<float> gap_ring_ = std::vector<float>( 4096, 0.0f );
	std::vector<float> ttft_ring_ = std::vector<float>( 256, 0.0f );
	size_t gap_n_ = 0, ttft_n_ = 0;
	mutable std::deque<HistorySample> history_;
	mutable std::chrono::steady_clock::time_point hist_last_;
	mutable uint64_t hist_out_ = 0, hist_pf_ = 0;
	ExpertStats experts_;
	bool heat_off_ = false;
	bool heat_pending_ = false;   // the model ran since the last fold
	bool published_active_ = false;
	std::chrono::steady_clock::time_point heat_last_;
};

} // namespace engine::runtime
