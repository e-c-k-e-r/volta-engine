#pragma once

#include "engine/runtime/spec_controller.h"
#include "engine/runtime/sampling.h"
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/runtime/cluster.h"
#include "engine/runtime/prefix_cache.h"
#include "engine/loader/tokenizer.h"
#include "engine/loader/image.h"
#include "engine/loader/chat_template.h"
#include "engine/server/config.h"

#include "engine/backend/types.h"
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
};

enum class FinishReason { Stop, Length, StopString, Cancelled };

struct GenerationRequest {
	uint64_t req_id = 0;
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
	std::vector<size_t> gpu_vram_used_bytes;
	PrefixCache::Stats prefix;
	bool prefix_enabled = false;
};

class Generator {
public:
	Generator();
	~Generator();

	bool init( const engine::server::ServerConfig& config );
	void shutdown();

	bool submit_request( GenerationRequest req );

	EngineMetrics get_metrics() const;

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
		bool force_close = false;   // budget reached: close the reasoning block before the next step
		std::string held_text;   // detokenized text withheld while it could still begin a stop string
		std::vector<int> all_tokens;
		std::vector<int> key_tokens;   // prompt_tokens with image placeholders replaced by content keys (empty without images)
		SlotMetrics metrics;
		SpecController ctrl;
	};

	void scheduler_loop();
	void admit_pending();
	void step_batch();
	int adapt_steps_ = 0;
	// Feeds up to `budget` prompt tokens of slot s (0 = all of them); true once the prompt is done and the first token is out
	bool prefill_slot( int s, int budget );
	int pf_rr_ = 0;   // the slot after which the next prefilling slot is looked for
	PrefixCache::Entry* store_prefix( int s, PrefixCache::Entry* parent, int tokens, bool pinned, bool with_taps );
	static constexpr int kPrefixMinTokens = 512;
	// Commits `tok` to slot s's output; returns true when the request finished.
	bool emit_token( int s, int tok );
	void finish_slot( int s, FinishReason reason );
	void apply_forced_close( int s );
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

	std::mutex queue_mtx_;
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
	uint64_t spec_accepted_ = 0;
};

} // namespace engine::runtime
