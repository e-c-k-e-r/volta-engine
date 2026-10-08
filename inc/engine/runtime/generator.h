#pragma once

#include "engine/runtime/spec_controller.h"
#include "engine/runtime/sampling.h"
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/runtime/cluster.h"
#include "engine/runtime/prefix_cache.h"
#include "engine/loader/tokenizer.h"
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

namespace engine::runtime {

struct SlotMetrics {
	std::chrono::steady_clock::time_point start_time;
	std::chrono::steady_clock::time_point first_token_time;
	int prompt_tokens = 0;
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
	std::vector<Checkpoint> checkpoint_at;   // ascending prompt positions to snapshot into the prefix cache (system-prompt end, turn boundary)
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
		std::string held_text;   // detokenized text withheld while it could still begin a stop string
		std::vector<int> all_tokens;
		SlotMetrics metrics;
		SpecController ctrl;
	};

	void scheduler_loop();
	void admit_pending();
	void step_batch();
	int adapt_steps_ = 0;
	void prefill_slot( int s );
	PrefixCache::Entry* store_prefix( int s, PrefixCache::Entry* parent, int tokens, bool pinned, bool with_taps );
	static constexpr int kPrefixMinTokens = 512;
	// Commits `tok` to slot s's output; returns true when the request finished.
	bool emit_token( int s, int tok );
	void finish_slot( int s, FinishReason reason );

	engine::server::ServerConfig config_;
	ModelCluster cluster_;
	engine::loader::Tokenizer tokenizer_;
	engine::loader::ChatTemplate chat_template_;
	std::unique_ptr<PrefixCache> prefix_cache_;

	std::thread scheduler_thread_;
	std::atomic<bool> running_{ false };

	static constexpr int MAX_SLOTS = 8;
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
