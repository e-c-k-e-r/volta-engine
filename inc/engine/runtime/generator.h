#pragma once

#include "engine/runtime/spec_controller.h"
#include "engine/runtime/sampling.h"
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/runtime/cluster.h"
#include "engine/loader/tokenizer.h"
#include "engine/loader/chat_template.h"
#include "engine/server/config.h"

#include <cuda_runtime.h>
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

struct GenerationRequest {
	uint64_t req_id = 0;
	std::string prompt_text;
	std::vector<int> prompt_tokens;
	int max_tokens = 64;
	float temperature = 0.0f;
	float top_p = 1.0f;
	int top_k = 16;          // capped at 16 (kSampleTopK)
	uint64_t seed = 0;       // 0 = derive from req_id
	bool stream = false;
	std::function<void( const std::string& token_str, bool is_done )> stream_callback;
	std::function<void( const std::string& full_text, const SlotMetrics& metrics )> completion_callback;
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
		std::string accumulated_text;
		std::vector<int> all_tokens;
		SlotMetrics metrics;
		SpecController ctrl;
	};

	void scheduler_loop();
	void admit_pending();
	void step_batch();
	void prefill_slot( int s );
	// Commits `tok` to slot s's output; returns true when the request finished.
	bool emit_token( int s, int tok );
	void finish_slot( int s );

	engine::server::ServerConfig config_;
	ModelCluster cluster_;
	engine::loader::Tokenizer tokenizer_;
	engine::loader::ChatTemplate chat_template_;

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
