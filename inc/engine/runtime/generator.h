#pragma once

#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/models/model.h"
#include "engine/runtime/kv_cache.h"
#include "engine/loader/tokenizer.h"
#include "engine/loader/chat_template.h"
#include "engine/spec/dflash.h"
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
	void worker_thread_func( int rank );
	void scheduler_loop();
	void step_batch();
	void run_gpu_step_token( int tok, int pos, int slot_id = 0 );

	engine::server::ServerConfig config_;
	std::unique_ptr<engine::models::ModelPipeline> models_[ENGINE_NUM_GPUS];
	KVCachePool kv_pools_[ENGINE_NUM_GPUS];
	pod::MemoryArena arenas_[ENGINE_NUM_GPUS];
	engine::comm::CollectiveRing rings_[ENGINE_NUM_GPUS];
	cudaStream_t streams_[ENGINE_NUM_GPUS];
	void* staging_ptrs_[ENGINE_NUM_GPUS] = { nullptr };
	void* flag_ptrs_[ENGINE_NUM_GPUS] = { nullptr };

	engine::loader::Tokenizer tokenizer_;
	engine::loader::ChatTemplate chat_template_;
	engine::spec::DFlashSpeculator speculator_;

	std::thread worker_threads_[ENGINE_NUM_GPUS];
	std::thread scheduler_thread_;
	pthread_barrier_t step_barrier_;
	std::atomic<bool> running_{ false };

	enum class WorkerTask {
		NONE,
		STEP_FORWARD,
		STOP
	};
	WorkerTask worker_task_ = WorkerTask::NONE;
	int worker_token_ = -1;
	int worker_pos_ = 0;
	int worker_slot_id_ = 0;

	struct Slot {
		bool active = false;
		GenerationRequest req;
		int current_pos = 0;
		int tokens_generated = 0;
		int last_token = -1;
		bool prefill_done = false;
		std::string accumulated_text;
		SlotMetrics metrics;
	};

	static constexpr int MAX_SLOTS = 8;
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
