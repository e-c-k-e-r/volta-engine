#pragma once

#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/models/qwen38_27b.h"
#include "engine/models/glm53_flash.h"
#include "engine/spec/dflash.h"
#include "engine/loader/tokenizer.h"
#include "engine/loader/chat_template.h"

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <sys/epoll.h>

namespace engine::server {

enum class WorkerTask {
	NONE,
	STEP_FORWARD,
	STOP
};

struct GPUCoordinator {
	std::mutex mtx;
	std::condition_variable cv_worker;
	std::condition_variable cv_main;

	WorkerTask current_task = WorkerTask::NONE;
	uint64_t task_generation = 0;
	int step_token = -1;
	int step_pos = 0;
	int completed_count = 0;
	bool shutdown = false;
};

constexpr int MAX_BATCH_SIZE = 8;
constexpr int MAX_CLIENTS = 128;
constexpr size_t CLIENT_BUF_SIZE = 65536;

enum class ClientState {
	UNUSED = 0,
	READING_HTTP,
	PREFILLING,
	DECODING,
	SENDING_RESPONSE,
	CLOSING
};

struct BatchSlot {
	bool active = false;
	int client_fd = -1;
	uint64_t req_id = 0;
	char req_id_str[64] = { 0 };
	char model[64] = { 0 };
	bool stream = false;
	int max_tokens = 64;
	int tokens_generated = 0;
	int current_pos = 0;
	std::vector<int> prompt_tokens;
	size_t prompt_idx = 0;
	bool prefill_done = false;
	int last_token = -1;
	std::string accumulated_content;
	bool finish_sent = false;
	std::chrono::steady_clock::time_point start_time;
};

struct ClientContext {
	int fd = -1;
	ClientState state = ClientState::UNUSED;
	char in_buf[CLIENT_BUF_SIZE];
	size_t in_len = 0;
	int slot_id = -1;
};

struct ServerConfig {
	std::string host = "0.0.0.0";
	int port = 8000;
	std::string default_model = "qwen38";
	std::string qwen_model_dir = "/home/mrq/Documents/models/Qwen3.8-27B-exl3";
	std::string glm_model_dir = "/home/mrq/Documents/models/GLM-5.3-Flash-exl3-3.05bpw";
	std::string qwen_spec_dir = "/home/mrq/Documents/models/Qwen3.8-27B-DFlash2-EXL3-5.0bpw";
	std::string glm_spec_dir = "/home/mrq/Documents/models/GLM-5.3-Flash-DFlash2";
	bool speculative = false;
	bool benchmark_mode = false;
	bool load_weights = true;
	int tp_size = 4;
};

class VoltaServer {
public:
	VoltaServer();
	~VoltaServer();

	bool init( const ServerConfig& config );
	void run();
	void request_stop();
	void stop();

private:
	bool setup_socket();
	void set_nonblocking( int fd );
	void handle_new_connection();
	void handle_client_read( int fd );
	void process_http_request( ClientContext& client );
	void handle_health_request( int client_fd );
	void handle_chat_completion( ClientContext& client, const char* body, size_t body_len );

	void step_prefill();
	void step_decode();
	void advance_slot( int slot_idx, int next_token );
	void send_sse_chunk( BatchSlot& slot, const char* text_content, bool is_finish );
	void send_json_response( BatchSlot& slot );
	void release_slot( int slot_idx );
	void close_client( int fd );

	void write_all( int fd, const char* data, size_t len );
	void run_gpu_step_token( int tok, int pos );
	void gpu_worker_loop( int rank );

	ServerConfig config_;
	int listen_fd_ = -1;
	int epoll_fd_ = -1;
	std::atomic<bool> running_{ false };

	GPUCoordinator coord_;
	std::thread workers_[ENGINE_NUM_GPUS];

	std::unique_ptr<ClientContext[]> clients_;
	BatchSlot slots_[MAX_BATCH_SIZE];
	uint64_t next_req_id_ = 1;

	// Tokenizers & Chat templates
	engine::loader::Tokenizer qwen_tok_;
	engine::loader::ChatTemplate qwen_tmpl_;
	bool qwen_loaded_ = false;

	engine::loader::Tokenizer glm_tok_;
	engine::loader::ChatTemplate glm_tmpl_;
	bool glm_loaded_ = false;

	// GPU Resources (TP=4)
	int num_gpus_ = 4;
	cudaStream_t streams_[ENGINE_NUM_GPUS];
	pod::MemoryArena arenas_[ENGINE_NUM_GPUS];
	engine::comm::CollectiveRing rings_[ENGINE_NUM_GPUS];
	void* staging_ptrs_[ENGINE_NUM_GPUS] = { nullptr };
	void* flag_ptrs_[ENGINE_NUM_GPUS] = { nullptr };

	std::unique_ptr<engine::models::Qwen38Pipeline[]> qwen_pipes_;
	std::unique_ptr<engine::models::GLM53Pipeline[]> glm_pipes_;
	engine::spec::DFlashSpeculator qwen_spec_;
	engine::spec::DFlashSpeculator glm_spec_;

	pod::TensorView qwen_in_tv_[ENGINE_NUM_GPUS];
	pod::TensorView qwen_out_tv_[ENGINE_NUM_GPUS];
	pod::TensorView glm_in_tv_[ENGINE_NUM_GPUS];
	pod::TensorView glm_out_tv_[ENGINE_NUM_GPUS];

	bool models_initialized_ = false;
	bool real_weights_loaded_ = false;
};

} // namespace engine::server
