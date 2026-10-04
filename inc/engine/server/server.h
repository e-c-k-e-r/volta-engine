#pragma once

#include "engine/config.h"
#include "engine/server/config.h"
#include "engine/runtime/generator.h"

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <sys/epoll.h>

namespace engine::server {

constexpr int MAX_CLIENTS = 128;
constexpr size_t CLIENT_BUF_SIZE = 65536;

enum class ClientState {
	UNUSED = 0,
	READING_HTTP,
	PROCESSING,
	CLOSING
};

struct ClientContext {
	int fd = -1;
	ClientState state = ClientState::UNUSED;
	char in_buf[CLIENT_BUF_SIZE];
	size_t in_len = 0;
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
	void handle_metrics_request( int client_fd );
	void handle_chat_completion( ClientContext& client, const char* body, size_t body_len );

	void write_all( int fd, const char* data, size_t len );
	void close_client( int fd );

	ServerConfig config_;
	int listen_fd_ = -1;
	int epoll_fd_ = -1;
	std::atomic<bool> running_{ false };

	std::unique_ptr<engine::runtime::Generator> generator_;
	std::unique_ptr<ClientContext[]> clients_;
	std::mutex client_mtx_;
	std::atomic<uint64_t> next_req_id_{ 1 };
};

} // namespace engine::server
