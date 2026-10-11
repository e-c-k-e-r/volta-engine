#pragma once

#include "engine/config.h"
#include "engine/runtime/generator.h"
#include "engine/server/chat.h"
#include "engine/server/config.h"
#include "engine/server/http.h"
#include "engine/server/system_stats.h"

#include <atomic>
#include <map>
#include <memory>
#include <string>

namespace engine::server {

constexpr size_t MAX_HEADER_BYTES = 64 * 1024;
constexpr size_t MAX_BODY_BYTES = 64u << 20;

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
	void handle_new_connection();
	void handle_readable( int fd );
	void sweep_closed();
	// Returns true when a complete request was parsed out of conn->in; false when more input is needed or the request was rejected.
	bool parse_request( const ConnPtr& conn, HttpRequest* req );
	void register_routes();
	void dispatch( const ConnPtr& conn, const HttpRequest& req );
	void handle_health( const ConnPtr& conn );
	void handle_metrics( const ConnPtr& conn );
	void handle_models( const ConnPtr& conn );
	void handle_metrics_prometheus( const ConnPtr& conn );
	void handle_monitor( const ConnPtr& conn );
	void handle_api_stats( const ConnPtr& conn );
	void handle_api_experts( const ConnPtr& conn );
	void handle_chat( const ConnPtr& conn, const HttpRequest& req, ChatProtocol& protocol );
	void handle_count_tokens( const ConnPtr& conn, const HttpRequest& req, ChatProtocol& protocol );

	ServerConfig config_;
	int listen_fd_ = -1;
	int epoll_fd_ = -1;
	std::atomic<bool> running_{ false };

	std::unique_ptr<engine::runtime::Generator> generator_;
	std::map<int, ConnPtr> conns_; // epoll thread only
	Router router_;
	std::unique_ptr<ChatProtocol> openai_;
	std::unique_ptr<ChatProtocol> anthropic_;
	std::atomic<uint64_t> next_req_id_{ 1 };
	std::string model_name_;
	SystemStats sys_;
};

} // namespace engine::server
