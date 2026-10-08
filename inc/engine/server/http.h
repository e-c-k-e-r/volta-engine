#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

namespace engine::server {

// One client socket. Every fd operation (read, write, close) happens under mtx and checks closed, so the scheduler
// thread (streaming responses) and the epoll thread (reads, disconnects) can both use it, and a closed fd number reused
// by a later accept() can never be touched through a stale Connection.
class Connection {
public:
	Connection( int fd, int epoll_fd ) : fd_( fd ), epoll_fd_( epoll_fd ) {}
	~Connection() { close_now(); }

	bool closed() const { return closed_.load(); }

	// Appends whatever is readable to in. Returns false when the peer is gone (the connection is then closed).
	bool read_some( std::string& in );
	// Blocking-ish write (bounded wait for a slow peer). Returns false and closes the connection on failure.
	bool write_all( const char* data, size_t len );
	bool write_all( const std::string& s ) { return write_all( s.data(), s.size() ); }
	void close_now();

	std::string in;            // epoll thread only
	bool dispatched = false;   // epoll thread only: request handed to a handler, further input is ignored
	bool sent_continue = false;

private:
	int fd_;
	int epoll_fd_;
	std::atomic<bool> closed_{ false };
	std::mutex mtx_;
};

using ConnPtr = std::shared_ptr<Connection>;

const char* http_reason( int status );

// Complete response with Content-Length; closes the connection afterwards.
void send_response( const ConnPtr& c, int status, const std::string& content_type, const std::string& body );
void send_json( const ConnPtr& c, int status, const std::string& json );
// OpenAI/Anthropic-shaped error body: {"error": {"message", "type"}} / {"type":"error","error":{...}}
void send_error( const ConnPtr& c, int status, const std::string& type, const std::string& message, bool anthropic = false );
// Headers of a server-sent-events response (no Content-Length, connection closes at the end).
bool start_sse( const ConnPtr& c );

} // namespace engine::server
