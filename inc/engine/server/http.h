#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace engine::server {

// One client socket. Every fd operation (read, write, close) happens under mtx and checks closed, so the scheduler
// thread (streaming responses) and the epoll thread (reads, writes, disconnects) can both use it, and a closed fd number
// reused by a later accept() can never be touched through a stale Connection. Writes never block the caller: what the
// socket does not take is queued and flushed by the epoll thread on EPOLLOUT; a peer that lets the queue grow past
// kMaxQueued bytes is dropped.
class Connection {
public:
	static constexpr size_t kMaxQueued = size_t( 32 ) << 20;

	Connection( int fd, int epoll_fd ) : fd_( fd ), epoll_fd_( epoll_fd ), last_activity( std::chrono::steady_clock::now() ) {}
	~Connection() { close_now(); }

	bool closed() const { return closed_.load(); }
	// Bytes accepted by write_all() that the socket has not taken yet (for callers that stream a large body and wait for the queue to drain).
	size_t queued();

	// Appends whatever is readable to in. Returns false when the peer is gone (the connection is then closed).
	bool read_some( std::string& in );
	// Sends what fits and queues the rest. False (and closed) if the queue limit is exceeded or the socket failed.
	bool write_all( const char* data, size_t len );
	bool write_all( const std::string& s ) { return write_all( s.data(), s.size() ); }
	// Epoll thread: the socket became writable.
	void on_writable();
	// Close once everything queued has been sent / close right away.
	void finish();
	void close_now();

	std::string in;            // epoll thread only
	bool dispatched = false;   // epoll thread only: request handed to a handler, further input is ignored
	bool sent_continue = false;
	std::chrono::steady_clock::time_point last_activity; // epoll thread only: last byte received

private:
	void close_locked();
	void set_epollout_locked( bool on );

	int fd_;
	int epoll_fd_;
	std::atomic<bool> closed_{ false };
	std::mutex mtx_;
	std::string out_;
	bool want_close_ = false;
	bool epollout_ = false;
};

using ConnPtr = std::shared_ptr<Connection>;

// A parsed request: the query string is split off the path, header names are lower-cased.
struct HttpRequest {
	std::string method;
	std::string path;
	std::string query;
	std::string body;
	std::map<std::string, std::string> headers;

	std::string header( const std::string& name ) const {
		auto it = headers.find( name );
		return it == headers.end() ? std::string() : it->second;
	}
};

using RouteHandler = std::function<void( const ConnPtr&, const HttpRequest& )>;

// Method + path table of a server. Exact paths win over prefixes; a method of "*" matches any (the handler answers 405 itself). Handlers run on the epoll thread unless the route
// is added with blocking = true, which hands the request to a short-lived thread (decoding a large upload, reading a file): the Connection is safe to use from any thread.
class Router {
public:
	void add( const std::string& method, const std::string& path, RouteHandler h, bool blocking = false );
	void add_prefix( const std::string& method, const std::string& prefix, RouteHandler h, bool blocking = false );
	// False when no route matches (the caller answers 404).
	bool dispatch( const ConnPtr& conn, const HttpRequest& req ) const;

private:
	struct Route {
		std::string method, path;
		bool prefix = false, blocking = false;
		RouteHandler handler;
	};
	std::vector<Route> routes_;
};

const char* http_reason( int status );

// Complete response with Content-Length; closes the connection afterwards.
void send_response( const ConnPtr& c, int status, const std::string& content_type, const std::string& body, const std::string& extra_headers = "" );
// A file, honouring a single-range "Range: bytes=a-b" request (206 + Content-Range). Streams in chunks and waits for the socket, so it blocks the calling
// thread: use it from a blocking route. 404 when the file cannot be opened.
void send_file( const ConnPtr& c, const HttpRequest& req, const std::string& path, const std::string& content_type );
void send_json( const ConnPtr& c, int status, const std::string& json );
// OpenAI/Anthropic-shaped error body: {"error": {"message", "type"}} / {"type":"error","error":{...}}
void send_error( const ConnPtr& c, int status, const std::string& type, const std::string& message, bool anthropic = false );
// Headers of a server-sent-events response (no Content-Length, connection closes at the end).
bool start_sse( const ConnPtr& c );

} // namespace engine::server
