#include "engine/server/http.h"

#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>

#include <errno.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <thread>

namespace engine::server {

void Connection::close_locked() {
	if ( closed_.exchange( true ) ) return;
	epoll_ctl( epoll_fd_, EPOLL_CTL_DEL, fd_, nullptr );
	::shutdown( fd_, SHUT_WR );
	::close( fd_ );
	out_.clear();
}

void Connection::set_epollout_locked( bool on ) {
	if ( epollout_ == on ) return;
	epollout_ = on;
	epoll_event ev = {};
	ev.events = EPOLLIN | EPOLLRDHUP | ( on ? EPOLLOUT : 0u );
	ev.data.fd = fd_;
	epoll_ctl( epoll_fd_, EPOLL_CTL_MOD, fd_, &ev );
}

bool Connection::read_some( std::string& buf ) {
	std::lock_guard<std::mutex> lock( mtx_ );
	if ( closed_.load() ) return false;
	char tmp[16384];
	for ( ;; ) {
		const ssize_t n = ::read( fd_, tmp, sizeof( tmp ) );
		if ( n > 0 ) {
			buf.append( tmp, static_cast<size_t>( n ) );
			last_activity = std::chrono::steady_clock::now();
			if ( buf.size() > ( size_t( 64 ) << 20 ) ) break;
			continue;
		}
		if ( n < 0 && ( errno == EAGAIN || errno == EWOULDBLOCK ) ) return true;
		if ( n < 0 && errno == EINTR ) continue;
		close_locked(); // EOF or error
		return false;
	}
	return true;
}

bool Connection::write_all( const char* data, size_t len ) {
	std::lock_guard<std::mutex> lock( mtx_ );
	if ( closed_.load() ) return false;
	size_t off = 0;
	if ( out_.empty() ) {
		while ( off < len ) {
			const ssize_t n = ::send( fd_, data + off, len - off, MSG_NOSIGNAL | MSG_DONTWAIT );
			if ( n > 0 ) {
				off += static_cast<size_t>( n );
				continue;
			}
			if ( n < 0 && errno == EINTR ) continue;
			if ( n < 0 && ( errno == EAGAIN || errno == EWOULDBLOCK ) ) break;
			close_locked();
			return false;
		}
	}
	if ( off < len ) {
		out_.append( data + off, len - off );
		if ( out_.size() > kMaxQueued ) {
			close_locked();
			return false;
		}
		set_epollout_locked( true );
	}
	return true;
}

void Connection::on_writable() {
	std::lock_guard<std::mutex> lock( mtx_ );
	if ( closed_.load() ) return;
	size_t off = 0;
	while ( off < out_.size() ) {
		const ssize_t n = ::send( fd_, out_.data() + off, out_.size() - off, MSG_NOSIGNAL | MSG_DONTWAIT );
		if ( n > 0 ) {
			off += static_cast<size_t>( n );
			continue;
		}
		if ( n < 0 && errno == EINTR ) continue;
		if ( n < 0 && ( errno == EAGAIN || errno == EWOULDBLOCK ) ) break;
		close_locked();
		return;
	}
	out_.erase( 0, off );
	if ( out_.empty() ) {
		set_epollout_locked( false );
		if ( want_close_ ) close_locked();
	}
}

size_t Connection::queued() {
	std::lock_guard<std::mutex> lock( mtx_ );
	return out_.size();
}

void Connection::finish() {
	std::lock_guard<std::mutex> lock( mtx_ );
	if ( closed_.load() ) return;
	if ( out_.empty() ) close_locked();
	else want_close_ = true;
}

void Connection::close_now() {
	std::lock_guard<std::mutex> lock( mtx_ );
	close_locked();
}

const char* http_reason( int status ) {
	switch ( status ) {
		case 200: return "OK";
		case 206: return "Partial Content";
		case 400: return "Bad Request";
		case 404: return "Not Found";
		case 405: return "Method Not Allowed";
		case 408: return "Request Timeout";
		case 411: return "Length Required";
		case 413: return "Payload Too Large";
		case 422: return "Unprocessable Entity";
		case 500: return "Internal Server Error";
		case 501: return "Not Implemented";
		case 503: return "Service Unavailable";
		default: return "Error";
	}
}

void send_response( const ConnPtr& c, int status, const std::string& content_type, const std::string& body, const std::string& extra_headers ) {
	std::string head = "HTTP/1.1 " + std::to_string( status ) + " " + http_reason( status ) + "\r\nContent-Type: " + content_type +
		"\r\nContent-Length: " + std::to_string( body.size() ) + "\r\nConnection: close\r\n" + extra_headers + "\r\n";
	c->write_all( head + body );
	c->finish();
}

void send_json( const ConnPtr& c, int status, const std::string& json ) {
	send_response( c, status, "application/json", json );
}

void send_error( const ConnPtr& c, int status, const std::string& type, const std::string& message, bool anthropic ) {
	nlohmann::json e;
	if ( anthropic ) {
		e["type"] = "error";
		e["error"] = { { "type", type }, { "message", message } };
	}
	else {
		e["error"] = { { "message", message }, { "type", type }, { "param", nullptr }, { "code", nullptr } };
	}
	send_json( c, status, e.dump() );
}

void send_file( const ConnPtr& c, const HttpRequest& req, const std::string& path, const std::string& content_type ) {
	std::ifstream f( path, std::ios::binary | std::ios::ate );
	if ( !f ) return send_error( c, 404, "not_found_error", "no such file" );
	const size_t size = static_cast<size_t>( f.tellg() );
	size_t a = 0, b = size ? size - 1 : 0;
	bool range = false;
	const std::string spec = req.header( "range" );
	if ( spec.compare( 0, 6, "bytes=" ) == 0 && size > 0 ) {
		const size_t dash = spec.find( '-' );
		if ( dash != std::string::npos ) {
			if ( dash > 6 ) a = std::strtoull( spec.c_str() + 6, nullptr, 10 );
			if ( dash + 1 < spec.size() ) b = std::min<size_t>( std::strtoull( spec.c_str() + dash + 1, nullptr, 10 ), size - 1 );
			range = a <= b && a < size;
		}
	}
	if ( !range ) {
		a = 0;
		b = size ? size - 1 : 0;
	}
	const size_t length = size ? b - a + 1 : 0;
	std::string head = std::string( "HTTP/1.1 " ) + ( range ? "206 Partial Content" : "200 OK" ) + "\r\nContent-Type: " + content_type + "\r\nContent-Length: " + std::to_string( length ) +
		"\r\nAccept-Ranges: bytes\r\nConnection: close\r\n";
	if ( range ) head += "Content-Range: bytes " + std::to_string( a ) + "-" + std::to_string( b ) + "/" + std::to_string( size ) + "\r\n";
	head += "\r\n";
	c->write_all( head );
	constexpr size_t kChunk = size_t( 1 ) << 20, kHighWater = size_t( 4 ) << 20;
	std::string chunk;
	f.seekg( static_cast<std::streamoff>( a ) );
	for ( size_t left = length; left > 0 && !c->closed(); ) {
		while ( c->queued() > kHighWater && !c->closed() ) std::this_thread::sleep_for( std::chrono::milliseconds( 2 ) );
		chunk.resize( std::min( kChunk, left ) );
		f.read( chunk.data(), static_cast<std::streamsize>( chunk.size() ) );
		if ( !c->write_all( chunk ) ) return;
		left -= chunk.size();
	}
	c->finish();
}

bool start_sse( const ConnPtr& c ) {
	static const char head[] = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nX-Accel-Buffering: no\r\n\r\n";
	return c->write_all( head, sizeof( head ) - 1 );
}

void Router::add( const std::string& method, const std::string& path, RouteHandler h, bool blocking ) {
	routes_.push_back( { method, path, false, blocking, std::move( h ) } );
}

void Router::add_prefix( const std::string& method, const std::string& prefix, RouteHandler h, bool blocking ) {
	routes_.push_back( { method, prefix, true, blocking, std::move( h ) } );
}

bool Router::dispatch( const ConnPtr& conn, const HttpRequest& req ) const {
	const Route* hit = nullptr;
	for ( const Route& r : routes_ ) {
		if ( r.method != "*" && r.method != req.method ) continue;
		if ( r.prefix ? req.path.compare( 0, r.path.size(), r.path ) != 0 : r.path != req.path ) continue;
		if ( !r.prefix ) {
			hit = &r;
			break;
		}
		if ( !hit || r.path.size() > hit->path.size() ) hit = &r; // the longest prefix
	}
	if ( !hit ) return false;
	if ( hit->blocking ) {
		RouteHandler h = hit->handler;
		std::thread( [h, conn, req]() { h( conn, req ); } ).detach();
	} else {
		hit->handler( conn, req );
	}
	return true;
}

} // namespace engine::server
