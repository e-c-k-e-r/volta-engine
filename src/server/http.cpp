#include "engine/server/http.h"

#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>

#include <errno.h>
#include <poll.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace engine::server {

bool Connection::read_some( std::string& buf ) {
	std::lock_guard<std::mutex> lock( mtx_ );
	if ( closed_.load() ) return false;
	char tmp[16384];
	for ( ;; ) {
		const ssize_t n = ::read( fd_, tmp, sizeof( tmp ) );
		if ( n > 0 ) {
			buf.append( tmp, static_cast<size_t>( n ) );
			if ( buf.size() > ( size_t( 64 ) << 20 ) ) break;
			continue;
		}
		if ( n < 0 && ( errno == EAGAIN || errno == EWOULDBLOCK ) ) return true;
		if ( n < 0 && errno == EINTR ) continue;
		// EOF or error
		if ( closed_.exchange( true ) ) return false;
		epoll_ctl( epoll_fd_, EPOLL_CTL_DEL, fd_, nullptr );
		::close( fd_ );
		return false;
	}
	return true;
}

bool Connection::write_all( const char* data, size_t len ) {
	std::lock_guard<std::mutex> lock( mtx_ );
	if ( closed_.load() ) return false;
	size_t written = 0;
	int stalls = 0;
	while ( written < len ) {
		const ssize_t n = ::send( fd_, data + written, len - written, MSG_NOSIGNAL );
		if ( n > 0 ) {
			written += static_cast<size_t>( n );
			stalls = 0;
			continue;
		}
		if ( n < 0 && errno == EINTR ) continue;
		if ( n < 0 && ( errno == EAGAIN || errno == EWOULDBLOCK ) ) {
			pollfd p = { fd_, POLLOUT, 0 };
			if ( ::poll( &p, 1, 1000 ) > 0 ) continue;
			if ( ++stalls < 30 ) continue; // a peer that accepts nothing for 30 s is dropped
		}
		if ( !closed_.exchange( true ) ) {
			epoll_ctl( epoll_fd_, EPOLL_CTL_DEL, fd_, nullptr );
			::close( fd_ );
		}
		return false;
	}
	return true;
}

void Connection::close_now() {
	std::lock_guard<std::mutex> lock( mtx_ );
	if ( closed_.exchange( true ) ) return;
	epoll_ctl( epoll_fd_, EPOLL_CTL_DEL, fd_, nullptr );
	::shutdown( fd_, SHUT_WR );
	::close( fd_ );
}

const char* http_reason( int status ) {
	switch ( status ) {
		case 200: return "OK";
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

void send_response( const ConnPtr& c, int status, const std::string& content_type, const std::string& body ) {
	std::string head = "HTTP/1.1 " + std::to_string( status ) + " " + http_reason( status ) + "\r\nContent-Type: " + content_type +
		"\r\nContent-Length: " + std::to_string( body.size() ) + "\r\nConnection: close\r\n\r\n";
	c->write_all( head + body );
	c->close_now();
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

bool start_sse( const ConnPtr& c ) {
	static const char head[] = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nX-Accel-Buffering: no\r\n\r\n";
	return c->write_all( head, sizeof( head ) - 1 );
}

} // namespace engine::server
