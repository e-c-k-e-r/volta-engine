#include "engine/server/tui/http_get.h"

#include <cerrno>
#include <cstring>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>

namespace engine::server::tui {

namespace {

int connect_to( const std::string& host, int port, int timeout_ms, std::string* error ) {
	addrinfo hints = {};
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	addrinfo* res = nullptr;
	const int rc = getaddrinfo( host.c_str(), std::to_string( port ).c_str(), &hints, &res );
	if ( rc != 0 ) {
		*error = "cannot resolve " + host + ": " + gai_strerror( rc );
		return -1;
	}
	int fd = -1;
	std::string last = "no address";
	for ( addrinfo* a = res; a; a = a->ai_next ) {
		fd = socket( a->ai_family, a->ai_socktype, a->ai_protocol );
		if ( fd < 0 ) continue;
		fcntl( fd, F_SETFL, fcntl( fd, F_GETFL, 0 ) | O_NONBLOCK );
		int c = connect( fd, a->ai_addr, a->ai_addrlen );
		if ( c != 0 && errno == EINPROGRESS ) {
			pollfd p = { fd, POLLOUT, 0 };
			if ( poll( &p, 1, timeout_ms ) == 1 ) {
				int err = 0;
				socklen_t len = sizeof( err );
				getsockopt( fd, SOL_SOCKET, SO_ERROR, &err, &len );
				c = err == 0 ? 0 : -1;
				errno = err;
			}
			else {
				c = -1;
				errno = ETIMEDOUT;
			}
		}
		if ( c == 0 ) break;
		last = std::strerror( errno );
		close( fd );
		fd = -1;
	}
	freeaddrinfo( res );
	if ( fd < 0 ) {
		*error = "cannot connect to " + host + ":" + std::to_string( port ) + ": " + last;
		return -1;
	}
	fcntl( fd, F_SETFL, fcntl( fd, F_GETFL, 0 ) & ~O_NONBLOCK );
	timeval tv = { timeout_ms / 1000, ( timeout_ms % 1000 ) * 1000 };
	setsockopt( fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof( tv ) );
	setsockopt( fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof( tv ) );
	return fd;
}

} // namespace

bool http_get( const std::string& host, int port, const std::string& path, int timeout_ms, std::string* body, std::string* error ) {
	const int fd = connect_to( host, port, timeout_ms, error );
	if ( fd < 0 ) return false;
	const std::string req = "GET " + path + " HTTP/1.0\r\nHost: " + host + "\r\nConnection: close\r\n\r\n";
	for ( size_t sent = 0; sent < req.size(); ) {
		const ssize_t n = send( fd, req.data() + sent, req.size() - sent, MSG_NOSIGNAL );
		if ( n <= 0 ) {
			*error = std::string( "send failed: " ) + std::strerror( errno );
			close( fd );
			return false;
		}
		sent += static_cast<size_t>( n );
	}
	std::string resp;
	char buf[8192];
	for ( ;; ) {
		const ssize_t n = recv( fd, buf, sizeof( buf ), 0 );
		if ( n > 0 ) { resp.append( buf, static_cast<size_t>( n ) ); continue; }
		if ( n < 0 && errno == EINTR ) continue;
		if ( n < 0 ) {
			*error = errno == EAGAIN ? std::string( "timed out waiting for the response" ) : std::string( "receive failed: " ) + std::strerror( errno );
			close( fd );
			return false;
		}
		break;
	}
	close( fd );
	if ( resp.compare( 0, 5, "HTTP/" ) != 0 ) {
		*error = "not an HTTP response";
		return false;
	}
	const size_t sp = resp.find( ' ' );
	const int status = sp == std::string::npos ? 0 : std::atoi( resp.c_str() + sp + 1 );
	const size_t split = resp.find( "\r\n\r\n" );
	if ( status != 200 || split == std::string::npos ) {
		*error = "HTTP status " + std::to_string( status ) + " from " + path;
		return false;
	}
	*body = resp.substr( split + 4 );
	return true;
}

} // namespace engine::server::tui
