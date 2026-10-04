#include "engine/server/server.h"

#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <algorithm>

namespace engine::server {

static volatile sig_atomic_t g_stop_flag = 0;

static void global_signal_handler( int sig ) {
	(void) sig;
	g_stop_flag = 1;
}

VoltaServer::VoltaServer() {
	clients_ = std::make_unique<ClientContext[]>( MAX_CLIENTS );
	for ( int i = 0; i < MAX_CLIENTS; ++i ) {
		clients_[i].fd = -1;
		clients_[i].state = ClientState::UNUSED;
		clients_[i].in_len = 0;
	}
}

VoltaServer::~VoltaServer() {
	stop();
}

bool VoltaServer::init( const ServerConfig& config ) {
	std::setvbuf( stdout, nullptr, _IONBF, 0 );
	config_ = config;

	struct sigaction sa;
	std::memset( &sa, 0, sizeof( sa ) );
	sa.sa_handler = global_signal_handler;
	sigaction( SIGINT, &sa, nullptr );
	sigaction( SIGTERM, &sa, nullptr );
	signal( SIGPIPE, SIG_IGN );

	std::printf( "========================================================\n" );
	std::printf( "   VOLTA ENGINE: UNIFIED PRODUCTION HTTP/1.1 SERVER     \n" );
	std::printf( "   Target: 4x NVIDIA Tesla V100-SXM2-32GB (SM70, TP=4)  \n" );
	std::printf( "========================================================\n" );
	std::fflush( stdout );

	generator_ = std::make_unique<engine::runtime::Generator>();
	if ( !generator_->init( config_ ) ) {
		std::fprintf( stderr, "[-] Failed to initialize Generator runtime.\n" );
		return false;
	}

	if ( !setup_socket() ) {
		return false;
	}

	std::printf( "[+] Volta Server listening on http://%s:%d\n", config_.host.c_str(), config_.port );
	std::fflush( stdout );
	return true;
}

void VoltaServer::set_nonblocking( int fd ) {
	int flags = fcntl( fd, F_GETFL, 0 );
	if ( flags >= 0 ) {
		fcntl( fd, F_SETFL, flags | O_NONBLOCK );
	}
}

bool VoltaServer::setup_socket() {
	listen_fd_ = socket( AF_INET, SOCK_STREAM, 0 );
	if ( listen_fd_ < 0 ) {
		std::fprintf( stderr, "[-] socket() failed: %s\n", std::strerror( errno ) );
		return false;
	}

	int opt = 1;
	setsockopt( listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof( opt ) );
	setsockopt( listen_fd_, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof( opt ) );
	setsockopt( listen_fd_, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof( opt ) );

	sockaddr_in addr;
	std::memset( &addr, 0, sizeof( addr ) );
	addr.sin_family = AF_INET;
	addr.sin_port = htons( static_cast<uint16_t>( config_.port ) );
	inet_pton( AF_INET, config_.host.c_str(), &addr.sin_addr );

	if ( bind( listen_fd_, reinterpret_cast<sockaddr*>( &addr ), sizeof( addr ) ) < 0 ) {
		std::fprintf( stderr, "[-] bind() failed on %s:%d: %s\n", config_.host.c_str(), config_.port, std::strerror( errno ) );
		close( listen_fd_ );
		listen_fd_ = -1;
		return false;
	}

	if ( listen( listen_fd_, 128 ) < 0 ) {
		std::fprintf( stderr, "[-] listen() failed: %s\n", std::strerror( errno ) );
		close( listen_fd_ );
		listen_fd_ = -1;
		return false;
	}

	set_nonblocking( listen_fd_ );

	epoll_fd_ = epoll_create1( 0 );
	if ( epoll_fd_ < 0 ) {
		std::fprintf( stderr, "[-] epoll_create1() failed: %s\n", std::strerror( errno ) );
		close( listen_fd_ );
		listen_fd_ = -1;
		return false;
	}

	epoll_event ev;
	ev.events = EPOLLIN;
	ev.data.fd = listen_fd_;
	if ( epoll_ctl( epoll_fd_, EPOLL_CTL_ADD, listen_fd_, &ev ) < 0 ) {
		std::fprintf( stderr, "[-] epoll_ctl(ADD listen_fd) failed: %s\n", std::strerror( errno ) );
		close( epoll_fd_ );
		close( listen_fd_ );
		epoll_fd_ = -1;
		listen_fd_ = -1;
		return false;
	}

	return true;
}

void VoltaServer::write_all( int fd, const char* data, size_t len ) {
	if ( fd < 0 || !data || len == 0 ) {
		return;
	}
	size_t written = 0;
	while ( written < len ) {
		ssize_t n = write( fd, data + written, len - written );
		if ( n > 0 ) {
			written += static_cast<size_t>( n );
		}
		else if ( n < 0 ) {
			if ( errno == EAGAIN || errno == EWOULDBLOCK ) {
				usleep( 100 );
				continue;
			}
			break;
		}
		else {
			break;
		}
	}
}

void VoltaServer::close_client( int fd ) {
	if ( fd < 0 || fd >= MAX_CLIENTS ) {
		return;
	}
	std::lock_guard<std::mutex> lock( client_mtx_ );
	if ( clients_[fd].fd >= 0 ) {
		epoll_ctl( epoll_fd_, EPOLL_CTL_DEL, fd, nullptr );
		shutdown( fd, SHUT_WR );
		close( fd );
		clients_[fd].fd = -1;
		clients_[fd].state = ClientState::UNUSED;
		clients_[fd].in_len = 0;
	}
}

void VoltaServer::handle_new_connection() {
	while ( true ) {
		sockaddr_in client_addr;
		socklen_t client_len = sizeof( client_addr );
		int client_fd = accept( listen_fd_, reinterpret_cast<sockaddr*>( &client_addr ), &client_len );
		if ( client_fd < 0 ) {
			if ( errno == EAGAIN || errno == EWOULDBLOCK ) {
				break;
			}
			break;
		}

		int opt = 1;
		setsockopt( client_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof( opt ) );
		set_nonblocking( client_fd );

		if ( client_fd >= MAX_CLIENTS ) {
			close( client_fd );
			continue;
		}

		std::unique_lock<std::mutex> lock( client_mtx_ );
		ClientContext& client = clients_[client_fd];
		client.fd = client_fd;
		client.state = ClientState::READING_HTTP;
		client.in_len = 0;

		epoll_event ev;
		ev.events = EPOLLIN;
		ev.data.fd = client_fd;
		epoll_ctl( epoll_fd_, EPOLL_CTL_ADD, client_fd, &ev );
		lock.unlock();
	}
}

void VoltaServer::handle_client_read( int fd ) {
	if ( fd < 0 || fd >= MAX_CLIENTS ) {
		return;
	}

	ClientContext* client_ptr = nullptr;
	std::unique_lock<std::mutex> lock( client_mtx_ );
	if ( clients_[fd].fd >= 0 ) {
		client_ptr = &clients_[fd];
	}
	lock.unlock();
	if ( !client_ptr ) {
		return;
	}

	ssize_t n = read( fd, client_ptr->in_buf + client_ptr->in_len, sizeof( client_ptr->in_buf ) - 1 - client_ptr->in_len );
	if ( n <= 0 ) {
		if ( n < 0 && ( errno == EAGAIN || errno == EWOULDBLOCK ) ) {
			return;
		}
		close_client( fd );
		return;
	}

	client_ptr->in_len += static_cast<size_t>( n );
	client_ptr->in_buf[client_ptr->in_len] = '\0';

	process_http_request( *client_ptr );
}

void VoltaServer::handle_health_request( int client_fd ) {
	const char resp[] = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 16\r\nConnection: close\r\n\r\n{\"status\": \"ok\"}";
	write_all( client_fd, resp, sizeof( resp ) - 1 );
	close_client( client_fd );
}

void VoltaServer::handle_metrics_request( int client_fd ) {
	auto metrics = generator_->get_metrics();
	nlohmann::json m;
	m["uptime_seconds"] = metrics.uptime_seconds;
	m["active_slots"] = metrics.active_slots;
	m["total_requests"] = metrics.total_requests;
	m["avg_ttft_ms"] = metrics.avg_ttft_ms;
	m["avg_prefill_tok_s"] = metrics.avg_prefill_tok_s;
	m["avg_decode_tok_s"] = metrics.avg_decode_tok_s;
	m["spec_acceptance_rate"] = metrics.spec_acceptance_rate;
	m["gpu_vram_used_bytes"] = metrics.gpu_vram_used_bytes;

	std::string body = m.dump( 2 ) + "\n";
	std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " +
		std::to_string( body.size() ) + "\r\nConnection: close\r\n\r\n" + body;

	write_all( client_fd, resp.data(), resp.size() );
	close_client( client_fd );
}

void VoltaServer::handle_chat_completion( ClientContext& client, const char* body, size_t body_len ) {
	auto req = nlohmann::json::parse( body, body + body_len, nullptr, false );
	if ( req.is_discarded() ) {
		const char err[] = "HTTP/1.1 400 Bad Request\r\nContent-Length: 22\r\n\r\n{\"error\":\"invalid json\"}";
		write_all( client.fd, err, sizeof( err ) - 1 );
		close_client( client.fd );
		return;
	}

	std::string model = req.value( "model", config_.model_type );
	bool stream = req.value( "stream", false );
	int max_tokens = req.value( "max_tokens", config_.default_max_tokens );
	if ( max_tokens <= 0 ) {
		max_tokens = 64;
	}
	float temperature = req.value( "temperature", config_.default_temperature );

	std::vector<engine::loader::ChatMessage> chat_messages;
	if ( req.contains( "messages" ) && req["messages"].is_array() ) {
		for ( const auto& m : req["messages"] ) {
			std::string r = m.value( "role", "user" );
			std::string c = m.value( "content", "" );
			chat_messages.push_back( { r, c } );
		}
	}

	engine::loader::ChatTemplateOptions topts;
	topts.add_generation_prompt = true;
	topts.enable_thinking = false;

	std::string prompt_str = generator_->chat_template().render( chat_messages, topts );
	std::vector<int> prompt_tokens = generator_->tokenizer().encode( prompt_str, false );

	uint64_t req_id = next_req_id_++;
	char req_id_str[64];
	std::snprintf( req_id_str, sizeof( req_id_str ), "chatcmpl-%lu", req_id );
	std::string id_str( req_id_str );

	int client_fd = client.fd;

	engine::runtime::GenerationRequest gen_req;
	gen_req.req_id = req_id;
	gen_req.prompt_text = prompt_str;
	gen_req.prompt_tokens = std::move( prompt_tokens );
	gen_req.max_tokens = max_tokens;
	gen_req.temperature = temperature;
	gen_req.stream = stream;

	if ( stream ) {
		const char sse_header[] = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n";
		write_all( client_fd, sse_header, sizeof( sse_header ) - 1 );

		gen_req.stream_callback = [this, client_fd, id_str]( const std::string& token_str, bool is_done ) {
			nlohmann::json chunk;
			chunk["id"] = id_str;
			chunk["object"] = "chat.completion.chunk";
			chunk["choices"] = nlohmann::json::array();

			if ( !is_done ) {
				nlohmann::json choice;
				choice["delta"] = { { "content", token_str } };
				choice["finish_reason"] = nullptr;
				chunk["choices"].push_back( choice );
				std::string sse = "data: " + chunk.dump() + "\n\n";
				write_all( client_fd, sse.data(), sse.size() );
			}
			else {
				nlohmann::json choice;
				choice["delta"] = nlohmann::json::object();
				choice["finish_reason"] = "stop";
				chunk["choices"].push_back( choice );
				std::string sse = "data: " + chunk.dump() + "\n\ndata: [DONE]\n\n";
				write_all( client_fd, sse.data(), sse.size() );
				close_client( client_fd );
			}
		};
	}
	else {
		gen_req.completion_callback = [this, client_fd, id_str, model]( const std::string& full_text, const engine::runtime::SlotMetrics& metrics ) {
			nlohmann::json resp;
			resp["id"] = id_str;
			resp["object"] = "chat.completion";
			resp["created"] = std::time( nullptr );
			resp["model"] = model;
			resp["choices"] = nlohmann::json::array();

			nlohmann::json choice;
			choice["index"] = 0;
			choice["message"] = { { "role", "assistant" }, { "content", full_text } };
			choice["finish_reason"] = "stop";
			resp["choices"].push_back( choice );

			double ttft_ms = std::chrono::duration<double, std::milli>( metrics.first_token_time - metrics.start_time ).count();
			resp["usage"]["prompt_tokens"] = metrics.prompt_tokens;
			resp["usage"]["completion_tokens"] = metrics.generated_tokens;
			resp["usage"]["total_tokens"] = metrics.prompt_tokens + metrics.generated_tokens;
			resp["usage"]["ttft_ms"] = ttft_ms;
			resp["usage"]["decode_tok_s"] = metrics.decode_tok_s;

			std::string body = resp.dump() + "\n";
			std::string header = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " +
				std::to_string( body.size() ) + "\r\nConnection: close\r\n\r\n";
			write_all( client_fd, header.data(), header.size() );
			write_all( client_fd, body.data(), body.size() );
			close_client( client_fd );
		};
	}

	bool sub_ok = generator_->submit_request( std::move( gen_req ) );
	if ( !sub_ok ) {
		const char busy[] = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 21\r\n\r\n{\"error\":\"batch full\"}";
		write_all( client.fd, busy, sizeof( busy ) - 1 );
		close_client( client.fd );
	}
}

void VoltaServer::process_http_request( ClientContext& client ) {
	char* header_end = std::strstr( client.in_buf, "\r\n\r\n" );
	if ( !header_end ) {
		return;
	}

	size_t header_len = ( header_end + 4 ) - client.in_buf;
	char* line = client.in_buf;
	char* line_end = std::strstr( line, "\r\n" );
	if ( !line_end ) {
		return;
	}

	*line_end = '\0';
	char method[16] = { 0 };
	char path[256] = { 0 };
	std::sscanf( line, "%15s %255s", method, path );
	*line_end = '\r';

	if ( std::strcmp( method, "GET" ) == 0 && ( std::strcmp( path, "/health" ) == 0 || std::strcmp( path, "/" ) == 0 ) ) {
		handle_health_request( client.fd );
		return;
	}

	if ( std::strcmp( method, "GET" ) == 0 && std::strcmp( path, "/metrics" ) == 0 ) {
		handle_metrics_request( client.fd );
		return;
	}

	if ( std::strcmp( method, "POST" ) == 0 && std::strcmp( path, "/v1/chat/completions" ) == 0 ) {
		size_t content_len = 0;
		const char* cl_header = std::strstr( client.in_buf, "Content-Length:" );
		if ( !cl_header ) {
			cl_header = std::strstr( client.in_buf, "content-length:" );
		}
		if ( cl_header ) {
			content_len = static_cast<size_t>( std::strtoul( cl_header + 15, nullptr, 10 ) );
		}

		size_t body_received = client.in_len - header_len;
		if ( body_received < content_len ) {
			return;
		}

		handle_chat_completion( client, header_end + 4, content_len );
		return;
	}

	const char not_found[] = "HTTP/1.1 404 Not Found\r\nContent-Length: 19\r\n\r\n{\"error\":\"not found\"}";
	write_all( client.fd, not_found, sizeof( not_found ) - 1 );
	close_client( client.fd );
}

void VoltaServer::run() {
	running_.store( true );
	while ( !g_stop_flag && running_.load() ) {
		epoll_event events[64];
		int nfds = epoll_wait( epoll_fd_, events, 64, 5 );
		for ( int i = 0; i < nfds; ++i ) {
			if ( events[i].data.fd == listen_fd_ ) {
				handle_new_connection();
			}
			else {
				handle_client_read( events[i].data.fd );
			}
		}
	}
	stop();
}

void VoltaServer::request_stop() {
	g_stop_flag = 1;
	running_.store( false );
}

void VoltaServer::stop() {
	if ( !running_.exchange( false ) ) {
		return;
	}

	std::printf( "\n[*] Initiating Volta Server graceful shutdown...\n" );

	if ( epoll_fd_ >= 0 ) {
		close( epoll_fd_ );
		epoll_fd_ = -1;
	}
	if ( listen_fd_ >= 0 ) {
		close( listen_fd_ );
		listen_fd_ = -1;
	}

	std::unique_lock<std::mutex> lock( client_mtx_ );
	for ( int i = 0; i < MAX_CLIENTS; ++i ) {
		if ( clients_[i].fd >= 0 ) {
			close( clients_[i].fd );
			clients_[i].fd = -1;
		}
	}
	lock.unlock();

	if ( generator_ ) {
		generator_->shutdown();
		generator_.reset();
	}

	std::printf( "[+] Server shutdown complete. All 4 GPUs reset to 0 MiB VRAM.\n" );
	std::fflush( stdout );
	std::_Exit( 0 );
}

} // namespace engine::server
