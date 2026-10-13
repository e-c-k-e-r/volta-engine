#include "engine/server/server.h"
#include "engine/core/log.h"
#include "engine/runtime/options.h"

#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>

#include <sys/epoll.h>
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

namespace {

std::string lower( std::string s ) {
	for ( char& c : s ) c = static_cast<char>( std::tolower( static_cast<unsigned char>( c ) ) );
	return s;
}

} // namespace

VoltaServer::VoltaServer() = default;

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
#if defined(USE_ROCM)
	std::printf( "   Target: AMD RDNA (ROCm), tensor parallel              \n" );
#else
	std::printf( "   Target: NVIDIA SM70 / SM86 / SM89, tensor parallel    \n" );
#endif
	std::printf( "========================================================\n" );
	std::fflush( stdout );

	std::string dir = config_.model_dir;
	while ( dir.size() > 1 && dir.back() == '/' ) dir.pop_back();
	const size_t slash = dir.find_last_of( '/' );
	model_name_ = slash == std::string::npos ? dir : dir.substr( slash + 1 );
	if ( config_.model_type == "minimax_h3" ) {
		// video + audio generation: the job service owns the GPUs, there is no language-model runtime
		if ( !init_video() ) return false;
	} else {
		generator_ = std::make_unique<engine::runtime::Generator>();
		if ( !generator_->init( config_ ) ) {
			LOG_ERROR( "Failed to initialize Generator runtime.\n" );
			return false;
		}
		openai_ = make_openai_protocol();
		anthropic_ = make_anthropic_protocol();
	}
	register_routes();

	if ( !setup_socket() ) {
		return false;
	}

	LOG_INFO( "Volta Server listening on http://%s:%d\n", config_.host.c_str(), config_.port );
	std::fflush( stdout );
	return true;
}

bool VoltaServer::init_video() {
#ifdef ENGINE_BACKEND_CUDA
	video_ = std::make_unique<VideoService>();
	if ( video_->init( config_ ) ) return true;
	LOG_ERROR( "Failed to initialize the video service.\n" );
#else
	LOG_ERROR( "MiniMax-H3 video generation needs the CUDA backend.\n" );
#endif
	return false;
}

bool VoltaServer::setup_socket() {
	listen_fd_ = socket( AF_INET, SOCK_STREAM, 0 );
	if ( listen_fd_ < 0 ) {
		LOG_ERROR( "socket() failed: %s\n", std::strerror( errno ) );
		return false;
	}

	int opt = 1;
	setsockopt( listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof( opt ) );
	setsockopt( listen_fd_, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof( opt ) );

	sockaddr_in addr;
	std::memset( &addr, 0, sizeof( addr ) );
	addr.sin_family = AF_INET;
	addr.sin_port = htons( static_cast<uint16_t>( config_.port ) );
	inet_pton( AF_INET, config_.host.c_str(), &addr.sin_addr );

	if ( bind( listen_fd_, reinterpret_cast<sockaddr*>( &addr ), sizeof( addr ) ) < 0 ) {
		LOG_ERROR( "bind() failed on %s:%d: %s\n", config_.host.c_str(), config_.port, std::strerror( errno ) );
		close( listen_fd_ );
		listen_fd_ = -1;
		return false;
	}
	if ( listen( listen_fd_, 128 ) < 0 ) {
		LOG_ERROR( "listen() failed: %s\n", std::strerror( errno ) );
		close( listen_fd_ );
		listen_fd_ = -1;
		return false;
	}
	fcntl( listen_fd_, F_SETFL, fcntl( listen_fd_, F_GETFL, 0 ) | O_NONBLOCK );

	epoll_fd_ = epoll_create1( 0 );
	if ( epoll_fd_ < 0 ) {
		LOG_ERROR( "epoll_create1() failed: %s\n", std::strerror( errno ) );
		close( listen_fd_ );
		listen_fd_ = -1;
		return false;
	}
	epoll_event ev = {};
	ev.events = EPOLLIN;
	ev.data.fd = listen_fd_;
	if ( epoll_ctl( epoll_fd_, EPOLL_CTL_ADD, listen_fd_, &ev ) < 0 ) {
		LOG_ERROR( "epoll_ctl(ADD listen_fd) failed: %s\n", std::strerror( errno ) );
		close( epoll_fd_ );
		close( listen_fd_ );
		epoll_fd_ = -1;
		listen_fd_ = -1;
		return false;
	}
	return true;
}

void VoltaServer::handle_new_connection() {
	for ( ;; ) {
		const int fd = accept( listen_fd_, nullptr, nullptr );
		if ( fd < 0 ) break;
		int opt = 1;
		setsockopt( fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof( opt ) );
		fcntl( fd, F_SETFL, fcntl( fd, F_GETFL, 0 ) | O_NONBLOCK );
		epoll_event ev = {};
		ev.events = EPOLLIN | EPOLLRDHUP;
		ev.data.fd = fd;
		if ( epoll_ctl( epoll_fd_, EPOLL_CTL_ADD, fd, &ev ) < 0 ) {
			close( fd );
			continue;
		}
		conns_[fd] = std::make_shared<Connection>( fd, epoll_fd_ );
	}
}

void VoltaServer::sweep_closed() {
	// Connections that never complete a request are dropped (request_timeout_s since the last byte)
	const auto now = std::chrono::steady_clock::now();
	for ( auto it = conns_.begin(); it != conns_.end(); ) {
		if ( !it->second->closed() && !it->second->dispatched && now - it->second->last_activity > std::chrono::seconds( config_.request_timeout_s ) ) {
			send_error( it->second, 408, "invalid_request_error", "timed out waiting for the request" );
		}
		if ( it->second->closed() ) it = conns_.erase( it );
		else ++it;
	}
}

void VoltaServer::handle_readable( int fd ) {
	auto it = conns_.find( fd );
	if ( it == conns_.end() ) return;
	ConnPtr conn = it->second;
	if ( !conn->read_some( conn->in ) ) return;
	if ( conn->dispatched ) {
		conn->in.clear();
		return;
	}
	HttpRequest req;
	if ( parse_request( conn, &req ) ) {
		conn->dispatched = true;
		conn->in.clear();
		dispatch( conn, req );
	}
}

bool VoltaServer::parse_request( const ConnPtr& conn, HttpRequest* req ) {
	const size_t header_end = conn->in.find( "\r\n\r\n" );
	if ( header_end == std::string::npos ) {
		if ( conn->in.size() > MAX_HEADER_BYTES ) send_error( conn, 413, "invalid_request_error", "request headers too large" );
		return false;
	}
	const std::string head = conn->in.substr( 0, header_end );
	const size_t line_end = head.find( "\r\n" );
	const std::string line = head.substr( 0, line_end );
	char method[16] = { 0 }, target[2048] = { 0 };
	if ( std::sscanf( line.c_str(), "%15s %2047s", method, target ) != 2 ) {
		send_error( conn, 400, "invalid_request_error", "malformed request line" );
		return false;
	}
	size_t content_length = 0;
	bool expect_continue = false;
	size_t pos = line_end == std::string::npos ? head.size() : line_end + 2;
	while ( pos < head.size() ) {
		size_t e = head.find( "\r\n", pos );
		if ( e == std::string::npos ) e = head.size();
		const std::string h = head.substr( pos, e - pos );
		pos = e + 2;
		const size_t colon = h.find( ':' );
		if ( colon == std::string::npos ) continue;
		const std::string key = lower( h.substr( 0, colon ) );
		size_t vb = colon + 1;
		while ( vb < h.size() && ( h[vb] == ' ' || h[vb] == '\t' ) ) ++vb;
		const std::string value = h.substr( vb );
		req->headers[key] = value;
		if ( key == "content-length" ) {
			content_length = static_cast<size_t>( std::strtoull( value.c_str(), nullptr, 10 ) );
		}
		else if ( key == "transfer-encoding" && lower( value ).find( "chunked" ) != std::string::npos ) {
			send_error( conn, 501, "invalid_request_error", "chunked request bodies are not supported; send Content-Length" );
			return false;
		}
		else if ( key == "expect" && lower( value ) == "100-continue" ) {
			expect_continue = true;
		}
	}
	if ( content_length > MAX_BODY_BYTES ) {
		send_error( conn, 413, "invalid_request_error", "request body too large" );
		return false;
	}
	const size_t body_start = header_end + 4;
	if ( conn->in.size() - body_start < content_length ) {
		if ( expect_continue && !conn->sent_continue ) {
			conn->sent_continue = true;
			conn->write_all( "HTTP/1.1 100 Continue\r\n\r\n" );
		}
		return false;
	}
	req->method = method;
	req->path = target;
	const size_t q = req->path.find( '?' );
	if ( q != std::string::npos ) {
		req->query = req->path.substr( q + 1 );
		req->path.erase( q );
	}
	req->body = conn->in.substr( body_start, content_length );
	return true;
}

void VoltaServer::register_routes() {
#ifdef ENGINE_BACKEND_CUDA
	if ( video_ ) { // the video service answers /health and /metrics itself; the chat and monitor routes need a language-model runtime
		video_->register_routes( router_ );
		return;
	}
#endif
	router_.add( "GET", "/health", [this]( const ConnPtr& c, const HttpRequest& ) { handle_health( c ); } );
	router_.add( "GET", "/", [this]( const ConnPtr& c, const HttpRequest& ) { handle_health( c ); } );
	router_.add( "GET", "/metrics", [this]( const ConnPtr& c, const HttpRequest& ) { handle_metrics( c ); } );
	router_.add( "GET", "/metrics/prometheus", [this]( const ConnPtr& c, const HttpRequest& ) { handle_metrics_prometheus( c ); } );
	router_.add( "GET", "/monitor", [this]( const ConnPtr& c, const HttpRequest& ) { handle_monitor( c ); } );
	router_.add( "GET", "/monitor/", [this]( const ConnPtr& c, const HttpRequest& ) { handle_monitor( c ); } );
	router_.add( "GET", "/api/stats", [this]( const ConnPtr& c, const HttpRequest& ) { handle_api_stats( c ); } );
	router_.add( "GET", "/api/experts", [this]( const ConnPtr& c, const HttpRequest& ) { handle_api_experts( c ); } );
	router_.add( "GET", "/v1/models", [this]( const ConnPtr& c, const HttpRequest& ) { handle_models( c ); } );
	router_.add( "*", "/v1/chat/completions", [this]( const ConnPtr& c, const HttpRequest& r ) {
		if ( r.method != "POST" ) return send_error( c, 405, "invalid_request_error", "use POST" );
		handle_chat( c, r, *openai_ );
	} );
	if ( anthropic_ ) {
		router_.add( "*", "/v1/messages", [this]( const ConnPtr& c, const HttpRequest& r ) {
			if ( r.method != "POST" ) return send_error( c, 405, "invalid_request_error", "use POST", true );
			handle_chat( c, r, *anthropic_ );
		} );
		router_.add( "*", "/v1/messages/count_tokens", [this]( const ConnPtr& c, const HttpRequest& r ) {
			if ( r.method != "POST" ) return send_error( c, 405, "invalid_request_error", "use POST", true );
			handle_count_tokens( c, r, *anthropic_ );
		} );
	}
}

void VoltaServer::dispatch( const ConnPtr& conn, const HttpRequest& req ) {
	if ( !router_.dispatch( conn, req ) ) send_error( conn, 404, "not_found_error", "unknown endpoint " + req.method + " " + req.path );
}

void VoltaServer::handle_health( const ConnPtr& conn ) {
	send_json( conn, 200, "{\"status\":\"ok\"}" );
}

void VoltaServer::handle_models( const ConnPtr& conn ) {
	nlohmann::json m = { { "object", "list" }, { "data", nlohmann::json::array( { { { "id", model_name_ }, { "object", "model" }, { "created", 0 }, { "owned_by", "volta" },
		{ "context_length", generator_->max_context() } } } ) } };
	send_json( conn, 200, m.dump() );
}

void VoltaServer::handle_metrics( const ConnPtr& conn ) {
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
	if ( metrics.prefix_enabled ) {
		m["prefix_cache"] = { { "lookups", metrics.prefix.lookups }, { "hits", metrics.prefix.hits }, { "hit_tokens", metrics.prefix.hit_tokens },
			{ "inserts", metrics.prefix.inserts }, { "evictions", metrics.prefix.evictions }, { "entries", metrics.prefix.entries }, { "bytes", metrics.prefix.bytes } };
	}
	send_json( conn, 200, m.dump( 2 ) + "\n" );
}

void VoltaServer::handle_metrics_prometheus( const ConnPtr& conn ) {
	const auto m = generator_->get_metrics();
	char buf[4096];
	int n = 0;
	auto line = [&]( const char* name, const char* type, double v, const char* labels = "" ) {
		n += std::snprintf( buf + n, sizeof( buf ) - static_cast<size_t>( n ), "# TYPE %s %s\n%s%s %.6g\n", name, type, name, labels, v );
	};
	line( "volta_uptime_seconds", "gauge", m.uptime_seconds );
	line( "volta_active_slots", "gauge", m.active_slots );
	line( "volta_requests_total", "counter", static_cast<double>( m.total_requests ) );
	line( "volta_ttft_milliseconds_avg", "gauge", m.avg_ttft_ms );
	line( "volta_prefill_tokens_per_second_avg", "gauge", m.avg_prefill_tok_s );
	line( "volta_decode_tokens_per_second_avg", "gauge", m.avg_decode_tok_s );
	line( "volta_speculation_acceptance_ratio", "gauge", m.spec_acceptance_rate );
	if ( m.prefix_enabled ) {
		line( "volta_prefix_cache_lookups_total", "counter", static_cast<double>( m.prefix.lookups ) );
		line( "volta_prefix_cache_hits_total", "counter", static_cast<double>( m.prefix.hits ) );
		line( "volta_prefix_cache_hit_tokens_total", "counter", static_cast<double>( m.prefix.hit_tokens ) );
		line( "volta_prefix_cache_evictions_total", "counter", static_cast<double>( m.prefix.evictions ) );
		line( "volta_prefix_cache_entries", "gauge", static_cast<double>( m.prefix.entries ) );
		line( "volta_prefix_cache_bytes", "gauge", static_cast<double>( m.prefix.bytes ) );
	}
	send_response( conn, 200, "text/plain; version=0.0.4", std::string( buf, static_cast<size_t>( n ) ) );
}

void VoltaServer::handle_chat( const ConnPtr& conn, const HttpRequest& req, ChatProtocol& protocol ) {
	const bool anthropic = protocol.anthropic_errors();
	ChatRequest cr;
	ApiError err;
	if ( !runtime::options().dump_requests.empty() ) {
		const std::string path = runtime::options().dump_requests + "/req_" + std::to_string( next_req_id_.load() ) + ".json";
		if ( FILE* f = std::fopen( path.c_str(), "wb" ) ) {
			std::fwrite( req.body.data(), 1, req.body.size(), f );
			std::fclose( f );
		}
	}
	if ( !protocol.parse( req.body, &cr, &err ) ) return send_error( conn, err.status, err.type, err.message, anthropic );
	if ( cr.model.empty() ) cr.model = model_name_;
	const uint64_t id = next_req_id_++;
	std::shared_ptr<Responder> responder = protocol.make_responder( cr, conn, id );
	if ( !start_chat( *generator_, config_, std::move( cr ), id, std::move( responder ), conn, &err ) ) {
		send_error( conn, err.status, err.type, err.message, anthropic );
	}
}

void VoltaServer::handle_count_tokens( const ConnPtr& conn, const HttpRequest& req, ChatProtocol& protocol ) {
	ChatRequest cr;
	ApiError err;
	int tokens = 0;
	if ( !protocol.parse( req.body, &cr, &err ) || !count_chat_tokens( *generator_, config_, cr, &tokens, &err ) ) {
		return send_error( conn, err.status, err.type, err.message, true );
	}
	send_json( conn, 200, nlohmann::json( { { "input_tokens", tokens } } ).dump() );
}

void VoltaServer::run() {
	running_.store( true );
	while ( !g_stop_flag && running_.load() ) {
		epoll_event events[64];
		const int nfds = epoll_wait( epoll_fd_, events, 64, 5 );
		for ( int i = 0; i < nfds; ++i ) {
			const int fd = events[i].data.fd;
			if ( fd == listen_fd_ ) {
				handle_new_connection();
				continue;
			}
			if ( events[i].events & ( EPOLLHUP | EPOLLERR | EPOLLRDHUP ) ) {
				auto it = conns_.find( fd );
				if ( it != conns_.end() ) it->second->close_now();
				continue;
			}
			if ( events[i].events & EPOLLOUT ) {
				auto it = conns_.find( fd );
				if ( it != conns_.end() ) it->second->on_writable();
			}
			if ( events[i].events & EPOLLIN ) handle_readable( fd );
		}
		sweep_closed();
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

	LOG_INFO( "\n[*] Initiating Volta Server graceful shutdown...\n" );

	for ( auto& kv : conns_ ) kv.second->close_now();
	conns_.clear();
	if ( epoll_fd_ >= 0 ) {
		close( epoll_fd_ );
		epoll_fd_ = -1;
	}
	if ( listen_fd_ >= 0 ) {
		close( listen_fd_ );
		listen_fd_ = -1;
	}

#ifdef ENGINE_BACKEND_CUDA
	if ( video_ ) video_->stop();
#endif
	if ( generator_ ) {
		generator_->shutdown();
		generator_.reset();
	}

	LOG_INFO( "Server shutdown complete. All GPUs reset to 0 MiB VRAM.\n" );
	std::fflush( stdout );
	std::_Exit( 0 );
}

} // namespace engine::server
