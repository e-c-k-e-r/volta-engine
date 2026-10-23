#include "engine/server/tui/app.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <unistd.h>

namespace engine::server::tui {

namespace {

double now_ms() {
	return std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now().time_since_epoch() ).count();
}

constexpr size_t LOG_LINES = 2000;

// A captured line made printable: carriage-return progress keeps its last state, tabs become spaces, other control bytes go
std::string clean_line( const std::string& raw ) {
	const size_t cr = raw.rfind( '\r' );
	const std::string& s = cr == std::string::npos ? raw : raw.substr( cr + 1 );
	std::string out;
	for ( size_t i = 0; i < s.size(); ++i ) {
		const unsigned char c = static_cast<unsigned char>( s[i] );
		if ( c == '\t' ) out += "    ";
		else if ( c == 0x1b ) {   // drop colour codes the child code may print
			if ( i + 1 < s.size() && s[i + 1] == '[' ) {
				i += 2;
				while ( i < s.size() && !( s[i] >= 0x40 && s[i] <= 0x7e ) ) ++i;
			}
		}
		else if ( c >= 0x20 && c != 0x7f ) out += static_cast<char>( c );
	}
	return out;
}

} // namespace

void run_viewer( Terminal& term, const ViewerOptions& opt, const FetchFn& fetch, const LogFn& log, const std::atomic<bool>& stop ) {
	UiState ui;
	nlohmann::json stats, experts;
	bool have_stats = false, have_experts = false, connected = false;
	std::string error, experts_error;
	std::vector<std::string> lines;
	double last_fetch = -1.0e18, last_experts = -1.0e18;
	int last_rows = 0, last_cols = 0;
	bool dirty = true;
	while ( !stop.load() ) {
		if ( !ui.paused && now_ms() - last_fetch >= opt.interval_ms ) {
			nlohmann::json fresh;
			std::string err;
			if ( fetch( "stats", &fresh, &err ) ) {
				stats = std::move( fresh );
				have_stats = connected = true;
			}
			else {
				connected = false;
				error = err;
			}
			last_fetch = now_ms();
			dirty = true;
		}
		// the routing counters are a few hundred KB of JSON: only while their tab is open, and not every second
		if ( !ui.paused && ui.tab == TAB_EXPERTS && ( !have_experts || now_ms() - last_experts >= 2.0 * opt.interval_ms ) ) {
			nlohmann::json fresh;
			std::string err;
			if ( fetch( "experts", &fresh, &err ) ) {
				experts = std::move( fresh );
				have_experts = true;
				experts_error.clear();
			}
			else {
				experts_error = err;
			}
			last_experts = now_ms();
			dirty = true;
		}
		int rows = 0, cols = 0;
		term.size( &rows, &cols );
		if ( rows != last_rows || cols != last_cols ) {
			last_rows = rows;
			last_cols = cols;
			dirty = true;
		}
		if ( log && ui.tab == TAB_LOG && !ui.paused ) {
			const size_t before = lines.size();
			const std::string tail = lines.empty() ? std::string() : lines.back();
			log( &lines );
			if ( lines.size() != before || ( !lines.empty() && lines.back() != tail ) ) dirty = true;
		}
		if ( dirty ) {
			ViewData data;
			data.stats = have_stats ? &stats : nullptr;
			data.experts = have_experts ? &experts : nullptr;
			data.experts_error = experts_error;
			data.connected = connected;
			data.error = error;
			data.source = opt.source;
			data.log = log ? &lines : nullptr;
			term.write( render_frame( data, ui, rows, cols ) );
			dirty = false;
		}
		const int key = term.read_key( 100 );
		if ( key != KEY_NONE ) {
			if ( !apply_key( ui, key, std::max( 1, rows - 4 ) ) ) return;
			dirty = true;
		}
	}
}

bool StdioCapture::start() {
	if ( read_fd_ >= 0 ) return true;
	int fds[2];
	if ( pipe( fds ) != 0 ) return false;
	std::fflush( stdout );
	std::fflush( stderr );
	saved_out_ = dup( STDOUT_FILENO );
	saved_err_ = dup( STDERR_FILENO );
	if ( saved_out_ < 0 || saved_err_ < 0 ) {
		close( fds[0] );
		close( fds[1] );
		if ( saved_out_ >= 0 ) close( saved_out_ );
		if ( saved_err_ >= 0 ) close( saved_err_ );
		saved_out_ = saved_err_ = -1;
		return false;
	}
	dup2( fds[1], STDOUT_FILENO );
	dup2( fds[1], STDERR_FILENO );
	close( fds[1] );
	read_fd_ = fds[0];
	thread_ = std::thread( [this] { reader(); } );
	return true;
}

void StdioCapture::stop() {
	if ( read_fd_ < 0 ) return;
	std::fflush( stdout );
	std::fflush( stderr );
	dup2( saved_out_, STDOUT_FILENO );   // the last writer of the pipe goes away here, so the reader sees the end of the stream
	dup2( saved_err_, STDERR_FILENO );
	close( saved_out_ );
	close( saved_err_ );
	saved_out_ = saved_err_ = -1;
	if ( thread_.joinable() ) thread_.join();
	close( read_fd_ );
	read_fd_ = -1;
}

void StdioCapture::reader() {
	char buf[4096];
	for ( ;; ) {
		const ssize_t n = read( read_fd_, buf, sizeof( buf ) );
		if ( n < 0 && errno == EINTR ) continue;
		if ( n <= 0 ) break;
		std::lock_guard<std::mutex> lock( mu_ );
		for ( ssize_t i = 0; i < n; ++i ) {
			if ( buf[i] != '\n' ) { partial_ += buf[i]; continue; }
			lines_.push_back( clean_line( partial_ ) );
			partial_.clear();
		}
		if ( lines_.size() > LOG_LINES + 200 ) lines_.erase( lines_.begin(), lines_.begin() + static_cast<std::ptrdiff_t>( lines_.size() - LOG_LINES ) );
	}
}

void StdioCapture::snapshot( std::vector<std::string>* lines ) const {
	std::lock_guard<std::mutex> lock( mu_ );
	*lines = lines_;
	if ( !partial_.empty() ) lines->push_back( clean_line( partial_ ) );
}

bool ServerTui::start( FetchFn fetch, const std::string& source ) {
	if ( started_ ) return true;
	if ( !term_.open() ) return false;
	if ( !capture_.start() ) {
		term_.close();
		return false;
	}
	stop_.store( false );
	started_ = true;
	thread_ = std::thread( [this, fetch = std::move( fetch ), source] {
		ViewerOptions opt;
		opt.source = source;
		run_viewer( term_, opt, fetch, [this]( std::vector<std::string>* l ) { capture_.snapshot( l ); }, stop_ );
		term_.close();       // back to the normal screen first, then the console output returns
		capture_.stop();
	} );
	return true;
}

void ServerTui::stop() {
	if ( !started_ ) return;
	stop_.store( true );
	if ( thread_.joinable() ) thread_.join();
	started_ = false;
}

} // namespace engine::server::tui
