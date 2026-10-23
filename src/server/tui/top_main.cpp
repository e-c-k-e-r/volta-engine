// volta_top: the serving dashboard as a stand-alone program. It polls a running volta_server's /api/stats, so it can attach to
// (and leave) a server at any time, including one started without --tui.
//   volta_top [host[:port]] [--interval ms]
#include "engine/server/tui/app.h"
#include "engine/server/tui/http_get.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace engine::server::tui;

int main( int argc, char** argv ) {
	std::string host = "127.0.0.1";
	int port = 8000;
	ViewerOptions opt;
	for ( int i = 1; i < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--help" ) || !std::strcmp( argv[i], "-h" ) ) {
			std::printf( "usage: volta_top [host[:port]] [--interval ms]\nShows a running volta_server's slots, throughput and GPUs (reads /api/stats). Keys: Tab or 1-4 switch tabs, arrows scroll, q quits.\n" );
			return 0;
		}
		if ( !std::strcmp( argv[i], "--interval" ) && i + 1 < argc ) {
			opt.interval_ms = std::max( 100, std::atoi( argv[++i] ) );
			continue;
		}
		std::string a = argv[i];
		const size_t colon = a.rfind( ':' );
		if ( colon != std::string::npos && a.find( ':' ) == colon ) {   // not a bare IPv6 address
			port = std::atoi( a.c_str() + colon + 1 );
			a.resize( colon );
		}
		if ( !a.empty() ) host = a;
	}
	opt.source = host + ":" + std::to_string( port );

	Terminal term;
	if ( !term.open() ) {
		std::fprintf( stderr, "volta_top needs a terminal\n" );
		return 1;
	}
	std::atomic<bool> stop{ false };
	auto fetch = [&]( nlohmann::json* stats, std::string* error ) {
		std::string body;
		if ( !http_get( host, port, "/api/stats", 800, &body, error ) ) return false;
		*stats = nlohmann::json::parse( body, nullptr, false );
		if ( stats->is_discarded() ) {
			*error = "/api/stats did not return JSON (is this a language-model server?)";
			return false;
		}
		return true;
	};
	run_viewer( term, opt, fetch, LogFn(), stop );
	return 0;
}
