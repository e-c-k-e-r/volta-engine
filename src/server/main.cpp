#include "engine/server/server.h"
#include "engine/server/config.h"
#include <cstdio>
#include <memory>

int main( int argc, char** argv ) {
	engine::server::ServerConfig cfg;
	cfg.parse_cli_args( argc, argv );

	auto server = std::make_unique<engine::server::VoltaServer>();
	if ( !server->init( cfg ) ) {
		std::fprintf( stderr, "[-] FATAL: Failed to initialize Volta Server.\n" );
		return 1;
	}

	server->run();
	return 0;
}
