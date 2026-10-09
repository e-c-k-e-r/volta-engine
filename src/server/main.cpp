#include "engine/server/server.h"
#include "engine/core/log.h"
#include "engine/server/config.h"
#include <cstdio>
#include <memory>
#include <sched.h>

// Keep the server and its HTTP threads off the cores that drive the GPUs (rank workers sit on 0-3, the
// scheduler on 4): a preempted driver thread stalls every rank at the next all-reduce. Threads spawned
// from here inherit this mask; workers and the scheduler set their own.
static void isolate_server_threads() {
	cpu_set_t allowed;
	if ( sched_getaffinity( 0, sizeof( allowed ), &allowed ) != 0 ) return;
	cpu_set_t host;
	CPU_ZERO( &host );
	for ( int c = 5; c < 10; ++c ) {
		if ( CPU_ISSET( c, &allowed ) ) CPU_SET( c, &host );
	}
	if ( CPU_COUNT( &host ) > 0 ) sched_setaffinity( 0, sizeof( host ), &host );
}

int main( int argc, char** argv ) {
	isolate_server_threads();
	engine::server::ServerConfig cfg;
	cfg.parse_cli_args( argc, argv );

	auto server = std::make_unique<engine::server::VoltaServer>();
	if ( !server->init( cfg ) ) {
		LOG_ERROR( "FATAL: Failed to initialize Volta Server.\n" );
		return 1;
	}

	server->run();
	return 0;
}
