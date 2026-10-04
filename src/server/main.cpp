#include "engine/server/server.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main( int argc, char** argv ) {
	engine::server::ServerConfig cfg;

	for ( int i = 1; i < argc; ++i ) {
		if ( std::strcmp( argv[i], "--port" ) == 0 && i + 1 < argc ) {
			cfg.port = std::atoi( argv[++i] );
		}
		else if ( std::strcmp( argv[i], "--host" ) == 0 && i + 1 < argc ) {
			cfg.host = argv[++i];
		}
		else if ( std::strcmp( argv[i], "--model" ) == 0 && i + 1 < argc ) {
			cfg.default_model = argv[++i];
		}
		else if ( std::strcmp( argv[i], "--speculative" ) == 0 ) {
			cfg.speculative = true;
		}
		else if ( std::strcmp( argv[i], "--benchmark" ) == 0 ) {
			cfg.benchmark_mode = true;
		}
		else if ( std::strcmp( argv[i], "--load-weights" ) == 0 ) {
			cfg.load_weights = true;
		}
		else if ( ( std::strcmp( argv[i], "--model-dir" ) == 0 || std::strcmp( argv[i], "--weights-dir" ) == 0 ) && i + 1 < argc ) {
			cfg.qwen_model_dir = argv[++i];
		}
		else if ( std::strcmp( argv[i], "--help" ) == 0 || std::strcmp( argv[i], "-h" ) == 0 ) {
			std::printf( "Usage: %s [options]\n", argv[0] );
			std::printf( "Options:\n" );
			std::printf( "  --port <port>         Server port (default: 8000)\n" );
			std::printf( "  --host <ip>           Bind host address (default: 0.0.0.0)\n" );
			std::printf( "  --model <name>        Default model (qwen38 or glm53, default: qwen38)\n" );
			std::printf( "  --speculative         Enable DFlash2 speculative decoding\n" );
			std::printf( "  --benchmark           Run in pure hardware line-rate benchmark mode\n" );
			std::printf( "  --load-weights        Load full safetensors model weights into VRAM\n" );
			std::printf( "  --help, -h            Show this help message\n" );
			return 0;
		}
	}

	auto server = std::make_unique<engine::server::VoltaServer>();
	if ( !server->init( cfg ) ) {
		std::fprintf( stderr, "[-] FATAL: Failed to initialize Volta Server.\n" );
		return 1;
	}

	server->run();
	return 0;
}
