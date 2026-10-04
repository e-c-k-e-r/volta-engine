#include "engine/server/config.h"

#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

namespace engine::server {

bool ServerConfig::load_from_json( const char* json_path ) {
	if ( !json_path ) {
		return false;
	}

	FILE* f = std::fopen( json_path, "rb" );
	if ( !f ) {
		std::fprintf( stderr, "[-] Notice: Could not open config file: %s\n", json_path );
		return false;
	}

	std::fseek( f, 0, SEEK_END );
	long size = std::ftell( f );
	std::fseek( f, 0, SEEK_SET );
	if ( size <= 0 ) {
		std::fclose( f );
		return false;
	}

	std::string buffer( size, '\0' );
	size_t r = std::fread( buffer.data(), 1, size, f );
	std::fclose( f );
	if ( r != static_cast<size_t>( size ) ) {
		return false;
	}

	auto j = nlohmann::json::parse( buffer, nullptr, false );
	if ( j.is_discarded() ) {
		std::fprintf( stderr, "[-] Error: Failed to parse JSON config: %s\n", json_path );
		return false;
	}

	if ( j.contains( "host" ) && j["host"].is_string() ) {
		host = j["host"].get<std::string>();
	}
	if ( j.contains( "port" ) && j["port"].is_number_integer() ) {
		port = j["port"].get<int>();
	}
	if ( j.contains( "max_batch_size" ) && j["max_batch_size"].is_number_integer() ) {
		max_batch_size = j["max_batch_size"].get<int>();
	}
	if ( j.contains( "default_max_tokens" ) && j["default_max_tokens"].is_number_integer() ) {
		default_max_tokens = j["default_max_tokens"].get<int>();
	}
	if ( j.contains( "default_temperature" ) && j["default_temperature"].is_number() ) {
		default_temperature = j["default_temperature"].get<float>();
	}
	if ( j.contains( "model_type" ) && j["model_type"].is_string() ) {
		model_type = j["model_type"].get<std::string>();
	}
	if ( j.contains( "model" ) && j["model"].is_string() ) {
		model_type = j["model"].get<std::string>();
	}
	if ( j.contains( "model_dir" ) && j["model_dir"].is_string() ) {
		model_dir = j["model_dir"].get<std::string>();
	}
	if ( j.contains( "draft_model_dir" ) && j["draft_model_dir"].is_string() ) {
		draft_model_dir = j["draft_model_dir"].get<std::string>();
	}
	if ( j.contains( "use_speculation" ) && j["use_speculation"].is_boolean() ) {
		use_speculation = j["use_speculation"].get<bool>();
	}
	if ( j.contains( "speculative" ) && j["speculative"].is_boolean() ) {
		use_speculation = j["speculative"].get<bool>();
	}
	if ( j.contains( "load_weights" ) && j["load_weights"].is_boolean() ) {
		load_weights = j["load_weights"].get<bool>();
	}
	if ( j.contains( "benchmark_mode" ) && j["benchmark_mode"].is_boolean() ) {
		benchmark_mode = j["benchmark_mode"].get<bool>();
	}
	if ( j.contains( "tp_size" ) && j["tp_size"].is_number_integer() ) {
		tp_size = j["tp_size"].get<int>();
	}

	return true;
}

void ServerConfig::parse_cli_args( int argc, char** argv ) {
	// First pass: look for --config to load base configuration
	for ( int i = 1; i < argc; ++i ) {
		if ( std::strcmp( argv[i], "--config" ) == 0 && i + 1 < argc ) {
			load_from_json( argv[++i] );
		}
	}

	bool model_dir_overridden = false;

	// Second pass: apply CLI overrides
	for ( int i = 1; i < argc; ++i ) {
		if ( ( std::strcmp( argv[i], "--port" ) == 0 ) && i + 1 < argc ) {
			port = std::atoi( argv[++i] );
		}
		else if ( ( std::strcmp( argv[i], "--host" ) == 0 ) && i + 1 < argc ) {
			host = argv[++i];
		}
		else if ( ( std::strcmp( argv[i], "--model" ) == 0 || std::strcmp( argv[i], "--model-type" ) == 0 ) && i + 1 < argc ) {
			model_type = argv[++i];
		}
		else if ( ( std::strcmp( argv[i], "--model-dir" ) == 0 || std::strcmp( argv[i], "--weights-dir" ) == 0 ) && i + 1 < argc ) {
			model_dir = argv[++i];
			model_dir_overridden = true;
		}
		else if ( ( std::strcmp( argv[i], "--draft-model-dir" ) == 0 || std::strcmp( argv[i], "--spec-dir" ) == 0 ) && i + 1 < argc ) {
			draft_model_dir = argv[++i];
		}
		else if ( std::strcmp( argv[i], "--speculative" ) == 0 ) {
			use_speculation = true;
		}
		else if ( std::strcmp( argv[i], "--benchmark" ) == 0 ) {
			benchmark_mode = true;
		}
		else if ( std::strcmp( argv[i], "--load-weights" ) == 0 ) {
			load_weights = true;
		}
		else if ( std::strcmp( argv[i], "--no-load-weights" ) == 0 ) {
			load_weights = false;
		}
		else if ( ( std::strcmp( argv[i], "--max-batch-size" ) == 0 ) && i + 1 < argc ) {
			max_batch_size = std::atoi( argv[++i] );
		}
		else if ( ( std::strcmp( argv[i], "--max-tokens" ) == 0 || std::strcmp( argv[i], "--default-max-tokens" ) == 0 ) && i + 1 < argc ) {
			default_max_tokens = std::atoi( argv[++i] );
		}
		else if ( ( std::strcmp( argv[i], "--temperature" ) == 0 ) && i + 1 < argc ) {
			default_temperature = static_cast<float>( std::atof( argv[++i] ) );
		}
		else if ( ( std::strcmp( argv[i], "--tp-size" ) == 0 ) && i + 1 < argc ) {
			tp_size = std::atoi( argv[++i] );
		}
	}

	// Adjust default paths if model_type is glm53 and model_dir wasn't explicitly supplied
	if ( !model_dir_overridden && ( model_type == "glm53" || model_type == "glm" ) ) {
		model_dir = "/home/mrq/Documents/models/GLM-5.3-Flash-exl3-3.05bpw";
		draft_model_dir = "/home/mrq/Documents/models/GLM-5.3-Flash-DFlash2";
	}
}

} // namespace engine::server
