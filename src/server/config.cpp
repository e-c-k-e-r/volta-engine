#include "engine/server/config.h"
#include "engine/models/model.h"
#include "engine/runtime/options.h"

#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

namespace engine::server {

namespace {

bool apply_option( const std::string& key, const std::string& value ) {
	if ( runtime::set_option( key, value ) ) return true;
	std::fprintf( stderr, "[-] Error: unknown option or invalid value: %s = %s\n", key.c_str(), value.c_str() );
	return false;
}

} // namespace

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
	if ( j.contains( "default_top_p" ) && j["default_top_p"].is_number() ) default_top_p = j["default_top_p"].get<float>();
	if ( j.contains( "default_top_k" ) && j["default_top_k"].is_number_integer() ) default_top_k = j["default_top_k"].get<int>();
	if ( j.contains( "default_temperature" ) && j["default_temperature"].is_number() ) {
		default_temperature = j["default_temperature"].get<float>();
	}
	if ( j.contains( "default_enable_thinking" ) && j["default_enable_thinking"].is_boolean() ) {
		default_enable_thinking = j["default_enable_thinking"].get<bool>();
	}
	if ( j.contains( "model_type" ) || j.contains( "model" ) ) {
		std::fprintf( stderr, "[-] Error: model_type is read from the checkpoint config.json; remove it from the launch config\n" );
		return false;
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
	// "options": { "max_seq": 32768, "kv_format": "fp8", ... } -> runtime::Options fields
	if ( j.contains( "options" ) && j["options"].is_object() ) {
		for ( auto& [key, val] : j["options"].items() ) {
			const std::string text = val.is_string() ? val.get<std::string>() : val.dump();
			if ( !apply_option( key, text ) ) return false;
		}
	}

	return true;
}

void ServerConfig::parse_cli_args( int argc, char** argv ) {
	// First pass: look for --config to load base configuration
	for ( int i = 1; i < argc; ++i ) {
		if ( std::strcmp( argv[i], "--config" ) == 0 && i + 1 < argc ) {
			if ( !load_from_json( argv[++i] ) ) {
				std::fprintf( stderr, "[-] FATAL: invalid launch config %s\n", argv[i] );
				std::exit( 2 );
			}
		}
	}

	// Second pass: apply CLI overrides
	for ( int i = 1; i < argc; ++i ) {
		if ( ( std::strcmp( argv[i], "--port" ) == 0 ) && i + 1 < argc ) {
			port = std::atoi( argv[++i] );
		}
		else if ( ( std::strcmp( argv[i], "--host" ) == 0 ) && i + 1 < argc ) {
			host = argv[++i];
		}
		else if ( std::strcmp( argv[i], "--model" ) == 0 || std::strcmp( argv[i], "--model-type" ) == 0 ) {
			std::fprintf( stderr, "[-] FATAL: %s was removed; model_type is read from the checkpoint config.json\n", argv[i] );
			std::exit( 2 );
		}
		else if ( ( std::strcmp( argv[i], "--model-dir" ) == 0 || std::strcmp( argv[i], "--weights-dir" ) == 0 ) && i + 1 < argc ) {
			model_dir = argv[++i];
		}
		else if ( ( std::strcmp( argv[i], "--draft-model-dir" ) == 0 || std::strcmp( argv[i], "--spec-dir" ) == 0 ) && i + 1 < argc ) {
			draft_model_dir = argv[++i];
		}
		else if ( std::strcmp( argv[i], "--speculative" ) == 0 || std::strcmp( argv[i], "--use-speculation" ) == 0 ) {
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
		else if ( std::strcmp( argv[i], "--set" ) == 0 && i + 1 < argc ) {
			const std::string kv = argv[++i];
			const size_t eq = kv.find( '=' );
			if ( eq == std::string::npos || !apply_option( kv.substr( 0, eq ), kv.substr( eq + 1 ) ) ) {
				std::fprintf( stderr, "[-] FATAL: --set expects key=value for a known option, got '%s'\n", kv.c_str() );
				std::exit( 2 );
			}
		}
	}

	if ( !resolve() ) std::exit( 2 );
}

bool ServerConfig::resolve() {
	if ( model_dir.empty() ) {
		std::fprintf( stderr, "[-] FATAL: model_dir is required (--model-dir or \"model_dir\" in the launch config)\n" );
		return false;
	}
	{
		const std::string path = model_dir + "/config.json";
		FILE* f = std::fopen( path.c_str(), "rb" );
		if ( !f ) {
			std::fprintf( stderr, "[-] FATAL: %s is unreadable\n", path.c_str() );
			return false;
		}
		std::string text;
		char buf[4096];
		size_t n;
		while ( ( n = std::fread( buf, 1, sizeof( buf ), f ) ) > 0 ) text.append( buf, n );
		std::fclose( f );
		auto j = nlohmann::json::parse( text, nullptr, false );
		if ( j.is_discarded() || !j.contains( "model_type" ) || !j["model_type"].is_string() ) {
			std::fprintf( stderr, "[-] FATAL: %s has no model_type\n", path.c_str() );
			return false;
		}
		model_type = j["model_type"].get<std::string>();
	}
	if ( !models::is_known_model_type( model_type ) ) {
		std::fprintf( stderr, "[-] FATAL: unsupported model_type '%s' (supported: qwen3_5, glm5_next)\n", model_type.c_str() );
		return false;
	}
	// Sampling defaults: explicit config wins, then the checkpoint's generation_config.json
	{
		FILE* f = std::fopen( ( model_dir + "/generation_config.json" ).c_str(), "rb" );
		if ( f ) {
			std::string text;
			char buf[4096];
			size_t n;
			while ( ( n = std::fread( buf, 1, sizeof( buf ), f ) ) > 0 ) text.append( buf, n );
			std::fclose( f );
			auto g = nlohmann::json::parse( text, nullptr, false );
			if ( !g.is_discarded() && g.value( "do_sample", true ) ) {
				if ( default_temperature < 0.0f && g.contains( "temperature" ) && g["temperature"].is_number() ) default_temperature = g["temperature"].get<float>();
				if ( default_top_p < 0.0f && g.contains( "top_p" ) && g["top_p"].is_number() ) default_top_p = g["top_p"].get<float>();
				if ( default_top_k < 0 && g.contains( "top_k" ) && g["top_k"].is_number_integer() ) default_top_k = g["top_k"].get<int>();
			}
		}
		if ( default_temperature < 0.0f ) default_temperature = 0.0f;
		if ( default_top_p < 0.0f ) default_top_p = 1.0f;
		if ( default_top_k < 0 ) default_top_k = 16;
		std::printf( "[*] Sampling defaults: temperature %.2f, top_p %.2f, top_k %d\n", default_temperature, default_top_p, default_top_k );
	}
	if ( use_speculation && draft_model_dir.empty() ) {
		std::fprintf( stderr, "[-] FATAL: use_speculation requires draft_model_dir\n" );
		return false;
	}
	return true;
}

} // namespace engine::server
