#pragma once

#include <string>

namespace engine::server {

struct ServerConfig {
	std::string host = "0.0.0.0";
	int port = 8000;
	int max_batch_size = 8;
	int default_max_tokens = 256;
	float default_temperature = 0.0f;

	std::string model_type;      // empty: read from <model_dir>/config.json; otherwise qwen3_5 | glm5_next
	std::string model_dir;       // required
	std::string draft_model_dir; // required with use_speculation
	bool use_speculation = false;
	bool load_weights = true;
	bool benchmark_mode = false;
	int tp_size = 4;

	bool load_from_json( const char* json_path );
	// Parses --config and the CLI, then resolves and validates (model_dir, model_type, draft dir). Exits with status 2 on error.
	void parse_cli_args( int argc, char** argv );
	bool resolve();
};

} // namespace engine::server
