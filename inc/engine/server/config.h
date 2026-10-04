#pragma once

#include <string>

namespace engine::server {

struct ServerConfig {
	std::string host = "0.0.0.0";
	int port = 8000;
	int max_batch_size = 8;
	int default_max_tokens = 256;
	float default_temperature = 0.0f;

	std::string model_type = "qwen38"; // "qwen38" or "glm53"
	std::string model_dir = "/home/mrq/Documents/models/Qwen3.8-27B-exl3";
	std::string draft_model_dir = "/home/mrq/Documents/models/Qwen3.8-27B-DFlash2-EXL3-5.0bpw";
	bool use_speculation = false;
	bool load_weights = true;
	bool benchmark_mode = false;
	int tp_size = 4;

	bool load_from_json( const char* json_path );
	void parse_cli_args( int argc, char** argv );
};

} // namespace engine::server
