#include "engine/server/config.h"
#include "engine/core/log.h"
#include "engine/models/model.h"
#include "engine/runtime/options.h"

#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

namespace engine::server {

namespace {

bool apply_option( const std::string& key, const std::string& value ) {
	if ( runtime::set_option( key, value ) ) return true;
	LOG_ERROR( "Error: unknown option or invalid value: %s = %s\n", key.c_str(), value.c_str() );
	return false;
}

} // namespace

bool ServerConfig::load_from_json( const char* json_path ) {
	if ( !json_path ) {
		return false;
	}

	FILE* f = std::fopen( json_path, "rb" );
	if ( !f ) {
		LOG_ERROR( "Notice: Could not open config file: %s\n", json_path );
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
		LOG_ERROR( "Error: Failed to parse JSON config: %s\n", json_path );
		return false;
	}

	if ( j.contains( "host" ) && j["host"].is_string() ) {
		host = j["host"].get<std::string>();
	}
	if ( j.contains( "port" ) && j["port"].is_number_integer() ) {
		port = j["port"].get<int>();
	}
	if ( j.contains( "request_timeout_s" ) && j["request_timeout_s"].is_number_integer() ) request_timeout_s = j["request_timeout_s"].get<int>();
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
		LOG_ERROR( "Error: model_type is read from the checkpoint config.json; remove it from the launch config\n" );
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
	// "video": { "dit": ..., "dit_ref2va": ..., "lora_dirs": [...], "out_dir": ..., "devices": [...], "max_rows": N }
	if ( j.contains( "video" ) ) {
		if ( !j["video"].is_object() ) {
			LOG_ERROR( "Error: \"video\" must be an object\n" );
			return false;
		}
		const auto& v = j["video"];
		for ( auto it = v.begin(); it != v.end(); ++it ) {
			const std::string& k = it.key();
			if ( k == "dit" && it->is_string() ) video.dit = it->get<std::string>();
			else if ( k == "dit_ref2va" && it->is_string() ) video.dit_ref2va = it->get<std::string>();
			else if ( k == "out_dir" && it->is_string() ) video.out_dir = it->get<std::string>();
			else if ( k == "max_rows" && it->is_number_integer() ) video.max_rows = it->get<int>();
			else if ( k == "resident" && it->is_boolean() ) video.resident = it->get<bool>();
			else if ( k == "resident_idle_s" && it->is_number_integer() ) video.resident_idle_s = it->get<int>();
			else if ( k == "tensor_parallel" && it->is_boolean() ) video.tensor_parallel = it->get<bool>();
			else if ( k == "resident_decoders" && it->is_boolean() ) video.resident_decoders = it->get<bool>();
			else if ( k == "captioner" && it->is_boolean() ) video.captioner = it->get<bool>();
			else if ( k == "text_encoder" && it->is_string() ) video.text_encoder = it->get<std::string>();
			else if ( k == "lora_dirs" && it->is_array() ) {
				for ( const auto& d : *it ) if ( d.is_string() ) video.lora_dirs.push_back( d.get<std::string>() );
			}
			else if ( k == "devices" && it->is_array() ) {
				for ( const auto& d : *it ) if ( d.is_number_integer() ) video.devices.push_back( d.get<int>() );
			}
			else {
				LOG_ERROR( "Error: unknown or invalid \"video\" key: %s\n", k.c_str() );
				return false;
			}
		}
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

std::string server_config_markdown() {
	return std::string( R"(# Configuration

Generated by `volta_server --print-config-docs`; do not edit by hand (the static gate checks that this file is current).

A launch is configured by a JSON file (`--config file.json`) and command-line flags; flags win over the file, the file wins
over environment variables, environment variables win over built-in defaults. Unknown keys, unknown options and invalid
values are fatal errors. The model type is never configured: it is read from the checkpoint's `config.json`.

## Launch config keys (top level of the JSON file)

| Key | CLI flag | Default | Description |
|---|---|---|---|
| `model_dir` | `--model-dir` | required | checkpoint directory |
| `draft_model_dir` | `--draft-model-dir` | none | DFlash2 drafter checkpoint; required with `use_speculation` |
| `use_speculation` | `--speculative` | false | enable speculative decoding |
| `tp_size` | `--tp-size` | 4 | GPUs (1, 2 or 4; GLM runs on 1, 2 or 4) |
| `host`, `port` | `--host`, `--port` | 0.0.0.0, 8000 | listen address |
| `max_batch_size` | `--max-batch-size` | 8 | concurrent decode slots the scheduler may fill (capped by `max_slots`) |
| `default_max_tokens` | `--max-tokens` | 256 | `max_tokens` for requests that omit it |
| `default_temperature`, `default_top_p`, `default_top_k` | `--temperature` | from the checkpoint's `generation_config.json` | sampling defaults for requests that omit them |
| `default_enable_thinking` | | false | thinking on for requests that do not say |
| `request_timeout_s` | | 30 | seconds a connection may stay silent before its request is complete |
| `video` | see below | | MiniMax-H3 settings |
| `options` | `--set key=value` | | the options below |

## MiniMax-H3 (video and audio generation)

A `model_dir` that is a MiniMax-H3 pipeline directory (it holds `model_index.json`; the FL2VA or Ref2VA component directory with `text_encoder`, `video_vae`,
`audio_vae`, `tokenizer`) starts the generation service instead of the language-model runtime: a job queue with one worker that owns the GPUs, the job API
(`/api/jobs`, `/api/info`) and a web page at `/`. `tp_size` is the number of GPUs the transformer is split over unless `devices` is given.

| `video` key | CLI flag | Default | Description |
|---|---|---|---|
| `dit` | `--dit` | required | INT8 ConvRot transformer checkpoint of the FL2VA partition (text and keyframe tasks) |
| `dit_ref2va` | `--dit-ref2va` | none | the Ref2VA partition; enables reference images, videos and audio |
| `lora_dirs` | `--lora-dir` (repeatable) | none | directories scanned for adapters (turbo and community LoRAs) |
| `out_dir` | `--out-dir` | `h3_jobs` | job history and results |
| `devices` | `--devices 0,1,2,3` | the first `tp_size` GPUs | GPUs of the transformer |
| `max_rows` | `--max-rows` | 36000 | largest packed sequence one job may need (a resident transformer is sized for it) |
| `resident` | `--resident` / `--no-resident` | true | keep the transformer (weights and adapters, on every GPU) loaded between jobs; off loads it for each job and frees it after (about 10 to 15 s slower per job, for machines that share the GPUs). The VAEs, the image encoder and the language-model stack are always loaded per stage: next to the transformer the first GPU has room for one at a time |
| `resident_idle_s` | `--resident-idle-s` | 0 | with `resident`: free the transformer after this many idle seconds (0 = never) |
| `resident_decoders` | `--resident-decoders` / `--no-resident-decoders` | true | with `resident` and `tensor_parallel`: keep the video decoders and the audio VAE loaded between jobs too (about 6 GB per GPU; saves the 3.6 s decoder load of every job). Ignored with sequence parallelism, where the full-weight transformer leaves no room |
| `text_encoder` | `--text-encoder` | none | directory of an EXL3 conversion of the Qwen3-VL-32B-Instruct language model (e.g. `turboderp/Qwen3-VL-32B-Instruct-exl3`): its 50 layers stay resident, sharded over `devices` (about 4.6 GB per GPU at 6 bpw on four), instead of streaming 49 GB of BF16 per job; the text stage of a short prompt drops from about 7 s to under a second. The vision tower still loads from the model directory. Use it with `tensor_parallel`: next to the full-weight sequence-parallel transformer it does not fit in 32 GB |
| `captioner` | `--captioner` / `--no-captioner` | false | the resident text tower also writes text: the page's Caption button has the Qwen3-VL language model describe a reference image as a video prompt (the prompt box, if filled, steers it). Loads the last 14 layers, the LM head and a KV cache: about 1.3 GB more per GPU on four (1.9 GB on the first, which holds the head); needs `text_encoder` with all 64 layers (an EXL3 conversion) |
| `tensor_parallel` | `--tensor-parallel` / `--no-tensor-parallel` | false | shard the transformer over `devices` by attention heads and MLP width (a quarter of the weights and of the activations per GPU, so the VAEs and the language model fit next to it) instead of by rows; about 1 s slower per step, needs a build with NCCL; text prompts up to about 8.6K tokens |

## Options

Every option can be set in the launch config's `options` object, with `--set key=value`, or through the environment variable
shown. Debug options use `ENGINE_DEBUG_<NAME>`. Test-only variables are listed at the end.

)" ) + runtime::options_markdown() + R"(
## Test and tool variables

| Variable | Description |
|---|---|
| `ENGINE_MODEL_ROOT` | directory holding the checkpoints used by tests and benches (also read from `tests/local.env`) |
| `ENGINE_TEST_QWEN_DIR`, `ENGINE_TEST_DRAFT_DIR`, `ENGINE_TEST_GLM_DIR`, `ENGINE_TEST_GLM_DRAFT_DIR` | override one checkpoint |
| `ENGINE_CPU_THREADS` | worker threads of the CPU backend (default: the process's allowed cores) |
| `ENGINE_DEBUG_CPU_TRACE` | per-call timing lines in the CPU backend |
| `ENGINE_VK_DEVICE` | Vulkan device index |
| `ENGINE_SPV_DIR` | directory of compiled SPIR-V modules (default: the build directory) |
| `BENCH_*`, `AR_ROWS`, `MOE_ROWS`, `MOE_ACTIVE`, `MLA_TIME`, `SPEC_STATE_MODEL` | inputs of individual benches and parity tests (see the test source) |
)";
}

void ServerConfig::parse_cli_args( int argc, char** argv ) {
	for ( int i = 1; i < argc; ++i ) {
		if ( std::strcmp( argv[i], "--print-config-docs" ) == 0 ) {
			std::fputs( server_config_markdown().c_str(), stdout );
			std::exit( 0 );
		}
	}
	// First pass: look for --config to load base configuration
	for ( int i = 1; i < argc; ++i ) {
		if ( std::strcmp( argv[i], "--config" ) == 0 && i + 1 < argc ) {
			if ( !load_from_json( argv[++i] ) ) {
				LOG_ERROR( "FATAL: invalid launch config %s\n", argv[i] );
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
			LOG_ERROR( "FATAL: %s was removed; model_type is read from the checkpoint config.json\n", argv[i] );
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
		else if ( std::strcmp( argv[i], "--dit" ) == 0 && i + 1 < argc ) {
			video.dit = argv[++i];
		}
		else if ( std::strcmp( argv[i], "--dit-ref2va" ) == 0 && i + 1 < argc ) {
			video.dit_ref2va = argv[++i];
		}
		else if ( std::strcmp( argv[i], "--lora-dir" ) == 0 && i + 1 < argc ) {
			video.lora_dirs.push_back( argv[++i] );
		}
		else if ( std::strcmp( argv[i], "--out-dir" ) == 0 && i + 1 < argc ) {
			video.out_dir = argv[++i];
		}
		else if ( std::strcmp( argv[i], "--resident" ) == 0 ) {
			video.resident = true;
		}
		else if ( std::strcmp( argv[i], "--no-resident" ) == 0 ) {
			video.resident = false;
		}
		else if ( std::strcmp( argv[i], "--resident-decoders" ) == 0 ) {
			video.resident_decoders = true;
		}
		else if ( std::strcmp( argv[i], "--no-resident-decoders" ) == 0 ) {
			video.resident_decoders = false;
		}
		else if ( std::strcmp( argv[i], "--captioner" ) == 0 ) {
			video.captioner = true;
		}
		else if ( std::strcmp( argv[i], "--no-captioner" ) == 0 ) {
			video.captioner = false;
		}
		else if ( std::strcmp( argv[i], "--text-encoder" ) == 0 && i + 1 < argc ) {
			video.text_encoder = argv[++i];
		}
		else if ( std::strcmp( argv[i], "--tensor-parallel" ) == 0 ) {
			video.tensor_parallel = true;
		}
		else if ( std::strcmp( argv[i], "--no-tensor-parallel" ) == 0 ) {
			video.tensor_parallel = false;
		}
		else if ( std::strcmp( argv[i], "--resident-idle-s" ) == 0 && i + 1 < argc ) {
			video.resident_idle_s = std::atoi( argv[++i] );
		}
		else if ( std::strcmp( argv[i], "--max-rows" ) == 0 && i + 1 < argc ) {
			video.max_rows = std::atoi( argv[++i] );
		}
		else if ( std::strcmp( argv[i], "--devices" ) == 0 && i + 1 < argc ) {
			video.devices.clear();
			const std::string v = argv[++i];
			for ( size_t p = 0; p < v.size(); ) {
				video.devices.push_back( std::atoi( v.c_str() + p ) );
				p = v.find( ',', p );
				if ( p == std::string::npos ) break;
				++p;
			}
		}
		else if ( std::strcmp( argv[i], "--set" ) == 0 && i + 1 < argc ) {
			const std::string kv = argv[++i];
			const size_t eq = kv.find( '=' );
			if ( eq == std::string::npos || !apply_option( kv.substr( 0, eq ), kv.substr( eq + 1 ) ) ) {
				LOG_ERROR( "FATAL: --set expects key=value for a known option, got '%s'\n", kv.c_str() );
				std::exit( 2 );
			}
		}
	}

	if ( !resolve() ) std::exit( 2 );
}

bool ServerConfig::resolve() {
	if ( model_dir.empty() ) {
		LOG_ERROR( "FATAL: model_dir is required (--model-dir or \"model_dir\" in the launch config)\n" );
		return false;
	}
	// a MiniMax-H3 pipeline directory (diffusers layout) is recognized by its model_index.json
	{
		std::ifstream mi( model_dir + "/model_index.json" );
		if ( mi ) {
			std::stringstream ss;
			ss << mi.rdbuf();
			auto j = nlohmann::json::parse( ss.str(), nullptr, false );
			if ( !j.is_discarded() && j.value( "_class_name", std::string() ) == "MiniMaxH3Pipeline" ) {
				model_type = "minimax_h3";
				if ( video.dit.empty() ) {
					LOG_ERROR( "FATAL: a MiniMax-H3 server needs the transformer checkpoint (--dit or video.dit in the launch config)\n" );
					return false;
				}
				if ( video.devices.empty() ) {
					for ( int d = 0; d < tp_size; ++d ) video.devices.push_back( d );
				}
				if ( video.max_rows < 1024 ) {
					LOG_ERROR( "FATAL: video.max_rows must be at least 1024\n" );
					return false;
				}
				return true;
			}
		}
	}
	{
		const std::string path = model_dir + "/config.json";
		FILE* f = std::fopen( path.c_str(), "rb" );
		if ( !f ) {
			LOG_ERROR( "FATAL: %s is unreadable\n", path.c_str() );
			return false;
		}
		std::string text;
		char buf[4096];
		size_t n;
		while ( ( n = std::fread( buf, 1, sizeof( buf ), f ) ) > 0 ) text.append( buf, n );
		std::fclose( f );
		auto j = nlohmann::json::parse( text, nullptr, false );
		if ( j.is_discarded() || !j.contains( "model_type" ) || !j["model_type"].is_string() ) {
			LOG_ERROR( "FATAL: %s has no model_type\n", path.c_str() );
			return false;
		}
		model_type = j["model_type"].get<std::string>();
	}
	if ( !models::is_known_model_type( model_type ) ) {
		LOG_ERROR( "FATAL: unsupported model_type '%s' (supported: qwen3_5, glm5_next, qwen4_exp)\n", model_type.c_str() );
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
		LOG_INFO( "Sampling defaults: temperature %.2f, top_p %.2f, top_k %d\n", default_temperature, default_top_p, default_top_k );
	}
	if ( use_speculation && draft_model_dir.empty() && runtime::options().speculator != "mtp" ) {
		LOG_ERROR( "FATAL: use_speculation requires draft_model_dir\n" );
		return false;
	}
	return true;
}

} // namespace engine::server
