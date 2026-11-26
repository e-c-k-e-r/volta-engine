// vall-e text to speech in volta_server: POST /v1/audio/speech and GET /v1/audio/voices (engine/server/tts.h). A request runs on a blocking
// route's thread; engine::tts::Tts serializes syntheses itself. Voices are the WAV files of tts.voices_dir, by file name without the extension,
// read once and kept.
#include "engine/server/tts.h"
#include "engine/core/log.h"
#include "engine/tts/tts.h"
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <dirent.h>
#include <map>
#include <mutex>

namespace engine::server {

struct TtsService::Impl {
	TtsConfig cfg;
	engine::tts::Tts tts;
	std::mutex voice_mu;
	struct Voice { std::vector<float> audio; int rate = 0; bool loaded = false; };
	std::map<std::string, Voice> voices;  // name -> its audio, read on first use

	void scan_voices() {
		if ( cfg.voices_dir.empty() ) { return; }
		DIR* d = opendir( cfg.voices_dir.c_str() );
		if ( !d ) { LOG_WARN( "tts: can't read voices_dir %s\n", cfg.voices_dir.c_str() ); return; }
		while ( dirent* e = readdir( d ) ) {
			const std::string n = e->d_name;
			if ( n.size() > 4 && ( n.compare( n.size() - 4, 4, ".wav" ) == 0 || n.compare( n.size() - 4, 4, ".WAV" ) == 0 ) ) { voices[n.substr( 0, n.size() - 4 )]; }
		}
		closedir( d );
	}

	const Voice* voice( const std::string& name, std::string& error ) {
		std::lock_guard<std::mutex> lock( voice_mu );
		auto it = voices.find( name );
		if ( it == voices.end() ) { error = "no voice \"" + name + "\" (GET /v1/audio/voices lists them)"; return nullptr; }
		if ( !it->second.loaded ) {
			if ( !engine::tts::Tts::read_wav( cfg.voices_dir + "/" + name + ".wav", it->second.audio, it->second.rate, &error ) ) { return nullptr; }
			it->second.loaded = true;
		}
		return &it->second;
	}

	void handle_speech( const ConnPtr& c, const HttpRequest& r ) {
		if ( r.method != "POST" ) { return send_error( c, 405, "invalid_request_error", "use POST" ); }
		const auto j = nlohmann::json::parse( r.body, nullptr, false );
		if ( j.is_discarded() || !j.is_object() ) { return send_error( c, 400, "invalid_request_error", "the body must be a JSON object" ); }
		if ( !j.contains( "input" ) || !j["input"].is_string() || j["input"].get<std::string>().empty() ) {
			return send_error( c, 400, "invalid_request_error", "input (the text to speak) is required" );
		}
		const std::string format = j.contains( "response_format" ) && j["response_format"].is_string() ? j["response_format"].get<std::string>() : "wav";
		if ( format != "wav" ) { return send_error( c, 400, "invalid_request_error", "response_format: only \"wav\" is supported" ); }
		// typed reads (the server builds without exceptions, where a mistyped get would abort): false names the bad field
		std::string bad;
		auto num = [&]( const char* k, float def ) {
			if ( !j.contains( k ) || j[k].is_null() ) { return def; }
			if ( !j[k].is_number() ) { bad = k; return def; }
			return j[k].get<float>();
		};
		engine::tts::SynthOptions o;
		if ( j.contains( "language" ) ) {
			if ( j["language"].is_string() ) { o.language = j["language"].get<std::string>(); } else { bad = "language"; }
		}
		o.cfg = num( "cfg", -1.0f );
		o.cfg_prompt = num( "cfg_prompt", -1.0f );
		o.steps = static_cast<int>( num( "steps", 25.0f ) );
		o.duration = num( "duration", -1.0f );
		if ( j.contains( "seed" ) && !j["seed"].is_null() ) {
			if ( j["seed"].is_number_unsigned() ) { o.seed = j["seed"].get<uint64_t>(); } else { bad = "seed"; }
		}
		const float speed = num( "speed", 1.0f );   // OpenAI's speed: 0.25 to 4, here the inverse of the length scale
		if ( !bad.empty() ) { return send_error( c, 400, "invalid_request_error", "bad type for " + bad ); }
		if ( speed < 0.25f || speed > 4.0f ) { return send_error( c, 400, "invalid_request_error", "speed must be between 0.25 and 4" ); }
		o.length_scale = 1.0f / speed;
		if ( o.steps < 1 || o.steps > 100 ) { return send_error( c, 400, "invalid_request_error", "steps must be between 1 and 100" ); }
		std::string error;
		const Voice* v = nullptr;
		if ( j.contains( "voice" ) && j["voice"].is_string() && !j["voice"].get<std::string>().empty() ) {
			v = voice( j["voice"].get<std::string>(), error );
			if ( !v ) { return send_error( c, 400, "invalid_request_error", error ); }
		}
		cudaSetDevice( cfg.device );
		const auto t0 = std::chrono::steady_clock::now();
		std::vector<float> audio;
		static const std::vector<float> none;
		if ( !tts.synthesize( j["input"].get<std::string>(), v ? v->audio : none, v ? v->rate : 0, o, audio, &error ) ) {
			return send_error( c, 400, "invalid_request_error", error );
		}
		const double dt = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
		const double secs = audio.size() / static_cast<double>( tts.sample_rate() );
		LOG_INFO( "tts: %.1f s of speech in %.2f s (%.1fx real time)\n", secs, dt, secs / std::max( dt, 1e-9 ) );
		send_response( c, 200, "audio/wav", engine::tts::Tts::wav_bytes( audio, tts.sample_rate() ) );
	}

	void handle_voices( const ConnPtr& c ) {
		nlohmann::json names = nlohmann::json::array();
		{
			std::lock_guard<std::mutex> lock( voice_mu );
			for ( const auto& kv : voices ) { names.push_back( kv.first ); }
		}
		send_json( c, 200, nlohmann::json( { { "voices", names } } ).dump() );
	}
};

TtsService::TtsService() : p_( std::make_unique<Impl>() ) {}
TtsService::~TtsService() = default;

bool TtsService::init( const ServerConfig& config ) {
	p_->cfg = config.tts;
	if ( p_->cfg.codec.empty() ) {
		LOG_ERROR( "tts: \"codec\" (python -m vall_e.export codec) is required with \"model\"\n" );
		return false;
	}
	cudaSetDevice( p_->cfg.device );
	if ( !p_->tts.load( p_->cfg.model, p_->cfg.codec, p_->cfg.device ) ) {
		LOG_ERROR( "tts: failed to load %s / %s\n", p_->cfg.model.c_str(), p_->cfg.codec.c_str() );
		return false;
	}
	p_->scan_voices();
	const auto& s = p_->tts.model().default_sampling();
	LOG_INFO( "tts: %s on GPU %d (guidance %.1f / %.1f), %zu voices\n", p_->cfg.model.c_str(), p_->cfg.device, s.cfg, s.cfg_prompt, p_->voices.size() );
	return true;
}

void TtsService::register_routes( Router& router, bool alone ) {
	Impl* p = p_.get();
	router.add( "*", "/v1/audio/speech", [p]( const ConnPtr& c, const HttpRequest& r ) { p->handle_speech( c, r ); }, true );
	router.add( "GET", "/v1/audio/voices", [p]( const ConnPtr& c, const HttpRequest& ) { p->handle_voices( c ); } );
	if ( alone ) {
		router.add( "GET", "/health", []( const ConnPtr& c, const HttpRequest& ) { send_json( c, 200, R"({"status":"ok","model":"vall_e"})" ); } );
		router.add( "GET", "/", []( const ConnPtr& c, const HttpRequest& ) { send_json( c, 200, R"({"status":"ok","model":"vall_e"})" ); } );
	}
}

} // namespace engine::server
