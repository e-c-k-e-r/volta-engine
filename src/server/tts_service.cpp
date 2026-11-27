// vall-e text to speech in volta_server: POST /v1/audio/speech and GET /v1/audio/voices (engine/server/tts.h). A request runs on a blocking
// route's thread; engine::tts::Tts serializes syntheses itself. Voices are the WAV files of tts.voices_dir, by file name without the extension,
// read once and kept.
#include "engine/server/tts.h"
#include "engine/backend/runtime.h"
#include "engine/core/log.h"
#include "engine/loader/image.h"
#include "engine/tts/audio.h"
#include "engine/tts/tts.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <dirent.h>
#include <map>
#include <mutex>
#include <random>

namespace engine::server {

struct TtsService::Impl {
	TtsConfig cfg;
	engine::tts::Tts tts;
	mutable std::mutex voice_mu;
	using Codes = std::shared_ptr<const std::vector<uint8_t>>;
	std::map<std::string, Codes> voices;  // name -> its prompt (null: a voices_dir file, read and encoded on first use)

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

	Codes voice( const std::string& name, std::string& error ) {
		std::lock_guard<std::mutex> lock( voice_mu );
		auto it = voices.find( name );
		if ( it == voices.end() ) { error = "no voice \"" + name + "\""; return nullptr; }
		if ( !it->second ) {
			std::vector<float> audio;
			int rate = 0;
			if ( !engine::tts::read_wav( cfg.voices_dir + "/" + name + ".wav", audio, rate, &error ) ) { return nullptr; }
			auto codes = std::make_shared<std::vector<uint8_t>>();
			engine::backend::set_device( cfg.device );
			if ( !tts.encode_prompt( audio, rate, *codes, &error ) ) { return nullptr; }
			it->second = std::move( codes );
		}
		return it->second;
	}

	bool add_voice( const std::string& name, const std::vector<float>& pcm, int rate, std::string& error ) {
		if ( name.empty() ) { error = "a voice needs a name"; return false; }
		auto codes = std::make_shared<std::vector<uint8_t>>();
		engine::backend::set_device( cfg.device );
		if ( !tts.encode_prompt( pcm, rate, *codes, &error ) ) { return false; }
		std::lock_guard<std::mutex> lock( voice_mu );
		voices[name] = std::move( codes );
		return true;
	}

	std::vector<std::string> voice_names() const {
		std::lock_guard<std::mutex> lock( voice_mu );
		std::vector<std::string> names;
		for ( const auto& kv : voices ) { names.push_back( kv.first ); }
		return names;
	}

	// The speech of a parsed body: its voice (by name, or the WAV in the request) encoded, then the synthesis (one at a time, Tts's lock)
	bool speak( const SpeechBody& body, std::vector<float>& audio, std::string& error ) {
		Codes v;
		if ( !body.voice_wav.empty() ) {
			std::vector<float> pcm;
			int rate = 0;
			if ( !engine::tts::parse_wav( body.voice_wav, pcm, rate, &error ) ) { error = "voice_audio: " + error; return false; }
			auto codes = std::make_shared<std::vector<uint8_t>>();
			engine::backend::set_device( cfg.device );
			if ( !tts.encode_prompt( pcm, rate, *codes, &error ) ) { error = "voice_audio: " + error; return false; }
			v = std::move( codes );
		} else if ( !body.voice.empty() ) {
			v = voice( body.voice, error );
			if ( !v ) { return false; }
		}
		engine::backend::set_device( cfg.device );
		const auto t0 = std::chrono::steady_clock::now();
		static const std::vector<uint8_t> none;
		if ( !tts.synthesize( body.input, v ? *v : none, body.options, audio, &error ) ) { return false; }
		const double dt = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
		const double secs = audio.size() / static_cast<double>( tts.sample_rate() );
		LOG_INFO( "tts: %.1f s of speech in %.2f s (%.1fx real time)\n", secs, dt, secs / std::max( dt, 1e-9 ) );
		return true;
	}

	void handle_speech( const ConnPtr& c, const HttpRequest& r ) {
		if ( r.method != "POST" ) { return send_error( c, 405, "invalid_request_error", "use POST" ); }
		SpeechBody body;
		std::string error;
		if ( !parse_speech_body( r.body, &body, &error ) ) { return send_error( c, 400, "invalid_request_error", error ); }
		const auto t0 = std::chrono::steady_clock::now();
		std::vector<float> audio;
		if ( !speak( body, audio, error ) ) {
			if ( !body.voice.empty() && error.rfind( "no voice", 0 ) == 0 ) { error += " (GET /v1/audio/voices lists them)"; }
			return send_error( c, 400, "invalid_request_error", error );
		}
		const double dt = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
		const double secs = audio.size() / static_cast<double>( tts.sample_rate() );
		const std::string wav = engine::tts::wav_bytes( audio, tts.sample_rate() );
		if ( body.format == "json" ) {
			return send_json( c, 200, nlohmann::json( { { "audio", engine::loader::base64_encode( wav ) }, { "format", "wav" }, { "sample_rate", tts.sample_rate() }, { "duration", secs },
				{ "seed", body.options.seed }, { "elapsed", dt } } ).dump() );
		}
		send_response( c, 200, "audio/wav", wav );
	}

	void handle_info( const ConnPtr& c, bool key_required ) {
		const auto& s = tts.model().default_sampling();
		std::string name = cfg.model;
		const size_t slash = name.find_last_of( '/' );
		if ( slash != std::string::npos ) { name = name.substr( slash + 1 ); }
		nlohmann::json names = nlohmann::json::array();
		{
			std::lock_guard<std::mutex> lock( voice_mu );
			for ( const auto& kv : voices ) { names.push_back( kv.first ); }
		}
		const auto& mc = tts.model().config();
		send_json( c, 200, nlohmann::json( { { "model", name }, { "layers", mc.layers }, { "d_model", mc.d_model }, { "sample_rate", tts.sample_rate() },
			{ "cfg", s.cfg }, { "cfg_prompt", s.cfg_prompt }, { "steps", s.steps }, { "voices", names }, { "key_required", key_required } } ).dump() );
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

bool parse_speech_body( const std::string& text, SpeechBody* out, std::string* error ) {
	const auto j = nlohmann::json::parse( text, nullptr, false );
	if ( j.is_discarded() || !j.is_object() ) { *error = "the body must be a JSON object"; return false; }
	if ( !j.contains( "input" ) || !j["input"].is_string() || j["input"].get<std::string>().empty() ) { *error = "input (the text to speak) is required"; return false; }
	out->input = j["input"].get<std::string>();
	out->format = j.contains( "response_format" ) && j["response_format"].is_string() ? j["response_format"].get<std::string>() : "wav";
	// "wav": the WAV file itself; "json": {"audio": base64 WAV, "sample_rate", "duration"} (for pages that would rather not handle a binary body)
	if ( out->format != "wav" && out->format != "json" ) { *error = "response_format: \"wav\" or \"json\""; return false; }
	// typed reads (the server builds without exceptions, where a mistyped get would abort): false names the bad field
	std::string bad;
	auto num = [&]( const char* k, float def ) {
		if ( !j.contains( k ) || j[k].is_null() ) { return def; }
		if ( !j[k].is_number() ) { bad = k; return def; }
		return j[k].get<float>();
	};
	engine::tts::SynthOptions& o = out->options;
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
	o.flow_steps = static_cast<int>( num( "flow_steps", 8.0f ) );
	o.sway = num( "sway", -1.0f );
	o.temperature = num( "temperature", 1.0f );
	o.pause = num( "pause", 0.25f );
	if ( j.contains( "trim" ) && !j["trim"].is_null() ) {
		if ( j["trim"].is_boolean() ) { o.trim = j["trim"].get<bool>(); } else { bad = "trim"; }
	}
	if ( !bad.empty() ) { *error = "bad type for " + bad; return false; }
	if ( o.flow_steps < 1 || o.flow_steps > 64 ) { *error = "flow_steps must be between 1 and 64"; return false; }
	if ( o.temperature < 0.0f || o.temperature > 2.0f ) { *error = "temperature must be between 0 and 2"; return false; }
	if ( o.sway < -1.0f || o.sway > 1.0f ) { *error = "sway must be between -1 and 1"; return false; }
	if ( o.pause < 0.0f || o.pause > 5.0f ) { *error = "pause must be between 0 and 5 seconds"; return false; }
	if ( o.duration > 300.0f ) { *error = "duration is at most 300 seconds"; return false; }
	if ( speed < 0.25f || speed > 4.0f ) { *error = "speed must be between 0.25 and 4"; return false; }
	o.length_scale = 1.0f / speed;
	if ( o.steps < 1 || o.steps > 100 ) { *error = "steps must be between 1 and 100"; return false; }
	const bool named = j.contains( "voice" ) && j["voice"].is_string() && !j["voice"].get<std::string>().empty();
	if ( j.contains( "voice_audio" ) && !j["voice_audio"].is_null() ) {
		// the voice prompt in the request: a base64 WAV, bare or as a data URL ("data:audio/wav;base64,...")
		if ( named ) { *error = "give voice or voice_audio, not both"; return false; }
		if ( !j["voice_audio"].is_string() ) { *error = "bad type for voice_audio"; return false; }
		std::string b64 = j["voice_audio"].get<std::string>();
		const size_t comma = b64.find( "base64," );
		if ( comma != std::string::npos ) { b64 = b64.substr( comma + 7 ); }
		if ( !engine::loader::base64_decode( b64, &out->voice_wav ) || out->voice_wav.empty() ) { *error = "voice_audio holds invalid base64"; return false; }
	} else if ( named ) {
		out->voice = j["voice"].get<std::string>();
	}
	if ( !o.seed ) { o.seed = ( static_cast<uint64_t>( std::random_device{}() ) << 20 ) ^ std::random_device{}(); o.seed &= ( 1ull << 52 ) - 1; }   // known, so it can be returned (and fits a JS number)
	return true;
}

TtsService::TtsService() : p_( std::make_unique<Impl>() ) {}
TtsService::~TtsService() = default;

bool TtsService::init( const ServerConfig& config ) {
	p_->cfg = config.tts;
	if ( p_->cfg.codec.empty() ) {
		LOG_ERROR( "tts: \"codec\" (python -m vall_e.export codec) is required with \"model\"\n" );
		return false;
	}
	engine::backend::set_device( p_->cfg.device );
	const size_t free0 = engine::backend::device_free_bytes( p_->cfg.device );
	if ( !p_->tts.load( p_->cfg.model, p_->cfg.codec, p_->cfg.device ) ) {
		LOG_ERROR( "tts: failed to load %s / %s\n", p_->cfg.model.c_str(), p_->cfg.codec.c_str() );
		return false;
	}
	const size_t free1 = engine::backend::device_free_bytes( p_->cfg.device );
	p_->scan_voices();
	const auto& s = p_->tts.model().default_sampling();
	LOG_INFO( "tts: %s on GPU %d, %.2f GiB (guidance %.1f / %.1f), %zu voices\n", p_->cfg.model.c_str(), p_->cfg.device, ( free0 - free1 ) / 1073741824.0, s.cfg,
		s.cfg_prompt, p_->voices.size() );
	return true;
}

bool TtsService::speak( const SpeechBody& body, std::vector<float>* audio, std::string* error ) { return p_->speak( body, *audio, *error ); }
bool TtsService::add_voice( const std::string& name, const std::vector<float>& pcm, int rate, std::string* error ) { return p_->add_voice( name, pcm, rate, *error ); }
std::vector<std::string> TtsService::voice_names() const { return p_->voice_names(); }
int TtsService::sample_rate() const { return p_->tts.sample_rate(); }

const char* tts_page_html();

void TtsService::register_routes( Router& router, bool alone, bool key_required ) {
	Impl* p = p_.get();
	router.add( "GET", "/tts", []( const ConnPtr& c, const HttpRequest& ) { send_response( c, 200, "text/html; charset=utf-8", tts_page_html() ); } );
	router.add( "GET", "/api/tts/info", [p, key_required]( const ConnPtr& c, const HttpRequest& ) { p->handle_info( c, key_required ); } );
	router.add( "*", "/v1/audio/speech", [p]( const ConnPtr& c, const HttpRequest& r ) { p->handle_speech( c, r ); }, true );
	router.add( "GET", "/v1/audio/voices", [p]( const ConnPtr& c, const HttpRequest& ) { p->handle_voices( c ); } );
	if ( alone ) {
		router.add( "GET", "/health", []( const ConnPtr& c, const HttpRequest& ) { send_json( c, 200, R"({"status":"ok","model":"vall_e"})" ); } );
		router.add( "GET", "/", []( const ConnPtr& c, const HttpRequest& ) { send_response( c, 200, "text/html; charset=utf-8", tts_page_html() ); } );
	}
}

} // namespace engine::server
