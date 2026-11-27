#pragma once

#include "engine/server/config.h"
#include "engine/server/http.h"
#include "engine/tts/tts.h"
#include <memory>
#include <string>

// vall-e text to speech in volta_server (the launch config's "tts" section; see docs/CONFIG.md): POST /v1/audio/speech in OpenAI's shape (input,
// voice, response_format "wav", speed) with vall-e's own fields (language, seed, cfg, cfg_prompt, steps, duration), and GET /v1/audio/voices.
// Served next to a language model, or alone when the config has no model_dir. Needs propan2ol at build time (ENGINE_TTS).
namespace engine::server {

// A /v1/audio/speech body (the endpoint's and the library's): the text, the voice by name or as a WAV file in the request (voice_audio, base64),
// the synthesis options (a random seed when none is given) and the response format. False with *error naming the bad field
struct SpeechBody {
	std::string input;
	std::string voice;       // a voice by name (empty: none, or voice_wav)
	std::string voice_wav;   // voice_audio decoded: a WAV file's bytes
	std::string format = "wav";
	engine::tts::SynthOptions options;
};
bool parse_speech_body( const std::string& body, SpeechBody* out, std::string* error );

class TtsService {
public:
	TtsService();
	~TtsService();

	// Loads the model and codec onto tts.device and lists tts.voices_dir's voices.
	bool init( const ServerConfig& config );
	// A parsed body's speech at sample_rate() (blocks; one synthesis at a time). False with *error set
	bool speak( const SpeechBody& body, std::vector<float>* audio, std::string* error );
	// A voice from PCM, encoded once and used by name; replaces one of the same name
	bool add_voice( const std::string& name, const std::vector<float>& pcm, int rate, std::string* error );
	std::vector<std::string> voice_names() const;
	int sample_rate() const;
	// Adds /v1/audio/speech, /v1/audio/voices, the page at /tts (and /, with /health, when alone) and /api/tts/info (key_required: /v1/* need the api_key).
	void register_routes( Router& router, bool alone, bool key_required );

private:
	struct Impl;
	std::unique_ptr<Impl> p_;
};

} // namespace engine::server
