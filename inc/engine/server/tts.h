#pragma once

#include "engine/server/config.h"
#include "engine/server/http.h"
#include <memory>

// vall-e text to speech in volta_server (the launch config's "tts" section; see docs/CONFIG.md): POST /v1/audio/speech in OpenAI's shape (input,
// voice, response_format "wav", speed) with vall-e's own fields (language, seed, cfg, cfg_prompt, steps, duration), and GET /v1/audio/voices.
// Served next to a language model, or alone when the config has no model_dir. Needs libespeak-ng at build time (ENGINE_TTS).
namespace engine::server {

class TtsService {
public:
	TtsService();
	~TtsService();

	// Loads the model and codec onto tts.device and lists tts.voices_dir's voices.
	bool init( const ServerConfig& config );
	// Adds /v1/audio/speech and /v1/audio/voices (and /health when alone).
	void register_routes( Router& router, bool alone );

private:
	struct Impl;
	std::unique_ptr<Impl> p_;
};

} // namespace engine::server
