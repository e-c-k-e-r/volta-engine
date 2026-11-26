// vall-e text to speech end to end (vall_e/tts.py TTS.generate): the text front-end (G2P), the model (TtsModel) and the
// codec (NemoCodec). The voice prompt's audio is averaged to mono, resampled to 44.1 kHz as torchaudio does and cut to its
// first 10 s; texts over 160 phonemes are generated sentence by sentence (merged up to that), each its predicted length
// (or a share of a given duration), trailing silence trimmed, joined with short pauses.
#pragma once

#include "engine/tts/codec.h"
#include "engine/tts/g2p.h"
#include "engine/tts/tts_model.h"
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace engine::tts {

struct SynthOptions {
	std::string language = "en";
	float cfg = -1.0f, cfg_prompt = -1.0f;  // < 0: the export's own (TtsModel::default_sampling)
	int steps = 25;
	int flow_steps = 8;         // Euler steps of the flow head per revealed patch
	float sway = -1.0f;         // the flow steps' sway grid (< 0: more steps near the noise end, as F5-TTS)
	float temperature = 1.0f;   // the flow head's starting noise scale
	float length_scale = 1.0f;  // on each sentence's predicted length
	float duration = -1.0f;     // seconds for the whole text (shared out by phonemes); < 0: predicted
	float pause = 0.25f;        // seconds between sentences
	bool trim = true;           // drop each sentence's trailing silence
	uint64_t seed = 0;          // 0: a random one
};

class Tts {
public:
	bool load( const std::string& model, const std::string& codec, int device = 0 );
	// prompt: the voice's audio at prompt_rate (any channel count averaged beforehand; empty: no voice prompt). Returns the
	// speech at sample_rate(), or false with *error set. One synthesis at a time (the model's workspaces are shared)
	bool synthesize( const std::string& text, const std::vector<float>& prompt, int prompt_rate, const SynthOptions& o, std::vector<float>& out,
		std::string* error = nullptr );
	int sample_rate() const { return codec_.config().sample_rate; }
	const TtsModel& model() const { return model_; }

	// WAV files: PCM 8 / 16 / 24 / 32-bit or IEEE float, channels averaged (as TTS._load); 16-bit PCM out
	static bool read_wav( const std::string& path, std::vector<float>& mono, int& rate, std::string* error = nullptr );
	static bool parse_wav( const std::string& bytes, std::vector<float>& mono, int& rate, std::string* error = nullptr );
	static std::string wav_bytes( const std::vector<float>& x, int rate );
	// torchaudio.functional.resample (sinc_interp_hann, lowpass_filter_width 6, rolloff 0.99), in FP32 as it computes it
	static std::vector<float> resample( const std::vector<float>& x, int from, int to );

private:
	std::vector<std::vector<int>> split( const std::string& text, const std::string& language );

	TtsModel model_;
	NemoCodec codec_;
	G2P g2p_;
	std::mutex mu_;
};

} // namespace engine::tts
