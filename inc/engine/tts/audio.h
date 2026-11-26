// Audio in and out for the TTS pipeline (host side, built without the text front-end): WAV files and the prompt resampler.
#pragma once

#include <string>
#include <vector>

namespace engine::tts {

// WAV files: PCM 8 / 16 / 24 / 32-bit or IEEE float, channels averaged (as vall_e's TTS._load); 16-bit PCM out
bool read_wav( const std::string& path, std::vector<float>& mono, int& rate, std::string* error = nullptr );
bool parse_wav( const std::string& bytes, std::vector<float>& mono, int& rate, std::string* error = nullptr );
std::string wav_bytes( const std::vector<float>& x, int rate );
bool write_wav( const std::string& path, const std::vector<float>& x, int rate );
// torchaudio.functional.resample (sinc_interp_hann, lowpass_filter_width 6, rolloff 0.99), in FP32 as it computes it
std::vector<float> resample( const std::vector<float>& x, int from, int to );

} // namespace engine::tts
