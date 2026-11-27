#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace engine::loader {

// The audio feature extractor of the checkpoint (Gemma 4's Gemma4AudioFeatureExtractor): log-mel frames of mono audio at sample_rate,
// frame_length samples a frame (a periodic Hann window, zero-padded to fft_length) every hop samples, mel_bins HTK filters over
// [min_frequency, max_frequency], log( mel + mel_floor ); half a frame of silence before the first. Clips longer than chunk_seconds split into
// chunks of their own (the processor's limit of one clip).
struct AudioParams {
	int sample_rate = 16000;
	int mel_bins = 128;
	int frame_length = 320;
	int hop = 160;
	int fft_length = 512;
	double min_frequency = 0.0;
	double max_frequency = 8000.0;
	double mel_floor = 1e-3;
	int chunk_seconds = 30;
	// the tower's two stride-2 convolutions: mel frames -> tokens
	int tokens( int frames ) const {
		const int t = ( frames - 1 ) / 2 + 1;
		return frames > 0 ? ( t - 1 ) / 2 + 1 : 0;
	}
};

// One chunk of a clip, ready for the audio tower
struct PreparedAudio {
	int frames = 0;              // mel frames
	int tokens = 0;              // language-model tokens
	std::vector<float> mel;      // [frames, mel_bins]
	uint64_t hash = 0;           // of the samples and the chunk
};

// Mono samples at `rate` (resampled to the extractor's rate as torchaudio does) cut into chunks and their features
bool prepare_audio( const std::vector<float>& pcm, int rate, const AudioParams& p, std::vector<PreparedAudio>* out, std::string* error );

// An audio file as mono samples (channels averaged) at its own rate: WAV here, other containers and codecs through FFmpeg's libraries
// where the build has them
bool decode_audio( const uint8_t* data, size_t size, std::vector<float>* pcm, int* rate, std::string* error );
bool decode_audio_ffmpeg( const uint8_t* data, size_t size, std::vector<float>* pcm, int* rate, std::string* error );

} // namespace engine::loader
