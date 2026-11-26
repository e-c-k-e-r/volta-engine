// The TTS prompt resampler (engine::tts::Tts::resample, torchaudio.functional.resample's sinc_interp_hann) on a WAV file: writes the mono
// samples resampled to the given rate as raw FP32, for comparing against torchaudio.
//   test_tts_resample <in.wav> <rate> <out.f32>
#include "engine/tts/tts.h"
#include <cstdio>
#include <cstdlib>

int main( int argc, char** argv ) {
	if ( argc < 4 ) {
		std::fprintf( stderr, "usage: test_tts_resample <in.wav> <rate> <out.f32>\n" );
		return 2;
	}
	std::vector<float> x;
	int rate = 0;
	std::string err;
	if ( !engine::tts::Tts::read_wav( argv[1], x, rate, &err ) ) { std::fprintf( stderr, "%s\n", err.c_str() ); return 1; }
	const auto y = engine::tts::Tts::resample( x, rate, std::atoi( argv[2] ) );
	FILE* f = std::fopen( argv[3], "wb" );
	if ( !f ) { return 1; }
	std::fwrite( y.data(), 4, y.size(), f );
	std::fclose( f );
	std::printf( "%zu samples at %d Hz -> %zu at %s Hz\n", x.size(), rate, y.size(), argv[2] );
	return 0;
}
