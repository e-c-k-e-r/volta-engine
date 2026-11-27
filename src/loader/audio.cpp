// Audio input: Gemma 4's feature extractor (transformers' feature_extraction_gemma4.py: semicausal framing, a periodic Hann window, |rfft|,
// HTK mel filters without normalization, natural log over a floor), in double precision as numpy computes it; WAV files here, others decoded by
// video_decode.cpp's FFmpeg reader.
#include "engine/loader/audio.h"
#include "engine/loader/image.h"
#include "engine/tts/audio.h"

#include <algorithm>
#include <cmath>
#include <complex>

namespace engine::loader {

namespace {

double hz_to_mel( double f ) { return 2595.0 * std::log10( 1.0 + f / 700.0 ); }
double mel_to_hz( double m ) { return 700.0 * ( std::pow( 10.0, m / 2595.0 ) - 1.0 ); }

// transformers.audio_utils.mel_filter_bank( norm=None, mel_scale="htk" ): [bins, mel] triangles over the FFT bins
std::vector<double> mel_filters( const AudioParams& p ) {
	const int bins = p.fft_length / 2 + 1, n = p.mel_bins;
	const double lo = hz_to_mel( p.min_frequency ), hi = hz_to_mel( p.max_frequency );
	std::vector<double> edge( n + 2 ), fft_hz( bins );
	for ( int i = 0; i < n + 2; ++i ) edge[i] = mel_to_hz( lo + ( hi - lo ) * i / ( n + 1 ) );
	for ( int k = 0; k < bins; ++k ) fft_hz[k] = static_cast<double>( p.sample_rate / 2 ) * k / ( bins - 1 );
	std::vector<double> f( static_cast<size_t>( bins ) * n, 0.0 );
	for ( int k = 0; k < bins; ++k ) {
		for ( int m = 0; m < n; ++m ) {
			const double down = ( fft_hz[k] - edge[m] ) / ( edge[m + 1] - edge[m] );
			const double up = ( edge[m + 2] - fft_hz[k] ) / ( edge[m + 2] - edge[m + 1] );
			f[static_cast<size_t>( k ) * n + m] = std::max( 0.0, std::min( down, up ) );
		}
	}
	return f;
}

// In-place radix-2 FFT of a power-of-two length
void fft( std::vector<std::complex<double>>& a ) {
	const size_t n = a.size();
	for ( size_t i = 1, j = 0; i < n; ++i ) {
		size_t bit = n >> 1;
		for ( ; j & bit; bit >>= 1 ) j ^= bit;
		j ^= bit;
		if ( i < j ) std::swap( a[i], a[j] );
	}
	for ( size_t len = 2; len <= n; len <<= 1 ) {
		const double ang = -2.0 * M_PI / static_cast<double>( len );
		for ( size_t i = 0; i < n; i += len ) {
			for ( size_t k = 0; k < len / 2; ++k ) {
				const std::complex<double> w( std::cos( ang * k ), std::sin( ang * k ) );
				const std::complex<double> u = a[i + k], v = a[i + k + len / 2] * w;
				a[i + k] = u + v;
				a[i + k + len / 2] = u - v;
			}
		}
	}
}

// The log-mel frames of one chunk: half a frame of zeros before it, frames of frame_length + 1 samples every hop (the last sample unused)
void log_mel( const float* x, size_t n, const AudioParams& p, const std::vector<double>& filters, PreparedAudio* out ) {
	const int pad = p.frame_length / 2, bins = p.fft_length / 2 + 1, nm = p.mel_bins;
	const long padded = static_cast<long>( n ) + pad;
	const long frames = padded >= p.frame_length + 1 ? ( padded - ( p.frame_length + 1 ) ) / p.hop + 1 : 0;
	out->frames = static_cast<int>( frames );
	out->tokens = p.tokens( out->frames );
	out->mel.assign( static_cast<size_t>( frames ) * nm, 0.0f );
	std::vector<float> window( p.frame_length );
	for ( int i = 0; i < p.frame_length; ++i ) window[i] = static_cast<float>( 0.5 - 0.5 * std::cos( 2.0 * M_PI * i / p.frame_length ) );
	std::vector<std::complex<double>> buf( p.fft_length );
	std::vector<double> mag( bins );
	for ( long f = 0; f < frames; ++f ) {
		std::fill( buf.begin(), buf.end(), std::complex<double>( 0.0, 0.0 ) );
		for ( int i = 0; i < p.frame_length; ++i ) {
			const long s = f * p.hop + i - pad;
			const float v = s >= 0 && s < static_cast<long>( n ) ? x[s] : 0.0f;
			buf[i] = static_cast<double>( v * window[i] );   // the frame and the window are FP32, their product too
		}
		fft( buf );
		for ( int k = 0; k < bins; ++k ) mag[k] = std::abs( buf[k] );
		float* row = &out->mel[static_cast<size_t>( f ) * nm];
		for ( int m = 0; m < nm; ++m ) {
			double acc = 0.0;
			for ( int k = 0; k < bins; ++k ) acc += mag[k] * filters[static_cast<size_t>( k ) * nm + m];
			row[m] = static_cast<float>( std::log( acc + p.mel_floor ) );
		}
	}
}

} // namespace

bool prepare_audio( const std::vector<float>& pcm, int rate, const AudioParams& p, std::vector<PreparedAudio>* out, std::string* error ) {
	if ( pcm.empty() || rate <= 0 ) {
		if ( error ) *error = "the audio has no samples";
		return false;
	}
	const std::vector<float> x = rate == p.sample_rate ? pcm : engine::tts::resample( pcm, rate, p.sample_rate );
	const std::vector<double> filters = mel_filters( p );
	const size_t chunk = static_cast<size_t>( p.chunk_seconds ) * p.sample_rate;
	out->clear();
	for ( size_t off = 0; off < x.size(); off += chunk ) {
		PreparedAudio a;
		const size_t n = std::min( chunk, x.size() - off );
		log_mel( x.data() + off, n, p, filters, &a );
		if ( a.tokens == 0 ) {
			if ( out->empty() ) {
				if ( error ) *error = "the audio is shorter than one feature frame";
				return false;
			}
			break;   // a tail shorter than a frame
		}
		const uint64_t idx = off;
		a.hash = hash_bytes( &idx, sizeof( idx ), hash_bytes( x.data() + off, n * sizeof( float ) ) );
		out->push_back( std::move( a ) );
	}
	return true;
}

bool decode_audio( const uint8_t* data, size_t size, std::vector<float>* pcm, int* rate, std::string* error ) {
	if ( size >= 12 && std::equal( data, data + 4, "RIFF" ) && std::equal( data + 8, data + 12, "WAVE" ) ) {
		std::string why;
		if ( engine::tts::parse_wav( std::string( reinterpret_cast<const char*>( data ), size ), *pcm, *rate, &why ) ) return true;
		// an encoding parse_wav does not read: FFmpeg's, when the build has it
		if ( decode_audio_ffmpeg( data, size, pcm, rate, nullptr ) ) return true;
		if ( error ) *error = "WAV: " + why;
		return false;
	}
	return decode_audio_ffmpeg( data, size, pcm, rate, error );
}

} // namespace engine::loader
