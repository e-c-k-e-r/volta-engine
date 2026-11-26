// WAV files and the prompt resampler of the TTS pipeline (torchaudio.functional.resample for prompts at other rates).
#include "engine/tts/audio.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>

namespace engine::tts {

namespace {

void set_error( std::string* e, const std::string& m ) { if ( e ) { *e = m; } }

template <typename T> T rd( const std::string& b, size_t at ) { T v; std::memcpy( &v, b.data() + at, sizeof( T ) ); return v; }

}  // namespace

std::vector<float> resample( const std::vector<float>& x, int from, int to ) {
	if ( from == to || x.empty() ) { return x; }
	const int g = std::gcd( from, to ), orig = from / g, nw = to / g;
	const int lpw = 6;
	const float base = static_cast<float>( std::min( orig, nw ) ) * 0.99f;
	const int width = static_cast<int>( std::ceil( lpw * orig / static_cast<double>( base ) ) );
	const int K = 2 * width + orig;
	// kernels [nw][K] in FP32, as _get_sinc_resample_kernel computes them for an FP32 waveform
	std::vector<float> ker( static_cast<size_t>( nw ) * K );
	for ( int j = 0; j < nw; ++j ) {
		for ( int i = 0; i < K; ++i ) {
			const float idx = static_cast<float>( i - width ) / static_cast<float>( orig );
			float t = ( static_cast<float>( -j ) / static_cast<float>( nw ) + idx ) * base;
			t = std::min( static_cast<float>( lpw ), std::max( -static_cast<float>( lpw ), t ) );
			const float c = std::cos( t * static_cast<float>( M_PI ) / lpw / 2.0f ), window = c * c;
			t *= static_cast<float>( M_PI );
			const float sinc = t == 0.0f ? 1.0f : std::sin( t ) / t;
			ker[static_cast<size_t>( j ) * K + i] = sinc * ( window * ( base / orig ) );
		}
	}
	// zero-padded by width on the left and width + orig on the right, strided by orig, the phases interleaved
	const long L = static_cast<long>( x.size() );
	const long frames = ( L + 2 * width + orig - K ) / orig + 1;
	const long target = static_cast<long>( std::ceil( static_cast<double>( nw ) * L / orig ) );
	std::vector<float> y( std::min( frames * nw, target ) );
	for ( long f = 0; f < frames; ++f ) {
		for ( int j = 0; j < nw; ++j ) {
			const long o = f * nw + j;
			if ( o >= static_cast<long>( y.size() ) ) { break; }
			float acc = 0.0f;
			const float* kj = ker.data() + static_cast<size_t>( j ) * K;
			for ( int i = 0; i < K; ++i ) {
				const long s = f * orig + i - width;
				if ( s >= 0 && s < L ) { acc += kj[i] * x[s]; }
			}
			y[o] = acc;
		}
	}
	return y;
}

bool parse_wav( const std::string& b, std::vector<float>& mono, int& rate, std::string* error ) {
	if ( b.size() < 12 || b.compare( 0, 4, "RIFF" ) || b.compare( 8, 4, "WAVE" ) ) { set_error( error, "not a WAV file" ); return false; }
	int format = 0, channels = 0, bits = 0;
	size_t data = 0, data_len = 0;
	for ( size_t at = 12; at + 8 <= b.size(); ) {
		const std::string id = b.substr( at, 4 );
		const size_t len = rd<uint32_t>( b, at + 4 );
		if ( id == "fmt " && len >= 16 && at + 8 + len <= b.size() ) {
			format = rd<uint16_t>( b, at + 8 );
			channels = rd<uint16_t>( b, at + 10 );
			rate = static_cast<int>( rd<uint32_t>( b, at + 12 ) );
			bits = rd<uint16_t>( b, at + 22 );
			if ( format == 0xfffe && len >= 26 ) { format = rd<uint16_t>( b, at + 32 ); }  // WAVE_FORMAT_EXTENSIBLE: the subformat's tag
		} else if ( id == "data" ) {
			data = at + 8;
			data_len = std::min( len, b.size() - data );
			break;
		}
		at += 8 + len + ( len & 1 );
	}
	if ( !data || !channels || rate <= 0 ) { set_error( error, "a WAV file without its format or data" ); return false; }
	if ( !( ( format == 1 && ( bits == 8 || bits == 16 || bits == 24 || bits == 32 ) ) || ( format == 3 && bits == 32 ) ) ) {
		set_error( error, "WAV: only PCM 8 / 16 / 24 / 32-bit and 32-bit float are read" );
		return false;
	}
	const size_t bytes = bits / 8, n = data_len / ( bytes * channels );
	mono.assign( n, 0.0f );
	for ( size_t i = 0; i < n; ++i ) {
		float sum = 0.0f;
		for ( int ch = 0; ch < channels; ++ch ) {
			const size_t at = data + ( i * channels + ch ) * bytes;
			float v;
			if ( format == 3 ) { v = rd<float>( b, at ); }
			else if ( bits == 8 ) { v = ( static_cast<unsigned char>( b[at] ) - 128 ) / 128.0f; }
			else if ( bits == 16 ) { v = rd<int16_t>( b, at ) / 32768.0f; }
			else if ( bits == 24 ) {
				const int32_t s = static_cast<int32_t>( ( static_cast<uint32_t>( static_cast<unsigned char>( b[at] ) ) << 8 ) |
					( static_cast<uint32_t>( static_cast<unsigned char>( b[at + 1] ) ) << 16 ) | ( static_cast<uint32_t>( static_cast<unsigned char>( b[at + 2] ) ) << 24 ) ) >> 8;
				v = s / 8388608.0f;
			} else { v = rd<int32_t>( b, at ) / 2147483648.0f; }
			sum += v;
		}
		mono[i] = sum / channels;
	}
	return true;
}

bool read_wav( const std::string& path, std::vector<float>& mono, int& rate, std::string* error ) {
	std::ifstream f( path, std::ios::binary );
	if ( !f ) { set_error( error, "can't open " + path ); return false; }
	std::stringstream ss;
	ss << f.rdbuf();
	return parse_wav( ss.str(), mono, rate, error );
}

std::string wav_bytes( const std::vector<float>& x, int rate ) {
	std::string b;
	auto put32 = [&]( uint32_t v ) { b.append( reinterpret_cast<const char*>( &v ), 4 ); };
	auto put16 = [&]( uint16_t v ) { b.append( reinterpret_cast<const char*>( &v ), 2 ); };
	const uint32_t data = static_cast<uint32_t>( x.size() * 2 );
	b += "RIFF"; put32( 36 + data ); b += "WAVEfmt "; put32( 16 ); put16( 1 ); put16( 1 ); put32( rate ); put32( rate * 2 ); put16( 2 ); put16( 16 );
	b += "data"; put32( data );
	for ( float v : x ) { put16( static_cast<uint16_t>( static_cast<int16_t>( std::lrint( std::max( -1.0f, std::min( 1.0f, v ) ) * 32767.0f ) ) ) ); }
	return b;
}

bool write_wav( const std::string& path, const std::vector<float>& x, int rate ) {
	FILE* f = std::fopen( path.c_str(), "wb" );
	if ( !f ) { return false; }
	const std::string b = wav_bytes( x, rate );
	const bool ok = std::fwrite( b.data(), 1, b.size(), f ) == b.size();
	return std::fclose( f ) == 0 && ok;
}

}  // namespace engine::tts
