// vall-e text to speech end to end (vall_e/tts.py TTS.generate / _split / _trim_silence / encode_prompt, and
// torchaudio.functional.resample for prompts at other rates).
#include "engine/tts/tts.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>

namespace engine::tts {

namespace {

constexpr int MAX_PHONEMES = 160;    // about 10 s of speech: longer texts go sentence by sentence
constexpr double PROMPT_S = 10.0;    // the longest prompt in training
constexpr double FPS = 44100.0 / 512.0;

void set_error( std::string* e, const std::string& m ) { if ( e ) { *e = m; } }

// re.split(r"(?<=[.!?。！？])\s*", text.strip()): cut after every sentence end, dropping the whitespace that follows it
std::vector<std::string> sentences( const std::string& text ) {
	auto is_ws = []( unsigned char c ) { return c == ' ' || ( c >= '\t' && c <= '\r' ); };
	size_t a = 0, b = text.size();
	while ( a < b && is_ws( text[a] ) ) { ++a; }
	while ( b > a && is_ws( text[b - 1] ) ) { --b; }
	const std::string t = text.substr( a, b - a );
	static const char* ends[] = { ".", "!", "?", "\xe3\x80\x82", "\xef\xbc\x81", "\xef\xbc\x9f" };  // . ! ? 。 ！ ？
	std::vector<std::string> out;
	std::string cur;
	for ( size_t i = 0; i < t.size(); ) {
		size_t n = 0;
		for ( const char* e : ends ) {
			const size_t l = std::strlen( e );
			if ( t.compare( i, l, e ) == 0 ) { n = l; break; }
		}
		if ( !n ) { cur += t[i++]; continue; }
		cur += t.substr( i, n );
		i += n;
		out.push_back( cur );
		cur.clear();
		while ( i < t.size() && is_ws( t[i] ) ) { ++i; }
	}
	out.push_back( cur );
	return out;
}

std::string strip( const std::string& s ) {
	size_t a = 0, b = s.size();
	while ( a < b && ( s[a] == ' ' || ( s[a] >= '\t' && s[a] <= '\r' ) ) ) { ++a; }
	while ( b > a && ( s[b - 1] == ' ' || ( s[b - 1] >= '\t' && s[b - 1] <= '\r' ) ) ) { --b; }
	return s.substr( a, b - a );
}

// _trim_silence: trailing 20 ms windows under -50 dBFS dropped, 50 ms kept after the last loud one
void trim_silence( std::vector<float>& w ) {
	const size_t window = 882, n = w.size() / window * window;
	long last = -1;
	for ( size_t k = 0; k * window < n; ++k ) {
		double e = 0;
		for ( size_t i = k * window; i < ( k + 1 ) * window; ++i ) { e += static_cast<double>( w[i] ) * w[i]; }
		if ( std::sqrt( e / window ) >= 0.003 ) { last = static_cast<long>( k ); }
	}
	if ( last < 0 ) { return; }
	w.resize( std::min( w.size(), static_cast<size_t>( last + 1 ) * window + static_cast<size_t>( 0.05 * 44100 ) ) );
}

template <typename T> T rd( const std::string& b, size_t at ) { T v; std::memcpy( &v, b.data() + at, sizeof( T ) ); return v; }

}  // namespace

bool Tts::load( const std::string& model, const std::string& codec, int device ) {
	return model_.load( model, device ) && codec_.load( codec, device ) && g2p_.init( model_.config().vocab );
}

std::vector<std::vector<int>> Tts::split( const std::string& text, const std::string& language ) {
	std::vector<int> ids = g2p_.encode( text, language );
	if ( static_cast<int>( ids.size() ) <= MAX_PHONEMES ) { return { ids }; }
	std::vector<std::string> parts;
	std::string cur;
	for ( const std::string& s : sentences( text ) ) {
		if ( !cur.empty() && static_cast<int>( g2p_.encode( cur + " " + s, language ).size() ) > MAX_PHONEMES ) {
			parts.push_back( cur );
			cur = s;
		} else {
			cur = strip( cur + " " + s );
		}
	}
	if ( !cur.empty() ) { parts.push_back( cur ); }
	std::vector<std::vector<int>> out;
	for ( const std::string& p : parts ) { if ( !strip( p ).empty() ) { out.push_back( g2p_.encode( p, language ) ); } }
	return out;
}

bool Tts::synthesize( const std::string& text, const std::vector<float>& prompt_audio, int prompt_rate, const SynthOptions& o, std::vector<float>& out,
	std::string* error ) {
	std::lock_guard<std::mutex> lock( mu_ );
	const auto& c = model_.config();
	TtsModel::Sampling s = model_.default_sampling();
	if ( o.cfg >= 0.0f ) { s.cfg = o.cfg; }
	if ( o.cfg_prompt >= 0.0f ) { s.cfg_prompt = o.cfg_prompt; }
	s.steps = o.steps;
	// the prompt: 44.1 kHz, its first 10 s, encoded, in patches (the last frame repeated to a whole patch, as to_patches)
	std::vector<uint8_t> prompt;
	if ( !prompt_audio.empty() ) {
		std::vector<float> a = prompt_rate == sample_rate() ? prompt_audio : resample( prompt_audio, prompt_rate, sample_rate() );
		a.resize( std::min( a.size(), static_cast<size_t>( PROMPT_S * sample_rate() ) ) );
		if ( a.size() < static_cast<size_t>( 4 * codec_.config().samples_per_frame ) ) { set_error( error, "the voice prompt is too short" ); return false; }
		prompt = codec_.encode( a );
		while ( prompt.size() / 32 % c.patch ) { prompt.insert( prompt.end(), prompt.end() - 32, prompt.end() ); }
	}
	const auto parts = split( text, o.language );
	size_t total = 0;
	for ( const auto& p : parts ) { total += p.size(); }
	if ( parts.empty() || total == 0 ) { set_error( error, "nothing to say: the text gave no phonemes" ); return false; }
	std::mt19937_64 rng( o.seed ? o.seed : std::random_device{}() );
	std::normal_distribution<float> normal( 0.0f, 1.0f );
	out.clear();
	for ( size_t k = 0; k < parts.size(); ++k ) {
		const auto& t = parts[k];
		if ( t.empty() ) { continue; }
		int n;
		if ( o.duration > 0.0f ) {
			n = std::max( 44, static_cast<int>( o.duration * FPS * t.size() / std::max<size_t>( 1, total ) ) );
		} else if ( c.length ) {
			n = std::max( 44, static_cast<int>( model_.predict_length( t, prompt ) * o.length_scale ) );
		} else {
			set_error( error, "this export has no length head: give a duration" );
			return false;
		}
		const int La = ( n + c.patch - 1 ) / c.patch;
		if ( static_cast<int>( t.size() ) + static_cast<int>( prompt.size() ) / c.values() + La > TtsModel::MAX_LEN ) {
			set_error( error, "the text and prompt are too long for one generation" );
			return false;
		}
		std::vector<int> order( La );
		std::iota( order.begin(), order.end(), 0 );
		std::shuffle( order.begin(), order.end(), rng );
		std::vector<std::vector<float>> noise( s.steps );
		int done = 0;
		for ( int st = 0; st < s.steps; ++st ) {
			const int target = TtsModel::revealed_after( st, s.steps, La );
			noise[st].resize( static_cast<size_t>( target - done ) * c.values() );
			for ( float& v : noise[st] ) { v = normal( rng ); }
			done = target;
		}
		const auto digits = model_.generate( t, prompt, n, order, noise, s );
		std::vector<float> w = codec_.decode( digits, n );
		if ( o.trim ) { trim_silence( w ); }
		if ( !out.empty() ) { out.resize( out.size() + static_cast<size_t>( o.pause * sample_rate() ), 0.0f ); }
		out.insert( out.end(), w.begin(), w.end() );
	}
	return true;
}

std::vector<float> Tts::resample( const std::vector<float>& x, int from, int to ) {
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

bool Tts::parse_wav( const std::string& b, std::vector<float>& mono, int& rate, std::string* error ) {
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

bool Tts::read_wav( const std::string& path, std::vector<float>& mono, int& rate, std::string* error ) {
	std::ifstream f( path, std::ios::binary );
	if ( !f ) { set_error( error, "can't open " + path ); return false; }
	std::stringstream ss;
	ss << f.rdbuf();
	return parse_wav( ss.str(), mono, rate, error );
}

std::string Tts::wav_bytes( const std::vector<float>& x, int rate ) {
	std::string b;
	auto put32 = [&]( uint32_t v ) { b.append( reinterpret_cast<const char*>( &v ), 4 ); };
	auto put16 = [&]( uint16_t v ) { b.append( reinterpret_cast<const char*>( &v ), 2 ); };
	const uint32_t data = static_cast<uint32_t>( x.size() * 2 );
	b += "RIFF"; put32( 36 + data ); b += "WAVEfmt "; put32( 16 ); put16( 1 ); put16( 1 ); put32( rate ); put32( rate * 2 ); put16( 2 ); put16( 16 );
	b += "data"; put32( data );
	for ( float v : x ) { put16( static_cast<uint16_t>( static_cast<int16_t>( std::lrint( std::max( -1.0f, std::min( 1.0f, v ) ) * 32767.0f ) ) ) ); }
	return b;
}

}  // namespace engine::tts
