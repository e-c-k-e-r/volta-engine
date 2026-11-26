// vall-e text to speech end to end (vall_e/tts.py TTS.generate / _split / _trim_silence / encode_prompt).
#include "engine/tts/tts.h"
#include "engine/tts/audio.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>

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

}  // namespace

bool Tts::load( const std::string& model, const std::string& codec, int device ) {
	// the codec's activations up front for the longest prompt (a sentence of up to as long decodes in them too): taken at startup, before a
	// language model beside it sizes its arena from what is left
	return model_.load( model, device ) && codec_.load( codec, device ) && g2p_.init( model_.config().vocab ) &&
		codec_.reserve( static_cast<int>( std::ceil( PROMPT_S * FPS ) ) );
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
	s.flow_steps = o.flow_steps;
	s.sway = o.sway;
	// the prompt: 44.1 kHz, its first 10 s, encoded, in patches (the last frame repeated to a whole patch, as to_patches)
	std::vector<uint8_t> prompt;
	if ( !prompt_audio.empty() ) {
		std::vector<float> a = prompt_rate == sample_rate() ? prompt_audio : resample( prompt_audio, prompt_rate, sample_rate() );
		a.resize( std::min( a.size(), static_cast<size_t>( PROMPT_S * sample_rate() ) ) );
		if ( a.size() < static_cast<size_t>( 4 * codec_.config().samples_per_frame ) ) { set_error( error, "the voice prompt is too short" ); return false; }
		prompt = codec_.encode( a );
		if ( prompt.empty() ) { set_error( error, "not enough GPU memory to encode the voice prompt" ); return false; }
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
			for ( float& v : noise[st] ) { v = normal( rng ) * o.temperature; }   // flow_sample: randn * temperature
			done = target;
		}
		const auto digits = model_.generate( t, prompt, n, order, noise, s );
		std::vector<float> w = codec_.decode( digits, n );
		if ( w.empty() ) {
			set_error( error, "not enough GPU memory to decode " + std::to_string( static_cast<int>( n / FPS ) ) + " s of speech at once" );
			return false;
		}
		if ( o.trim ) { trim_silence( w ); }
		if ( !out.empty() ) { out.resize( out.size() + static_cast<size_t>( o.pause * sample_rate() ), 0.0f ); }
		out.insert( out.end(), w.begin(), w.end() );
	}
	return true;
}

}  // namespace engine::tts
