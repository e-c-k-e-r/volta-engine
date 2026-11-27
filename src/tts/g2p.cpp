// The vall-e text front-end: vall_e/emb/g2p.py (normalize_en, coerce_language, encode) ported as it is, quirks included,
// over propan2ol (espeak-ng 1.52.0 plus phonemizer 3.3's espeak backend, character for character), so the output matches
// vall_e's (tests/2_parity/test_tts_g2p checks it against vall_e on store texts). Text is handled as code points; Python's
// \s, \w and isupper() are Unicode-aware, so they're taken from the C.UTF-8 locale's classification.
#include "engine/tts/g2p.h"
#include "engine/core/log.h"
#include <propan2ol/propan2ol.h>
#include <clocale>
#include <cwctype>
#include <locale.h>
#include <string_view>

namespace engine::tts {

namespace {

std::u32string utf8_to_u32( const std::string& s ) {
	std::u32string out;
	for ( size_t i = 0; i < s.size(); ) {
		const unsigned char c = static_cast<unsigned char>( s[i] );
		char32_t cp;
		int n;
		if ( c < 0x80 ) { cp = c; n = 1; }
		else if ( ( c >> 5 ) == 6 ) { cp = c & 0x1f; n = 2; }
		else if ( ( c >> 4 ) == 14 ) { cp = c & 0x0f; n = 3; }
		else if ( ( c >> 3 ) == 30 ) { cp = c & 0x07; n = 4; }
		else { cp = 0xfffd; n = 1; }
		if ( i + n > s.size() ) { cp = 0xfffd; n = 1; }
		for ( int k = 1; k < n; ++k ) { cp = ( cp << 6 ) | ( static_cast<unsigned char>( s[i + k] ) & 0x3f ); }
		out += cp;
		i += n;
	}
	return out;
}

std::string u32_to_utf8( const std::u32string& s ) {
	std::string out;
	for ( char32_t c : s ) {
		if ( c < 0x80 ) { out += static_cast<char>( c ); }
		else if ( c < 0x800 ) { out += static_cast<char>( 0xc0 | ( c >> 6 ) ); out += static_cast<char>( 0x80 | ( c & 0x3f ) ); }
		else if ( c < 0x10000 ) {
			out += static_cast<char>( 0xe0 | ( c >> 12 ) ); out += static_cast<char>( 0x80 | ( ( c >> 6 ) & 0x3f ) ); out += static_cast<char>( 0x80 | ( c & 0x3f ) );
		} else {
			out += static_cast<char>( 0xf0 | ( c >> 18 ) ); out += static_cast<char>( 0x80 | ( ( c >> 12 ) & 0x3f ) );
			out += static_cast<char>( 0x80 | ( ( c >> 6 ) & 0x3f ) ); out += static_cast<char>( 0x80 | ( c & 0x3f ) );
		}
	}
	return out;
}

locale_t utf8_locale() {
	static locale_t l = newlocale( LC_CTYPE_MASK, "C.UTF-8", static_cast<locale_t>( 0 ) );
	return l;
}

// Python's str.isspace() (what \s and strip() use)
bool is_space( char32_t c ) {
	return ( c >= 0x09 && c <= 0x0d ) || ( c >= 0x1c && c <= 0x20 ) || c == 0x85 || c == 0xa0 || c == 0x1680 || ( c >= 0x2000 && c <= 0x200a ) ||
		c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}
bool is_word( char32_t c ) { return c == U'_' || iswalnum_l( static_cast<wint_t>( c ), utf8_locale() ); }
bool is_upper( char32_t c ) { return iswupper_l( static_cast<wint_t>( c ), utf8_locale() ); }
bool str_is_upper( const std::u32string& s ) {
	// str.isupper(): at least one cased character and no lowercase ones (here a single character or none)
	bool cased = false;
	for ( char32_t c : s ) {
		if ( iswlower_l( static_cast<wint_t>( c ), utf8_locale() ) ) { return false; }
		cased |= is_upper( c );
	}
	return cased;
}

std::u32string rstrip( const std::u32string& s ) {
	size_t b = s.size();
	while ( b > 0 && is_space( s[b - 1] ) ) { --b; }
	return s.substr( 0, b );
}

// ---- normalize_en ----
struct Abbrev { const char32_t* word; const char32_t* full; };
// the regex's alternation order: Mr before Mrs, so "Mrs." fails "Mr" at the "." and matches "Mrs"
constexpr Abbrev ABBREVIATIONS[] = {
	{ U"Mr", U"Mister" }, { U"Mrs", U"Missus" }, { U"Ms", U"Miz" }, { U"Dr", U"Doctor" }, { U"Jr", U"Junior" }, { U"Sr", U"Senior" },
	{ U"Lt", U"Lieutenant" }, { U"Sgt", U"Sergeant" }, { U"Capt", U"Captain" }, { U"Prof", U"Professor" }, { U"Mt", U"Mount" },
	{ U"vs", U"versus" }, { U"etc", U"et cetera" }, { U"St", nullptr },
};

// \b(word)\.(\s+(?=\S)|$) at i: the match's end and its gap's start, or false
bool match_abbrev( const std::u32string& t, size_t i, const char32_t* word, size_t& end, size_t& gap ) {
	const std::u32string_view w( word );
	if ( i > 0 && is_word( t[i - 1] ) ) { return false; }
	if ( t.compare( i, w.size(), w ) != 0 || i + w.size() >= t.size() || t[i + w.size()] != U'.' ) { return false; }
	gap = i + w.size() + 1;
	size_t j = gap;
	while ( j < t.size() && is_space( t[j] ) ) { ++j; }
	if ( j > gap && j < t.size() ) { end = j; return true; }   // \s+ then a non-space
	// $: the end, or before a final newline
	if ( gap == t.size() || ( gap + 1 == t.size() && t[gap] == U'\n' ) ) { end = gap; return true; }
	return false;
}

}  // namespace

std::string G2P::normalize_en( const std::string& text ) {
	const std::u32string t = utf8_to_u32( text );
	std::u32string out;
	size_t i = 0, last = 0;
	while ( i < t.size() ) {
		size_t end = 0, gap = 0;
		const Abbrev* hit = nullptr;
		for ( const Abbrev& a : ABBREVIATIONS ) {
			if ( match_abbrev( t, i, a.word, end, gap ) ) { hit = &a; break; }
		}
		if ( !hit ) { ++i; continue; }
		const std::u32string g1 = hit->word, gp = t.substr( gap, end - gap );
		const std::u32string nxt = t.substr( end, 1 );
		std::u32string word;
		if ( !hit->full ) {
			// "Baker St." / "Main St. Then": after a name it's a street; "on St. Patrick" / "St. Louis": a saint
			const std::u32string before = rstrip( t.substr( 0, i ) );
			const size_t sp = before.rfind( U' ' );
			const std::u32string prev = sp == std::u32string::npos ? before : before.substr( sp + 1 );
			word = ( !prev.empty() && str_is_upper( prev.substr( 0, 1 ) ) ) || !str_is_upper( nxt ) ? U"Street" : U"Saint";
		} else {
			word = hit->full;
		}
		const bool at_end = nxt.empty() || ( ( g1 == U"etc" || g1 == U"St" ) && word != U"Saint" && str_is_upper( nxt ) );
		out += t.substr( last, i - last ) + word + ( at_end ? U"." : U"" ) + gp;
		i = last = end;
	}
	out += t.substr( last );
	return u32_to_utf8( out );
}

std::string G2P::coerce_language( const std::string& lang ) {
	if ( lang == "en" ) { return "en-us"; }
	if ( lang == "fr" ) { return "fr-fr"; }
	if ( lang.compare( 0, 2, "zh" ) == 0 ) { return "cmn-latn-pinyin"; }
	return lang;
}

G2P::G2P() = default;
G2P::~G2P() = default;

bool G2P::init( const std::vector<std::string>& vocab ) {
	for ( size_t i = 0; i < vocab.size(); ++i ) {
		const std::u32string c = utf8_to_u32( vocab[i] );
		if ( c.size() == 1 ) { ids_[c[0]] = static_cast<int>( i ) + 1; }
	}
	return true;
}

const propan2ol::Phonemizer* G2P::phonemizer( const std::string& code ) {
	auto it = phonemizers_.find( code );
	if ( it == phonemizers_.end() ) {
		std::string error;
		auto p = propan2ol::Phonemizer::create( propan2ol::DEFAULT_PROFILE, code, &error );
		if ( !p ) {
			LOG_ERROR( "g2p: no phonemizer for \"%s\": %s", code.c_str(), error.c_str() );
			return nullptr;
		}
		it = phonemizers_.emplace( code, std::move( p ) ).first;
	}
	return it->second.get();
}

std::string G2P::phonemize( const std::string& text_in, const std::string& language ) {
	const std::string lang = coerce_language( language );
	std::string text = text_in;
	if ( lang.compare( 0, 2, "en" ) == 0 ) { text = normalize_en( text ); }
	if ( lang.compare( 0, 2, "ja" ) == 0 ) {
		// coerce_to_hiragana without pykakasi: katakana to hiragana; kanji need its dictionary
		std::u32string t = utf8_to_u32( text );
		for ( char32_t& c : t ) {
			if ( c >= 0x30a1 && c <= 0x30f6 ) { c -= 0x60; }
			else if ( ( c >= 0x4e00 && c <= 0x9fff ) || ( c >= 0x3400 && c <= 0x4dbf ) ) {
				LOG_ERROR( "g2p: Japanese text with kanji needs pykakasi's readings, which aren't ported: give it as kana" );
				return "";
			}
		}
		text = u32_to_utf8( t );
	}
	const propan2ol::Phonemizer* phonemizer = this->phonemizer( lang );
	if ( !phonemizer ) { return ""; }
	// vall_e's backend: EspeakBackend( language, preserve_punctuation=True, with_stress=True ).phonemize( [text], strip=True ), the
	// language-switch flags removed
	propan2ol::Options o;
	o.preserve_punctuation = true;
	o.with_stress = true;
	o.strip = true;
	o.language_switch = propan2ol::LanguageSwitch::RemoveFlags;
	const std::vector<std::string> lines = phonemizer->phonemize( std::vector<std::string>{ text }, o );
	if ( lines.empty() ) {
		LOG_ERROR( "g2p: nothing to phonemize in \"%s\"", text_in.c_str() );
		return "";
	}
	std::u32string ph = utf8_to_u32( lines[0] );   // encode() keeps the first of phonemize()'s outputs
	if ( lang == "cmn-latn-pinyin" ) {
		for ( char32_t& c : ph ) {
			switch ( c ) {
				case U'1': c = U'ˇ'; break;
				case U'2': c = U'ˉ'; break;
				case U'3': c = U'ˊ'; break;
				case U'4': c = U'ˋ'; break;
				case U'5': c = U'_'; break;
				default: break;
			}
		}
	}
	return u32_to_utf8( ph );
}

std::vector<int> G2P::encode( const std::string& text, const std::string& language ) {
	std::vector<int> out;
	for ( char32_t c : utf8_to_u32( phonemize( text, language ) ) ) {
		const auto it = ids_.find( c );
		if ( it != ids_.end() ) { out.push_back( it->second ); }
	}
	return out;
}

}  // namespace engine::tts
