// The vall-e text front-end: vall_e/emb/g2p.py (normalize_en, coerce_language, encode) and phonemizer 3.3's espeak
// backend (punctuation.py preserve / restore, espeak.py _postprocess_line, wrapper.py text_to_phonemes / set_voice) ported
// as they are, quirks included, so the output matches vall_e's character for character (tests/2_parity/tts_g2p checks it
// against vall_e on store texts). Text is handled as code points; Python's \s, \w and isupper() are Unicode-aware, so
// they're taken from the C.UTF-8 locale's classification.
#include "engine/tts/g2p.h"
#include "engine/core/log.h"
#include <espeak-ng/speak_lib.h>
#include <clocale>
#include <cwctype>
#include <locale.h>
#include <mutex>
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

std::u32string strip( const std::u32string& s ) {
	size_t a = 0, b = s.size();
	while ( a < b && is_space( s[a] ) ) { ++a; }
	while ( b > a && is_space( s[b - 1] ) ) { --b; }
	return s.substr( a, b - a );
}
std::u32string rstrip( const std::u32string& s ) {
	size_t b = s.size();
	while ( b > 0 && is_space( s[b - 1] ) ) { --b; }
	return s.substr( 0, b );
}
std::u32string replace_all( std::u32string s, const std::u32string& from, const std::u32string& to ) {  // str.replace
	std::u32string out;
	size_t i = 0;
	for ( size_t j; ( j = s.find( from, i ) ) != std::u32string::npos; i = j + from.size() ) { out += s.substr( i, j - i ) + to; }
	return out + s.substr( i );
}
bool starts_with( const std::u32string& s, const std::u32string& p ) { return s.size() >= p.size() && s.compare( 0, p.size(), p ) == 0; }
bool ends_with( const std::u32string& s, const std::u32string& p ) { return s.size() >= p.size() && s.compare( s.size() - p.size(), p.size(), p ) == 0; }

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

// ---- phonemizer: punctuation ----
constexpr std::u32string_view MARKS = U";:,.!?¡¿—…\"«»“”(){}[]";
bool is_mark( char32_t c ) { return MARKS.find( c ) != std::u32string_view::npos; }

struct Mark { int index; std::u32string mark; char position; };

// (\s*[marks]+\s*)+ found left to right: the maximal runs of whitespace and marks that hold a mark
std::vector<std::pair<size_t, size_t>> mark_runs( const std::u32string& s ) {
	std::vector<std::pair<size_t, size_t>> out;
	for ( size_t i = 0; i < s.size(); ) {
		if ( !is_space( s[i] ) && !is_mark( s[i] ) ) { ++i; continue; }
		size_t j = i;
		bool has = false;
		while ( j < s.size() && ( is_space( s[j] ) || is_mark( s[j] ) ) ) { has |= is_mark( s[j] ); ++j; }
		if ( has ) {
			// the run's leading whitespace belongs to the match, its trailing whitespace too; a whitespace-only stretch
			// before the first mark that isn't followed by one can't occur inside a run
			out.emplace_back( i, j );
		}
		i = j;
	}
	return out;
}

void preserve_line( const std::u32string& line0, int num, std::vector<std::u32string>& chunks, std::vector<Mark>& marks ) {
	std::u32string line = line0;
	const auto runs = mark_runs( line );
	if ( runs.empty() ) { chunks.push_back( line ); return; }
	if ( runs.size() == 1 && runs[0].first == 0 && runs[0].second == line.size() ) { marks.push_back( { num, line, 'A' } ); return; }
	std::vector<Mark> ms;
	for ( size_t k = 0; k < runs.size(); ++k ) {
		const std::u32string g = line.substr( runs[k].first, runs[k].second - runs[k].first );
		char position = 'I';
		if ( k == 0 && starts_with( line, g ) ) { position = 'B'; }
		else if ( k + 1 == runs.size() && ends_with( line, g ) ) { position = 'E'; }
		ms.push_back( { num, g, position } );
	}
	for ( const Mark& m : ms ) {
		// line.split(mark): the text before its first occurrence, and the rest
		const size_t at = line.find( m.mark );
		if ( at == std::u32string::npos ) { chunks.push_back( line ); line.clear(); continue; }
		chunks.push_back( line.substr( 0, at ) );
		line = line.substr( at + m.mark.size() );
	}
	chunks.push_back( line );
	marks.insert( marks.end(), ms.begin(), ms.end() );
}

std::vector<std::u32string> restore( std::vector<std::u32string> text, std::vector<Mark> marks ) {
	// Punctuation.restore with the default separator (word " ") and strip=True
	const std::u32string sep = U" ";
	std::vector<std::u32string> out;
	int pos = 0;
	size_t mi = 0, ti = 0;
	while ( ti < text.size() || mi < marks.size() ) {
		if ( mi == marks.size() ) {
			for ( ; ti < text.size(); ++ti ) { out.push_back( text[ti] ); }
		} else if ( ti == text.size() ) {
			std::u32string all;
			for ( ; mi < marks.size(); ++mi ) { all += marks[mi].mark; }
			out.push_back( all );
		} else {
			const Mark& cur = marks[mi];
			if ( cur.index == pos ) {
				const std::u32string mark = cur.mark;   // re.sub(' ', ' ', ...)
				++mi;
				if ( ends_with( text[ti], sep ) ) { text[ti] = text[ti].substr( 0, text[ti].size() - sep.size() ); }
				if ( cur.position == 'B' ) {
					text[ti] = mark + text[ti];
				} else if ( cur.position == 'E' ) {
					out.push_back( text[ti] + mark );
					++ti;
					++pos;
				} else if ( cur.position == 'A' ) {
					out.push_back( mark );
					++pos;
				} else if ( ti + 1 == text.size() ) {
					text[ti] = text[ti] + mark;
				} else {
					const std::u32string first = text[ti];
					++ti;
					text[ti] = first + mark + text[ti];
				}
			} else {
				out.push_back( text[ti] );
				++ti;
				++pos;
			}
		}
	}
	return out;
}

// espeak.py _postprocess_line with stress kept, no tie, the default separator and strip=True
std::u32string postprocess_line( std::u32string line ) {
	line = replace_all( replace_all( strip( line ), U"\n", U" " ), U"  ", U" " );
	std::u32string a;
	for ( char32_t c : line ) { if ( !( c == U'_' && !a.empty() && a.back() == U'_' ) ) { a += c; } }   // _+ -> _
	line = replace_all( a, U"_ ", U" " );
	// remove-flags: every \(.+?\)
	std::u32string b;
	for ( size_t i = 0; i < line.size(); ) {
		if ( line[i] == U'(' ) {
			const size_t close = line.find( U')', i + 2 );
			if ( i + 1 < line.size() && line[i + 1] != U'\n' && close != std::u32string::npos && line.find( U'\n', i + 1 ) > close ) { i = close + 1; continue; }
		}
		b += line[i++];
	}
	line = b;
	if ( line.empty() ) { return line; }
	std::u32string out;
	size_t i = 0;
	while ( true ) {
		const size_t j = line.find( U' ', i );
		std::u32string word = strip( line.substr( i, j == std::u32string::npos ? std::u32string::npos : j - i ) );
		word = replace_all( word, U"_", U"" );
		out += word + U" ";
		if ( j == std::u32string::npos ) { break; }
		i = j + 1;
	}
	out.pop_back();
	return out;
}

std::once_flag espeak_once;
bool espeak_ok = false;

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

bool G2P::init( const std::vector<std::string>& vocab ) {
	for ( size_t i = 0; i < vocab.size(); ++i ) {
		const std::u32string c = utf8_to_u32( vocab[i] );
		if ( c.size() == 1 ) { ids_[c[0]] = static_cast<int>( i ) + 1; }
	}
	std::call_once( espeak_once, [] {
		espeak_ok = espeak_Initialize( AUDIO_OUTPUT_SYNCHRONOUS, 0, nullptr, 0 ) > 0;
	} );
	if ( !espeak_ok ) {
		LOG_ERROR( "g2p: espeak-ng failed to initialize" );
		return false;
	}
	// wrapper.py set_voice: the first listed voice of each language code (a voice's first language, after its priority byte)
	for ( const espeak_VOICE** v = espeak_ListVoices( nullptr ); v && *v; ++v ) {
		if ( !( *v )->languages || !( *v )->identifier ) { continue; }
		const std::string code = ( *v )->languages + 1;
		voices_.emplace( code, ( *v )->identifier );
	}
	return true;
}

bool G2P::set_voice( const std::string& code ) {
	if ( code == voice_ ) { return true; }
	const auto it = voices_.find( code );
	if ( it == voices_.end() ) {
		LOG_ERROR( "g2p: espeak has no voice for \"%s\"", code.c_str() );
		return false;
	}
	if ( espeak_SetVoiceByName( it->second.c_str() ) != EE_OK ) {
		LOG_ERROR( "g2p: failed to load the voice for \"%s\"", code.c_str() );
		return false;
	}
	voice_ = code;
	return true;
}

std::string G2P::espeak_phonemes( const std::string& text ) {
	// wrapper.py text_to_phonemes: UTF-8 in, IPA out with '_' between phonemes, a clause per call, joined by spaces
	const void* p = text.c_str();
	std::string out;
	bool first = true;
	while ( p ) {
		const char* ph = espeak_TextToPhonemes( &p, espeakCHARS_UTF8, ( '_' << 8 ) | 0x02 );
		if ( ph && *ph ) {
			if ( !first ) { out += ' '; }
			out += ph;
			first = false;
		}
	}
	return out;
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
	if ( !set_voice( lang ) ) { return ""; }
	// BaseBackend.phonemize with preserve_punctuation: the pieces between runs of marks, each phonemized as a line
	std::vector<std::u32string> chunks;
	std::vector<Mark> marks;
	preserve_line( utf8_to_u32( text ), 0, chunks, marks );
	std::vector<std::u32string> kept;
	for ( auto& c : chunks ) { if ( !c.empty() ) { kept.push_back( c ); } }
	std::vector<std::u32string> phonemized;
	for ( const auto& c : kept ) { phonemized.push_back( postprocess_line( utf8_to_u32( espeak_phonemes( u32_to_utf8( c ) ) ) ) ); }
	const auto lines = restore( phonemized, marks );
	if ( lines.empty() ) {
		LOG_ERROR( "g2p: nothing to phonemize in \"%s\"", text_in.c_str() );
		return "";
	}
	std::u32string ph = lines[0];   // encode() keeps the first of phonemize()'s outputs
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
