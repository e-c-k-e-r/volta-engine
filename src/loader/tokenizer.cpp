#include "engine/loader/tokenizer.h"
#include "engine/core/log.h"

#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cctype>

namespace engine::loader {

// Pre-tokenizer: an exact matcher for the two split regexes the supported checkpoints use (tokenizer.json is checked
// against them at load time):
//   Qwen: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//   GLM:  (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// Alternatives are tried in order, each greedy with the regex's backtracking.
namespace unicode_util {

#include "unicode_tables.inc"

static const char* const kQwenPattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
// Qwen3.8-Flash-Next: the Qwen split with marks no longer part of letters (as GLM) and single digits
static const char* const kFlashPattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
static const char* const kGlmPattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

template <size_t N>
static bool in_table( const uint32_t ( &t )[N][2], uint32_t cp ) {
	size_t lo = 0, hi = N;
	while ( lo < hi ) {
		const size_t mid = ( lo + hi ) / 2;
		if ( cp < t[mid][0] ) hi = mid;
		else if ( cp > t[mid][1] ) lo = mid + 1;
		else return true;
	}
	return false;
}

static inline bool is_letter( uint32_t cp ) { return in_table( kUnicodeLetter, cp ); }
static inline bool is_mark( uint32_t cp ) { return in_table( kUnicodeMark, cp ); }
static inline bool is_number( uint32_t cp ) { return in_table( kUnicodeNumber, cp ); }
static inline bool is_space( uint32_t cp ) { return in_table( kUnicodeSpace, cp ); }

// Decodes the code point at text[i] and advances i; malformed bytes decode as themselves (one byte).
static inline uint32_t utf8_next( const std::string& str, size_t& i ) {
	const unsigned char c = static_cast<unsigned char>( str[i] );
	auto cont = [&]( size_t k ) { return i + k < str.size() && ( static_cast<unsigned char>( str[i + k] ) & 0xC0 ) == 0x80; };
	uint32_t cp = c;
	size_t len = 1;
	if ( c >= 0xC2 && c < 0xE0 && cont( 1 ) ) {
		cp = ( ( c & 0x1Fu ) << 6 ) | ( static_cast<unsigned char>( str[i + 1] ) & 0x3Fu );
		len = 2;
	}
	else if ( c >= 0xE0 && c < 0xF0 && cont( 1 ) && cont( 2 ) ) {
		cp = ( ( c & 0x0Fu ) << 12 ) | ( ( static_cast<unsigned char>( str[i + 1] ) & 0x3Fu ) << 6 ) | ( static_cast<unsigned char>( str[i + 2] ) & 0x3Fu );
		len = 3;
	}
	else if ( c >= 0xF0 && c < 0xF5 && cont( 1 ) && cont( 2 ) && cont( 3 ) ) {
		cp = ( ( c & 0x07u ) << 18 ) | ( ( static_cast<unsigned char>( str[i + 1] ) & 0x3Fu ) << 12 ) | ( ( static_cast<unsigned char>( str[i + 2] ) & 0x3Fu ) << 6 ) | ( static_cast<unsigned char>( str[i + 3] ) & 0x3Fu );
		len = 4;
	}
	i += len;
	return cp;
}

struct Splitter {
	const std::string& s;
	bool marks_are_letters;   // Qwen: [\p{L}\p{M}]+ and no \p{M} in the symbol class
	int max_digits;

	bool letter( uint32_t cp ) const { return is_letter( cp ) || ( marks_are_letters && is_mark( cp ) ); }

	// code point at position p (p < size) and the position after it
	uint32_t at( size_t p, size_t* next ) const {
		size_t q = p;
		const uint32_t cp = utf8_next( s, q );
		*next = q;
		return cp;
	}

	// 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
	size_t contraction( size_t i ) const {
		if ( s[i] != '\'' || i + 1 >= s.size() ) return 0;
		auto lc = [&]( size_t k ) { return k < s.size() ? static_cast<char>( std::tolower( static_cast<unsigned char>( s[k] ) ) ) : '\0'; };
		const char a = lc( i + 1 ), b = lc( i + 2 );
		if ( a == 's' || a == 't' || a == 'm' || a == 'd' ) return 2;
		if ( ( a == 'r' && b == 'e' ) || ( a == 'v' && b == 'e' ) || ( a == 'l' && b == 'l' ) ) return 3;
		return 0;
	}

	// run of letters starting at p; returns the end (p if none)
	size_t letters( size_t p ) const {
		while ( p < s.size() ) {
			size_t nx;
			if ( !letter( at( p, &nx ) ) ) break;
			p = nx;
		}
		return p;
	}

	// 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
	size_t word( size_t i ) const {
		size_t nx;
		const uint32_t c = at( i, &nx );
		if ( c != '\r' && c != '\n' && !is_letter( c ) && !is_number( c ) ) {
			const size_t e = letters( nx );
			if ( e > nx ) return e;
		}
		const size_t e = letters( i );
		return e > i ? e : 0;
	}

	// 4. ' '?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
	size_t symbols( size_t i ) const {
		size_t p = i;
		if ( s[p] == ' ' ) ++p;
		const size_t start = p;
		while ( p < s.size() ) {
			size_t nx;
			const uint32_t c = at( p, &nx );
			if ( is_space( c ) || is_letter( c ) || is_number( c ) || ( marks_are_letters && is_mark( c ) ) ) break;
			p = nx;
		}
		if ( p == start ) return 0;
		while ( p < s.size() && ( s[p] == '\r' || s[p] == '\n' ) ) ++p;
		return p;
	}

	size_t space_run_end( size_t i ) const {
		size_t p = i;
		while ( p < s.size() ) {
			size_t nx;
			if ( !is_space( at( p, &nx ) ) ) break;
			p = nx;
		}
		return p;
	}

	size_t next( size_t i ) const {
		if ( size_t n = contraction( i ) ) return i + n;
		if ( size_t e = word( i ) ) return e;
		size_t nx;
		const uint32_t c = at( i, &nx );
		if ( is_number( c ) ) {
			size_t p = nx;
			for ( int d = 1; d < max_digits && p < s.size(); ++d ) {
				size_t nn;
				if ( !is_number( at( p, &nn ) ) ) break;
				p = nn;
			}
			return p;
		}
		if ( size_t e = symbols( i ) ) return e;
		if ( is_space( c ) ) {
			const size_t e = space_run_end( i );
			// \s*[\r\n]+ : the last newline in the run is where \s* stops; newlines directly after it join the match
			size_t last_nl = std::string::npos;
			for ( size_t p = i; p < e; ++p ) {
				if ( s[p] == '\r' || s[p] == '\n' ) last_nl = p;
			}
			if ( last_nl != std::string::npos ) {
				size_t p = last_nl;
				while ( p < e && ( s[p] == '\r' || s[p] == '\n' ) ) ++p;
				return p;
			}
			// \s+(?!\S) : all of the run at the end of the text, else all but its last character
			if ( e == s.size() ) return e;
			size_t last = i, p = i;
			while ( p < e ) {
				last = p;
				at( p, &p );
			}
			if ( last > i ) return last;
			return e; // \s+
		}
		return nx; // a code point no alternative matches: the regex skips it, the splitter keeps it alone
	}
};

// Unicode NFC (canonical decomposition, canonical reordering, canonical composition), as the Qwen tokenizer.json normalizer
static int combining_class( uint32_t cp ) {
	size_t lo = 0, hi = sizeof( kCombiningClass ) / sizeof( kCombiningClass[0] );
	while ( lo < hi ) {
		const size_t mid = ( lo + hi ) / 2;
		if ( cp < kCombiningClass[mid][0] ) hi = mid;
		else if ( cp > kCombiningClass[mid][1] ) lo = mid + 1;
		else return static_cast<int>( kCombiningClass[mid][2] );
	}
	return 0;
}

static void append_decomposed( uint32_t cp, std::vector<uint32_t>& out ) {
	constexpr uint32_t SBase = 0xAC00, LBase = 0x1100, VBase = 0x1161, TBase = 0x11A7, LCount = 19, VCount = 21, TCount = 28, NCount = VCount * TCount, SCount = LCount * NCount;
	if ( cp >= SBase && cp < SBase + SCount ) {
		const uint32_t si = cp - SBase;
		out.push_back( LBase + si / NCount );
		out.push_back( VBase + ( si % NCount ) / TCount );
		if ( si % TCount ) out.push_back( TBase + si % TCount );
		return;
	}
	size_t lo = 0, hi = sizeof( kNfdIndex ) / sizeof( kNfdIndex[0] );
	while ( lo < hi ) {
		const size_t mid = ( lo + hi ) / 2;
		if ( cp < kNfdIndex[mid][0] ) hi = mid;
		else if ( cp > kNfdIndex[mid][0] ) lo = mid + 1;
		else {
			for ( uint32_t k = 0; k < kNfdIndex[mid][2]; ++k ) out.push_back( kNfdData[kNfdIndex[mid][1] + k] );
			return;
		}
	}
	out.push_back( cp );
}

static uint32_t compose_pair( uint32_t a, uint32_t b ) {
	constexpr uint32_t SBase = 0xAC00, LBase = 0x1100, VBase = 0x1161, TBase = 0x11A7, LCount = 19, VCount = 21, TCount = 28, NCount = VCount * TCount;
	if ( a >= LBase && a < LBase + LCount && b >= VBase && b < VBase + VCount ) return SBase + ( ( a - LBase ) * VCount + ( b - VBase ) ) * TCount;
	if ( a >= SBase && a < SBase + LCount * NCount && ( a - SBase ) % TCount == 0 && b > TBase && b < TBase + TCount ) return a + ( b - TBase );
	size_t lo = 0, hi = sizeof( kComposePairs ) / sizeof( kComposePairs[0] );
	while ( lo < hi ) {
		const size_t mid = ( lo + hi ) / 2;
		const uint32_t* e = kComposePairs[mid];
		if ( a < e[0] || ( a == e[0] && b < e[1] ) ) hi = mid;
		else if ( a > e[0] || b > e[1] ) lo = mid + 1;
		else return e[2];
	}
	return 0;
}

static void append_utf8( uint32_t cp, std::string& out ) {
	if ( cp < 0x80 ) {
		out.push_back( static_cast<char>( cp ) );
	}
	else if ( cp < 0x800 ) {
		out.push_back( static_cast<char>( 0xC0 | ( cp >> 6 ) ) );
		out.push_back( static_cast<char>( 0x80 | ( cp & 0x3F ) ) );
	}
	else if ( cp < 0x10000 ) {
		out.push_back( static_cast<char>( 0xE0 | ( cp >> 12 ) ) );
		out.push_back( static_cast<char>( 0x80 | ( ( cp >> 6 ) & 0x3F ) ) );
		out.push_back( static_cast<char>( 0x80 | ( cp & 0x3F ) ) );
	}
	else {
		out.push_back( static_cast<char>( 0xF0 | ( cp >> 18 ) ) );
		out.push_back( static_cast<char>( 0x80 | ( ( cp >> 12 ) & 0x3F ) ) );
		out.push_back( static_cast<char>( 0x80 | ( ( cp >> 6 ) & 0x3F ) ) );
		out.push_back( static_cast<char>( 0x80 | ( cp & 0x3F ) ) );
	}
}

static std::string nfc( const std::string& text ) {
	bool ascii = true;
	for ( unsigned char c : text ) {
		if ( c >= 0x80 ) {
			ascii = false;
			break;
		}
	}
	if ( ascii ) return text;
	std::vector<uint32_t> cps;
	cps.reserve( text.size() );
	for ( size_t i = 0; i < text.size(); ) {
		const size_t before = i;
		const uint32_t cp = utf8_next( text, i );
		if ( cp >= 0x80 && i == before + 1 ) return text; // malformed UTF-8: leave untouched
		append_decomposed( cp, cps );
	}
	// canonical ordering: stable sort of each run of non-starters by combining class
	for ( size_t i = 0; i < cps.size(); ) {
		if ( combining_class( cps[i] ) == 0 ) {
			++i;
			continue;
		}
		size_t j = i;
		while ( j < cps.size() && combining_class( cps[j] ) != 0 ) ++j;
		std::stable_sort( cps.begin() + static_cast<long>( i ), cps.begin() + static_cast<long>( j ),
			[]( uint32_t a, uint32_t b ) { return combining_class( a ) < combining_class( b ); } );
		i = j;
	}
	// canonical composition
	std::vector<uint32_t> out;
	out.reserve( cps.size() );
	long starter = -1;
	int last_class = -1;
	for ( uint32_t cp : cps ) {
		const int cc = combining_class( cp );
		if ( starter >= 0 && ( last_class < cc || last_class == 0 ) ) {
			const uint32_t comp = compose_pair( out[static_cast<size_t>( starter )], cp );
			if ( comp != 0 && ( last_class < cc || static_cast<size_t>( starter ) == out.size() - 1 ) ) {
				out[static_cast<size_t>( starter )] = comp;
				continue;
			}
		}
		if ( cc == 0 ) {
			starter = static_cast<long>( out.size() );
			last_class = 0;
		}
		else {
			last_class = cc;
		}
		out.push_back( cp );
	}
	std::string res;
	res.reserve( text.size() );
	for ( uint32_t cp : out ) append_utf8( cp, res );
	return res;
}

static std::vector<std::string> bpe_split( const std::string& text, int max_digits, bool marks_are_letters ) {
	std::vector<std::string> words;
	Splitter sp{ text, marks_are_letters, max_digits };
	size_t i = 0;
	while ( i < text.size() ) {
		const size_t e = sp.next( i );
		words.push_back( text.substr( i, e - i ) );
		i = e;
	}
	return words;
}

} // namespace unicode_util

Tokenizer::Tokenizer() {
}

Tokenizer::~Tokenizer() {
}

void Tokenizer::init_byte_mappings() {
	byte_to_token_.assign( 256, -1 );
	codepoint_to_byte_.clear();

	std::vector<int> bs;
	for ( int b = '!'; b <= '~'; ++b ) bs.push_back( b );
	for ( int b = 161; b <= 172; ++b ) bs.push_back( b );
	for ( int b = 174; b <= 255; ++b ) bs.push_back( b );

	std::vector<int> cs = bs;
	int n = 0;
	for ( int b = 0; b < 256; ++b ) {
		bool found = false;
		for ( int x : bs ) {
			if ( x == b ) {
				found = true;
				break;
			}
		}
		if ( !found ) {
			bs.push_back( b );
			cs.push_back( 256 + n );
			n++;
		}
	}

	for ( size_t i = 0; i < bs.size(); ++i ) {
		codepoint_to_byte_[cs[i]] = static_cast<uint8_t>( bs[i] );
	}

	auto cp_to_utf8 = []( int cp ) -> std::string {
		std::string s;
		if ( cp < 0x80 ) {
			s.push_back( static_cast<char>( cp ) );
		} else if ( cp < 0x800 ) {
			s.push_back( static_cast<char>( 0xC0 | ( cp >> 6 ) ) );
			s.push_back( static_cast<char>( 0x80 | ( cp & 0x3F ) ) );
		} else {
			s.push_back( static_cast<char>( 0xE0 | ( cp >> 12 ) ) );
			s.push_back( static_cast<char>( 0x80 | ( ( cp >> 6 ) & 0x3F ) ) );
			s.push_back( static_cast<char>( 0x80 | ( cp & 0x3F ) ) );
		}
		return s;
	};

	for ( size_t i = 0; i < bs.size(); ++i ) {
		int b = bs[i];
		int cp = cs[i];
		std::string u = cp_to_utf8( cp );
		auto it = vocab_.find( u );
		if ( it != vocab_.end() ) {
			byte_to_token_[b] = it->second;
		}
	}
}

bool Tokenizer::load( const char* model_dir ) {
	if ( !model_dir ) {
		return false;
	}

	std::string dir( model_dir );
	if ( !dir.empty() && dir.back() != '/' ) {
		dir.push_back( '/' );
	}

	std::string tok_json_path = dir + "tokenizer.json";
	std::string vocab_json_path = dir + "vocab.json";
	std::string merges_txt_path = dir + "merges.txt";
	std::string tok_cfg_path = dir + "tokenizer_config.json";

	std::ifstream tok_file( tok_json_path );
	nlohmann::json tok_json;
	bool has_tok_json = false;

	if ( tok_file.is_open() ) {
		tok_json = nlohmann::json::parse( tok_file, nullptr, false );
		if ( !tok_json.is_discarded() ) {
			has_tok_json = true;
		}
	}

	vocab_.clear();
	special_tokens_.clear();
	id_to_token_.clear();
	is_special_.clear();
	merge_rank_.clear();
	merge_result_.clear();

	int max_id = 0;

	if ( has_tok_json && tok_json.contains( "model" ) && tok_json["model"].contains( "vocab" ) ) {
		const auto& v = tok_json["model"]["vocab"];
		for ( auto& el : v.items() ) {
			int id = el.value().get<int>();
			vocab_[el.key()] = id;
			if ( id > max_id ) max_id = id;
		}
	} else {
		std::ifstream vf( vocab_json_path );
		if ( !vf.is_open() ) {
			return false;
		}
		nlohmann::json vj = nlohmann::json::parse( vf, nullptr, false );
		if ( vj.is_discarded() ) {
			return false;
		}
		for ( auto& el : vj.items() ) {
			int id = el.value().get<int>();
			vocab_[el.key()] = id;
			if ( id > max_id ) max_id = id;
		}
	}

	if ( has_tok_json && tok_json.contains( "added_tokens" ) ) {
		for ( auto& at : tok_json["added_tokens"] ) {
			std::string content = at["content"].get<std::string>();
			int id = at["id"].get<int>();
			special_tokens_[content] = id;
			if ( id > max_id ) max_id = id;
		}
	}

	std::ifstream cfg_file( tok_cfg_path );
	if ( cfg_file.is_open() ) {
		nlohmann::json cfg = nlohmann::json::parse( cfg_file, nullptr, false );
		if ( !cfg.is_discarded() ) {
			if ( cfg.contains( "added_tokens_decoder" ) && cfg["added_tokens_decoder"].is_object() ) {
				for ( auto& el : cfg["added_tokens_decoder"].items() ) {
					int id = std::stoi( el.key() );
					if ( el.value().contains( "content" ) ) {
						std::string content = el.value()["content"].get<std::string>();
						special_tokens_[content] = id;
						if ( id > max_id ) max_id = id;
					}
				}
			}
			if ( cfg.contains( "eos_token" ) ) {
				if ( cfg["eos_token"].is_string() ) {
					std::string eos = cfg["eos_token"].get<std::string>();
					auto it = special_tokens_.find( eos );
					if ( it != special_tokens_.end() ) {
						eos_token_id_ = it->second;
					} else {
						auto vit = vocab_.find( eos );
						if ( vit != vocab_.end() ) eos_token_id_ = vit->second;
					}
				} else if ( cfg["eos_token"].is_object() && cfg["eos_token"].contains( "content" ) ) {
					std::string eos = cfg["eos_token"]["content"].get<std::string>();
					auto it = special_tokens_.find( eos );
					if ( it != special_tokens_.end() ) eos_token_id_ = it->second;
				}
			}
			if ( cfg.contains( "pad_token" ) ) {
				if ( cfg["pad_token"].is_string() ) {
					std::string pad = cfg["pad_token"].get<std::string>();
					auto it = special_tokens_.find( pad );
					if ( it != special_tokens_.end() ) {
						pad_token_id_ = it->second;
					} else {
						auto vit = vocab_.find( pad );
						if ( vit != vocab_.end() ) pad_token_id_ = vit->second;
					}
				} else if ( cfg["pad_token"].is_object() && cfg["pad_token"].contains( "content" ) ) {
					std::string pad = cfg["pad_token"]["content"].get<std::string>();
					auto it = special_tokens_.find( pad );
					if ( it != special_tokens_.end() ) pad_token_id_ = it->second;
				}
			}
			if ( cfg.contains( "bos_token" ) ) {
				if ( cfg["bos_token"].is_string() ) {
					std::string bos = cfg["bos_token"].get<std::string>();
					auto it = special_tokens_.find( bos );
					if ( it != special_tokens_.end() ) {
						bos_token_id_ = it->second;
					} else {
						auto vit = vocab_.find( bos );
						if ( vit != vocab_.end() ) bos_token_id_ = vit->second;
					}
				}
			}
		}
	}

	id_to_token_.assign( max_id + 1, "" );
	is_special_.assign( max_id + 1, false );
	for ( const auto& pair : vocab_ ) {
		id_to_token_[pair.second] = pair.first;
	}
	for ( const auto& pair : special_tokens_ ) {
		id_to_token_[pair.second] = pair.first;
		is_special_[pair.second] = true;
	}

	init_byte_mappings();

	int rank = 0;
	if ( has_tok_json && tok_json.contains( "model" ) && tok_json["model"].contains( "merges" ) ) {
		const auto& merges = tok_json["model"]["merges"];
		for ( const auto& m : merges ) {
			std::string m1, m2;
			if ( m.is_array() ) {
				m1 = m[0].get<std::string>();
				m2 = m[1].get<std::string>();
			} else {
				std::string s = m.get<std::string>();
				size_t sp = s.find( ' ' );
				if ( sp == std::string::npos ) continue;
				m1 = s.substr( 0, sp );
				m2 = s.substr( sp + 1 );
			}
			auto it1 = vocab_.find( m1 );
			auto it2 = vocab_.find( m2 );
			auto itc = vocab_.find( m1 + m2 );
			if ( it1 != vocab_.end() && it2 != vocab_.end() && itc != vocab_.end() ) {
				uint64_t key = ( static_cast<uint64_t>( it1->second ) << 32 ) | static_cast<uint64_t>( it2->second );
				merge_rank_[key] = rank++;
				merge_result_[key] = itc->second;
			}
		}
	} else {
		std::ifstream mf( merges_txt_path );
		if ( mf.is_open() ) {
			std::string line;
			while ( std::getline( mf, line ) ) {
				if ( line.empty() || line[0] == '#' ) continue;
				size_t sp = line.find( ' ' );
				if ( sp == std::string::npos ) continue;
				std::string m1 = line.substr( 0, sp );
				std::string m2 = line.substr( sp + 1 );
				auto it1 = vocab_.find( m1 );
				auto it2 = vocab_.find( m2 );
				auto itc = vocab_.find( m1 + m2 );
				if ( it1 != vocab_.end() && it2 != vocab_.end() && itc != vocab_.end() ) {
					uint64_t key = ( static_cast<uint64_t>( it1->second ) << 32 ) | static_cast<uint64_t>( it2->second );
					merge_rank_[key] = rank++;
					merge_result_[key] = itc->second;
				}
			}
		}
	}

	nfc_ = false;
	if ( has_tok_json && tok_json.contains( "normalizer" ) && !tok_json["normalizer"].is_null() ) {
		if ( tok_json["normalizer"].value( "type", std::string() ) == "NFC" ) {
			nfc_ = true;
		}
		else {
			LOG_ERROR( "tokenizer.json uses a normalizer this engine does not implement\n" );
			return false;
		}
	}
	// The split regex must be one the matcher implements exactly
	max_digits_ = 1;
	marks_are_letters_ = true;
	if ( has_tok_json && tok_json.contains( "pre_tokenizer" ) ) {
		const std::string pt_dump = tok_json["pre_tokenizer"].dump();
		// the regex appears JSON-escaped inside the dump, so compare against the escaped form
		const std::string qwen = nlohmann::json( unicode_util::kQwenPattern ).dump();
		const std::string glm = nlohmann::json( unicode_util::kGlmPattern ).dump();
		const std::string flash = nlohmann::json( unicode_util::kFlashPattern ).dump();
		if ( pt_dump.find( glm.substr( 1, glm.size() - 2 ) ) != std::string::npos ) {
			max_digits_ = 3;
			marks_are_letters_ = false;
		}
		else if ( pt_dump.find( flash.substr( 1, flash.size() - 2 ) ) != std::string::npos ) {
			max_digits_ = 1;
			marks_are_letters_ = false;
		}
		else if ( pt_dump.find( qwen.substr( 1, qwen.size() - 2 ) ) == std::string::npos ) {
			LOG_ERROR( "tokenizer.json uses a pre-tokenizer pattern this engine does not implement\n" );
			return false;
		}
	}

	decoded_tokens_.resize( id_to_token_.size() );
	for ( size_t id = 0; id < id_to_token_.size(); ++id ) {
		decoded_tokens_[id] = decode_token( static_cast<int>( id ) );
	}

	return true;
}

void Tokenizer::bpe_merge( std::vector<int>& ids ) const {
	while ( ids.size() > 1 ) {
		int min_rank = 1000000000;
		int min_idx = -1;
		uint64_t min_key = 0;

		for ( size_t i = 0; i < ids.size() - 1; ++i ) {
			uint64_t key = ( static_cast<uint64_t>( ids[i] ) << 32 ) | static_cast<uint64_t>( ids[i + 1] );
			auto it = merge_rank_.find( key );
			if ( it != merge_rank_.end() && it->second < min_rank ) {
				min_rank = it->second;
				min_idx = static_cast<int>( i );
				min_key = key;
			}
		}

		if ( min_idx == -1 ) break;

		ids[min_idx] = merge_result_.at( min_key );
		ids.erase( ids.begin() + min_idx + 1 );
	}
}

std::vector<int> Tokenizer::encode_piece( const std::string& piece ) const {
	std::vector<int> tids;
	tids.reserve( piece.size() );
	for ( unsigned char c : piece ) {
		tids.push_back( byte_to_token_[c] );
	}
	bpe_merge( tids );
	return tids;
}

std::vector<int> Tokenizer::encode_text_segment( const std::string& text ) const {
	std::vector<int> result;
	if ( text.empty() ) return result;

	const std::string normalized = nfc_ ? unicode_util::nfc( text ) : text;
	std::vector<std::string> words = unicode_util::bpe_split( normalized, max_digits_, marks_are_letters_ );
	for ( const auto& word : words ) {
		auto piece_toks = encode_piece( word );
		result.insert( result.end(), piece_toks.begin(), piece_toks.end() );
	}

	return result;
}

std::vector<int> Tokenizer::encode( const std::string& text, bool add_special ) const {
	std::vector<int> result;
	if ( text.empty() ) return result;

	if ( add_special && bos_token_id_ != -1 ) {
		result.push_back( bos_token_id_ );
	}

	size_t pos = 0;
	while ( pos < text.size() ) {
		size_t best_pos = std::string::npos;
		size_t best_len = 0;
		int best_id = -1;

		for ( const auto& st : special_tokens_ ) {
			size_t p = text.find( st.first, pos );
			if ( p != std::string::npos ) {
				if ( p < best_pos || ( p == best_pos && st.first.size() > best_len ) ) {
					best_pos = p;
					best_len = st.first.size();
					best_id = st.second;
				}
			}
		}

		if ( best_pos == std::string::npos ) {
			auto seg_toks = encode_text_segment( text.substr( pos ) );
			result.insert( result.end(), seg_toks.begin(), seg_toks.end() );
			break;
		} else {
			if ( best_pos > pos ) {
				auto seg_toks = encode_text_segment( text.substr( pos, best_pos - pos ) );
				result.insert( result.end(), seg_toks.begin(), seg_toks.end() );
			}
			result.push_back( best_id );
			pos = best_pos + best_len;
		}
	}

	if ( add_special && eos_token_id_ != -1 ) {
		result.push_back( eos_token_id_ );
	}

	return result;
}

std::string Tokenizer::decode_token( int token_id ) const {
	if ( token_id < 0 || token_id >= static_cast<int>( id_to_token_.size() ) ) {
		return "";
	}

	if ( is_special_[token_id] ) {
		return id_to_token_[token_id];
	}

	const std::string& s = id_to_token_[token_id];
	std::string raw;
	size_t i = 0;
	while ( i < s.size() ) {
		unsigned char c = static_cast<unsigned char>( s[i] );
		int cp = 0;
		int len = 0;
		if ( c < 0x80 ) {
			cp = c;
			len = 1;
		} else if ( ( c & 0xE0 ) == 0xC0 ) {
			cp = ( ( c & 0x1F ) << 6 ) | ( static_cast<unsigned char>( s[i + 1] ) & 0x3F );
			len = 2;
		} else if ( ( c & 0xF0 ) == 0xE0 ) {
			cp = ( ( c & 0x0F ) << 12 ) | ( ( static_cast<unsigned char>( s[i + 1] ) & 0x3F ) << 6 ) | ( static_cast<unsigned char>( s[i + 2] ) & 0x3F );
			len = 3;
		} else if ( ( c & 0xF8 ) == 0xF0 ) {
			cp = ( ( c & 0x07 ) << 18 ) | ( ( static_cast<unsigned char>( s[i + 1] ) & 0x3F ) << 12 ) | ( ( static_cast<unsigned char>( s[i + 2] ) & 0x3F ) << 6 ) | ( static_cast<unsigned char>( s[i + 3] ) & 0x3F );
			len = 4;
		} else {
			len = 1;
		}

		auto it = codepoint_to_byte_.find( cp );
		if ( it != codepoint_to_byte_.end() ) {
			raw.push_back( static_cast<char>( it->second ) );
		} else {
			raw.append( s.substr( i, len ) );
		}
		i += len;
	}
	return raw;
}

const std::string& Tokenizer::decode_token_fast( int token_id ) const {
	static const std::string empty;
	if ( token_id < 0 || token_id >= static_cast<int>( decoded_tokens_.size() ) ) {
		return empty;
	}
	return decoded_tokens_[token_id];
}

std::string Tokenizer::decode( const std::vector<int>& tokens ) const {
	std::string result;
	for ( int tok : tokens ) {
		result += decode_token( tok );
	}
	return result;
}

int Tokenizer::get_special_token( const std::string& name ) const {
	auto it = special_tokens_.find( name );
	if ( it != special_tokens_.end() ) {
		return it->second;
	}
	auto vit = vocab_.find( name );
	if ( vit != vocab_.end() ) {
		return vit->second;
	}
	return -1;
}

bool Tokenizer::has_special_token( const std::string& name ) const {
	return get_special_token( name ) != -1;
}

} // namespace engine::loader
