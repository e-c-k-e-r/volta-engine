#include "engine/loader/tokenizer.h"

#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cctype>

namespace engine::loader {

// Lightweight Unicode and BPE Pre-tokenizer Classifier
namespace unicode_util {

static inline uint32_t utf8_next( const std::string& str, size_t& i ) {
	if ( i >= str.size() ) return 0;
	unsigned char c = static_cast<unsigned char>( str[i++] );
	if ( c < 0x80 ) return c;
	if ( ( c & 0xE0 ) == 0xC0 && i < str.size() ) {
		return ( ( c & 0x1F ) << 6 ) | ( static_cast<unsigned char>( str[i++] ) & 0x3F );
	}
	if ( ( c & 0xF0 ) == 0xE0 && i + 1 < str.size() ) {
		uint32_t c1 = static_cast<unsigned char>( str[i++] ) & 0x3F;
		uint32_t c2 = static_cast<unsigned char>( str[i++] ) & 0x3F;
		return ( ( c & 0x0F ) << 12 ) | ( c1 << 6 ) | c2;
	}
	if ( ( c & 0xF8 ) == 0xF0 && i + 2 < str.size() ) {
		uint32_t c1 = static_cast<unsigned char>( str[i++] ) & 0x3F;
		uint32_t c2 = static_cast<unsigned char>( str[i++] ) & 0x3F;
		uint32_t c3 = static_cast<unsigned char>( str[i++] ) & 0x3F;
		return ( ( c & 0x07 ) << 18 ) | ( c1 << 12 ) | ( c2 << 6 ) | c3;
	}
	return c;
}

static inline bool is_number( uint32_t cp ) {
	if ( cp >= '0' && cp <= '9' ) return true;
	if ( cp >= 0x0660 && cp <= 0x0669 ) return true; // Arabic-Indic digits
	if ( cp >= 0xFF10 && cp <= 0xFF19 ) return true; // Fullwidth digits
	return false;
}

static inline bool is_letter( uint32_t cp ) {
	if ( ( cp >= 'a' && cp <= 'z' ) || ( cp >= 'A' && cp <= 'Z' ) ) return true;
	if ( cp >= 0x00C0 && cp <= 0x024F ) return true; // Latin extended
	if ( cp >= 0x0370 && cp <= 0x03FF ) return true; // Greek
	if ( cp >= 0x0400 && cp <= 0x04FF ) return true; // Cyrillic
	if ( cp >= 0x4E00 && cp <= 0x9FFF ) return true; // CJK Unified Ideographs
	if ( cp >= 0x3400 && cp <= 0x4DBF ) return true; // CJK Extension A
	if ( cp >= 0x3040 && cp <= 0x309F ) return true; // Hiragana
	if ( cp >= 0x30A0 && cp <= 0x30FF ) return true; // Katakana
	if ( cp >= 0xAC00 && cp <= 0xD7AF ) return true; // Hangul Syllables
	return false;
}

static inline bool is_whitespace( uint32_t cp ) {
	return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x00A0 || cp == 0x2000 || cp == 0x3000;
}

static inline bool is_punctuation_or_symbol( uint32_t cp ) {
	return !is_letter( cp ) && !is_number( cp ) && !is_whitespace( cp );
}

static inline bool check_contraction( const std::string& text, size_t pos, size_t& len ) {
	if ( pos + 1 >= text.size() ) return false;
	char c0 = text[pos];
	if ( c0 != '\'' ) return false;
	size_t apo_len = 1;
	if ( pos + apo_len >= text.size() ) return false;

	char c1 = static_cast<char>( std::tolower( static_cast<unsigned char>( text[pos + apo_len] ) ) );
	if ( c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd' ) {
		len = apo_len + 1;
		return true;
	}
	if ( pos + apo_len + 1 < text.size() ) {
		char c2 = static_cast<char>( std::tolower( static_cast<unsigned char>( text[pos + apo_len + 1] ) ) );
		if ( ( c1 == 'r' && c2 == 'e' ) || ( c1 == 'v' && c2 == 'e' ) || ( c1 == 'l' && c2 == 'l' ) ) {
			len = apo_len + 2;
			return true;
		}
	}
	return false;
}

// BPE word-level splitting matching standard BPE regex patterns
static std::vector<std::string> bpe_split( const std::string& text, int max_digits ) {
	std::vector<std::string> words;
	size_t i = 0;
	size_t n = text.size();

	while ( i < n ) {
		// 1. Contractions: 's|'t|'re|'ve|'m|'ll|'d
		size_t cont_len = 0;
		if ( check_contraction( text, i, cont_len ) ) {
			words.push_back( text.substr( i, cont_len ) );
			i += cont_len;
			continue;
		}

		// 2. Whitespace runs (newlines vs spaces)
		if ( text[i] == '\r' || text[i] == '\n' ) {
			size_t start = i;
			while ( i < n && ( text[i] == '\r' || text[i] == '\n' ) ) {
				i++;
			}
			words.push_back( text.substr( start, i - start ) );
			continue;
		}

		if ( text[i] == ' ' || text[i] == '\t' ) {
			size_t space_start = i;
			while ( i < n && ( text[i] == ' ' || text[i] == '\t' ) ) {
				i++;
			}
			// Check if space followed by letter or punctuation
			if ( i < n && ( text[i] != '\r' && text[i] != '\n' ) ) {
				// If followed by letter or number, include single leading space with word
				if ( i - space_start == 1 ) {
					// Leading single space attached to next word/punctuation
					size_t next_i = i;
					size_t temp = i;
					uint32_t next_cp = utf8_next( text, temp );
					if ( is_letter( next_cp ) ) {
						i = temp;
						while ( i < n ) {
							size_t chk = i;
							uint32_t cp = utf8_next( text, chk );
							if ( is_letter( cp ) ) i = chk;
							else break;
						}
						words.push_back( text.substr( space_start, i - space_start ) );
						continue;
					} else if ( is_number( next_cp ) ) {
						// Numbers: do not attach space to digits in standard BPE regex (" ?" is only for punctuation/symbols)
						words.push_back( text.substr( space_start, 1 ) );
						continue;
					} else if ( is_punctuation_or_symbol( next_cp ) ) {
						i = temp;
						while ( i < n ) {
							size_t chk = i;
							uint32_t cp = utf8_next( text, chk );
							if ( is_punctuation_or_symbol( cp ) ) i = chk;
							else break;
						}
						words.push_back( text.substr( space_start, i - space_start ) );
						continue;
					}
				}
			}
			words.push_back( text.substr( space_start, i - space_start ) );
			continue;
		}

		// 3. Letters: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
		size_t letter_start = i;
		uint32_t first_cp = utf8_next( text, i );
		if ( is_letter( first_cp ) ) {
			while ( i < n ) {
				size_t chk = i;
				uint32_t cp = utf8_next( text, chk );
				if ( is_letter( cp ) ) i = chk;
				else break;
			}
			words.push_back( text.substr( letter_start, i - letter_start ) );
			continue;
		}

		// 4. Numbers: \p{N} or \p{N}{1,3}
		if ( is_number( first_cp ) ) {
			int count = 1;
			while ( count < max_digits && i < n ) {
				size_t chk = i;
				uint32_t cp = utf8_next( text, chk );
				if ( is_number( cp ) ) {
					i = chk;
					count++;
				} else {
					break;
				}
			}
			words.push_back( text.substr( letter_start, i - letter_start ) );
			continue;
		}

		// 5. Punctuation / symbols: ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
		if ( is_punctuation_or_symbol( first_cp ) ) {
			while ( i < n ) {
				size_t chk = i;
				uint32_t cp = utf8_next( text, chk );
				if ( is_punctuation_or_symbol( cp ) ) i = chk;
				else break;
			}
			words.push_back( text.substr( letter_start, i - letter_start ) );
			continue;
		}

		// Fallback single character
		words.push_back( text.substr( letter_start, i - letter_start ) );
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

	// Detect digit splitting from pre_tokenizer pattern: \p{N}{1,3} -> 3, else 1
	max_digits_ = 1;
	if ( has_tok_json && tok_json.contains( "pre_tokenizer" ) ) {
		std::string pt_dump = tok_json["pre_tokenizer"].dump();
		if ( pt_dump.find( "\\p{N}{1,3}" ) != std::string::npos ) {
			max_digits_ = 3;
		}
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

	std::vector<std::string> words = unicode_util::bpe_split( text, max_digits_ );
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
