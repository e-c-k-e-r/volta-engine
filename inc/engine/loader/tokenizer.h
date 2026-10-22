#pragma once

#include "engine/config.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace engine::loader {

class Tokenizer {
public:
	Tokenizer();
	~Tokenizer();

	// Load tokenizer.json, vocab.json, merges.txt, tokenizer_config.json from model directory
	bool load( const char* model_dir );

	// Encode input text to token IDs using byte-level BPE
	std::vector<int> encode( const std::string& text, bool add_special = false ) const;
	// Encode text without recognising special-token strings in it (they become ordinary pieces)
	std::vector<int> encode_plain( const std::string& text ) const { return encode_text_segment( text ); }
	// Chat-control special tokens (role / turn markers, vision markers): their strings are escaped in message content
	std::vector<std::string> control_tokens() const;

	// Decode a single token ID or a sequence of token IDs to text
	std::string decode_token( int token_id ) const;
	const std::string& decode_token_fast( int token_id ) const;
	std::string decode( const std::vector<int>& tokens ) const;

	// Special token lookup
	int get_special_token( const std::string& name ) const;
	bool has_special_token( const std::string& name ) const;

	// Common special token IDs (-1 if not present)
	int im_start_id() const { return get_special_token( "<|im_start|>" ); }
	int im_end_id() const { return get_special_token( "<|im_end|>" ); }
	int think_start_id() const { return get_special_token( "<think>" ); }
	int think_end_id() const { return get_special_token( "</think>" ); }
	int gmask_id() const { return get_special_token( "[gMASK]" ); }
	int sop_id() const { return get_special_token( "<sop>" ); }
	int eos_id() const { return eos_token_id_; }
	int pad_id() const { return pad_token_id_; }
	int bos_id() const { return bos_token_id_; }

	size_t vocab_size() const { return id_to_token_.size(); }

private:
	void init_byte_mappings();
	void bpe_merge( std::vector<int>& ids ) const;
	void bpe_merge_heap( std::vector<int>& ids ) const;
	std::vector<int> encode_piece( const std::string& piece ) const;
	std::vector<int> encode_text_segment( const std::string& text ) const;

	std::unordered_map<std::string, int> vocab_;
	std::unordered_map<std::string, int> special_tokens_;
	std::vector<std::string> id_to_token_;
	std::vector<std::string> decoded_tokens_;
	std::vector<bool> is_special_;

	// Byte <-> Token mappings for BPE pretokenization
	std::vector<int> byte_to_token_;
	std::unordered_map<int, uint8_t> codepoint_to_byte_;

	// BPE merges: key = ((uint64_t)id1 << 32) | (uint64_t)id2
	std::unordered_map<uint64_t, int> merge_rank_;
	std::unordered_map<uint64_t, int> merge_result_;

	int eos_token_id_ = -1;
	int pad_token_id_ = -1;
	int bos_token_id_ = -1;
	int max_digits_ = 1; // 1 for Qwen (\p{N}), 3 for GLM (\p{N}{1,3})
	bool nfc_ = false;               // tokenizer.json normalizer: NFC
	bool deepseek_split_ = false;   // DeepSeek-V4.1: digit groups, CJK runs, then its own word pattern (three isolated splits)
	bool marks_are_letters_ = true; // Qwen pattern: [\p{L}\p{M}]+, GLM: \p{L}+
};

} // namespace engine::loader
