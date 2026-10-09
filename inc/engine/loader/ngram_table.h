#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engine::loader {

// Hashed n-gram embedding table of Qwen3.8-Flash-Next (ngram_embedding.safetensors, format exl3_ngram_trellis): 128 shards of
// 160-wide rows, each a tail-biting trellis ring over the mul1 codebook with a per-row fp16 scale and a per-head fp16 bias.
// The file is memory-mapped (39 GB); lookups touch a handful of rows per token. Thread safe once open.
class NgramTable {
public:
	~NgramTable();
	bool open( const std::string& path );
	bool is_open() const { return base_ != nullptr; }

	int ngram_size() const { return ngram_size_; }
	int heads_per_ngram() const { return heads_per_ngram_; }
	int num_heads() const { return num_heads_; }
	static constexpr int ROW_DIM = 160;

	// Global row ids of the (ngram_size - 1) * heads_per_ngram heads for one position. tokens[0] is the token at the position,
	// tokens[s] the one s positions earlier; valid[s] says the token lies in the same eos segment (else the eos id is used).
	void row_ids( const int64_t* tokens, uint64_t* uids ) const;

	// Decodes one row (+ the head's bias) to fp32
	void decode_row( uint64_t uid, int head, float* out ) const;

private:
	void* base_ = nullptr;
	size_t map_bytes_ = 0;
	int fd_ = -1;
	int ngram_size_ = 3, heads_per_ngram_ = 8, num_heads_ = 16, K_ = 6, words_ = 61;
	int64_t rows_per_shard_ = 0;
	std::vector<const uint16_t*> shards_;
	std::vector<float> head_bias_;       // [num_heads, 160]
	std::vector<int64_t> head_offsets_, head_sizes_, multipliers_;
	std::vector<float> codebook_;        // 65536 decoded mul1 values (fp16 rounded)
};

} // namespace engine::loader
