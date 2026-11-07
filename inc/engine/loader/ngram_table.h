#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engine::loader {

class SafeTensorsRepo;

// Hashed n-gram embedding table of Qwen3.8-Flash-Next (ngram_embedding.safetensors, format exl3_ngram_trellis): 128 shards of
// 160-wide rows, each a tail-biting trellis ring over the mul1 codebook with a per-row fp16 scale and a per-head fp16 bias.
// The file is memory-mapped (39 GB); lookups touch a handful of rows per token. Thread safe once open.
class NgramTable {
public:
	~NgramTable();
	bool open( const std::string& path );
	// The same table as FP8 E4M3 rows (the normal Flash-Next NVFP4 checkpoint: `<prefix>.ngram_embedding.shard_N.weight` [rows, 160] and a BF16
	// `weight_scale`, with the hash parameters `<prefix>.ngram_heads_offsets`, `.ngram_heads_vocab_sizes` and `.layer_multipliers` in the repo); no bias.
	// The repo's mapping is used as it is: it has to outlive the table.
	bool open_fp8( const SafeTensorsRepo& repo, const std::string& prefix );
	// The same table from a GGUF conversion (llama.cpp's qwen4exp: per_layer_token_embd, one tensor of every shard's rows in order, here under
	// `name`, block-quantized rows of 160 values), with the hash parameters from the file's metadata (`<arch>.ple.*`); no bias. Rows are
	// decoded on the host (IQ4_NL, Q8_0, F16, BF16, F32). The repo's mapping is used as it is: it has to outlive the table.
	bool open_gguf( const SafeTensorsRepo& repo, const std::string& name );
	bool is_open() const { return base_ != nullptr || fp8_ || gguf_rows_ != nullptr; }

	int ngram_size() const { return ngram_size_; }
	int heads_per_ngram() const { return heads_per_ngram_; }
	int num_heads() const { return num_heads_; }
	static constexpr int ROW_DIM = 160;

	// Global row ids of the (ngram_size - 1) * heads_per_ngram heads for one position. tokens[0] is the token at the position,
	// tokens[s] the one s positions earlier; valid[s] says the token lies in the same eos segment (else the eos id is used).
	void row_ids( const int64_t* tokens, uint64_t* uids ) const;

	// Decodes one row (+ the head's bias) to fp32
	void decode_row( uint64_t uid, int head, float* out ) const;
	// Asks the kernel to read the pages of a row (asynchronous readahead): a lookup that requests all its rows first and decodes them after
	// waits for the reads together instead of one fault after another
	void prefetch_row( uint64_t uid ) const;

private:
	void* base_ = nullptr;
	size_t map_bytes_ = 0;
	int fd_ = -1;
	int ngram_size_ = 3, heads_per_ngram_ = 8, num_heads_ = 16, K_ = 6, words_ = 61;
	int64_t rows_per_shard_ = 0;
	std::vector<const uint16_t*> shards_;
	std::vector<const uint8_t*> shards8_;   // FP8 mode
	bool fp8_ = false;
	float fp8_scale_ = 1.0f;
	const uint8_t* gguf_rows_ = nullptr;   // GGUF mode: every row, row_bytes each
	size_t gguf_row_bytes_ = 0;
	uint32_t gguf_type_ = 0;
	std::vector<float> head_bias_;       // [num_heads, 160]
	std::vector<int64_t> head_offsets_, head_sizes_, multipliers_;
	std::vector<float> codebook_;        // 65536 decoded mul1 values (fp16 rounded)
};

} // namespace engine::loader
