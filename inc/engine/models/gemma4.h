#pragma once

#include "engine/config.h"
#include "engine/models/model.h"
#include "engine/runtime/kv_cache.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/core/graph_cache.h"
#include "engine/modules/linear.h"
#include "engine/models/gemma4_vision.h"
#include "engine/backend/types.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::loader { class SafeTensorsRepo; }

namespace engine::models {

// A Gemma 4 text model (gemma4_text), its layout read from the checkpoint's config.json (read_gemma4_config); the defaults are the 31B's:
// 60 layers, every 6th full attention, the rest sliding-window attention
struct Gemma4Config {
	int num_layers = 60;
	std::vector<uint8_t> full;   // per layer: full attention (else sliding window), layer_types; empty: every 6th (layers 5, 11, ...)
	int hidden_size = 5376;
	int intermediate_size = 21504;
	bool double_wide_shared = false;   // the KV-sharing layers' MLPs are twice as wide (use_double_wide_mlp)
	int num_q_heads = 32;

	// Sliding-window layers
	int swa_kv_heads = 16;
	int swa_head_dim = 256;
	int swa_window = 1024;       // a row attends to its last 1024 positions, its own included
	float swa_rope_theta = 10000.0f;

	// Full-attention layers: "proportional" RoPE over 512 dims with 64 rotating pairs; K = V (attention_k_eq_v: no v_proj)
	int full_kv_heads = 4;
	int full_head_dim = 512;
	int full_rot_pairs = 64;
	float full_rope_theta = 1000000.0f;
	bool full_k_eq_v = true;

	// The last kv_shared_layers layers have no K / V of their own: each attends with those of the last earlier layer of its kind
	int kv_shared_layers = 0;
	// Per-layer input embeddings (hidden_size_per_layer_input): a [ple_vocab, num_layers * ple_dim] table, every layer's slice mixed in
	// after its MLP
	int ple_dim = 0;
	int ple_vocab = 0;

	bool bidirectional_images = true;   // an image's tokens attend to the whole image on the sliding-window layers (use_bidirectional_attention "vision")
	int max_seq = 32768;
	int max_slots = 8;
	int vocab_size = 262144;
	float rms_norm_eps = 1e-6f;
	float final_softcap = 30.0f;
	static constexpr int kTaps = 6;
	int tap_layers[kTaps] = { 1, 12, 23, 35, 46, 57 };   // the DFlash drafter's target_layer_ids (z-lab gemma-4-31B-it-DFlash)

	bool is_full( int layer ) const { return full.empty() ? ( layer + 1 ) % 6 == 0 : full[layer] != 0; }
	int first_shared() const { return num_layers - kv_shared_layers; }
};

// The text_config of a checkpoint's config.json into c (its fields over the defaults); false, with the reason logged, when the file is
// unreadable or describes something this pipeline doesn't implement (MoE blocks, attention biases, another activation or RoPE)
bool read_gemma4_config( const char* model_dir, Gemma4Config* c );

struct Gemma4LayerWeights {
	bool is_full = false;
	bool shared_kv = false;      // attends with the K / V of layer kv_source (no k_proj / v_proj of its own)
	int kv_source = 0;
	int kv_layer = 0;            // its (or its source's) layer in the pool of its kind
	int inter = 0;               // MLP width on this rank
	float layer_scalar = 1.0f;

	pod::TensorView input_norm;
	pod::TensorView post_attn_norm;
	pod::TensorView pre_ffn_norm;
	pod::TensorView post_ffn_norm;

	pod::LinearWeight q_proj;
	pod::LinearWeight k_proj;
	pod::LinearWeight v_proj;    // none on full layers with K = V
	pod::LinearWeight o_proj;
	pod::TensorView q_norm;
	pod::TensorView k_norm;

	pod::LinearWeight gate_proj;
	pod::LinearWeight up_proj;
	pod::LinearWeight down_proj;

	// Per-layer input: x += post_ple_norm( ple_proj( gelu( ple_gate( x ) ) * the layer's slice ) )
	pod::LinearWeight ple_gate;
	pod::LinearWeight ple_proj;
	pod::TensorView post_ple_norm;
};

class Gemma4Pipeline : public ModelPipeline {
public:
	Gemma4Pipeline();
	~Gemma4Pipeline() override;

	using ModelPipeline::forward;

	void set_model_dir( const char* dir ) override { model_dir_ = dir != nullptr ? dir : ""; }
	bool init( int device_id, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) override;
	bool init( const Gemma4Config& config, int rank, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring );
	bool load_weights( const char* model_dir, backend::stream_t stream = nullptr ) override;

	void embed_tokens( const int* token_ids, int count, backend::stream_t stream = nullptr ) override;
	bool embeds_device_tokens() const override { return config_.ple_dim == 0; }   // the per-layer table's rows are gathered on the host
	void embed_tokens_device( const int* d_tokens, int src_dev, int count, backend::stream_t stream = nullptr ) override;
	void forward( const pod::BatchRows& rows, backend::stream_t stream = nullptr ) override;
	void sample_tokens( int* out_preds, int count, backend::stream_t stream = nullptr ) override;
	bool sample_local( int* out_tokens, float* out_vals, int count, backend::stream_t stream = nullptr ) override;
	bool topk_local( float* out_vals, int* out_ids, int count, backend::stream_t stream = nullptr ) override;
	bool lse_local( float* out, int count, backend::stream_t stream = nullptr ) override;
	const backend::half_t* shard_logits_ptr() const override { return logits_.ptr<const backend::half_t>(); }
	int shard_logits_cols() const override { return lm_shard_cols_; }
	const pod::TensorView& gather_logits( int rows, const backend::half_t* const* peer_shard_ptrs, int peer_shard_cols, backend::stream_t stream = nullptr ) override;

	// No recurrent state: every cache is indexed by position
	void checkpoint_state( int, backend::stream_t = nullptr ) override {}
	void rollback_state( int, backend::stream_t = nullptr ) override {}
	void set_spec_snapshot( bool, backend::stream_t = nullptr ) override {}
	void commit_spec_row( int, int, backend::stream_t = nullptr ) override {}
	void commit_spec_rows( int, int, int, backend::stream_t = nullptr ) override {}
	bool spec_commit_always() const override { return true; }   // nothing to commit: joint verify rounds need no journal
	bool free_rollback() const override { return true; }

	const DraftHooks& draft_hooks() const override { return draft_hooks_; }
	void draft_head_topk( int rows, backend::stream_t stream = nullptr ) override;
	void set_snapshot_taps( bool on ) override { snap_taps_ = on; }
	bool read_taps( int slot_id, int first_pos, int count, backend::half_t* host_dst, backend::stream_t stream = nullptr ) override;
	void reset_state( int slot_id = 0 ) override;

	size_t seq_state_bytes( int from, int tokens, bool with_taps ) const override;
	bool save_seq_state( int slot_id, int from, int tokens, bool with_taps, void* host_dst, backend::stream_t stream = nullptr ) override;
	bool load_seq_state( int slot_id, int from, int tokens, bool with_taps, const void* host_src, bool kv_only, backend::stream_t stream = nullptr ) override;
	bool seq_rows_span( int from, int tokens, bool with_taps, size_t* offset, size_t* bytes ) const override;
	bool load_seq_rows( int slot_id, int from, int tokens, const void* rows, backend::stream_t stream = nullptr ) override;

	// Images: the tower runs on rank 0; its rows replace the embeddings of the image tokens (unscaled)
	bool inject_embeddings( const backend::half_t* host_rows, int dst_row, int rows, backend::stream_t stream = nullptr ) override;
	bool load_vision( const char* model_dir, backend::stream_t stream = nullptr ) override;
	const engine::loader::VisionParams* vision_params() const override { return vision_.loaded() ? &vision_.params() : nullptr; }
	bool encode_image( const engine::loader::PreparedImage& image, std::vector<backend::half_t>* out, backend::stream_t stream = nullptr ) override;
	bool bidirectional_images() const override { return config_.bidirectional_images; }
	bool image_mrope() const override { return false; }

	pod::TensorView& logits() override { return logits_; }
	const pod::TensorView& logits() const override { return logits_; }
	const ModelMetadata& get_metadata() const override { return metadata_; }

	void destroy();

private:
	void set_rows( int m );
	bool layer_forward( const Gemma4LayerWeights& lw, int layer, int m, backend::stream_t stream, bool input_normed = false, const pod::TensorView* next_norm = nullptr );
	bool load_ple_table( engine::loader::SafeTensorsRepo& repo, const char* model_dir, backend::stream_t stream );
	void ple_rows( const int* token_ids, int count, backend::stream_t stream );
	void ple_mix( int m, backend::stream_t stream );
	void forward_body( int m, backend::stream_t stream );
	void copy_kv_rows( const engine::runtime::KVCachePool& pool, int slot_id, int from, int tokens, char* host, bool to_host, backend::stream_t stream );
	size_t tap_snapshot_bytes( int tokens ) const;

	Gemma4VisionTower vision_;   // rank 0 only

	// DFlash drafter attachment: a tap ring per slot (6 taps per position), the head scratch
	DraftHooks draft_hooks_;
	DraftScratch draft_;
	bool snap_taps_ = false;
	int tap_shift_ = 0;

	Gemma4Config config_;
	std::string model_dir_;
	ModelMetadata metadata_;
	int rank_ = 0;
	pod::MemoryArena* arena_ = nullptr;
	engine::comm::CollectiveRing* ring_ = nullptr;
	bool weights_loaded_ = false;

	// Per-rank (TP-sharded) sizes
	int q_heads_ = 0;
	int swa_kv_heads_ = 0;
	int full_kv_heads_ = 0;
	int inter_size_ = 0;

	engine::runtime::KVCachePool swa_pool_;    // the sliding-window layers, in layer order
	engine::runtime::KVCachePool full_pool_;   // the full-attention layers

	pod::BatchRows* d_rows_ = nullptr;
	// A prefill chunk that needs one row's logits at most runs the KV-sharing layers (which store nothing) over its last kTailRows rows alone
	// (d_tail_rows_) or not at all; rows_ is the descriptor the layers read, ple_m_ / ple_row0_ where their per-layer inputs sit
	static constexpr int kTailRows = pod::MAX_BATCH + 1;
	pod::BatchRows* d_tail_rows_ = nullptr;
	pod::BatchRows tail_host_;
	const pod::BatchRows* rows_ = nullptr;
	bool tail_ = false;
	int ple_m_ = 0, ple_row0_ = 0;
	engine::core::GraphCache graphs_;
	int* d_token_ids_ = nullptr;
	int lm_mode_ = 0;
	int last_rows_ = 0;
	bool attn_tc_ = false;   // current forward: all rows share one slot
	int key_hint_ = 0;       // keys the current chunk can attend to (sizes the prefill attention split)

	pod::TensorView resid_;          // [MAX_ROWS, hidden] FP32 residual stream
	pod::TensorView hidden_states_;  // [MAX_ROWS, hidden] FP16 embedding rows
	pod::TensorView norm_hidden_;
	pod::TensorView q_buf_;
	pod::TensorView k_buf_;
	pod::TensorView v_buf_;
	pod::TensorView attn_out_;
	pod::TensorView gate_buf_;
	pod::TensorView up_buf_;
	pod::TensorView mlp_inter_;
	pod::TensorView mlp_out_;

	std::vector<Gemma4LayerWeights> layers_;
	pod::TensorView embed_tokens_;

	// Per-layer input embeddings: the table's rows stay in the checkpoint's mapping on the host (block-quantized GGUF rows, or BF16 / FP16 /
	// FP32 ones); a forward's rows are staged (ple_stage_, pinned), copied up and decoded to FP16 (ple_tok_), then mixed with the projection
	// of the embeddings into ple_ (FP16, layer-major [num_layers][rows][ple_dim])
	std::unique_ptr<engine::loader::SafeTensorsRepo> ple_repo_;   // keeps the table's file mapped
	struct PleTable {
		const uint8_t* rows = nullptr;
		size_t row_bytes = 0;
		pod::DType dtype = pod::DType::FP16;   // GGUF: block rows of gguf_type
		uint32_t gguf_type = 0;
	} ple_table_;
	pod::LinearWeight ple_model_proj_;
	pod::TensorView ple_proj_norm_;
	uint8_t* ple_stage_ = nullptr;
	void* ple_blocks_ = nullptr;     // the staged rows on the device (block rows only)
	pod::TensorView ple_tok_;        // [MAX_ROWS, num_layers * ple_dim] the rows' table values
	pod::TensorView ple_proj_;       // [MAX_ROWS, num_layers * ple_dim] the projection of their embeddings
	pod::TensorView ple_;            // [num_layers, rows, ple_dim] both mixed (the rows of the current forward)
	pod::TensorView final_norm_;
	pod::LinearWeight lm_head_;
	pod::TensorView logits_;
	pod::TensorView full_logits_;
	float embed_scale_ = 1.0f;

	int tp_size_ = 1;     // ranks of the group; above 1 the head is vocabulary-parallel
	int lm_shard_cols_ = 0;
	int lm_shard_real_cols_ = 0;
	int* d_argmax_idx_ = nullptr;
	float* d_argmax_val_ = nullptr;
	int* h_argmax_idx_ = nullptr;
	static constexpr int kTopkTile = 4096;
	float* samp_part_v_ = nullptr;
	int* samp_part_i_ = nullptr;
	int* samp_counters_ = nullptr;
	float* d_samp_vals_ = nullptr;
	int* d_samp_idx_ = nullptr;
};

} // namespace engine::models
