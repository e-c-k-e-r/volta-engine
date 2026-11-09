#pragma once

#include "engine/config.h"
#include "engine/models/model.h"
#include "engine/runtime/kv_cache.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/core/graph_cache.h"
#include "engine/modules/linear.h"
#include "engine/backend/types.h"
#include <cstdint>
#include <vector>

namespace engine::models {

// Gemma 4 31B (gemma4_text): 60 layers, every 6th full attention, the rest sliding-window attention
struct Gemma4Config {
	int num_layers = 60;
	int full_every = 6;          // layers 5, 11, ... are full attention
	int hidden_size = 5376;
	int intermediate_size = 21504;
	int num_q_heads = 32;

	// Sliding-window layers
	int swa_kv_heads = 16;
	int swa_head_dim = 256;
	int swa_window = 1024;       // a row attends to its last 1024 positions, its own included
	float swa_rope_theta = 10000.0f;

	// Full-attention layers: K = V (no v_proj), "proportional" RoPE over 512 dims with 64 rotating pairs
	int full_kv_heads = 4;
	int full_head_dim = 512;
	int full_rot_pairs = 64;
	float full_rope_theta = 1000000.0f;

	int max_seq = 32768;
	int max_slots = 8;
	int vocab_size = 262144;
	float rms_norm_eps = 1e-6f;
	float final_softcap = 30.0f;
	int tap_layers[6] = { 1, 12, 23, 35, 46, 57 };   // the DFlash drafter's target_layer_ids (z-lab gemma-4-31B-it-DFlash)
};

struct Gemma4LayerWeights {
	bool is_full = false;
	float layer_scalar = 1.0f;

	pod::TensorView input_norm;
	pod::TensorView post_attn_norm;
	pod::TensorView pre_ffn_norm;
	pod::TensorView post_ffn_norm;

	pod::LinearWeight q_proj;
	pod::LinearWeight k_proj;
	pod::LinearWeight v_proj;    // sliding layers only
	pod::LinearWeight o_proj;
	pod::TensorView q_norm;
	pod::TensorView k_norm;

	pod::LinearWeight gate_proj;
	pod::LinearWeight up_proj;
	pod::LinearWeight down_proj;
};

class Gemma4Pipeline : public ModelPipeline {
public:
	Gemma4Pipeline();
	~Gemma4Pipeline() override;

	using ModelPipeline::forward;

	bool init( int device_id, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) override;
	bool init( const Gemma4Config& config, int rank, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring );
	bool load_weights( const char* model_dir, backend::stream_t stream = nullptr ) override;

	void embed_tokens( const int* token_ids, int count, backend::stream_t stream = nullptr ) override;
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

	pod::TensorView& logits() override { return logits_; }
	const pod::TensorView& logits() const override { return logits_; }
	const ModelMetadata& get_metadata() const override { return metadata_; }

	void destroy();

private:
	void set_rows( int m );
	void layer_forward( const Gemma4LayerWeights& lw, int kv_layer, int m, backend::stream_t stream );
	void forward_body( int m, backend::stream_t stream );
	void copy_kv_rows( const engine::runtime::KVCachePool& pool, int slot_id, int from, int tokens, char* host, bool to_host, backend::stream_t stream );
	size_t tap_snapshot_bytes( int tokens ) const;

	// DFlash drafter attachment: a tap ring per slot (6 taps per position), the head scratch
	DraftHooks draft_hooks_;
	DraftScratch draft_;
	bool snap_taps_ = false;
	int tap_shift_ = 0;

	Gemma4Config config_;
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
	engine::core::GraphCache graphs_;
	int* d_token_ids_ = nullptr;
	int lm_mode_ = 0;
	int last_rows_ = 0;

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

	Gemma4LayerWeights layers_[60];
	pod::TensorView embed_tokens_;
	pod::TensorView final_norm_;
	pod::LinearWeight lm_head_;
	pod::TensorView logits_;
	pod::TensorView full_logits_;
	float embed_scale_ = 1.0f;

	int tp_size_ = 1;     // 1 when the head is not sharded
	int tp_ranks_ = 1;
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
