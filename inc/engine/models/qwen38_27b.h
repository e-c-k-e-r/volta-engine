#pragma once

#include "engine/config.h"
#include "engine/models/model.h"
#include "engine/runtime/kv_cache.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/modules/trellis.h"
#include "engine/modules/norm.h"
#include "engine/modules/swiglu.h"
#include "engine/modules/rope.h"
#include "engine/modules/attention.h"
#include "engine/modules/gdn.h"
#include "engine/modules/linear.h"
#include "engine/modules/embedding.h"
#include "engine/core/graph_cache.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

namespace engine::models {

struct Qwen38Config {
	int num_layers = 64;
	int hidden_size = 5120;
	int intermediate_size = 17408;
	int num_linear_layers = 48; // 48 GDN layers
	int num_attn_layers = 16;   // 16 full-attention layers (every 4th layer)

	// Full attention parameters
	int full_num_q_heads = 24;
	int full_num_kv_heads = 4;
	int full_head_dim = 256;
	int full_rot_dim = 64;

	// Linear attention parameters
	int linear_num_k_heads = 16;
	int linear_num_v_heads = 48;
	int linear_head_dim = 128;
	int linear_channels = 10240; // 16*128 + 16*128 + 48*128
	int max_seq = 8192;
	int max_slots = 8;
	int vocab_size = 248320;
	float rms_norm_eps = 1e-6f;
	float rms_norm_bias = 1.0f;
	float rope_theta = 10000000.0f;
	int tap_layers[5] = { 5, 19, 33, 47, 61 }; // DFlash2 tapped hidden states
};

// Per-rank recurrent state, laid out [slot][layer][...] so one slot is a contiguous block.
struct Qwen38State {
	float* gdn_states = nullptr;      // [slots, 48, v_heads, col 128, row 128] FP32 (row fastest)
	float* conv_history = nullptr;    // [slots, 48, channels, 3] FP32
	float* gdn_checkpoint = nullptr;  // one-slot snapshot
	float* conv_checkpoint = nullptr;
	// Per-row post-state snapshots for a speculative verify forward: [kDraftM, <slot layout>].
	// Row t holds the state after consuming rows 0..t; commit_spec_row copies one into the live slot.
	float* gdn_snap = nullptr;
	float* conv_snap = nullptr;
	int* d_snap_on = nullptr;         // device flag: snapshots are written only while nonzero
	size_t slot_gdn_floats = 0;
	size_t slot_conv_floats = 0;
};

struct Qwen38LayerWeights {
	pod::TensorView input_norm;
	pod::TensorView post_attn_norm;

	bool is_full_attention;

	// Linear attention (GDN) weights
	pod::LinearWeight in_proj_qkv;
	pod::LinearWeight in_proj_z;
	pod::LinearWeight out_proj;

	pod::TensorView conv1d_w;
	pod::TensorView A_log;
	pod::TensorView dt_bias;
	pod::TensorView in_proj_a;
	pod::TensorView in_proj_b;
	pod::TensorView linear_norm;

	// Full attention weights
	pod::LinearWeight q_proj;
	pod::LinearWeight k_proj;
	pod::LinearWeight v_proj;
	pod::LinearWeight o_proj;

	pod::TensorView q_norm;
	pod::TensorView k_norm;

	// MLP weights
	pod::LinearWeight gate_proj;
	pod::LinearWeight up_proj;
	pod::LinearWeight down_proj;
};

class Qwen38Pipeline : public ModelPipeline {
public:
	Qwen38Pipeline();
	virtual ~Qwen38Pipeline() override;

	using ModelPipeline::forward;

	bool init( int device_id, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) override;
	bool init( const Qwen38Config& config, int rank, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring );
	bool load_weights( const char* model_dir, cudaStream_t stream = nullptr ) override;

	void embed_tokens( const int* token_ids, int count, cudaStream_t stream = nullptr ) override;
	void forward( const pod::BatchRows& rows, cudaStream_t stream = nullptr ) override;
	void sample_tokens( int* out_preds, int count, cudaStream_t stream = nullptr ) override;
	bool sample_local( int* out_tokens, float* out_vals, int count, cudaStream_t stream = nullptr ) override;
	bool topk_local( float* out_vals, int* out_ids, int count, cudaStream_t stream = nullptr ) override;
	const half* shard_logits_ptr() const override { return logits_.ptr<const half>(); }
	int shard_logits_cols() const override { return lm_shard_cols_; }
	const pod::TensorView& gather_logits(
		int rows,
		const half* const* peer_shard_ptrs,
		int peer_shard_cols,
		cudaStream_t stream = nullptr
	) override;

	const DraftHooks& draft_hooks() const override { return draft_hooks_; }
	void draft_head_topk( int rows, cudaStream_t stream = nullptr ) override;

	void checkpoint_state( int slot_id, cudaStream_t stream = nullptr ) override;
	void rollback_state( int slot_id, cudaStream_t stream = nullptr ) override;
	void set_spec_snapshot( bool on, cudaStream_t stream = nullptr ) override;
	void commit_spec_row( int slot_id, int row, cudaStream_t stream = nullptr ) override;
	void reset_state( int slot_id = 0 ) override;

	pod::TensorView& logits() override { return logits_; }
	const pod::TensorView& logits() const override { return logits_; }
	const half* tapped_states( int row = 0 ) const override { return tapped_states_ + static_cast<size_t>( row ) * 5 * config_.hidden_size; }
	const ModelMetadata& get_metadata() const override { return metadata_; }

	void destroy();

private:
	void set_rows( int m );
	// Layer stack + LM head for m rows; reads slots/positions from d_rows_ only (graph body)
	void forward_body( int m, cudaStream_t stream );

	pod::BatchRows* d_rows_ = nullptr; // device-resident row descriptor of the current forward
	engine::core::GraphCache graphs_;

	// Per-rank (TP-sharded) head counts
	int q_heads_ = 0;
	int kv_heads_ = 0;
	int lin_k_heads_ = 0;
	int lin_v_heads_ = 0;
	int lin_channels_ = 0;
	int inter_size_ = 0;

	engine::runtime::KVCachePool kv_pool_;
	int last_slot_ = 0;
	bool attn_tc_ = false;    // current forward: all rows share one slot (attention shares K/V tiles across them)
	bool attn_tc_on_ = true;  // tensor-core attention enabled (EXL3_ATTN_TC=0 keeps the per-row kernel)
	int key_hint_ = 0;   // keys the current chunk can attend to; sizes the prefill attention split only
	int last_rows_ = 0;

	ModelMetadata metadata_;
	Qwen38Config config_;
	int rank_ = 0;
	pod::MemoryArena* arena_ = nullptr;
	engine::comm::CollectiveRing* ring_ = nullptr;
	Qwen38State state_;
	bool weights_loaded_ = false;

	half* tapped_states_ = nullptr; // [MAX_BATCH, 5, hidden_size]
	int* d_token_ids_ = nullptr;    // [MAX_ROWS]
	int lm_mode_ = 0;               // BatchRows::LogitsMode of the forward in flight
	pod::TensorView hidden_states_;
	pod::TensorView norm_hidden_;

	// Full-attention buffers
	pod::TensorView q_gate_buf_;
	pod::TensorView q_buf_;
	pod::TensorView gate_attn_buf_;
	pod::TensorView k_buf_;
	pod::TensorView v_buf_;
	pod::TensorView attn_out_;

	// Linear-attention buffers
	pod::TensorView qkv_buf_;
	pod::TensorView conv_out_;    // [MAX_ROWS, channels] GDN conv output for prefill chunks
	pod::TensorView z_buf_;
	pod::TensorView gdn_gate_;
	pod::TensorView gdn_beta_;
	pod::TensorView gdn_out_;

	// MLP buffers
	pod::TensorView gate_buf_;
	pod::TensorView up_buf_;
	pod::TensorView mlp_inter_;
	pod::TensorView mlp_out_;

	// Model level
	Qwen38LayerWeights layers_[64];

	// DFlash2 attachment (see DraftHooks): tap ring, head scratch and per-rank candidate top-k scratch.
	DraftHooks draft_hooks_;
	pod::TensorView draft_taps_;         // [taps_cap, 5 * hidden]
	pod::TensorView draft_lm_logits_;    // [kDraftM, lm_shard_cols]
	static constexpr int kTopkTile = 4096;   // TOPK_THREADS * 16
	float* draft_cand_part_v_ = nullptr;
	int* draft_cand_part_i_ = nullptr;
	int* draft_cand_counters_ = nullptr;
	// Sampling top-k scratch (target logits, up to MAX_BATCH rows).
	float* samp_part_v_ = nullptr;
	int* samp_part_i_ = nullptr;
	int* samp_counters_ = nullptr;
	float* d_samp_vals_ = nullptr;
	int* d_samp_idx_ = nullptr;
	int* d_taps_owner_ = nullptr;

	pod::TensorView embed_tokens_;
	pod::TensorView final_norm_;
	pod::LinearWeight lm_head_;
	pod::TensorView logits_;
	int* d_argmax_idx_ = nullptr;
	int* h_argmax_idx_ = nullptr;
	// Vocabulary-parallel head: every rank owns vocab_size / tp_size columns, padded up to a multiple
	// of 256 so the dp4a int8 GEMV engages. lm_shard_real_cols_ excludes the pad.
	int tp_size_ = 1;
	bool shard_lm_ = false;   // false -> single-rank full-vocabulary head (EXL3_LM_SHARD=0)
	int lm_shard_cols_ = 0;
	int lm_shard_real_cols_ = 0;
	float* d_argmax_val_ = nullptr;
	pod::TensorView full_logits_;   // rank 0 only, for consumers that need the whole distribution
};

} // namespace engine::models
