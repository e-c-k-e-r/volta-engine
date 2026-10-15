#pragma once

#include "engine/config.h"
#include "engine/models/model.h"
#include "engine/models/glm5_vision.h"
#include "engine/runtime/kv_cache.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/core/graph_cache.h"
#include "engine/modules/trellis.h"
#include "engine/modules/norm.h"
#include "engine/modules/swiglu.h"
#include "engine/modules/rope.h"
#include "engine/modules/mla.h"
#include "engine/modules/moe.h"
#include "engine/modules/expert_tier.h"
#include "engine/modules/mhc.h"
#include "engine/modules/linear.h"
#include "engine/backend/types.h"
#include <cstdint>
#include <vector>

namespace engine::loader { class SafeTensorsRepo; }

namespace engine::models {

struct Glm5NextConfig {
	int num_layers = 45;
	int hidden_size = 4096;
	int intermediate_size = 2048; // per expert
	int dense_intermediate_size = 12288;
	int num_experts = 288;
	int top_k = 8;
	int mla_dc = 512;
	int mla_dr = 64;
	int dsa_top_k = 64;
	int vocab_size = 154880;
	int tp_size = 4;
	int max_seq = 12288;
	int max_slots = 1; // absorbed MLA latent cache: ~135 MiB/slot/rank at 12K (expanded K/V: 369 MiB at 2K); weights fill the 31 GiB arena and the FP16 drafter needs the last GiB (ENGINE_MAX_SEQ=32768 fits without it, 16384 with the EXL3 drafter)
	int num_mla_layers = 11;
	int kda_heads = 16;
	int kda_head_dim = 128;
	int kda_channels = 6144;
	float rms_norm_eps = 1e-6f;
	float rms_norm_bias = 1.0f;
	int tap_layers[5] = { 5, 14, 24, 33, 42 };   // DFlash2 drafter taps (target_layer_ids)
};

// Per-rank recurrent state laid out [slot][layer][...] so one slot is contiguous.
struct Glm5NextState {
	float* kda_states = nullptr;      // [slots, 45, heads, 128, 128] FP32
	float* conv_history = nullptr;    // [slots, 45, channels, 3] FP32
	float* kda_snap = nullptr;        // [kDraftM, 45, heads, 128, 128] per-row post-state (speculative verify); null in journal mode
	unsigned char* kda_journal = nullptr;   // journal mode: [34, snap_rows_cap, record] the verify rows' k, v, decay, beta (GDN::kda_journal_row_bytes)
	size_t journal_row_stride = 0, journal_layer_stride = 0;
	int snap_rows_cap = 0;            // verify rows the snapshots / the journal hold
	float* conv_snap = nullptr;       // [snap_rows_cap, 45, channels, 3]
	int* d_snap_on = nullptr;         // device flag: forwards publish the per-row snapshots while non-zero
	float* kda_checkpoint = nullptr;
	float* conv_checkpoint = nullptr;
	size_t layer_kda_floats = 0;
	size_t layer_conv_floats = 0;
	size_t slot_kda_floats = 0;       // linear layers only
	size_t slot_conv_floats = 0;
	int* d_topk_indices = nullptr;
	backend::half_t* d_topk_weights = nullptr;
	pod::MoEScratch moe;
};

struct Glm5NextExpertWeights {
	pod::LinearWeight gate;
	pod::LinearWeight up;
	pod::LinearWeight down;
};

struct Glm5NextLayerWeights {
	bool is_moe = false;
	bool is_linear_attn = false;
	bool is_mla = false;
	pod::TensorView input_norm;
	pod::TensorView post_attn_norm;
	pod::TensorView router_weights; // [hidden_size, num_experts]
	pod::TensorView e_score_bias;   // [num_experts]

	// Linear Attention (KDA / GDN-class)
	pod::LinearWeight qkv;

	pod::TensorView conv1d_weights; // [6144 * 4] half
	pod::TensorView A_log;          // [16] half
	pod::TensorView dt_bias;        // [2048] half

	pod::TensorView b_proj;         // [16, 4096] half
	pod::TensorView f_a_proj;       // [128, 4096] half
	pod::TensorView f_b_proj;       // [2048, 128] half
	pod::TensorView g_a_proj;       // [128, 4096] half
	pod::TensorView g_b_proj;       // [2048, 128] half
	pod::TensorView o_norm;         // [128] half

	// MLA Attention (DeepSeek-style MLA)
	pod::LinearWeight q_a;
	pod::TensorView   q_a_norm;

	pod::LinearWeight q_b;

	pod::LinearWeight kv_a;
	pod::TensorView   kv_a_norm;

	pod::TensorView kv_b_proj; // [8192, 512] FP16

	// DSA lightning indexer (every MLA layer; replicated on all ranks, the selection is shared by all heads)
	pod::LinearWeight idx_wq_b;     // q_lora_norm [1536] -> 32 heads x 128
	pod::TensorView   idx_dense;    // [288, 4096] FP16: wk (128) | weights_proj (32) | k-pool gate (128)
	pod::TensorView   idx_ln_w, idx_ln_b;   // k_norm LayerNorm [128]
	pod::TensorView   idx_ape;      // [4, 128] k-pool in-pool bias

	// Attention output projection
	pod::LinearWeight o_proj;

	// Dense MLP (layers 0, 1, 2)
	Glm5NextExpertWeights dense_mlp;

	// 72 local routed experts for this rank: rank * 72 .. (rank + 1) * 72 - 1
	pod::MoELayerWeights moe_weights;

	// Shared expert
	bool has_shared_expert = false;
	bool dense_sliced = false;    // the dense MLP of layers 0-2 is cut over the ranks (all-reduce after the down projection) instead of replicated
	bool shared_sliced = false;   // each rank holds 1/tp of the shared expert's width and adds its partial output before the all-reduce; otherwise rank 0 runs all of it
	Glm5NextExpertWeights shared_expert;

	// mHC (Manifold-Constrained Hyper-Connections)
	pod::TensorView hc_attn_base;
	pod::TensorView hc_attn_scale;
	pod::TensorView hc_attn_fn;

	pod::TensorView hc_ffn_base;
	pod::TensorView hc_ffn_scale;
	pod::TensorView hc_ffn_fn;
};

class Glm5NextPipeline : public ModelPipeline {
public:
	Glm5NextPipeline();
	virtual ~Glm5NextPipeline() override;

	using ModelPipeline::forward;

	bool init( int device_id, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) override;
	bool init( const Glm5NextConfig& config, int rank, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring );
	bool load_weights( const char* model_dir, backend::stream_t stream = nullptr ) override;
	bool inject_embeddings( const backend::half_t* host_rows, int dst_row, int rows, backend::stream_t stream = nullptr ) override;
	bool load_vision( const char* model_dir, backend::stream_t stream = nullptr ) override;
	const engine::loader::VisionParams* vision_params() const override { return vision_.loaded() ? &vision_.params() : nullptr; }
	bool encode_image( const engine::loader::PreparedImage& image, std::vector<backend::half_t>* out, backend::stream_t stream = nullptr ) override;

	void embed_tokens( const int* token_ids, int count, backend::stream_t stream = nullptr ) override;
	void forward( const pod::BatchRows& rows, backend::stream_t stream = nullptr ) override;
	void sample_tokens( int* out_preds, int count, backend::stream_t stream = nullptr ) override;
	bool sample_local( int* out_tokens, float* out_vals, int count, backend::stream_t stream = nullptr ) override;
	bool topk_local( float* out_vals, int* out_ids, int count, backend::stream_t stream = nullptr ) override;
	const backend::half_t* shard_logits_ptr() const override { return logits_.ptr<const backend::half_t>(); }
	int shard_logits_cols() const override { return lm_shard_cols_; }
	const pod::TensorView& gather_logits(
		int rows,
		const backend::half_t* const* peer_shard_ptrs,
		int peer_shard_cols,
		backend::stream_t stream = nullptr
	) override;

	const DraftHooks& draft_hooks() const override { return draft_hooks_; }
	bool expert_heat( std::vector<uint32_t>& out, int& layers, int& experts, bool reset = false ) override;
	// ENGINE_EXPERT_CACHE < 1: fold the routing counters into the experts' scores and move hot experts into the device tier
	bool rebalance_experts( backend::stream_t stream = nullptr ) override;
	void draft_head_topk( int rows, backend::stream_t stream = nullptr ) override;
	void set_spec_snapshot( bool on, backend::stream_t stream = nullptr ) override;
	void commit_spec_row( int slot_id, int row, backend::stream_t stream = nullptr ) override;
	void commit_spec_rows( int slot_id, int first_row, int last_row, backend::stream_t stream = nullptr ) override;
	bool spec_commit_always() const override { return use_journal_; }
	void set_snapshot_taps( bool on ) override { snap_taps_ = on; }
	int seq_align() const override { return 4; }   // DSA::POOL
	size_t seq_state_bytes( int from, int tokens, bool with_taps ) const override;
	bool save_seq_state( int slot_id, int from, int tokens, bool with_taps, void* host_dst, backend::stream_t stream = nullptr ) override;
	bool load_seq_state( int slot_id, int from, int tokens, bool with_taps, const void* host_src, bool kv_only, backend::stream_t stream = nullptr ) override;

	void checkpoint_state( int slot_id, backend::stream_t stream = nullptr ) override;
	void rollback_state( int slot_id, backend::stream_t stream = nullptr ) override;
	void reset_state( int slot_id = 0 ) override;
	void set_reserve_after_weights( size_t bytes ) override { reserve_after_weights_ = bytes; }
	bool reserve_tokens( int slot_id, int tokens ) override;
	void release_tokens( int slot_id ) override;
	int pool_tokens() const override { return kv_pool_tokens_; }
	int free_pool_tokens() const override { return static_cast<int>( kv_free_.size() ) << kKvPageShift; }

	pod::TensorView& logits() override { return logits_; }
	const pod::TensorView& logits() const override { return logits_; }
	const backend::half_t* tapped_states( int row = 0 ) const override { (void) row; return nullptr; }
	const ModelMetadata& get_metadata() const override { return metadata_; }

	// MTP head (speculator=mtp, from the NVFP4 checkpoint's layer 45): see model.h 5d. The history rows are the final-norm hidden rows
	bool has_mtp() const override { return mtp_on_; }
	void mtp_step( const pod::BatchRows& rows, const int* tokens, int hidden_mode, int hist_row0, bool logits_last, backend::stream_t stream = nullptr ) override;
	void mtp_save_hist( int src_row, int rows, int dst_row, backend::stream_t stream = nullptr ) override;

	void destroy();

private:
	static constexpr int kTopkTile = 4096;   // TOPK_THREADS * 16, as in the Qwen head

	// 154880 = 1210 x 128 columns do not split four ways at 128-column granularity (the output Hadamard spans 128):
	// ranks 0..1 take 303 spans, ranks 2..3 take 302
	static int lm_shard_spans( int vocab, int tp, int rank ) { return vocab / 128 / tp + ( rank < vocab / 128 % tp ? 1 : 0 ); }
	static int lm_shard_offset( int vocab, int tp, int rank ) { return 128 * ( rank * ( vocab / 128 / tp ) + MIN( rank, vocab / 128 % tp ) ); }

	// Layer stack over m rows at once: embed_rows_[0..m) -> logits_[0..m); slot and position of each row are
	// read from d_rows_ on device so the launch sequence depends on m alone and is graph-replayable
	void forward_body( int m, backend::stream_t stream );
	// One decoder layer over the m rows of hidden_streams_ (mHC layers) or hidden_states_ (the MTP head's, which has no hyper-connection);
	// `layer` picks the recurrent and latent state (45 = the MTP head's MLA layer, the last latent slot)
	void layer_body( int layer, Glm5NextLayerWeights& lw, int m, backend::stream_t stream );
	void set_rows( int m );

	pod::BatchRows* d_rows_ = nullptr;
	backend::stream_t side_stream_ = nullptr;   // shared expert beside the fused MoE decode kernels (CUDA)
	engine::core::GraphCache graphs_;

	// Absorbed MLA: the cache holds the 512-wide KV latent per token, shared by all heads,
	// instead of expanded per-head K / V (1 KiB vs 16 KiB per token per layer, so kv_pool_ is not allocated)
	size_t reserve_after_weights_ = 0;
	bool moe_tp_ = true;                // routed experts sharded over the ranks instead of expert-parallel
	engine::modules::ExpertTier tier_;
	pod::MemoryArena host_arena_{};     // pinned cold-expert tier
	int tier_layers_ = 0;
	// Calibration (ENGINE_HESSIAN_DIR): x^T x of a projection input over the prefill rows, per layer and site (0 attention input, 1 FFN input,
	// 2 o_proj input (rank slice), 3 shared-expert down input, 4 q-lora norm output (MLA), 5 lm_head input with layer 0)
	struct HessianSite { float* h = nullptr; int k = 0; int64_t rows = 0; };
	std::vector<HessianSite> hess_;
	void hessian( int layer, int site, const void* x, int k, int m, backend::stream_t stream );
	void hessian_dump();
	void* latent_cache_ = nullptr;    // [slots][mla layers][max_seq][512] in latent_q_ (K and V are the same latent)
	void* latent_scales_ = nullptr;   // per-block scales of the quantized formats, same element indexing
	pod::KVQ latent_q_ = pod::KVQ::F16;
	size_t latent_slot_stride_ = 0;   // elements per slot
	pod::KVRef latent_ref() const;
	// Shared token pool (option kv_pool > 0): the latent cache and the pooled indexer keys of all slots are [layer][pool tokens]
	// arrays cut into pages of 1 << kKvPageShift tokens, handed to a slot as its positions grow and returned when its request ends
	static constexpr int kKvPageShift = 9;
	int kv_pool_tokens_ = 0;          // 0 = the per-slot layout
	int kv_pages_per_slot_ = 0;
	std::vector<int> kv_table_;       // [slots][pages per slot]: physical page of each logical page (0 where none)
	std::vector<int> kv_slot_pages_;  // pages held per slot
	std::vector<int> kv_free_;        // free physical pages
	int* d_kv_table_ = nullptr;
	bool kv_dirty_ = false;
	pod::KVPages kv_pages() const;
	// expert_cache=auto, before the experts of a MoE layer: the experts per layer the arena can hold for this and the layers after it, from the
	// bytes the layers loaded so far took, the checkpoint size of the layers still to come, the speculator and a margin
	int auto_resident( const engine::loader::SafeTensorsRepo& repo, int layer, size_t layers_begin, size_t experts_cost, int layers_done, int experts, size_t slot_bytes, int moe_layers ) const;
	void kv_prepare( const pod::BatchRows& rows, backend::stream_t stream );   // pages for the rows' positions, table on the device
	// One contiguous device range per page of the tokens [from, tokens) of a slot: fn( first token, tokens in the range )
	template <typename F> void for_kv_ranges( int slot_id, int from, int tokens, F&& fn ) const;
	size_t latent_data_bytes( size_t elems ) const { return pod::kv_data_bytes( latent_q_, elems ); }
	size_t latent_scale_bytes( size_t elems ) const { return pod::kv_scale_bytes( latent_q_, elems ); }
	float* latent_ws_ = nullptr;      // split-KV partials
	int ws_seq_ = 0;                  // context the workspace is sized for
	bool use_journal_ = false;        // speculative verify journals the KDA rows and replays the accepted ones instead of keeping a state snapshot per row
	pod::TensorView q_lat_;           // [rows, 16 * 512]
	// DSA (ENGINE_DSA, default on with the latent cache): indexer planes and selection scratch
	bool dsa_ = true;
	backend::half_t* dsa_plane_ = nullptr;       // [slots][mla layers][DSA::PLANE_CAP][256]  ring of per-token [k_norm | gate]
	backend::half_t* dsa_pool_ = nullptr;        // [slots][mla layers][max_seq / 4][128]  pooled keys
	size_t dsa_plane_stride_ = 0, dsa_pool_stride_ = 0;
	backend::half_t* dsa_proj_ = nullptr;        // [rows, 288]
	backend::half_t* dsa_w_ = nullptr;           // [rows, 32]
	backend::half_t* dsa_q_ = nullptr;           // [rows, 32 * 128]
	backend::half_t* dsa_scores_ = nullptr;      // [rows, max_seq / 4]
	int* dsa_idx_ = nullptr;          // [rows, DSA::KPAD]
	int* dsa_nk_ = nullptr;           // [rows]
	pod::TensorView o_lat_;
	int last_rows_ = 0;
	int mla_slots_ = 11;              // latent / indexer layers per slot: the MLA layers, plus the MTP head's
	bool mtp_on_ = false;
	Glm5NextLayerWeights mtp_layer_;
	pod::LinearWeight mtp_eh_;        // [hidden, 2 * hidden] over [normed embedding | normed hidden]
	pod::TensorView mtp_enorm_, mtp_hnorm_, mtp_norm_;
	pod::TensorView mtp_emb_, mtp_xe_, mtp_xh_, mtp_in_, mtp_stage_, mtp_out_, mtp_chain_, mtp_hist_;
	int lm_mode_ = pod::BatchRows::LOGITS_ALL;
	int key_hint_ = 0;   // keys the current chunk can attend to; sizes the prefill attention split only

	ModelMetadata metadata_;
	Glm5NextConfig config_;
	int rank_ = 0;
	int logit_rows_ = 16;   // rows the logits buffers hold
	int lh_ = 16;   // attention heads of this rank (MLA and KDA): 64 / tp_size
	pod::MemoryArena* arena_ = nullptr;
	engine::comm::CollectiveRing* ring_ = nullptr;
	Glm5NextState state_;
	bool weights_loaded_ = false;

	// DFlash2 attachment (see DraftHooks): tap ring, head scratch and per-rank candidate top-k scratch
	DraftHooks draft_hooks_;
	pod::TensorView draft_taps_;         // [taps_cap, 5 * hidden]
	bool snap_taps_ = false;             // sequence snapshots include the tap ring rows
	size_t tap_snapshot_bytes( int tokens ) const;
	void copy_seq_rows( int slot_id, int from, int tokens, char* host, bool to_host, backend::stream_t stream );
	pod::TensorView tap_tmp_;            // [MAX_ROWS, hidden] mean of the 4 residual streams at a tapped layer
	pod::TensorView draft_lm_logits_;    // [kDraftM, lm_shard_cols]
	float* draft_cand_part_v_ = nullptr;
	int* draft_cand_part_i_ = nullptr;
	int* draft_cand_counters_ = nullptr;
	int* d_taps_owner_ = nullptr;

	pod::TensorView embed_rows_;    // [MAX_ROWS, hidden] embedded inputs
	Glm5VisionTower vision_;        // rank 0 only
	int* d_token_ids_ = nullptr;    // [MAX_ROWS]
	pod::TensorView hidden_states_; // [rows, hidden]
	pod::TensorView norm_hidden_;

	// mHC residual stream buffers
	pod::TensorView hidden_streams_;
	pod::TensorView mhc_collapsed_;
	pod::TensorView mhc_inv_rms_;
	pod::TensorView mhc_mix_logits_;
	pod::TensorView mhc_pre_;
	pod::TensorView mhc_post_;
	pod::TensorView mhc_comb_;

	// MLA buffers
	pod::TensorView attn_out_;
	pod::TensorView q_lora_buf_;
	pod::TensorView q_lora_norm_buf_;
	pod::TensorView q_buf_;
	pod::TensorView kv_lora_buf_;
	pod::TensorView kv_lora_norm_buf_;
	pod::TensorView k_head_buf_;
	pod::TensorView v_head_buf_;
	pod::TensorView mla_heads_out_;

	// KDA linear attention buffers
	pod::TensorView qkv_buf_;       // [1, 6144] FP16
	pod::TensorView f_a_buf_;       // [1, 128] FP16
	pod::TensorView g_a_buf_;       // [1, 128] FP16
	pod::TensorView beta_buf_;      // [1, 16] FP32
	pod::TensorView g_decay_buf_;   // [1, 2048] FP32
	pod::TensorView z_buf_;         // [1, 2048] FP16
	pod::TensorView attn_head_out_; // [1, 2048] FP16

	// Dense / shared-expert FFN buffers
	pod::TensorView moe_gate_buf_;
	pod::TensorView moe_up_buf_;
	pod::TensorView moe_inter_buf_;
	pod::TensorView moe_expert_out_;
	pod::TensorView local_moe_out_;
	pod::TensorView ffn_out_;

	// Model level
	Glm5NextLayerWeights layers_[45];
	pod::TensorView embed_tokens_;
	pod::TensorView final_norm_;
	pod::LinearWeight lm_head_;
	pod::TensorView logits_;  // [MAX_BATCH, lm_shard_cols] this rank's vocabulary shard
	int* d_argmax_idx_ = nullptr;
	float* d_argmax_val_ = nullptr;
	// Vocabulary-parallel head: every rank owns vocab_size / tp_size columns padded to a multiple of 256
	int lm_shard_cols_ = 0;
	int lm_shard_real_cols_ = 0;
	int lm_shard_offset_ = 0;        // first global token id of this rank's slice
	pod::TensorView full_logits_;    // rank 0 only: whole-distribution consumers
	float* samp_part_v_ = nullptr;   // top-k sampling scratch
	int* samp_part_i_ = nullptr;
	int* samp_counters_ = nullptr;
	float* d_samp_vals_ = nullptr;
	int* d_samp_idx_ = nullptr;
	int* h_argmax_idx_ = nullptr;
};

} // namespace engine::models
