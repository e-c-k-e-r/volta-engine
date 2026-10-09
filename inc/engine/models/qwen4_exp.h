#pragma once

#include "engine/config.h"
#include "engine/models/model.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/modules/linear.h"
#include "engine/modules/moe.h"
#include "engine/core/graph_cache.h"
#include "engine/loader/ngram_table.h"
#include "engine/runtime/kv_cache.h"
#include "engine/backend/types.h"
#include <vector>

namespace engine::models {

// Qwen3.8-Flash-Next (qwen4_exp), see docs/FLASH_NEXT.md. Defaults are the shipped checkpoint's; load_weights checks them against config.json.
struct Qwen4ExpConfig {
	int num_layers = 48;
	int hidden_size = 2560;
	int vocab_size = 248320;
	int hc_mult = 4;              // residual streams
	int hc_rank = 320;            // low-rank gate of the gated residual

	// Full attention (every 4th layer, QSA indexer)
	int num_q_heads = 24;
	int num_kv_heads = 2;
	int head_dim = 256;
	int rot_dim = 64;
	int indexer_heads = 4;
	int indexer_head_dim = 128;
	int indexer_budget = 2048;
	int indexer_compress = 4;

	// Linear attention (GDN)
	int linear_num_k_heads = 16;
	int linear_num_v_heads = 48;
	int linear_head_dim = 128;
	int linear_channels = 10240;

	// MoE
	int num_experts = 512;
	int top_k = 10;
	int moe_intermediate = 640;
	int shared_intermediate = 640;

	// N-gram embedding (PLE) layer: inserted before decoder layer ple_layer
	int ple_layer = 1;
	int ple_embed_dim = 2560;

	int max_seq = 32768;
	int max_slots = 2;
	float rms_norm_eps = 1e-6f;
	float rope_theta = 10000000.0f;

	bool is_full_attention( int layer ) const { return ( layer + 1 ) % 4 == 0; }
};

struct Qwen4ExpGatedResidual {
	pod::TensorView norm;      // [hc_mult * hidden] fp16, +1 applied at use
	pod::TensorView down;      // [rank, hc_mult * hidden]
	pod::TensorView up;        // [hc_mult * hidden, rank]
	pod::TensorView inject;    // [hc_mult, hc_mult * hidden] (absent on the final mixer)
};

struct Qwen4ExpPLEWeights {
	pod::LinearWeight key_proj, value_proj;   // dense fp16
	pod::TensorView norm_key, norm_query, norm_conv, conv_w;
};

struct Qwen4ExpLayerWeights {
	bool is_full_attention = false;
	Qwen4ExpGatedResidual attn_hc;
	Qwen4ExpGatedResidual mlp_hc;

	// GDN
	pod::LinearWeight in_proj_qkv, in_proj_z, out_proj;
	pod::TensorView conv1d_w, A_log, dt_bias, in_proj_a, in_proj_b, linear_norm;

	// Full attention
	pod::LinearWeight q_proj, k_proj, v_proj, o_proj;
	pod::TensorView q_norm, k_norm;
	pod::LinearWeight idx_qk_proj;
	pod::TensorView idx_q_norm, idx_k_norm;

	// MoE: router, expert-parallel routed experts, shared expert + its gate
	pod::TensorView router;
	pod::MoELayerWeights experts;
	pod::LinearWeight shared_gate, shared_up, shared_down;
	pod::TensorView shared_expert_gate;
};

class Qwen4ExpPipeline : public ModelPipeline {
public:
	Qwen4ExpPipeline();
	~Qwen4ExpPipeline() override;

	using ModelPipeline::forward;

	bool init( int device_id, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) override;
	bool load_weights( const char* model_dir, backend::stream_t stream = nullptr ) override;

	void embed_tokens( const int* token_ids, int count, backend::stream_t stream = nullptr ) override;
	void forward( const pod::BatchRows& rows, backend::stream_t stream = nullptr ) override;
	void sample_tokens( int* out_preds, int count, backend::stream_t stream = nullptr ) override;

	void checkpoint_state( int slot_id, backend::stream_t stream = nullptr ) override;
	void rollback_state( int slot_id, backend::stream_t stream = nullptr ) override;
	void reset_state( int slot_id = 0 ) override;

	pod::TensorView& logits() override { return logits_; }
	const pod::TensorView& logits() const override { return logits_; }
	const backend::half_t* tapped_states( int row = 0 ) const override { (void) row; return nullptr; }
	const ModelMetadata& get_metadata() const override { return metadata_; }

private:
	void set_rows( int m );
	void forward_body( int m, backend::stream_t stream );
	// One sublayer's input: the gated-residual mix of the streams into mixed_ (and post_ when the site has an inject gate)
	void gr_mix( const Qwen4ExpGatedResidual& g, bool with_post, int m, backend::stream_t stream );
	// N-gram embedding layer: host gather of the rows for the forward (called from forward()), device gate / conv (from forward_body)
	void ple_prepare( const pod::BatchRows& rows, int m, backend::stream_t stream );
	void ple_forward( int m, backend::stream_t stream );
	// Full-attention sublayer (the QSA layers): mixed_ in, the rank's partial o_proj result in y_
	void attention_forward( const Qwen4ExpLayerWeights& lw, int attn_idx, int m, backend::stream_t stream );
	void gdn_forward( const Qwen4ExpLayerWeights& lw, int linear_idx, int m, backend::stream_t stream );
	// MoE sublayer: mixed_ in, this rank's partial result (routed experts of the rank, plus the shared expert on rank 0) in y_
	void moe_forward( const Qwen4ExpLayerWeights& lw, int m, backend::stream_t stream );
	// Debug trace (trace_layers): the first position of the forward, rows of streams or of an fp16 matrix
	void trace_route( int tag, int m, backend::stream_t stream );
	void trace_rows( int tag, const void* dev, bool streams, int m, backend::stream_t stream, int cols = 0 );

	Qwen4ExpConfig config_;
	ModelMetadata metadata_;
	int rank_ = 0;
	int tp_ = 1;
	pod::MemoryArena* arena_ = nullptr;
	engine::comm::CollectiveRing* ring_ = nullptr;
	bool weights_loaded_ = false;
	int key_hint_ = 0;

	// Per-rank head counts (tensor parallel over ranks)
	int lin_k_heads_ = 0;
	int lin_v_heads_ = 0;
	int lin_channels_ = 0;

	struct State {
		float* gdn_states = nullptr;    // [slots, linear layers, v heads, 128, 128] fp32
		float* conv_history = nullptr;  // [slots, linear layers, channels, 3] fp32
		float* gdn_snap = nullptr;      // per-row post-state snapshots for speculation (kDraftM rows)
		float* conv_snap = nullptr;
		int* d_snap_on = nullptr;
		size_t slot_gdn_floats = 0;
		size_t slot_conv_floats = 0;
	} state_;

	pod::BatchRows* d_rows_ = nullptr;
	int* d_token_ids_ = nullptr;
	pod::TensorView embed_table_;
	pod::TensorView embed_rows_;      // [R, hidden] fp16
	pod::TensorView streams_;         // [R, hc, hidden] fp32
	pod::TensorView mixed_;           // [R, hidden] fp16: the sublayer input
	pod::TensorView y_;               // [R, hidden] fp16: the sublayer output
	float* post_ = nullptr;           // [R, hc]
	float* gr_scratch_ = nullptr;
	// N-gram embedding (PLE) layer
	Qwen4ExpPLEWeights ple_;
	pod::TensorView ple_emb_, ple_key_, ple_value_;
	float* ple_gated_ = nullptr;              // [R, hc, hidden] fp32
	backend::half_t* ple_normed_ = nullptr;   // [R, hc * hidden]
	backend::half_t* ple_ring_ = nullptr;     // [slots, RING, hc * hidden]
	std::vector<int> host_tokens_;            // the ids of the last embed_tokens
	std::vector<std::vector<int>> hist_tok_, hist_seg_;   // per slot: token at each position, start of its eos segment
	// MoE buffers
	int* topk_idx_ = nullptr;                 // [R, top_k]
	backend::half_t* topk_w_ = nullptr;
	pod::MoEScratch moe_scratch_;
	pod::TensorView sh_gate_buf_, sh_up_buf_, sh_inter_buf_, sh_out_;
	// Full-attention buffers (this rank: q_heads_ query heads over kv_heads_ = 1 KV head)
	int q_heads_ = 0, kv_heads_ = 0;
	engine::runtime::KVCachePool kv_pool_;
	bool attn_tc_ = false;     // all rows of the forward share one slot
	pod::TensorView q_gate_buf_, q_buf_, gate_attn_buf_, k_buf_, v_buf_, attn_out_;
	// GDN buffers
	pod::TensorView qkv_buf_, conv_out_, z_buf_, gdn_out_, gdn_gate_, gdn_beta_;

	Qwen4ExpGatedResidual final_mixer_;
	pod::LinearWeight lm_head_;
	pod::TensorView logits_;
	int lm_shard_real_cols_ = 0;
	int lm_shard_cols_ = 0;

	std::vector<Qwen4ExpLayerWeights> layers_;
};

} // namespace engine::models
