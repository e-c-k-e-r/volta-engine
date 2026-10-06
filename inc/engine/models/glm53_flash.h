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
#include "engine/modules/mla.h"
#include "engine/modules/moe.h"
#include "engine/modules/mhc.h"
#include "engine/modules/linear.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

namespace engine::models {

struct GLM53Config {
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
	int max_seq = 2048;
	int max_slots = 1; // MLA KV is ~369 MiB/slot/rank; weights fill the 31 GiB arena
	int num_mla_layers = 11;
	int kda_heads = 16;
	int kda_head_dim = 128;
	int kda_channels = 6144;
	float rms_norm_eps = 1e-6f;
	float rms_norm_bias = 1.0f;
};

// Per-rank recurrent state laid out [slot][layer][...] so one slot is contiguous.
struct GLM53State {
	float* kda_states = nullptr;      // [slots, 45, heads, 128, 128] FP32
	float* conv_history = nullptr;    // [slots, 45, channels, 3] FP32
	float* kda_checkpoint = nullptr;
	float* conv_checkpoint = nullptr;
	size_t layer_kda_floats = 0;
	size_t layer_conv_floats = 0;
	int* d_topk_indices = nullptr;
	half* d_topk_weights = nullptr;
	pod::MoEScratch moe;
};

struct GLM53ExpertWeights {
	pod::LinearWeight gate;
	pod::LinearWeight up;
	pod::LinearWeight down;
};

struct GLM53LayerWeights {
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

	// Attention output projection
	pod::LinearWeight o_proj;

	// Dense MLP (layers 0, 1, 2)
	GLM53ExpertWeights dense_mlp;

	// 72 local routed experts for this rank: rank * 72 .. (rank + 1) * 72 - 1
	pod::MoELayerWeights moe_weights;

	// Shared expert
	bool has_shared_expert = false;
	GLM53ExpertWeights shared_expert;

	// mHC (Manifold-Constrained Hyper-Connections)
	pod::TensorView hc_attn_base;
	pod::TensorView hc_attn_scale;
	pod::TensorView hc_attn_fn;

	pod::TensorView hc_ffn_base;
	pod::TensorView hc_ffn_scale;
	pod::TensorView hc_ffn_fn;
};

class GLM53Pipeline : public ModelPipeline {
public:
	GLM53Pipeline();
	virtual ~GLM53Pipeline() override;

	using ModelPipeline::forward;

	bool init( int device_id, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) override;
	bool init( const GLM53Config& config, int rank, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring );
	bool load_weights( const char* model_dir, cudaStream_t stream = nullptr ) override;

	void embed_tokens( const int* token_ids, int count, cudaStream_t stream = nullptr ) override;
	void forward( const pod::BatchRows& rows, cudaStream_t stream = nullptr ) override;
	void sample_tokens( int* out_preds, int count, cudaStream_t stream = nullptr ) override;

	void checkpoint_state( int slot_id, cudaStream_t stream = nullptr ) override;
	void rollback_state( int slot_id, cudaStream_t stream = nullptr ) override;
	void reset_state( int slot_id = 0 ) override;

	pod::TensorView& logits() override { return logits_; }
	const pod::TensorView& logits() const override { return logits_; }
	const half* tapped_states( int row = 0 ) const override { (void) row; return nullptr; }
	const ModelMetadata& get_metadata() const override { return metadata_; }

	void destroy();

private:
	// Single-token layer stack for one row: embed_rows_[row] -> logits_[row]
	void forward_row( int row, int pos, int slot_id, cudaStream_t stream );

	engine::runtime::KVCachePool kv_pool_;
	int last_rows_ = 0;

	ModelMetadata metadata_;
	GLM53Config config_;
	int rank_ = 0;
	pod::MemoryArena* arena_ = nullptr;
	engine::comm::CollectiveRing* ring_ = nullptr;
	GLM53State state_;
	bool weights_loaded_ = false;

	pod::TensorView embed_rows_;    // [MAX_BATCH, hidden] embedded inputs
	int* d_token_ids_ = nullptr;    // [MAX_BATCH]
	pod::TensorView hidden_states_; // [1, hidden] current row
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
	GLM53LayerWeights layers_[45];
	pod::TensorView embed_tokens_;
	pod::TensorView final_norm_;
	pod::LinearWeight lm_head_;
	pod::TensorView logits_;  // [MAX_BATCH, vocab] (rank 0)
	int* d_argmax_idx_ = nullptr;
	int* h_argmax_idx_ = nullptr;
};

} // namespace engine::models
