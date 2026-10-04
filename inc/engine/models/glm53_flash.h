#pragma once

#include "engine/config.h"
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
};

struct GLM53State {
	half* latent_cache; // [num_pages, 16, mla_dc]
	half* rope_cache;   // [num_pages, 16, mla_dr]
	half* k_cache;      // [11, max_seq, 16 * 256]
	half* v_cache;      // [11, max_seq, 16 * 256]
	int32_t* block_table;
	int* d_topk_indices;
	half* d_topk_weights;
	int current_pos;
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
	pod::TensorView conv_history;   // [6144 * 3] float
	pod::TensorView A_log;          // [16] half
	pod::TensorView dt_bias;        // [2048] half

	pod::TensorView b_proj;         // [16, 4096] half
	pod::TensorView f_a_proj;       // [128, 4096] half
	pod::TensorView f_b_proj;       // [2048, 128] half
	pod::TensorView g_a_proj;       // [128, 4096] half
	pod::TensorView g_b_proj;       // [2048, 128] half
	pod::TensorView o_norm;         // [128] half

	float*          gdn_state = nullptr; // [16 * 128 * 128] float

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

	// 72 local experts for this rank: rank * 72 .. (rank + 1) * 72 - 1
	GLM53ExpertWeights experts[72];

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

class GLM53Pipeline {
public:
	GLM53Pipeline();
	~GLM53Pipeline();

	bool init( const GLM53Config& config, int rank, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring );
	bool load_weights( const char* model_dir, cudaStream_t stream = nullptr );
	void embed_token( int token_id, cudaStream_t stream = nullptr );
	int sample_next_token( cudaStream_t stream = nullptr );

	void forward_step(
		const pod::TensorView& input_tokens,
		pod::TensorView& output_logits,
		int pos,
		cudaStream_t stream = nullptr
	);

	bool capture_graph( int m, cudaStream_t stream = nullptr );
	void launch_graph( int m, cudaStream_t stream = nullptr );

	void reset_state();
	void destroy();

	const pod::TensorView& logits() const { return logits_; }

private:
	GLM53Config config_;
	int rank_;
	pod::MemoryArena* arena_;
	engine::comm::CollectiveRing* ring_;
	GLM53State state_;
	bool weights_loaded_;

	pod::TensorView hidden_states_;
	pod::TensorView residual_;
	pod::TensorView norm_hidden_;

	// mHC residual stream buffers
	pod::TensorView hidden_streams_;
	pod::TensorView residual_streams_;
	pod::TensorView mhc_collapsed_;
	pod::TensorView mhc_inv_rms_;
	pod::TensorView mhc_mix_logits_;
	pod::TensorView mhc_pre_;
	pod::TensorView mhc_post_;
	pod::TensorView mhc_comb_;

	// MLA buffers
	pod::TensorView q_lat_buf_;
	pod::TensorView q_pe_buf_;
	pod::TensorView mla_out_;
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

	// Router and MoE buffers
	pod::TensorView router_weights_;
	pod::TensorView moe_gate_buf_;
	pod::TensorView moe_up_buf_;
	pod::TensorView moe_inter_buf_;
	pod::TensorView moe_expert_out_;
	pod::TensorView local_moe_out_;
	pod::TensorView ffn_out_;

	// Fallback dummy descriptors
	pod::LinearWeight dummy_linear_;
	pod::TensorView   dummy_norm_;

	// Model level
	GLM53LayerWeights layers_[45];
	pod::TensorView embed_tokens_;
	pod::TensorView final_norm_;
	pod::LinearWeight lm_head_;
	pod::TensorView logits_;
	int* d_argmax_idx_;
	int* h_argmax_idx_;

	cudaGraph_t graph_m1_;
	cudaGraphExec_t instance_m1_;
	bool graph_m1_captured_;

	cudaGraph_t graph_m4_;
	cudaGraphExec_t instance_m4_;
	bool graph_m4_captured_;

	cudaGraph_t graph_m8_;
	cudaGraphExec_t instance_m8_;
	bool graph_m8_captured_;
};

} // namespace engine::models
