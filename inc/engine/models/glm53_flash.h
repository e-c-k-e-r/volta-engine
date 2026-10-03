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
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

namespace engine::models {

struct GLM53Config {
	int num_layers = 40;
	int hidden_size = 5120;
	int intermediate_size = 2048; // per expert
	int num_experts = 288;
	int top_k = 8;
	int mla_dc = 512;
	int mla_dr = 64;
	int dsa_top_k = 64;
	int vocab_size = 154880;
};

struct GLM53State {
	half* latent_cache; // [num_pages, 16, mla_dc]
	half* rope_cache;   // [num_pages, 16, mla_dr]
	int32_t* block_table;
	int* d_topk_indices;
	half* d_topk_weights;
	int current_pos;
};

struct GLM53LayerWeights {
	pod::TensorView input_norm;
	pod::TensorView post_attn_norm;
	pod::TensorView router_weights; // [hidden_size, num_experts]

	// MLA weights
	pod::TensorView q_proj_tr;
	pod::TensorView q_proj_suh;
	pod::TensorView q_proj_svh;
	int             q_proj_bits = 4;

	pod::TensorView kv_proj_tr;
	pod::TensorView kv_proj_suh;
	pod::TensorView kv_proj_svh;
	int             kv_proj_bits = 4;

	pod::TensorView o_proj_tr;
	pod::TensorView o_proj_suh;
	pod::TensorView o_proj_svh;
	int             o_proj_bits = 4;

	// MoE Expert weights
	pod::TensorView expert_gate_tr;
	pod::TensorView expert_gate_suh;
	pod::TensorView expert_gate_svh;
	int             expert_gate_bits = 4;

	pod::TensorView expert_up_tr;
	pod::TensorView expert_up_suh;
	pod::TensorView expert_up_svh;
	int             expert_up_bits = 4;

	pod::TensorView expert_down_tr;
	pod::TensorView expert_down_suh;
	pod::TensorView expert_down_svh;
	int             expert_down_bits = 4;
};

class GLM53Pipeline {
public:
	GLM53Pipeline();
	~GLM53Pipeline();

	bool init( const GLM53Config& config, int rank, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring );
	bool load_weights( const char* model_dir, cudaStream_t stream = nullptr );

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

	// MLA buffers
	pod::TensorView q_lat_buf_;
	pod::TensorView q_pe_buf_;
	pod::TensorView mla_out_;
	pod::TensorView attn_out_;

	// Router and MoE buffers
	pod::TensorView router_weights_;
	pod::TensorView moe_gate_buf_;
	pod::TensorView moe_up_buf_;
	pod::TensorView moe_inter_buf_;
	pod::TensorView moe_out_buf_;
	pod::TensorView expert_parts_buf_;
	pod::TensorView ffn_out_;

	// Model level
	GLM53LayerWeights layers_[45];
	pod::TensorView embed_tokens_;
	pod::TensorView final_norm_;
	pod::TensorView lm_head_trellis_;
	pod::TensorView lm_head_suh_;
	pod::TensorView lm_head_svh_;
	pod::TensorView logits_;

	cudaGraph_t graph_m1_;
	cudaGraphExec_t instance_m1_;
	bool graph_m1_captured_;

	cudaGraph_t graph_m8_;
	cudaGraphExec_t instance_m8_;
	bool graph_m8_captured_;
};

} // namespace engine::models
