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
	int max_seq = 2048;
};

struct Qwen38State {
	// 48 GDN layers: [48, 48, 128, 128] FP32 recurrent state matrix
	float* gdn_states;
	// 48 GDN layers: [48, 10240, 3] FP32 causal conv1d history
	float* conv_history;
	// 16 full-attention layers: [16, max_seq, 4, 256] FP16 KV cache
	half* k_cache;
	half* v_cache;
	int current_pos;
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
	pod::LinearWeight gate_up_proj;
	pod::LinearWeight down_proj;
};

class Qwen38Pipeline : public ModelPipeline {
public:
	Qwen38Pipeline();
	virtual ~Qwen38Pipeline() override;

	// ModelPipeline interface overrides
	bool init( int device_id, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring ) override;
	bool load_weights( const char* model_dir, cudaStream_t stream = nullptr ) override;
	void embed_token( int token_id, int pos, cudaStream_t stream = nullptr ) override;
	void forward_step( int token_id, int pos, int slot_id, cudaStream_t stream = nullptr ) override;
	void forward_step( int token_id, int pos, cudaStream_t stream = nullptr ) override;
	void forward_batch( const int* token_ids, int batch_size, int pos, cudaStream_t stream = nullptr ) override;
	int sample_next_token( cudaStream_t stream = nullptr ) override;
	bool capture_graph( int batch_size, cudaStream_t stream = nullptr ) override;
	void launch_graph( int batch_size, cudaStream_t stream = nullptr ) override;
	pod::TensorView get_logits() override;
	const ModelMetadata& get_metadata() const override;

	// KV Cache Management
	void set_kv_pool( engine::runtime::KVCachePool* pool ) override;
	void reset_kv_cache( int slot_id = 0 ) override;

	// Concrete interface
	bool init( const Qwen38Config& config, int rank, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring );
	void embed_token( int token_id, cudaStream_t stream = nullptr );
	void embed_tokens( const int* token_ids, int count, cudaStream_t stream = nullptr );
	int sample_next_token_impl( cudaStream_t stream = nullptr );
	void sample_next_tokens( int* h_preds, int count, cudaStream_t stream = nullptr );

	void forward_step(
		const pod::TensorView& input_tokens,
		pod::TensorView& output_logits,
		int pos,
		int slot_id = 0,
		cudaStream_t stream = nullptr
	);

	void forward_step(
		const pod::TensorView& input_tokens,
		pod::TensorView& output_logits,
		int pos,
		cudaStream_t stream = nullptr
	);

	void reset_state();
	void checkpoint_state( cudaStream_t stream = nullptr );
	void rollback_state( cudaStream_t stream = nullptr );
	void destroy();

	pod::TensorView& logits() { return logits_; }
	const pod::TensorView& logits() const { return logits_; }
	pod::TensorView& hidden_states() { return hidden_states_; }
	const pod::TensorView& hidden_states() const { return hidden_states_; }
	half* tapped_states( int row = 0 ) { return tapped_states_ + static_cast<size_t>( row ) * 5 * config_.hidden_size; }
	const half* tapped_states( int row = 0 ) const { return tapped_states_ + static_cast<size_t>( row ) * 5 * config_.hidden_size; }

private:
	engine::runtime::KVCachePool* kv_pool_ = nullptr;
	engine::runtime::KVCachePool internal_kv_pool_;
	int active_slot_id_ = 0;

	ModelMetadata metadata_;
	Qwen38Config config_;
	int rank_;
	pod::MemoryArena* arena_;
	engine::comm::CollectiveRing* ring_;
	Qwen38State state_;
	bool weights_loaded_;
	float* gdn_checkpoint_ = nullptr;
	float* conv_checkpoint_ = nullptr;

	half* tapped_states_; // [5, hidden_size] tapped hidden states for DFlash2
	pod::TensorView hidden_states_;
	pod::TensorView residual_;
	pod::TensorView norm_hidden_;

	// Benchmark dummy weights/buffers
	pod::TensorView dummy_proj_;
	pod::TensorView dummy_suh_;
	pod::TensorView dummy_svh_;
	pod::TensorView dummy_norm_;
	pod::TensorView dummy_down_;
	pod::TensorView dummy_suh_h_;
	pod::TensorView dummy_svh_h_;
	pod::TensorView dummy_norm_h_;

	// Full-attention buffers
	pod::TensorView q_gate_buf_;
	pod::TensorView q_buf_;
	pod::TensorView gate_attn_buf_;
	pod::TensorView k_buf_;
	pod::TensorView v_buf_;
	pod::TensorView attn_out_;

	// Linear-attention buffers
	pod::TensorView qkv_buf_;
	pod::TensorView z_buf_;
	pod::TensorView gdn_gate_;
	pod::TensorView gdn_beta_;
	pod::TensorView gdn_out_;

	// MLP buffers
	pod::TensorView gate_buf_;
	pod::TensorView up_buf_;
	pod::TensorView gate_up_buf_;
	pod::TensorView mlp_inter_;
	pod::TensorView mlp_out_;

	// Model level
	Qwen38LayerWeights layers_[64];
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
