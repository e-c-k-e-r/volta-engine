#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/modules/trellis.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

namespace engine::models {

struct Qwen38Config {
	int num_layers = 64;
	int hidden_size = 5120;
	int intermediate_size = 10240;
	int num_heads = 40;
	int head_dim = 128;
	int num_linear_layers = 48; // 48 GDN layers
	int num_attn_layers = 16;   // 16 SWA layers (every 4th layer)
	int swa_window = 16;
	int block_size = 16;
};

struct Qwen38State {
	// 48 GDN layers: [48, num_heads, 128, 128] FP32 recurrent state matrix
	float* gdn_states;
	// 16 SWA layers: paged KV cache [num_pages, 16, num_heads, 128]
	half* k_cache;
	half* v_cache;
	int32_t* block_table;
	int current_pos;
};

class Qwen38Pipeline {
public:
	Qwen38Pipeline();
	~Qwen38Pipeline();

	bool init( const Qwen38Config& config, int rank, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring );

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
	Qwen38Config config_;
	int rank_;
	pod::MemoryArena* arena_;
	engine::comm::CollectiveRing* ring_;
	Qwen38State state_;

	pod::TensorView hidden_states_;
	pod::TensorView residual_;
	pod::TensorView norm_weights_;
	pod::TensorView qkv_proj_;
	pod::TensorView out_proj_;
	pod::TensorView gate_up_proj_;
	pod::TensorView down_proj_;
	pod::TensorView suh_;
	pod::TensorView svh_;

	cudaGraph_t graph_m1_;
	cudaGraphExec_t instance_m1_;
	bool graph_m1_captured_;

	cudaGraph_t graph_m8_;
	cudaGraphExec_t instance_m8_;
	bool graph_m8_captured_;
};

} // namespace engine::models
