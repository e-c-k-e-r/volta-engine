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

struct GLM53Config {
	int num_layers = 40;
	int hidden_size = 5120;
	int intermediate_size = 2048; // per expert
	int num_experts = 288;
	int top_k = 8;
	int mla_dc = 512;
	int mla_dr = 64;
	int dsa_top_k = 64;
};

struct GLM53State {
	half* latent_cache; // [num_pages, 16, mla_dc]
	half* rope_cache;   // [num_pages, 16, mla_dr]
	int32_t* block_table;
	int* d_topk_indices;
	half* d_topk_weights;
	int current_pos;
};

class GLM53Pipeline {
public:
	GLM53Pipeline();
	~GLM53Pipeline();

	bool init( const GLM53Config& config, int rank, pod::MemoryArena& arena, engine::comm::CollectiveRing* ring );

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

	pod::TensorView hidden_states_;
	pod::TensorView residual_;
	pod::TensorView norm_weights_;
	pod::TensorView router_weights_; // [5120, 288]
	pod::TensorView expert_down_;
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
