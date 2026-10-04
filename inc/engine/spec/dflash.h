#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/linear.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace pod {

struct DFlashConfig {
	int block_size = 8;          // K = 7 draft tokens, M = 8 verify
	int conv_kernel_size = 2;    // taps = 2
	int conv_group_size = 16;
	int selector_rank = 256;
	int selector_top_k = 16;
	int hidden_size = 5120;
	int vocab_size = 248320;
	int num_hidden_layers = 5;
	int num_target_layers = 64;
	int target_layer_ids[8] = { 5, 19, 33, 47, 61, 0, 0, 0 };
	int num_target_layer_ids = 5;
	int mask_token_id = 248070;
	float rms_norm_eps = 1e-6f;
	float conf_threshold = 0.25f;
};

} // namespace pod

namespace engine::spec {

// Mathematical kernels:

// Pure CUDA grouped dynamic causal convolution:
// out[b, t, c] = sum_k (base[k, c] + dyn[b, t, k, c / group_size]) * x[b, t - k, c]
void dflash2_dynconv(
	const pod::TensorView& x,
	const pod::TensorView& dyn,
	const pod::TensorView& base,
	pod::TensorView& out,
	int group_size,
	bool accumulate,
	cudaStream_t stream = nullptr
);

// Pure CUDA candidate selector walk (greedy chained walk)
void dflash2_selector_walk(
	const float* unary,       // [bsz, rows, k]
	const int* cands,          // [bsz, rows, k]
	const half* gate,          // [bsz, rows, rank]
	const half* pred_cb,       // [vocab, rank]
	const half* succ_cb,       // [vocab, rank]
	const int* anchor,         // [bsz]
	int* out_tokens,           // [bsz, rows + 1] (out[0] = anchor, out[1..rows] = draft tokens)
	float* conf,               // [bsz, rows + 1] or nullptr
	int bsz,
	int rows,
	int k,
	int rank,
	cudaStream_t stream = nullptr
);

// Pure CUDA Top-K extraction from logits
void dflash2_topk(
	const half* logits,
	int bsz,
	int rows,
	int vocab,
	int k,
	float* values,
	int* indices,
	float scale = 1.0f,
	float softcap = 0.0f,
	cudaStream_t stream = nullptr
);

// In-stream tree-mask construction on cudaStream_t (zero host synchronization)
void construct_tree_mask(
	half* mask_matrix,         // [M, M]
	int M,
	cudaStream_t stream = nullptr
);

// GPU-side greedy verify kernel with logits:
// Compares target M=8 logits argmax against draft tokens to compute accepted_count (A in [1, 8])
// and emits next_token on stream.
void verify_greedy(
	const half* target_logits, // [M, vocab]
	const int* draft_tokens,   // [M] (anchor at 0, draft tokens at 1..M-1)
	int* d_accepted_count,     // [1] device scalar
	int* d_next_token,         // [1] device scalar
	int* h_accepted_count,     // [1] pinned host scalar
	int* h_next_token,         // [1] optional pinned host scalar
	int M,
	int vocab_size,
	cudaStream_t stream = nullptr
);

// Fast GPU-side greedy verify with precomputed target argmax
void verify_greedy_with_argmax(
	const int* target_argmax,  // [M]
	const int* draft_tokens,   // [M]
	int* d_accepted_count,     // [1] device scalar
	int* d_next_token,         // [1] device scalar
	int* h_accepted_count,     // [1] pinned host scalar
	int* h_next_token,         // [1] optional pinned host scalar
	int M,
	cudaStream_t stream = nullptr
);

struct SpeculativeResult {
	int accepted_count = 1; // A in [1, 8]
	int next_token = 0;
	float host_dispatch_us = 0.0f;
};

class DFlashSpeculator {
public:
	DFlashSpeculator();
	~DFlashSpeculator();

	bool init(
		const pod::DFlashConfig& config,
		int device,
		pod::MemoryArena& arena,
		cudaStream_t stream = nullptr
	);

	bool load_weights( const char* model_dir, cudaStream_t stream = nullptr );

	// Multi-token block drafting: produces K=7 draft tokens from anchor token
	void draft(
		int anchor_token,
		int current_pos,
		cudaStream_t stream = nullptr
	);

	// Construct in-stream tree mask
	void prepare_verify_mask( cudaStream_t stream = nullptr );

	// Verify target M=8 logits
	SpeculativeResult verify(
		const half* target_logits, // [8, vocab]
		cudaStream_t stream = nullptr
	);

	// Verify with target argmax
	SpeculativeResult verify_with_argmax(
		const int* target_argmax, // [8]
		cudaStream_t stream = nullptr
	);

	// In-place KV-cache rollback/advance helper
	void advance_pos( int& current_pos, int accepted_count ) {
		current_pos += accepted_count;
	}

	// Buffer accessors
	int* d_draft_tokens() { return d_draft_tokens_; }
	const int* d_draft_tokens() const { return d_draft_tokens_; }
	int* d_accepted_count() { return d_accepted_count_; }
	int* d_next_token() { return d_next_token_; }
	half* d_tree_mask() { return d_tree_mask_; }

	const pod::DFlashConfig& config() const { return config_; }

	void destroy();

private:
	pod::DFlashConfig config_;
	int device_;
	pod::MemoryArena* arena_;
	bool initialized_;
	bool weights_loaded_;

	// Weights loaded via SafeTensorsRepo
	pod::TensorView pred_codebook_;
	pod::TensorView succ_codebook_;
	pod::LinearWeight hidden_proj_;
	pod::TensorView base_attn_[5];
	pod::TensorView base_mlp_[5];
	pod::LinearWeight proj_attn_[5];
	pod::LinearWeight proj_mlp_[5];
	pod::TensorView attn_norms_[5];
	pod::TensorView mlp_norms_[5];
	pod::TensorView final_norm_;

	// Pointers allocated once at boot in static arena
	int* d_draft_tokens_;       // [8]: anchor at [0], 7 draft tokens at [1..7]
	float* d_conf_;             // [8]
	half* d_tree_mask_;         // [8, 8]
	int* d_accepted_count_;     // [1]
	int* d_next_token_;         // [1]
	int* d_target_argmax_;      // [8]
	int* h_accepted_count_;     // [1] pinned host
	int* h_next_token_;         // [1] pinned host

	// Scratch buffers for drafting
	pod::TensorView draft_hidden_;  // [8, hidden]
	pod::TensorView draft_dyn_;     // [1, 8, 2, groups]
	float* d_topk_values_;          // [7, 16]
	int* d_topk_indices_;           // [7, 16]
	float* d_topk_part_v_;          // [7 * 32 * 16]
	int* d_topk_part_ix_;           // [7 * 32 * 16]
	int* d_topk_counters_;          // [7]
	half* d_gate_;                  // [7, 256]
};

} // namespace engine::spec
