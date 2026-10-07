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
	// Token the drafter is fed in query slots it has no prediction for yet. The reference documents
	// this directly: "Masked query slots are fed to the draft as mask_token_id".
	int mask_token_id = 248070;

	int selector_rank = 256;
	int selector_top_k = 16;
	int hidden_size = 5120;
	int vocab_size = 248320;
	int num_hidden_layers = 5;
	int num_target_layers = 64;
	int target_layer_ids[8] = { 5, 19, 33, 47, 61, 0, 0, 0 };
	int num_target_layer_ids = 5;
	float rms_norm_eps = 1e-6f;
	float conf_threshold = 0.25f;

	// Attention geometry (Qwen3.8-27B drafter, from the checkpoint's own tensor shapes).
	int num_q_heads = 32;
	int num_kv_heads = 8;
	int head_dim = 128;
	int intermediate_size = 17408;
	int sliding_window = 2048;
	bool causal_in_block = false;   // dflash.attention.causal = false
	float rope_theta = 1.0e7f;
	// NeoX style. The drafter is full rotary: its checkpoint carries rope_theta with rope_type
	// "default" and no partial_rotary_factor or dimension_sections, so the reference's get_rope() runs
	// rotary_dim == head_dim, and its SM70 fast path is gated on exactly that. The previous {64,0,0,0}
	// default rotated only half of each head and silently made every draft wrong.
	int rope_dim_sections[4] = { 128, 0, 0, 0 };

	int conv_groups() const { return hidden_size / conv_group_size; }
	int conv_proj_out() const { return 2 * conv_kernel_size * conv_groups(); }
};

} // namespace pod

namespace engine::spec {

// One drafter decoder layer. Under TP the residual stream stays full width and replicated; only the
// attention heads and the MLP intermediate are split, so q/k/v/gate/up are column-parallel and
// o_proj/down_proj are row-parallel with an all-reduce on their full-width output.
struct DFlashLayer {
	pod::TensorView input_norm;      // [hidden]
	pod::TensorView post_attn_norm;  // [hidden]

	pod::LinearWeight q_proj;        // [hidden] -> q_heads * head_dim (sharded)
	pod::LinearWeight k_proj;        // [hidden] -> kv_heads * head_dim (sharded)
	pod::LinearWeight v_proj;        // [hidden] -> kv_heads * head_dim (sharded)
	pod::LinearWeight o_proj;        // q_heads * head_dim -> hidden (row-sharded)
	pod::TensorView q_norm;           // [head_dim]
	pod::TensorView k_norm;           // [head_dim]

	pod::LinearWeight gate_proj;     // [hidden] -> intermediate (sharded)
	pod::LinearWeight up_proj;       // [hidden] -> intermediate (sharded)
	pod::LinearWeight down_proj;     // intermediate -> hidden (row-sharded)

	// Sandwiched grouped dynamic convs. base_kernel is a static per-tap per-channel bias; the
	// projection emits per-token per-group coefficients for both sides of the sandwich.
	pod::TensorView attn_conv_base;   // [2, taps, hidden]
	pod::TensorView mlp_conv_base;    // [2, taps, hidden]
	pod::TensorView attn_conv_proj;   // [conv_proj_out, hidden] plain FP16
	pod::TensorView mlp_conv_proj;    // [conv_proj_out, hidden] plain FP16
};


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

// FP16 -> FP32 widening copy (the drafter widens o_proj / down_proj before their conv).
void dflash_h2f( const half* in, float* out, int n, cudaStream_t stream = nullptr );

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

// Per-row top-k over a *vocabulary shard*: scans the first `columns` entries of each row, whose
// stride is `row_stride` (the padded shard width), and adds `index_offset` so the winner is a global
// token id. The scratch is caller-owned, so this is safe inside a CUDA graph and per device, unlike
// dflash2_topk's process-global lazily-allocated scratch.
void dflash2_topk_shard(
	const half* logits,
	int rows,
	int row_stride,
	int columns,
	int k,
	float* values,
	int* indices,
	int index_offset,
	float* part_v,
	int* part_ix,
	int* counters,
	cudaStream_t stream = nullptr
);

// Merge the per-rank local top-k into the global top-k, per row. part_vals/part_ix are
// [tp_size, rows, k]; out_vals/out_ix are [rows, k]. Exact, not a heuristic: a rank's local top-k is
// a superset of the global top-k restricted to that shard, so the union contains the answer.
void dflash2_topk_merge(
	const float* part_vals,
	const int* part_ix,
	int tp_size,
	int rows,
	int k,
	float* out_vals,
	int* out_ix,
	int part_rows = 16,
	cudaStream_t stream = nullptr
);

// Plain FP16 GEMV, out[m, n] = sum_k w[n, k] * x[m, k], for up to 8 rows. The drafter's conv kernel
// projections and the selector's hidden projection are plain FP16 in the checkpoint, not trellis.
void dflash_f16_gemv_nt(
	const half* w,
	const half* x,
	half* out,
	int m,
	int n,
	int k,
	cudaStream_t stream = nullptr
);

// RMSNorm with the residual split the reference uses: sum = x + residual_in, normed = rmsnorm(sum) *
// (weight + constant_bias), residual_out = sum. residual_out is FP32.
void dflash_rmsnorm_residual(
	const pod::TensorView& x,
	const pod::TensorView& residual_in,
	const pod::TensorView& weight,
	pod::TensorView& normed_out,
	pod::TensorView& residual_out,
	int rows,
	float eps,
	float constant_bias,
	cudaStream_t stream = nullptr
);

// Per-head RMSNorm then NeoX RoPE, in place, on the drafter's Q or K tensor. `pairs` is
// head_dim*rotary_fraction/2; dims at or past 2*pairs pass through unrotated.
void dflash_qk_norm_rope(
	half* x,
	const half* weight,
	const float* rope_cos,
	const float* rope_sin,
	int rows,
	int heads,
	int head_dim,
	int pairs,
	int pos0,
	float eps,
	float constant_bias,
	cudaStream_t stream = nullptr
);

// GQA small-query attention for the drafter block: `rows` queries against `nctx` context slots
// starting at `first_ctx`, plus the block's own `rows` K/V starting at cache slot `own_slot`.
// Non-causal within the block; the caller windows the context via first_ctx/nctx.
void dflash_attn(
	const half* q,
	const half* kcache,
	const half* vcache,
	half* out,
	int rows,
	int q_heads,
	int kv_heads,
	int head_dim,
	int nctx,
	int first_ctx,
	int ring_slots,
	int ctx_layer,
	int stride_slot,
	int stride_layer,
	int stride_head,
	int own_slot,
	// First block row that contributes a key. 1 leaves the anchor to the context (its K/V comes from
	// the target's taps); 0 lets row 0 supply the anchor itself. See DFlashSpeculator::draft_backbone.
	int own_first_row = 1,
	cudaStream_t stream = nullptr
);

// Take one side out of a conv kernel projection's [rows, 2*taps*groups] output and reinterpret it as
// the [rows, taps, groups] the grouped conv indexes. Side 0 is the sandwich's entry side, side 1 its
// exit side; the projection emits both contiguously.
void dflash_conv_side_split(
	float* side,
	const half* proj,
	int rows,
	int taps,
	int groups,
	int side_idx,
	cudaStream_t stream = nullptr
);

// Scatter a [n, heads*head_dim] projection into one layer of the ring-shaped context cache.
void dflash_cache_store(
	half* cache,
	const half* src,
	int n,
	int heads,
	int head_dim,
	int slot0,
	int ring_slots,
	int layer,
	int stride_slot,
	int stride_layer,
	cudaStream_t stream = nullptr
);

// NeoX RoPE tables for [len, head_dim/2] cos/sin, with the rotary width taken from the sum of the
// non-zero dimension sections.
void build_rope_table(
	float* cos_table,
	float* sin_table,
	int len,
	int head_dim,
	float theta,
	const int dim_sections[4],
	cudaStream_t stream = nullptr
);

// Columns one dflash2_topk block-tile covers (TOPK_THREADS * 16); sizes the split-partial scratch.
static constexpr int kDflashTopkTile = 4096;

class DFlashSpeculator {
public:
	DFlashSpeculator();

	// Number of target LM-head ranks that contribute a candidate slice to the merge. The drafter's own
	// tp_size is 1 while the head is sharded, so the merge staging cannot be sized from tp_size_.
	void set_candidate_ranks( int ranks ) { cand_ranks_ = MAX( ranks, 1 ); }
	~DFlashSpeculator();

	bool init(
		const pod::DFlashConfig& config,
		int device,
		pod::MemoryArena& arena,
		cudaStream_t stream = nullptr,
		int rank = 0,
		int tp_size = 1
	);

	bool load_weights( const char* model_dir, cudaStream_t stream = nullptr );

	// ---- The real draft round, split so the per-row candidate top-k can cross the interconnect ----
	//
	// The drafter's hidden states are produced by the backbone, and each row's candidate set is the
	// top-k of that row's hidden state through the *target's* vocabulary-parallel LM head. So the
	// candidate top-k cannot happen before the backbone, and needs every rank's head shard. Hence:
	//
	//   draft_backbone()  -> drafter hidden [rows, hidden]
	//   (caller)          -> per-rank LM head + shard top-k, peer-copy back, merge
	//   draft_select()    -> selector gate projection + bilinear lattice walk -> tokens + conf
	//
	// The runner (DFlash2Runner) drives these three steps.

	// Runs the backbone for one draft block [anchor, mask x (block_size - 1)] and leaves its output in
	// draft_hidden(). `tapped_states` is the target's position-indexed tap ring (capacity taps_cap);
	// only the positions not yet ingested are projected into the drafter's context KV.
	void draft_backbone(
		const pod::TensorView& embed_table,
		int anchor_token,
		const half* tapped_states,
		int block_pos0,
		int taps_cap,
		cudaStream_t stream = nullptr
	);

	// Selector gate = the candidate selector's hidden projection of the drafter's output, for the first
	// `rows` rows. Written to d_gate_ as [rows, selector_rank].
	void project_gate( int rows, cudaStream_t stream = nullptr );

	// Seeds the walk with the anchor token (its step-0 predecessor).
	void set_walk_anchor( int anchor, cudaStream_t stream = nullptr );

	// Drop the projected context (one slot's worth: the ring fill level, KV length and ingested count
	// are single counters). Called when a new slot claims the drafter.
	void reset_context() {
		ctx_ingested_ = 0;
		kv_len_ = 0;
		kv_pos0_ = 0;
	}

	void draft_select(
		const float* vals,
		const int* ix,
		int rows,
		cudaStream_t stream = nullptr
	);

	const pod::TensorView& draft_hidden() const { return draft_hidden_; }

	// Merge staging for the distributed per-row top-k. Rank d's [rows, k] local top-k lands at
	// base + d * rows * k, and the merge reduces those tp_size * k candidates to the global top-k.
	float* merged_cand_values_ptr( int d ) { return merge_vals_.ptr<float>() + static_cast<size_t>( d ) * merge_stride(); }
	int* merged_cand_indices_ptr( int d ) { return merge_ix_.ptr<int>() + static_cast<size_t>( d ) * merge_stride(); }
	float* merged_final_values_ptr() { return merge_vals_.ptr<float>(); }
	int* merged_final_indices_ptr() { return merge_ix_.ptr<int>(); }
	int merge_stride() const { return config_.block_size * config_.selector_top_k; }
	int merge_rows() const { return config_.block_size; }

	// One draft round of the drafter backbone.
	//
	// Stage 1, context: the `n_ctx` most recent target positions arrive as their 5 tapped hidden
	// states, are compressed by fc (5*hidden -> hidden) and hidden_norm, then projected by every
	// layer's own k_proj/v_proj into the drafter's ring-shaped context KV cache. The drafter therefore
	// attends to the real target context instead of re-reading the target's KV cache.
	//
	// Stage 2, block: embed the `rows` input ids (row 0 the anchor, the rest the mask token) with the
	// *target's* embedding table, then run num_hidden_layers x (RMSNorm with residual split, grouped
	// dynamic conv, GQA attention, dynamic conv, RMSNorm with residual split, dynamic conv, SwiGLU
	// MLP, dynamic conv), then the final norm. Writes [rows, hidden].
	//
	// `tapped_states` is [n_ctx, 5*hidden] FP16 and `d_input_ids` is [rows] int, both device pointers.
	// Positions are absolute; `ctx_pos0` is the absolute position of the first context row and
	// `block_pos0` that of block row 0.
	// Parity tests install dense weights and read internals through DFlashTestAccess (below); the
	// engine never does.
	friend struct DFlashTestAccess;

	void forward_backbone(
		const pod::TensorView& embed_table,
		const int* d_input_ids,
		int rows,
		const half* tapped_states,
		int n_ctx,
		int ctx_pos0,
		int block_pos0,
		pod::TensorView& hidden_out,
		cudaStream_t stream = nullptr,
		int taps_ring_cap = 0
	);

	// Walk output: [anchor, d1..d7] tokens and per-step winning-candidate mass
	int* d_draft_tokens() { return d_draft_tokens_; }
	const int* d_draft_tokens() const { return d_draft_tokens_; }
	float* d_conf() { return d_conf_; }
	const float* d_conf() const { return d_conf_; }

	const pod::DFlashConfig& config() const { return config_; }

	void destroy();

private:
	pod::DFlashConfig config_;
	int device_;
	pod::MemoryArena* arena_;
	bool initialized_;
	bool weights_loaded_;
	int rank_ = 0;
	int tp_size_ = 1;

	// Weights loaded via SafeTensorsRepo
	pod::LinearWeight fc_;
	pod::TensorView fc_norm_;
	pod::TensorView pred_codebook_;
	pod::TensorView succ_codebook_;
	pod::LinearWeight hidden_proj_;
	pod::TensorView hidden_proj_w_;    // [rank, hidden] plain FP16, selector gate projection
	DFlashLayer layers_[8];        // num_hidden_layers of them are valid
	pod::TensorView hidden_norm_;
	pod::TensorView final_norm_;
	pod::TensorView embed_for_test_{};   // test-only, see DFlashTestAccess

	// Drafter KV cache, written from the fc projection of the tapped target states before the
	// backbone runs, exactly as the reference implementations do: the drafter attends to the real
	// target context rather than re-reading the target's KV cache. Context occupies a ring of
	// ring_len_ slots; the block's own transient K/V occupy the reserved tail from own_slot_.
	pod::TensorView drafter_k_;     // [ring_len + block_size, num_layers, kv_heads_local, head_dim]
	pod::TensorView drafter_v_;     // [ring_len + block_size, num_layers, kv_heads_local, head_dim]

	// Pointers allocated once at boot in static arena
	int cand_ranks_ = 1;         // contributing target-head ranks for the candidate merge

	int* d_draft_tokens_;       // [8]: anchor at [0], 7 draft tokens at [1..7]
	int* d_anchor_ = nullptr;    // [1]: walk seed, distinct from the walk's own output
	float* d_conf_;             // [8]
	int* d_input_ids_;          // [8] backbone input ids (row 0 = anchor, rest = mask)

	// Scratch buffers for drafting
	pod::TensorView draft_projected_; // [1, hidden]
	pod::TensorView draft_hidden_;  // [M, hidden] drafter output hidden states for the current block
	pod::TensorView merge_vals_;    // [tp_size * rows * k] float
	pod::TensorView merge_ix_;     // [tp_size * rows * k] int
	pod::TensorView draft_dyn_;     // [1, 8, 2, groups]
	float* d_topk_values_;          // [7, 16]
	int* d_topk_indices_;           // [7, 16]
	float* d_topk_part_v_;          // [7 * 32 * 16]
	int* d_topk_part_ix_;           // [7 * 32 * 16]
	int* d_topk_counters_;          // [7]
	half* d_gate_;                  // [7, 256]
	// Backbone activation scratch. Sized for block_size rows and this rank's shard widths; the
	// residual stream is always full width and replicated.
	pod::TensorView bb_residual_;   // [8, hidden] FP32
	pod::TensorView bb_normed_;     // [8, hidden] FP16
	pod::TensorView bb_stage_;      // [8, hidden] FP16 conv / attention staging

	pod::TensorView bb_stage_h_;     // [8, hidden] FP16 staging for the two convs fed by a normalized hidden
	pod::TensorView bb_stage_f32_;  // [8, hidden] FP32 carrier for conv outputs that can exceed FP16 range
	pod::TensorView bb_attn_out_;   // [8, hidden] FP16
	pod::TensorView bb_hidden_;     // [8, hidden] FP16 final normed hidden
	pod::TensorView bb_o_proj_;     // [8, hidden] FP16 o_proj output, before any all-reduce
	pod::TensorView bb_q_, bb_k_, bb_v_;   // [8, heads_local, head_dim]
	pod::TensorView bb_kv_stage_;   // [8, kv_heads_local * head_dim] before the cache scatter
	pod::TensorView bb_ffn_gate_, bb_ffn_up_, bb_ffn_mlp_;  // [8, intermediate_local]
	pod::TensorView bb_dyn_a_, bb_dyn_m_;     // [8, conv_proj_out] FP16, before the side split
	pod::TensorView bb_coef_a0_, bb_coef_a1_;  // [8, taps, groups] FP32, attention-conv sides
	pod::TensorView bb_coef_m0_, bb_coef_m1_;  // [8, taps, groups] FP32, mlp-conv sides
	pod::TensorView bb_ctx_fc_;     // [CTX_CHUNK, 5*hidden] FP16 context taps
	pod::TensorView bb_ctx_normed_; // [CTX_CHUNK, hidden] FP16
	pod::TensorView bb_ctx_resid_;  // [CTX_CHUNK, hidden] FP32, discarded
	// rope tables for the drafter's own positions
	pod::TensorView d_rope_cos_, d_rope_sin_;   // [max_pos, head_dim/2]
	int rope_len_ = 0;
	// The projected context is a ring of ring_len_ slots; kv_pos0_ is the absolute position of
	// logical slot 0. The block's own transient K/V live in the reserved tail starting at own_slot_.
	int ring_len_ = 0;
	int kv_len_ = 0;
	int kv_pos0_ = 0;
	int ctx_ingested_ = 0;       // target positions already projected into the drafter's KV ring
	int own_slot_ = 0;

	// First block row that contributes an attention key. 1 leaves the anchor's position to the
	// context (its K/V comes from the target's taps); 0 lets row 0 supply the anchor from the
	// drafter's own embedding. Chosen per round in draft_backbone; see EXL3_DFLASH2_ANCHORKV.
	int own_first_row_ = 1;
	int ctx_chunk_ = 0;
};

// Test-only access: installs dense FP16 layer weights, the context stage's `fc` / `hidden_norm`
// and the embedding table directly (bypassing the checkpoint) so a parity test can build a small
// backbone it can also evaluate on the CPU, and exposes the block ids and KV cache for checks.
struct DFlashTestAccess {
	static void set_dense_weights( DFlashSpeculator& s, const DFlashLayer* layers, const pod::TensorView& final_norm ) {
		for ( int l = 0; l < s.config_.num_hidden_layers; ++l ) s.layers_[l] = layers[l];
		s.final_norm_ = final_norm;
		s.weights_loaded_ = true;
	}
	static void set_context_weights( DFlashSpeculator& s, const pod::LinearWeight& fc, const pod::TensorView& fc_norm ) {
		s.fc_ = fc;
		s.fc_norm_ = fc_norm;
	}
	static int* input_ids( DFlashSpeculator& s ) { return s.d_input_ids_; }
	static void set_embed( DFlashSpeculator& s, const pod::TensorView& t ) { s.embed_for_test_ = t; }
	static const pod::TensorView& embed( const DFlashSpeculator& s ) { return s.embed_for_test_; }
	static void* cache_k( DFlashSpeculator& s ) { return s.drafter_k_.data; }
	static void* cache_v( DFlashSpeculator& s ) { return s.drafter_v_.data; }
};

} // namespace engine::spec
