#pragma once
#include <cuda_runtime_api.h>

// Host entry points of the Slang-generated CUDA kernels (src/kernels/slang/*.slang). Pointers are device pointers to FP16 data
// (BF16 bits for the embedding table); each call is one launch on the stream. Used by the parity test and as the reference
// for new backends; the CUDA build's own kernels remain the default.
void slang_rms_norm( void* src, void* dst, void* weight, int rows, int hidden, float eps, float bias, cudaStream_t s );
void slang_rms_norm_heads( void* qk, void* weight, int rows, int heads, int head_dim, float eps, float bias, cudaStream_t s );
void slang_add_residual( void* hidden, void* residual, int n, cudaStream_t s );
void slang_swiglu( void* gate, void* up, void* out, int n, float limit, cudaStream_t s );
void slang_swiglu_fused( void* gate_up, void* out, int inter_dim, int total, float limit, cudaStream_t s );
void slang_embed_tokens( void* table_bf16, void* token_ids, void* out, int hidden, int rows, cudaStream_t s );
void slang_rope( void* q, void* k, void* pos, int rows, int q_heads, int kv_heads, int head_dim, int rot_dim, int interleaved, float base, cudaStream_t s );
void slang_moe_sum_planes( void* planes, void* out, int n, int groups, cudaStream_t s );
void slang_moe_accumulate( void* accum, void* expert_out, float weight, int n, cudaStream_t s );
void slang_batched_argmax( void* logits, void* out_idx, void* out_val, int numel, int batch, int index_offset, int row_stride, cudaStream_t s );
void slang_router_select( void* partials, void* bias, void* topk_idx, void* topk_w, int rows, int num_experts, int top_k, int k_slices, int max_slices, float scale, cudaStream_t s );
void slang_l2_norm_heads( void* qkv, int rows, int num_k_heads, int head_dim, int scale_q, long long row_stride, cudaStream_t s );
void slang_gated_rmsnorm( void* attn_out, void* z, void* norm_weight, int rows, int num_v_heads, int head_dim, int use_sigmoid, cudaStream_t s );
void slang_project_gates( void* norm_hidden, void* in_proj_a, void* in_proj_b, void* A_log, void* dt_bias, void* gate_out, void* beta_out, int rows, int num_v_heads, int hidden_size, cudaStream_t s );
void slang_causal_conv1d_silu_rows( void* qkv, void* history, void* weights, void* slots, void* snap, void* snap_on, int channels, int rows, int snap_rows, long long slot_stride, long long snap_stride, cudaStream_t s );
void slang_causal_conv1d_silu_tiled( void* src, void* dst, void* history, void* weights, void* slots, int channels, int rows, int tile_rows, long long slot_stride, cudaStream_t s );
void slang_causal_conv1d_history( void* src, void* history, void* slots, int channels, int rows, long long slot_stride, cudaStream_t s );
void slang_recurrent_step_rows( void* state, void* qkv, void* gate, void* beta, void* output, void* slots, void* snap, void* snap_on, int h_k, int h_v, int rows, int snap_rows, long long slot_stride, long long snap_stride, cudaStream_t s );
void slang_kv_store_f16( void* data, void* src, void* slots, void* positions, long long slot_stride, int layer_idx, int count, int total_elems, int max_seq, cudaStream_t s );
void slang_init_streams( void* in, void* streams, int hidden_size, int rows, cudaStream_t s );
void slang_hc_apply( void* x_in, void* x_out, void* y, void* post, void* comb, int hidden_size, int rows, cudaStream_t s );
void slang_hc_mix_partials( void* streams, void* fn, void* partials, int row_len, int chunk_cols, int rows, cudaStream_t s );
void slang_hc_mix_finalize( void* streams, void* partials, void* base, void* scale, void* post, void* comb, void* collapsed, void* inv_rms_buf, void* mix_logits_buf, void* pre_buf,
	int hidden_size, int num_chunks, int chunk_cols_c, int sinkhorn_iters, float rms_eps, float hc_eps, int rows, cudaStream_t s );
void slang_hyperhead_mean( void* streams, void* final_hidden, int hidden_size, int rows, cudaStream_t s );
void slang_dsa_prep( void* proj, void* ln_w, void* ln_b, void* plane, void* w_out, void* slots, void* positions, int layer, long long plane_slot_stride, int rows, cudaStream_t s );
void slang_dsa_pool_update( void* plane, void* pool_plane, void* ape, void* slots, void* positions, int layer, int max_seq, long long plane_slot_stride, long long pool_slot_stride, int rows, cudaStream_t s );
void slang_dflash_store_taps( void* taps, void* hidden, void* slots, void* positions, void* owner, int m, int tap, int H, int cap, cudaStream_t s );
void slang_reduce_streams( void* streams, void* out, int m, int H, int mode, cudaStream_t s );
