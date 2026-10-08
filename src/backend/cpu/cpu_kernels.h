#pragma once

// Slang host-target kernels (build/slang/<entry>.cpp) as called from the CPU backend. The Params structs mirror
// src/shaders/*.slang (tests/2_parity/test_slang_cpu.cpp keeps its own copy). Kernels run serially: one call per launch.
#include <slang-cpp-prelude.h>
#include <cstdint>

extern "C" {
void rms_norm( ComputeVaryingInput*, void*, void* );
void rms_norm_heads( ComputeVaryingInput*, void*, void* );
void add_residual( ComputeVaryingInput*, void*, void* );
void swiglu( ComputeVaryingInput*, void*, void* );
void swiglu_fused( ComputeVaryingInput*, void*, void* );
void embed_tokens( ComputeVaryingInput*, void*, void* );
void rope( ComputeVaryingInput*, void*, void* );
void moe_sum_planes( ComputeVaryingInput*, void*, void* );
void moe_accumulate( ComputeVaryingInput*, void*, void* );
void batched_argmax( ComputeVaryingInput*, void*, void* );
void router_select( ComputeVaryingInput*, void*, void* );
void l2_norm_heads( ComputeVaryingInput*, void*, void* );
void gated_rmsnorm( ComputeVaryingInput*, void*, void* );
void project_gates( ComputeVaryingInput*, void*, void* );
void causal_conv1d_silu_rows( ComputeVaryingInput*, void*, void* );
void causal_conv1d_silu_tiled( ComputeVaryingInput*, void*, void* );
void causal_conv1d_history( ComputeVaryingInput*, void*, void* );
void recurrent_step_rows( ComputeVaryingInput*, void*, void* );
void kv_store_f16( ComputeVaryingInput*, void*, void* );
void init_streams( ComputeVaryingInput*, void*, void* );
void hc_apply( ComputeVaryingInput*, void*, void* );
void hc_mix_partials( ComputeVaryingInput*, void*, void* );
void hc_mix_finalize( ComputeVaryingInput*, void*, void* );
void hyperhead_mean( ComputeVaryingInput*, void*, void* );
void dsa_prep( ComputeVaryingInput*, void*, void* );
void dsa_pool_update( ComputeVaryingInput*, void*, void* );
void dflash_store_taps( ComputeVaryingInput*, void*, void* );
void reduce_streams( ComputeVaryingInput*, void*, void* );
}

struct NormParams { half* src; half* dst; half* weight; int hidden_size; int has_weight; float eps; float bias; };
struct HeadNormParams { half* qk; half* weight; int num_heads; int head_dim; int has_weight; float eps; float bias; };
struct AddParams { half* hidden; half* residual; int n; };
struct SwigluParams { half* gate; half* up; half* out_; int n; float limit; };
struct SwigluFusedParams { half* gate_up; half* out_; int inter_dim; int total; float limit; };
struct EmbedParams { uint16_t* table; int* token_ids; half* out_; int hidden_size; };
struct RopeParams { half* q; half* k; int* pos; int num_q_heads; int num_kv_heads; int head_dim; int rot_dim; int interleaved; float freq_base; };

struct SumPlanesParams { half* planes; half* out_; int n; int groups; };
struct AccumParams { half* accum; half* expert_out; float weight; int n; };
struct ArgmaxParams { half* logits; int* out_idx; float* out_val; int numel; int batch_size; int index_offset; int row_stride; int has_val; };
struct RouterParams { float* partials; half* bias; int* topk_idx; half* topk_w; int num_experts; int top_k; int k_slices; int max_slices; float scale; int has_bias; };

struct L2Params { half* qkv; int num_k_heads; int head_dim; int scale_q; int64_t row_stride; };
struct GatesParams { half* norm_hidden; half* in_proj_a; half* in_proj_b; half* A_log; half* dt_bias; float* gate_out; float* beta_out; int num_v_heads; int hidden_size; int has_a_log; int has_dt_bias; };
struct ConvRowsParams { half* qkv; float* history; half* weights; int* slots; float* snap; int* snap_on; int channels; int rows; int snap_rows; int has_weights; int64_t slot_stride; int64_t snap_stride; };
struct ConvTiledParams { half* src; half* dst; float* history; half* weights; int* slots; int channels; int rows; int tile_rows; int has_weights; int64_t slot_stride; };
struct RecurrentParams { float* state; half* qkv; float* gate; float* beta; half* output; int* slots; float* snap; int* snap_on; int h_k; int h_v; int rows; int snap_rows; float scale; int64_t slot_stride; int64_t snap_stride; };
struct KvStoreParams { half* data; half* src; int* slots; int* positions; int64_t slot_stride; int layer_idx; int count; int total_elems; int max_seq; };
struct InitStreamsParams { half* in_; float* streams; int hidden_size; };
struct ApplyParams { float* x_in; float* x_out; half* y; float* post; float* comb; int hidden_size; int has_comb; };
struct PartialsParams { float* streams; float* fn; float* partials; int row_len; int chunk_cols; int num_chunks; };
struct FinalizeParams { float* streams; float* partials; float* base; float* scale; float* post; float* comb; half* collapsed; float* inv_rms_buf; float* mix_logits_buf; float* pre_buf; int hidden_size; int num_chunks; int chunk_cols_c; int sinkhorn_iters; float rms_eps; float hc_eps; };
struct HeadMeanParams { float* streams; half* final_hidden; int hidden_size; };
struct PrepParams { half* proj; half* ln_w; half* ln_b; half* plane; half* w_out; int* slots; int* positions; int layer; int64_t plane_slot_stride; };
struct PoolParams { half* plane; half* pool_plane; half* ape; int* slots; int* positions; int layer; int max_seq; int64_t plane_slot_stride; int64_t pool_slot_stride; };
struct TapStoreParams { half* taps; half* hidden; int* slots; int* positions; int* owner; int m; int tap; int H; int cap; };
struct ReduceStreamsParams { float* streams; half* out_; int H; int mode; };
struct GatedNormParams { half* attn_out; half* z; half* norm_weight; int num_v_heads; int head_dim; int use_sigmoid; int has_z; int has_weight; };

namespace impl {

// Runs a kernel over a gx * gy * gz grid of thread groups
inline void cpu_dispatch( void ( *fn )( ComputeVaryingInput*, void*, void* ), void* params, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1 ) {
	if ( gx == 0 || gy == 0 || gz == 0 ) return;
	ComputeVaryingInput vi = {};
	vi.startGroupID = { 0, 0, 0 };
	vi.endGroupID = { gx, gy, gz };
	fn( &vi, params, nullptr );
}

} // namespace impl
