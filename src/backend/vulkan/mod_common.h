#pragma once

// Vulkan module layer: SPIR-V kernel cache and the push-constant blocks of the Slang kernels (src/kernels/slang/*.slang).
// A `uniform Params` is one push-constant block in natural C layout (8-byte device pointers, 4-byte scalars, int64 aligned to 8),
// so the structs below mirror the Slang structs field by field with every pointer as a uint64_t device address.
#include "engine/backend/vulkan/device.h"
#include "engine/config.h"
#include "engine/core/batch.h"
#include <cstddef>
#include <cstdint>

namespace engine::backend::impl {

inline uint64_t dp( const void* p ) { return reinterpret_cast<uint64_t>( p ); }

// BatchRows lives in device memory on this backend: address of its slot[] / pos[] arrays
inline uint64_t rows_slot( const pod::BatchRows* r ) { return dp( r ) + offsetof( pod::BatchRows, slot ); }
inline uint64_t rows_pos( const pod::BatchRows* r ) { return dp( r ) + offsetof( pod::BatchRows, pos ); }

// Pipeline of <spv dir>/<name>.spv (env ENGINE_SPV_DIR overrides the build directory), created once. Aborts when missing
vk::Pipeline* kernel( const char* name, uint32_t push_bytes );

// Device scratch buffer of at least `bytes` for slot `slot`; grows by reallocation (syncs `stream` first: recorded commands may reference the old buffer)
void* scratch( int slot, size_t bytes, stream_t stream );

template<typename P>
inline void launch( const char* name, const P& p, stream_t s, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1 ) {
	static_assert( sizeof( P ) <= 256, "push-constant block exceeds the 256-byte limit" );
	if ( gx == 0 || gy == 0 || gz == 0 ) return;
	vk::dispatch( s, kernel( name, sizeof( P ) ), &p, gx, gy, gz );
}

struct NormParams { uint64_t src, dst, weight; int hidden_size, has_weight; float eps, bias; };
struct HeadNormParams { uint64_t qk, weight; int num_heads, head_dim, has_weight; float eps, bias; };
struct AddParams { uint64_t hidden, residual; int n; };
struct SwigluParams { uint64_t gate, up, out_; int n; float limit; };
struct SwigluFusedParams { uint64_t gate_up, out_; int inter_dim, total; float limit; };
struct EmbedParams { uint64_t table, token_ids, out_; int hidden_size; };
struct RopeParams { uint64_t q, k, pos; int num_q_heads, num_kv_heads, head_dim, rot_dim, interleaved; float freq_base; };
struct ArgmaxParams { uint64_t logits, out_idx, out_val; int numel, batch_size, index_offset, row_stride, has_val; };

struct L2Params { uint64_t qkv; int num_k_heads, head_dim, scale_q; int64_t row_stride; };
struct GatesParams { uint64_t norm_hidden, in_proj_a, in_proj_b, A_log, dt_bias, gate_out, beta_out; int num_v_heads, hidden_size, has_a_log, has_dt_bias; };
struct ConvRowsParams { uint64_t qkv, history, weights, slots, snap, snap_on; int channels, rows, snap_rows, has_weights; int64_t slot_stride, snap_stride; };
struct ConvTiledParams { uint64_t src, dst, history, weights, slots; int channels, rows, tile_rows, has_weights; int64_t slot_stride; };
struct RecurrentParams { uint64_t state, qkv, gate, beta, output, slots, snap, snap_on; int h_k, h_v, rows, snap_rows; float scale; int64_t slot_stride, snap_stride; };
struct GatedNormParams { uint64_t attn_out, z, norm_weight; int num_v_heads, head_dim, use_sigmoid, has_z, has_weight; };
struct KvStoreParams { uint64_t data, src, slots, positions; int64_t slot_stride; int layer_idx, count, total_elems, max_seq; };
struct InitStreamsParams { uint64_t in_, streams; int hidden_size; };
struct HeadMeanParams { uint64_t streams, final_hidden; int hidden_size; };
struct PrepParams { uint64_t proj, ln_w, ln_b, plane, w_out, slots, positions; int layer; int64_t plane_slot_stride; };
struct PoolParams { uint64_t plane, pool_plane, ape, slots, positions; int layer, max_seq; int64_t plane_slot_stride, pool_slot_stride; };
struct TapStoreParams { uint64_t taps, hidden, slots, positions, owner; int m, tap, H, cap; };
struct ReduceStreamsParams { uint64_t streams, out_; int H, mode; };
struct RefGemmParams { uint64_t x, w, out_; int m, k, n; };

} // namespace engine::backend::impl

namespace engine::backend::impl {
struct RefReconParams { uint64_t codes, out_; int k_dim, n_dim, bits, cb; };
struct RefHadParams { uint64_t src, dst, pre, post; int cols; float r_scale; };
struct RefAttnParams { uint64_t q, k, v, gate, out_, slots, pos; int64_t slot_stride; int layer_idx, num_q_heads, num_kv_heads, head_dim, max_seq, has_gate; float scale; };
struct RefUnpackParams { uint64_t qg, q, gate; int total, head_dim; };
} // namespace engine::backend::impl

// mHC mix / apply kernels (src/kernels/slang/mhc.slang)
namespace engine::backend::impl {
struct HcPartialsParams { uint64_t streams, fn, partials; int row_len, chunk_cols, num_chunks; };
struct HcFinalizeParams { uint64_t streams, partials, base, scale, post, comb, collapsed, inv_rms_buf, mix_logits_buf, pre_buf; int hidden_size, num_chunks, chunk_cols_c, sinkhorn_iters; float rms_eps, hc_eps; };
struct HcApplyParams { uint64_t x_in, x_out, y, post, comb; int hidden_size, has_comb; };
} // namespace engine::backend::impl

// NVFP4 linear (src/kernels/slang/nvfp4.slang)
namespace engine::backend::impl {
struct Nvfp4Params { uint64_t codes, scales, x, out_; int m, k, n; float gm; };
} // namespace engine::backend::impl

// GLM kernels (src/kernels/slang/glm.slang)
namespace engine::backend::impl {
struct AbsorbParams { uint64_t q, kv_b, q_lat; int heads; };
struct ExpandParams { uint64_t o_lat, kv_b, out_; int heads; };
struct LatentStoreParams { uint64_t c, cache, slots, pos; int64_t slot_stride; int layer, max_seq; };
struct LatentAttnParams { uint64_t q_lat, cache, out_, slots, pos, sel_idx, sel_n; int64_t slot_stride; int layer, heads, max_seq, idx_stride, has_sel; float scale; };
struct DiagParams { uint64_t q_row, cache, scores; int n, heads; float scale; };
struct KvBParams { uint64_t x, w, k_out, v_out; int num_heads, head_dim; };
struct DsaScoreParams { uint64_t q, w, pool_plane, scores, slots, pos; int score_stride, layer, max_seq; int64_t pool_slot_stride; float norm; };
struct KdaSmallParams { uint64_t x, b_proj, f_a_proj, g_a_proj, beta, f_a_buf, g_a_buf; int hidden, has_b, has_fa, has_ga; };
struct KdaBigParams { uint64_t f_a_buf, g_a_buf, f_b_proj, g_b_proj, A_log, dt_bias, g_decay, z_out; int channels, has_fb, has_gb, has_a, has_dt; };
struct RouterPartialsParams { uint64_t x, w, partials; int hidden, num_experts, slices, max_slices; };
struct CombinePartsParams { uint64_t parts, w, shared_out, out_; int hidden, top_k, has_shared; };
struct SelectRouterParams { uint64_t partials, bias, topk_idx, topk_w; int num_experts, top_k, k_slices, max_slices; float scale; int has_bias; };
struct AccumParams { uint64_t accum, expert_out; float weight; int n; };
struct KdaStepParams { uint64_t state, qkv, g_decay, beta, z, o_norm, out_, slots, snap, snap_on; int64_t slot_stride, snap_stride; int num_heads, has_norm, has_z, has_o_norm, decay_first, snap_rows, rows; float eps; };
} // namespace engine::backend::impl

// Qwen4Exp kernels (src/kernels/slang/flash.slang)
namespace engine::backend::impl {
struct GrInitParams { uint64_t in_, x; int hidden, hc; };
struct GrNormParams { uint64_t x, w, normed, prev_post, prev_y, hi, lo; int hidden, hc; float eps; int has_prev, has_hi; };
struct GrGateParams { uint64_t normed, down, inject, t_out, post; int flat, rank, hc, has_inject; };
struct GrUpMixParams { uint64_t normed, up, t_in, mixed; int hidden, hc, rank, rows; };
struct GrApplyParams { uint64_t x, post, y; int hidden, hc; };
struct GrSigGatedParams { uint64_t y, sh, x, w; int hidden; };
struct GrStreamNormParams { uint64_t x, w, out_; int hidden, hc; float eps; };
struct GrBroadcastParams { uint64_t h, e, streams; int hidden, hc; };
struct GrGemmParams { uint64_t a, w, out_; int m, n, k, accumulate; };
struct GrGateActParams { uint64_t z, t_hi, t_lo, post; int ld, rank, hc, has_inject; };
struct GrUpEpilogueParams { uint64_t u, normed, mixed; int hidden, hc; };
struct PleGateParams { uint64_t key, value, streams, wk, wq, wc, gated, normed; int hidden, hc; float eps, gate_scale; };
struct PleConvParams { uint64_t normed, conv_w, ring, slots, pos, gated, streams; int channels, dilation; };
struct PleRingParams { uint64_t normed, ring, slots, pos; int channels; };
struct QsaPrepParams { uint64_t qk, qw, q_out, ring, slots, pos; int layer; int64_t ring_slot_stride; int rot_dim; float theta, eps; };
struct QsaPoolParams { uint64_t ring, pool_plane, kw, slots, pos; int layer; int64_t ring_slot_stride, pool_slot_stride; int max_pools, rot_dim; float theta, eps; };
struct QsaScoreParams { uint64_t q, pool_plane, scores, slots, pos; int score_stride, layer, max_pools; int64_t pool_slot_stride; float scale; };
struct SelectParams { uint64_t scores, idx, nkeys, pos; int score_stride, kpad; };
struct SparseAttnParams { uint64_t q, k, v, gate, out_, slots, pos, idx, nkeys; int64_t slot_stride; int layer_idx, num_q_heads, num_kv_heads, head_dim, max_seq, idx_stride, has_gate; float scale; };
struct RouterParams { uint64_t x, w, topk_idx, topk_w; int hidden, experts, top_k, round_fp16; };
struct MoeCombineParams { uint64_t route_out, topk_w, topk_idx, out_; int hidden, top_k, start_expert, end_expert; };
} // namespace engine::backend::impl
