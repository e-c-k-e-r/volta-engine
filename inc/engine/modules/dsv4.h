#pragma once

#include "engine/config.h"
#include "engine/core/batch.h"
#include "engine/backend/types.h"
#include <cstddef>
#include <cstdint>

namespace engine::modules {

// DeepSeek-V4.1 attention, routing and hyper-connection pieces (reference: DeepSeek's inference/model.py, vendored in tests/tools/dsv41).
// Every layer attends over a sliding window of its own raw KV (one 512-wide vector per token, shared by all heads as K and V) plus, on
// compressing layers, compressed KV produced by a source layer and chosen per query by an indexer. Rotary embedding is interleaved
// (adjacent pairs) on the last ROPE dims; the attention output gets the inverse rotation. The reference quantizes caches and some
// activations through FP8 / FP4 grids (quantize, dequantize); the kernels do the same so the values match:
//   window KV   : E4M3, one power-of-two scale per 32 values
//   compressed  : E2M1, one E4M3 scale per 16 values
//   index keys / queries : E2M1, one power-of-two scale per 32 values
//   attention output (wo_a input) : E4M3, one power-of-two scale per 32 values
// rope: the layer's rotary table [positions][ROPE / 2] of (cos, sin) pairs.
// Caches are position-indexed per slot, so a rewind needs no state copy:
//   window ring [slot][layer][RING][HD]  at pos % RING (a row reads the last WINDOW positions; RING covers a chunk on top)
//   compressor ring (ratio > 1) [slot][source][RING][HD] FP32 kv and score of the tokens of the group being filled
//   compressed KV [slot][source][capacity][HD] and index keys [slot][source][capacity][IDX_DIM] at entry pos / ratio
struct DSV4 {
	static constexpr int HD = 512;          // head dim = KV latent width
	static constexpr int ROPE = 64;         // rotary dims (the last ROPE of a head)
	static constexpr int WINDOW = 128;
	static constexpr int RING = 1024;
	static constexpr int IDX_HEADS = 32;
	static constexpr int IDX_DIM = 128;
	static constexpr int TOPK = 512;        // compressed entries a query keeps
	static constexpr int GROUP_HEADS = 8;   // heads per wo_a group
	static constexpr int SCORE_ROWS = 64;   // rows the indexer scores at a time

	// kv [rows, HD] (normed) -> rope at pos, E4M3 grid -> ring[slot][layer][pos % RING]
	static void window_store( const backend::half_t* kv, backend::half_t* ring, const float* rope, int layer, int rows,
		const pod::BatchRows* d_rows, size_t ring_slot_stride, backend::stream_t stream );

	// Compressor of a source layer. ratio 1: latent = rmsnorm(ckv); ratio r > 1: ckv / score rows go to the ring first, and the row
	// completing a group (pos % r == r - 1) pools latent[d] = sum_t softmax_t(score[t][d]) ckv[t][d] over the group's r tokens, then rmsnorm.
	// The completing row writes the unrotated latent to latent_out[row] (for the index keys) and the compressed entry pos / r: the latent
	// roped at the group's first position, on the E2M1 / E4M3-16 grid. Other rows leave latent_out untouched
	static void compress( const backend::half_t* ckv, const backend::half_t* cscore, const backend::half_t* norm_w, float* cring_kv, float* cring_sc,
		backend::half_t* latent_out, backend::half_t* comp_kv, const float* rope, int ratio, int rows, const pod::BatchRows* d_rows,
		size_t cring_slot_stride, size_t comp_slot_stride, float eps, backend::stream_t stream );

	// Index keys of the completing rows: rmsnorm(kraw) (kraw = wk latent), rope at the group's first position, E2M1 / E8M0-32 grid -> idx_k
	static void index_key_store( const backend::half_t* kraw, const backend::half_t* knorm_w, backend::half_t* idx_k, const float* rope, int ratio, int rows,
		const pod::BatchRows* d_rows, size_t idx_slot_stride, float eps, backend::stream_t stream );

	// Indexer of an index-source layer. q [rows, IDX_HEADS * IDX_DIM] (rope at pos, E2M1 / E8M0-32 grid here), w [rows, IDX_HEADS] (weights_proj x):
	//   score[t] = (IDX_DIM * IDX_HEADS)^-1/2 * sum_h w[h] relu(q[h] . k[t]),   t < n = (pos + 1) / ratio
	// sel[row][0..nsel) = the TOPK best t (all t when n <= TOPK). cand_mode 1 (the candidate source): before the top-k, cand[row] marks the best
	// cand_blocks blocks of cand_block entries by their best score (the block holding entry n - 1 always kept); cand_mode 2: scores outside
	// cand[row] do not compete. scores [SCORE_ROWS, cap] and cand [rows, cap / cand_block] are scratch (rows are scored in blocks of SCORE_ROWS)
	static void indexer( const backend::half_t* q, const backend::half_t* w, const backend::half_t* idx_k, float* scores, uint8_t* cand, int* sel, int* nsel,
		const float* rope, int ratio, int cap, int cand_mode, int cand_blocks, int cand_block, int rows, const pod::BatchRows* d_rows, size_t idx_slot_stride,
		backend::stream_t stream );

	// Attention of nh heads over the window (positions pos - WINDOW + 1 .. pos of the layer's ring) and, with comp_kv, the selected compressed
	// entries sel[row][0..nsel). q [rows, nh, HD] unrotated; logits scaled by HD^-1/2, a per-head sink in the denominator; the output is
	// inverse-roped at pos, put on the E4M3-32 grid and written group-major: o[g][row][(h % GROUP_HEADS) * HD + d], g = h / GROUP_HEADS
	// ws (attend_ws_floats( nh )) enables the split-KV decode path for up to MAX_BATCH rows; null runs one block per row
	static void attend( const backend::half_t* q, const backend::half_t* ring, const backend::half_t* comp_kv, const int* sel, const int* nsel,
		const float* sink, const float* rope, backend::half_t* o, int nh, int layer, int rows, const pod::BatchRows* d_rows, size_t ring_slot_stride,
		size_t comp_slot_stride, float* ws, backend::stream_t stream );
	static constexpr size_t attend_ws_floats( int nh ) { return static_cast<size_t>( 64 ) * nh * ( HD + 2 ); }   // rows x splits <= 64

	// DSpark drafter. Taps: the mean of the four hyper-connection streams of each row (times scale; the drafter's entry RMSNorm makes it scale-free),
	// kept for the last WINDOW positions of every slot: ring[slot][pos % WINDOW][taps][hidden], tap = this layer's index among the taps. Rows below
	// floor_pos are not stored (a long chunk would wrap onto itself)
	static void tap_store( const float* streams, backend::half_t* ring, int tap, int taps, int hidden, float scale, int floor_pos, int rows, const pod::BatchRows* d_rows,
		size_t ring_slot_stride, backend::stream_t stream );
	// out[r] = the taps row of position first + r of a slot, r < n
	static void tap_gather( const backend::half_t* ring, backend::half_t* out, int slot, int first, int n, int taps, int hidden, size_t ring_slot_stride, backend::stream_t stream );
	// Attention of a drafter block: nh heads, query rows `rows` (positions d_rows->pos, slot 0 of the scratch), over the window ring's positions
	// [w0, t] of `slot` (layer `layer` of ring) and all the block's own keys (blk: a scratch ring written by window_store at the block positions; the
	// block is not causal). Same scaling, sink, inverse rotation and output grid as attend
	static void dspark_attend( const backend::half_t* q, const backend::half_t* ring, const backend::half_t* blk, const float* sink, const float* rope, backend::half_t* o,
		int nh, int layer, int rows, const pod::BatchRows* d_rows, int slot, int w0, int t, size_t ring_slot_stride, backend::stream_t stream );

	// out[row][g * n + c] = in[g][row][c] for g < groups
	static void concat_groups( const backend::half_t* in, backend::half_t* out, int groups, int n, int rows, backend::stream_t stream );

	// sqrt-softplus routing: s = sqrt(softplus(x . W[e])), experts = top_k of s + bias, weights = s / sum(s of the chosen) * scale. Rows whose img_flags entry
	// is set (tokens of an image) take bias_vl instead of bias (both null-safe: without img_flags every row takes bias).
	// w_t [hidden, E]; the chosen ids are written through perm (checkpoint id -> engine id). scratch: route_scratch_floats( rows, experts )
	static void route( const backend::half_t* x, const backend::half_t* w_t, const float* bias, const float* bias_vl, const uint8_t* img_flags, const int* perm, int* idx_out, backend::half_t* w_out,
		float* scratch, int hidden, int experts, int top_k, float scale, int rows, backend::stream_t stream );
	static constexpr size_t route_scratch_floats( int rows, int experts ) { return static_cast<size_t>( rows ) * 4 * experts; }

	// Hyper-connection input with the lagged pre-mix: out = rmsnorm(sum_h pre[h] streams[h]) * norm_w (the sum in FP32, it exceeds FP16),
	// and post[row][h] *= post_scale (the sublayer output is stored scaled down by that factor)
	static void hc_collapse_norm( const float* streams, const float* pre, const backend::half_t* norm_w, backend::half_t* out, float* post, float post_scale,
		int hidden, float eps, int rows, backend::stream_t stream );

	// pre[row] = (1, 0, 0, 0)
	static void hc_identity_pre( float* pre, int rows, backend::stream_t stream );

	// Engram: kv [rows, 5 * hidden] = [key 0..3 | value] (wkv of the looked-up rows); for each stream h:
	//   dot = sum_d s[h][d] w[h][d] key[h][d] * rstd(s[h]) * rstd(key[h]) / sqrt(hidden),  gate = sigmoid(sign(dot) sqrt(max(|dot|, 1e-6)))
	//   s[h] += gate * value
	static void engram_gate( float* streams, const backend::half_t* kv, const float* w, int hidden, float eps, int rows, backend::stream_t stream );

};

} // namespace engine::modules
