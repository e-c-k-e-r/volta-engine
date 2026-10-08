#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/batch.h"
#include "engine/backend/types.h"

namespace engine::modules {

// DeepSeek sparse attention indexer for GLM-5.x with k-pool compression (reference: exllamav3 MLAttention._indexer_topk_kpool).
// Per layer a lightning indexer scores every complete pool of kpool consecutive tokens against the query and keeps the
// best topk / kpool pools; a row attends to those pools' tokens plus the raw tokens of its incomplete tail pool. A row
// whose visible context holds at most topk / kpool complete pools attends to everything (dense).
//
//   per token : k = LayerNorm(wk x) (128), gate = Wg x (128), w = Ww x (32)         -> plane [k || gate]
//   per pool  : key[d] = sum_j softmax_j(gate_j[d] + ape[j][d]) k_j[d]                -> pool plane (written by the row that completes it)
//   per query : score[p] = (128 * 32)^-1/2 * sum_h w[h] relu(q[h] . key[p]),   p < (pos + 1) / kpool
struct DSA {
	static constexpr int HEADS = 32;
	static constexpr int DIM = 128;
	static constexpr int POOL = 4;
	static constexpr int TOPK = 2048;
	static constexpr int SEL_POOLS = TOPK / POOL;                   // 512
	static constexpr int KPAD = TOPK + POOL - 1 + 29;               // list entries per row: selected tokens + tail, padded to 2080
	static constexpr int PROJ = DIM + HEADS + DIM;                  // fused dense projection: [k | w | gate]
	static constexpr int PLANE = 2 * DIM;                           // per-token plane: [k_norm | gate]
	// The plane is a ring of PLANE_CAP positions (index pos % PLANE_CAP): a pool only needs the kpool tokens that complete it, and a
	// chunk (at most MAX_ROWS rows) is prepared before its pools are built, so the ring outlives every consumer
	static constexpr int PLANE_CAP = 1024;

	// proj[r][j] = sum_k w[j][k] x[r][k] for rows <= 16 (graph-capturable); larger row counts use the caller's GEMM
	static void project_small( const backend::half_t* w, const backend::half_t* x, backend::half_t* proj, int rows, int n, int k, backend::stream_t stream );

	// Same product for larger row counts (prefill chunks): one dense GEMM
	static void project_large( const backend::half_t* w, const backend::half_t* x, backend::half_t* proj, int rows, int n, int k, backend::stream_t stream );

	// LayerNorm the key part of proj, append [k_norm | gate] to the plane at each row's position, copy the head weights to w_out
	static void prep( const backend::half_t* proj, const backend::half_t* ln_w, const backend::half_t* ln_b, backend::half_t* plane, backend::half_t* w_out, int layer, int max_seq, int rows,
		const pod::BatchRows* d_rows, size_t plane_slot_stride, backend::stream_t stream );

	// Build the pool each row completes (position % kpool == kpool - 1) from the plane
	static void pool_update( const backend::half_t* plane, backend::half_t* pool_plane, const backend::half_t* ape, int layer, int max_seq, int rows,
		const pod::BatchRows* d_rows, size_t plane_slot_stride, size_t pool_slot_stride, backend::stream_t stream );

	// scores[r][p] for p < (pos + 1) / kpool (fp16); rows already dense (<= SEL_POOLS pools) are skipped
	static void score( const backend::half_t* q_idx, const backend::half_t* w, const backend::half_t* pool_plane, backend::half_t* scores, int score_stride, int layer, int max_seq, int rows,
		const pod::BatchRows* d_rows, size_t pool_slot_stride, backend::stream_t stream );

	// Top SEL_POOLS pools per row, expanded to token indices + tail: idx[r][0 .. nkeys[r]); nkeys[r] = -1 marks a dense row
	static void select( const backend::half_t* scores, int score_stride, int* idx, int* nkeys, int rows, const pod::BatchRows* d_rows, backend::stream_t stream );
};

} // namespace engine::modules
