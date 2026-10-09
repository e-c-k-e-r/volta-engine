#pragma once

#include "engine/backend/types.h"
#include "engine/core/batch.h"

namespace engine::modules {

// QSA indexer of Qwen3.8-Flash-Next (reference: exllamav3 QSAIndexer; docs/FLASH_NEXT.md). Like the GLM DSA indexer it scores every
// complete pool of POOL tokens and keeps the best SEL_POOLS pools plus the incomplete tail; selection and expansion to token lists reuse
// DSA::select (same pool size, budget and list width). What differs:
//   per token : [q (4 heads x 128) | k (128)] = index_qk_proj x ; q = rmsnorm(q) * (1 + w_q), roped on the first rot_dim dims at the row's
//               position; the raw k is kept as is
//   per pool  : key = rope_at_pool_start( rmsnorm( mean of the pool's POOL raw keys ) * (1 + w_k) )
//   per query : score[p] = DIM^-1/2 * sum_h relu( q[h] . key[p] ),   p < (pos + 1) / POOL
struct QSA {
	static constexpr int HEADS = 4;
	static constexpr int DIM = 128;
	static constexpr int POOL = 4;
	static constexpr int PROJ = ( HEADS + 1 ) * DIM;
	static constexpr int RING = 1024;   // raw keys of the last RING positions per slot and layer (a chunk of up to MAX_ROWS rows is prepared before its pools are built)

	// qk [rows, PROJ] -> q_out [rows, HEADS, DIM] (normed and roped), raw k into ring[slot][layer][pos % RING]
	static void prep( const backend::half_t* qk, const backend::half_t* q_norm_w, backend::half_t* q_out, backend::half_t* ring, int layer, int rows,
		const pod::BatchRows* d_rows, size_t ring_slot_stride, int rot_dim, float theta, float eps, backend::stream_t stream );

	// Rows with pos % POOL == POOL - 1 complete the pool pos / POOL: mean, norm, rope at the pool's first position, into pool_plane[slot][layer][pool]
	static void pool_update( const backend::half_t* ring, backend::half_t* pool_plane, const backend::half_t* k_norm_w, int layer, int rows,
		const pod::BatchRows* d_rows, size_t ring_slot_stride, size_t pool_slot_stride, int max_pools, int rot_dim, float theta, float eps, backend::stream_t stream );

	// scores[r][p] (fp16, row stride score_stride) for p < (pos + 1) / POOL, for rows with more than DSA::SEL_POOLS such pools
	static void score( const backend::half_t* q, const backend::half_t* pool_plane, backend::half_t* scores, int score_stride, int layer, int max_pools, int rows,
		const pod::BatchRows* d_rows, size_t pool_slot_stride, backend::stream_t stream );
};

} // namespace engine::modules
