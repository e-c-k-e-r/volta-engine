#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/batch.h"
#include "engine/core/kv_format.h"
#include "engine/backend/types.h"
#include <cstdint>
#include <algorithm>

namespace engine::modules {

// Absorbed MLA (GLM-5.x: no decoupled rope key). The cache holds the normalized KV latent c (512 halves per token, shared by
// every head) instead of per-head expanded K / V, so a token costs 1 KiB per layer where the expanded cache costs 16 KiB.
// With W_UK[h] the 256 K rows and W_UV[h] the 256 V rows of kv_b_proj for head h:
//   score(h, t) = q_nope[h] . (W_UK[h] c_t) = (W_UK[h]^T q_nope[h]) . c_t        -> absorb_q
//   o_lat[h]    = sum_t softmax_t(score) c_t,   out[h] = W_UV[h] o_lat[h]         -> attend, expand_out
struct MLALatent {
	static constexpr int LORA = 512;
	static constexpr int HEAD = 256;   // q_nope / v head dimension

	// q [rows, heads * 256] -> q_lat, stored by groups of 16 heads ([heads / 16][rows][16 * 512]; [rows, heads * 512] for 16 heads); kv_b is this rank's dense [heads * 512, 512] FP16
	static void absorb_q( const pod::TensorView& q, const pod::TensorView& kv_b, pod::TensorView& q_lat, int heads, int rows, backend::stream_t stream );
	// o_lat (grouped as above) -> out [rows, heads * 256]
	static void expand_out( const pod::TensorView& o_lat, const pod::TensorView& kv_b, pod::TensorView& out, int heads, int rows, backend::stream_t stream );
	// c [rows, 512] -> cache rows at the positions slot.rows gives (cache layout [slot][layer][max_seq][512], slot stride in elements)
	// The cache is one KV tensor in a single format (kv.k data, kv.ks scales, kv.kq): K and V are the same latent.
	static void store( const pod::KVRef& kv, const pod::TensorView& c, int layer, int max_seq, int rows, const pod::SlotRef& slot, backend::stream_t stream );
	static void store( backend::half_t* cache, const pod::TensorView& c, int layer, int max_seq, int rows, const pod::SlotRef& slot, backend::stream_t stream ) {
		store( pod::kv_ref_f16( cache, cache ), c, layer, max_seq, rows, slot, stream );
	}
	// Attention of every head over the latent cache (a multiple of 16 heads, run 16 at a time); q_lat / o_lat grouped as above. Rows beyond MAX_BATCH run in groups.
	// workspace: MAX_BATCH * 16 * n_chunks * (512 + 2) floats with n_chunks = workspace_chunks( max_seq ).
	static void attend( const pod::KVRef& kv, const pod::TensorView& q_lat, pod::TensorView& o_lat, int layer, int max_seq, int rows, int heads,
		const pod::SlotRef& slot, float* workspace, int key_hint, backend::stream_t stream,
		const int* sel_idx = nullptr, const int* sel_n = nullptr,   // DSA: per-row token lists (stride DSA::KPAD) and counts (-1 = dense row)
		int ws_seq = 0 );   // context the workspace was sized for (workspace_floats( ws_seq ...)); 0 = max_seq. With DSA a dense row never has more than ~2.1K keys
	static void attend( const backend::half_t* cache, const pod::TensorView& q_lat, pod::TensorView& o_lat, int layer, int max_seq, int rows, int heads,
		const pod::SlotRef& slot, float* workspace, int key_hint, backend::stream_t stream,
		const int* sel_idx = nullptr, const int* sel_n = nullptr, int ws_seq = 0 ) {
		attend( pod::kv_ref_f16( cache, cache ), q_lat, o_lat, layer, max_seq, rows, heads, slot, workspace, key_hint, stream, sel_idx, sel_n, ws_seq );
	}
	// Debug: dense logits of one row (n keys of a slot's layer cache) into scores[heads][n]
	static void diag_scores( const backend::half_t* q_lat_row, const backend::half_t* cache_slot_layer, int n, int heads, float* scores, backend::stream_t stream );
	static int workspace_chunks( int max_seq ) { return ( max_seq + 1023 ) / 1024; }
	// Partial slots (row, head, split) of 514 floats: decode rows x chunks, or a prefill chunk's key splits (at least 8192)
	static size_t workspace_slots( int max_seq, int heads ) { return std::max<size_t>( static_cast<size_t>( pod::MAX_BATCH ) * heads * workspace_chunks( max_seq ), 8192 ); }
	static size_t workspace_floats( int max_seq, int heads ) { return workspace_slots( max_seq, heads ) * ( LORA + 2 ); }
};

} // namespace engine::modules
