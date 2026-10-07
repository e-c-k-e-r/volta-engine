#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/batch.h"
#include "engine/core/kv_format.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <algorithm>

namespace engine::modules {

struct MLADSA {
	static void store_cache(
		half* latent_cache,
		half* rope_cache,
		const half* k_lat,
		const half* k_pe,
		int layer_idx,
		int pos,
		int mla_dc = 512,
		int mla_dr = 64,
		int max_seq = 2048,
		cudaStream_t stream = nullptr
	);

	static void forward_decode(
		const pod::TensorView& q_lat,
		const pod::TensorView& q_pe,
		const half* latent_cache,
		const half* rope_cache,
		pod::TensorView& out,
		int layer_idx,
		int current_pos,
		int dsa_top_k = 64,
		int mla_dc = 512,
		int mla_dr = 64,
		int max_seq = 2048,
		int out_dim = 4096,
		cudaStream_t stream = nullptr
	);
};

struct MLA {
	static void project_kv_b(
		const pod::TensorView& kv_norm,
		const pod::TensorView& kv_b_proj,
		pod::TensorView& k_out,
		pod::TensorView& v_out,
		int num_heads = 16,
		int head_dim = 256,
		cudaStream_t stream = nullptr,
		int rows = 1                    // kv_norm [rows, 512] -> k/v [rows, heads * head_dim]
	);

	// Requires slot.rows (positions and slots are read on the device); K/V formats per the KVRef.
	static void store_kv_cache(
		const pod::KVRef& kv,
		const pod::TensorView& k,
		const pod::TensorView& v,
		int mla_layer_idx,
		int pos,
		int total_elems = 4096,
		int max_seq = 2048,
		cudaStream_t stream = nullptr,
		const pod::SlotRef& slot = {}, // if set: pos/slot read on device (graph-capturable)
		int rows = 1                   // consecutive rows slot.row .. slot.row + rows - 1
	);

	static void forward_decode(
		const pod::TensorView& q,
		const pod::KVRef& kv,
		pod::TensorView& out,
		int mla_layer_idx,
		int pos,
		int num_heads = 16,
		int head_dim = 256,
		int max_seq = 2048,
		cudaStream_t stream = nullptr,
		const pod::SlotRef& slot = {}, // if set: pos/slot read on device (graph-capturable)
		int rows = 1,                  // consecutive rows slot.row .. slot.row + rows - 1
		int key_hint = 0               // prefill chunks: keys the rows can attend to (sizes the split only)
	);
};


// Absorbed MLA (GLM-5.x: no decoupled rope key). The cache holds the normalized KV latent c (512 halves per token, shared by
// every head) instead of per-head expanded K / V, so a token costs 1 KiB per layer where the expanded cache costs 16 KiB.
// With W_UK[h] the 256 K rows and W_UV[h] the 256 V rows of kv_b_proj for head h:
//   score(h, t) = q_nope[h] . (W_UK[h] c_t) = (W_UK[h]^T q_nope[h]) . c_t        -> absorb_q
//   o_lat[h]    = sum_t softmax_t(score) c_t,   out[h] = W_UV[h] o_lat[h]         -> attend, expand_out
struct MLALatent {
	static constexpr int LORA = 512;
	static constexpr int HEAD = 256;   // q_nope / v head dimension

	// q [rows, heads * 256] -> q_lat [rows, heads * 512]; kv_b is this rank's dense [heads * 512, 512] FP16
	static void absorb_q( const pod::TensorView& q, const pod::TensorView& kv_b, pod::TensorView& q_lat, int heads, int rows, cudaStream_t stream );
	// o_lat [rows, heads * 512] -> out [rows, heads * 256]
	static void expand_out( const pod::TensorView& o_lat, const pod::TensorView& kv_b, pod::TensorView& out, int heads, int rows, cudaStream_t stream );
	// c [rows, 512] -> cache rows at the positions slot.rows gives (cache layout [slot][layer][max_seq][512], slot stride in elements)
	static void store( half* cache, const pod::TensorView& c, int layer, int max_seq, int rows, const pod::SlotRef& slot, cudaStream_t stream );
	// Attention of every head over the latent cache; q_lat / o_lat [rows, heads * 512]. Rows beyond MAX_BATCH run in groups.
	// workspace: MAX_BATCH * heads * n_chunks * (512 + 2) floats with n_chunks = workspace_chunks( max_seq ).
	static void attend( const half* cache, const pod::TensorView& q_lat, pod::TensorView& o_lat, int layer, int max_seq, int rows, int heads,
		const pod::SlotRef& slot, float* workspace, int key_hint, cudaStream_t stream );
	static int workspace_chunks( int max_seq ) { return ( max_seq + 1023 ) / 1024; }
	// Partial slots (row, head, split) of 514 floats: decode rows x chunks, or a prefill chunk's key splits (at least 8192)
	static size_t workspace_slots( int max_seq, int heads ) { return std::max<size_t>( static_cast<size_t>( pod::MAX_BATCH ) * heads * workspace_chunks( max_seq ), 8192 ); }
	static size_t workspace_floats( int max_seq, int heads ) { return workspace_slots( max_seq, heads ) * ( LORA + 2 ); }
};

} // namespace engine::modules
