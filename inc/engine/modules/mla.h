#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/batch.h"
#include "engine/core/kv_format.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

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
		int rows = 1                   // consecutive rows slot.row .. slot.row + rows - 1
	);
};

} // namespace engine::modules
