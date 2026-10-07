#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/batch.h"
#include "engine/core/kv_format.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace engine::modules {

// KV cache layout: [slot][layer][max_seq][num_kv_heads * head_dim] FP16.
// k_cache / v_cache point at slot 0; slot_stride is the per-slot element stride.
// Row m of every call reads/writes slot d_rows->slot[m] at position d_rows->pos[m]; d_rows is
// device-resident so launches carry no position/slot values (CUDA-graph replayable).
struct Attention {
	static void unpack_q_gate(
		const pod::TensorView& q_gate,
		pod::TensorView& q,
		pod::TensorView& gate,
		int num_q_heads,
		int head_dim,
		cudaStream_t stream = nullptr
	);

	// KV tensors are addressed through a KVRef (formats and scales in kv_format.h); element offsets and slot_stride are
	// in elements whatever the format.
	static void store_kv_cache(
		const pod::KVRef& kv,
		const pod::TensorView& k,
		const pod::TensorView& v,
		size_t slot_stride,
		int layer_idx,
		const pod::BatchRows* d_rows,
		int count,
		int num_kv_heads,
		int head_dim,
		int max_seq = 2048,
		cudaStream_t stream = nullptr
	);

	static void forward_decode(
		const pod::KVRef& kv,
		const pod::TensorView& q,
		const pod::TensorView& gate,
		pod::TensorView& out,
		size_t slot_stride,
		int layer_idx,
		const pod::BatchRows* d_rows,
		int count,
		int num_q_heads,
		int num_kv_heads,
		int head_dim,
		int max_seq = 2048,
		cudaStream_t stream = nullptr
	);

	// Prefill form for chunks above the decode sizes (one slot, head_dim 256): K/V tiles are shared across queries.
	// Same arguments and result as forward_decode.
	static void forward_prefill(
		const pod::KVRef& kv,
		const pod::TensorView& q,
		const pod::TensorView& gate,
		pod::TensorView& out,
		size_t slot_stride,
		int layer_idx,
		const pod::BatchRows* d_rows,
		int count,
		int num_q_heads,
		int num_kv_heads,
		int head_dim,
		int max_seq = 2048,
		cudaStream_t stream = nullptr
	);
};

} // namespace engine::modules
