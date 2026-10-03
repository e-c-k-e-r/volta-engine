#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace engine::modules {

struct Attention {
	static void unpack_q_gate(
		const pod::TensorView& q_gate,
		pod::TensorView& q,
		pod::TensorView& gate,
		int num_q_heads,
		int head_dim,
		cudaStream_t stream = nullptr
	);

	static void store_kv_cache(
		half* k_cache,
		half* v_cache,
		const pod::TensorView& k,
		const pod::TensorView& v,
		int layer_idx,
		int pos,
		int num_kv_heads,
		int head_dim,
		int max_seq = 2048,
		cudaStream_t stream = nullptr
	);

	static void forward_decode(
		const pod::TensorView& q,
		const half* k_cache,
		const half* v_cache,
		const pod::TensorView& gate,
		pod::TensorView& out,
		int layer_idx,
		int pos,
		int num_q_heads,
		int num_kv_heads,
		int head_dim,
		int max_seq = 2048,
		cudaStream_t stream = nullptr
	);
};

} // namespace engine::modules
