#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/batch.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace engine::modules {

struct RoPE {
	static void forward(
		pod::TensorView& q,
		pod::TensorView& k,
		int num_q_heads,
		int num_kv_heads,
		int head_dim,
		int rot_dim,
		const pod::BatchRows& rows,
		float freq_base = 10000000.0f,
		cudaStream_t stream = nullptr
	);
};

} // namespace engine::modules
