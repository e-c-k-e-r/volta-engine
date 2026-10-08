#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/batch.h"
#include "engine/backend/types.h"

namespace engine::modules {

struct RoPE {
	static void forward(
		pod::TensorView& q,
		pod::TensorView& k,
		int num_q_heads,
		int num_kv_heads,
		int head_dim,
		int rot_dim,
		const pod::BatchRows* d_rows, // device-resident row descriptor (pos per row)
		int count,                    // rows to process (grid size)
		float freq_base = 10000000.0f,
		backend::stream_t stream = nullptr,
		bool interleaved = false      // pair channels (2i, 2i+1) instead of (i, i + rot_dim/2)
	);
};

} // namespace engine::modules
