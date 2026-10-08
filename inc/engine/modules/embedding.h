#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/backend/types.h"
#include <cuda_fp16.h>

namespace engine::modules {

struct Embedding {
	// Gathers `count` rows of a BF16 [vocab, hidden] table into FP16 out [count, hidden].
	// token_ids must be a device pointer.
	static void forward(
		const pod::TensorView& table,
		const int* token_ids,
		int count,
		pod::TensorView& out,
		backend::stream_t stream = nullptr
	);
};

} // namespace engine::modules
