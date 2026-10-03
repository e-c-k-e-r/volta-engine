#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
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
		const int32_t* block_table,
		int pos,
		int mla_dc = 512,
		int mla_dr = 64,
		int page_size = 16,
		cudaStream_t stream = nullptr
	);

	static void forward_decode(
		const pod::TensorView& q_lat,
		const pod::TensorView& q_pe,
		const half* latent_cache,
		const half* rope_cache,
		const int32_t* block_table,
		pod::TensorView& out,
		int current_pos,
		int dsa_top_k = 64,
		int mla_dc = 512,
		int mla_dr = 64,
		int page_size = 16,
		cudaStream_t stream = nullptr
	);
};

} // namespace engine::modules
