#pragma once

#include "engine/config.h"
#include "engine/core/arena.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace engine::runtime {

struct KVCachePool {
	void* k_data = nullptr;
	void* v_data = nullptr;
	size_t slot_stride_bytes = 0;
	int max_slots = 0;
	int max_seq_len = 0;
	int num_layers = 0;
	int num_heads = 0;
	int head_dim = 0;

	bool init( int device_id, pod::MemoryArena& arena, int max_slots, int max_seq_len, int num_layers, int num_heads, int head_dim );
	void* get_k_ptr( int slot_id, int layer_idx, int pos );
	void* get_v_ptr( int slot_id, int layer_idx, int pos );
	void reset_slot( int slot_id, cudaStream_t stream = nullptr );
};

} // namespace engine::runtime
