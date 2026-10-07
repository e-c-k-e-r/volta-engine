#pragma once

#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/core/kv_format.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace engine::runtime {

struct KVCachePool {
	void* k_data = nullptr;
	void* v_data = nullptr;
	void* k_scale = nullptr;
	void* v_scale = nullptr;
	pod::KVFormat format = pod::KVFormat::FP16;
	size_t slot_stride_elems = 0;      // elements per slot (format independent): layers * max_seq * heads * head_dim
	size_t slot_stride_bytes = 0;      // FP16-equivalent bytes per slot, kept for existing callers
	size_t k_slot_bytes = 0, v_slot_bytes = 0, k_scale_slot_bytes = 0, v_scale_slot_bytes = 0;
	int max_slots = 0;
	int max_seq_len = 0;
	int num_layers = 0;
	int num_heads = 0;
	int head_dim = 0;

	// format: EXL3_KV = fp16 | fp8 | k8v4 | nvfp4 overrides the argument.
	bool init( int device_id, pod::MemoryArena& arena, int max_slots, int max_seq_len, int num_layers, int num_heads, int head_dim,
		pod::KVFormat format = pod::KVFormat::FP16 );
	pod::KVRef ref() const;
	void* get_k_ptr( int slot_id, int layer_idx, int pos );
	void* get_v_ptr( int slot_id, int layer_idx, int pos );
	void reset_slot( int slot_id, cudaStream_t stream = nullptr );
};

} // namespace engine::runtime
