#pragma once

// The Vulkan backend's two attention paths, for test_vulkan_attention to hold one against the other (Attention::forward_decode / _prefill
// pick between them with vulkan_attn_ref)
#include "engine/core/batch.h"
#include "engine/core/kv_format.h"
#include "engine/core/tensor.h"
#include "engine/backend/types.h"

namespace engine::modules::vk_attn {

// The portable reference kernel (ref_attention.slang): a workgroup per (query head, row)
void reference( const pod::KVRef& kv, const pod::TensorView& q, const pod::TensorView& gate, pod::TensorView& out, size_t slot_stride, int layer_idx,
	const pod::BatchRows* d_rows, int count, int num_q_heads, int num_kv_heads, int head_dim, int max_seq, backend::stream_t stream, float scale, int window );

// The tiled kernels (fa.slang); one_slot: every row in one slot (key_hint as Attention::forward_prefill's)
void tiled( const pod::KVRef& kv, const pod::TensorView& q, const pod::TensorView& gate, pod::TensorView& out, size_t slot_stride, int layer_idx,
	const pod::BatchRows* d_rows, int count, int num_q_heads, int num_kv_heads, int head_dim, int max_seq, backend::stream_t stream, float scale, int window,
	bool one_slot, int key_hint );

} // namespace engine::modules::vk_attn
