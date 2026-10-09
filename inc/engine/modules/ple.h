#pragma once

#include "engine/backend/types.h"
#include "engine/core/batch.h"

namespace engine::modules {

// Device side of the Qwen4Exp n-gram embedding layer (PLE); see docs/FLASH_NEXT.md. All streams are fp32 [rows, hc, hidden].
struct PLE {
	static constexpr int RING = 16;   // per-slot ring of the last normed columns (conv state: (taps - 1) * dilation = 9 positions)

	// key [rows, hc * hidden] fp16 (key_proj output), value [rows, hidden] fp16 (value_proj output), streams [rows, hc, hidden]:
	//   gate[r, s]  = dot( rmsnorm_k(key[r, s]), rmsnorm_q(streams[r, s]) )          (grouped norms, weight applied as 1 + w)
	//   gated[r, s] = sigmoid( signed_sqrt( gate * gate_scale ) ) * value[r]          fp32
	//   normed      = rmsnorm_c( gated ), fp16 [rows, hc * hidden]
	static void gate_and_norm(
		const backend::half_t* key, const backend::half_t* value, const float* streams,
		const backend::half_t* norm_key_w, const backend::half_t* norm_query_w, const backend::half_t* norm_conv_w,
		float* gated, backend::half_t* normed,
		int rows, int hidden, int hc, float eps, float gate_scale, backend::stream_t stream );

	// streams += gated + silu( causal dilated depthwise conv of normed ): taps at positions pos, pos - d, pos - 2d, pos - 3d; a tap
	// comes from this chunk's normed when the same slot's earlier row holds that position, else from the slot's ring (zero below 0)
	static void conv_apply(
		const backend::half_t* normed, const backend::half_t* conv_w, const backend::half_t* ring, const pod::BatchRows* d_rows,
		const float* gated, float* streams, int rows, int channels, int dilation, backend::stream_t stream );

	// ring[slot][pos % RING] = normed[row], for every row
	static void ring_update( const backend::half_t* normed, backend::half_t* ring, const pod::BatchRows* d_rows, int rows, int channels, backend::stream_t stream );
};

} // namespace engine::modules
