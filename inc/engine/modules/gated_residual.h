#pragma once

#include "engine/backend/types.h"

namespace engine::modules {

// Qwen4Exp gated residual over fp32 streams x [rows, hc, hidden] (see docs/FLASH_NEXT.md):
//   normed = x * rsqrt(mean_d(x^2) + eps) * (1 + norm_w)            per stream
//   t      = silu(normed.flat @ down^T / hc)                         [rows, rank]
//   mixed  = mean_s( sigmoid(t @ up^T) * normed )                    [rows, hidden] fp16, the sublayer input
//   post   = 2 * sigmoid(normed.flat @ inject^T / hc)                [rows, hc]    (only with inject)
// and after the sublayer: x[:, s, :] += post[:, s] * y.
struct GatedResidual {
	// Floats of scratch mix() needs: normed [rows, hc * hidden] and t [rows, rank]
	static size_t scratch_floats( int rows, int hidden, int hc, int rank ) {
		return static_cast<size_t>( rows ) * hc * hidden + static_cast<size_t>( rows ) * rank;
	}

	// x[:, s, :] = in (fp16 [rows, hidden]) for every stream
	static void init_streams( const backend::half_t* in, float* x, int rows, int hidden, int hc, backend::stream_t stream );

	static void mix(
		const float* x,
		const backend::half_t* norm_w,
		const backend::half_t* down,
		const backend::half_t* up,
		const backend::half_t* inject,   // null: no post gate
		backend::half_t* mixed,
		float* post,                     // [rows, hc], unused without inject
		float* scratch,
		int rows, int hidden, int hc, int rank, float eps,
		backend::stream_t stream
	);

	static void apply( float* x, const float* post, const backend::half_t* y, int rows, int hidden, int hc, backend::stream_t stream );
};

} // namespace engine::modules
