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
		backend::stream_t stream,
		// A deferred apply() folded into the norm pass: x[r, s, :] += prev_post[r, s] * prev_y[r, :] is applied (and stored) first when prev_y is set.
		// x must then be writable (the caller's own stream buffer).
		const float* prev_post = nullptr,
		const backend::half_t* prev_y = nullptr
	);

	// y[r, :] += sigmoid( x[r] . w ) * s[r, :]  (the shared expert's output gate; w [hidden] fp16)
	static void add_sigmoid_gated( backend::half_t* y, const backend::half_t* s, const backend::half_t* x, const backend::half_t* w, int rows, int hidden, backend::stream_t stream );

	// out[r * hc + s, :] = fp16( x[r, s, :] * rsqrt(mean(x^2) + eps) * (1 + w[s, :]) ): the per-stream grouped RMS norm (the MTP head's hidden input)
	static void stream_norm( const float* x, const backend::half_t* w, backend::half_t* out, int rows, int hidden, int hc, float eps, backend::stream_t stream );

	// streams[r, s, :] = h[r * hc + s, :] + e[r, :]  (the embedding branch broadcast into every stream)
	static void add_broadcast( const backend::half_t* h, const backend::half_t* e, float* streams, int rows, int hidden, int hc, backend::stream_t stream );

	static void apply( float* x, const float* post, const backend::half_t* y, int rows, int hidden, int hc, backend::stream_t stream );
};

} // namespace engine::modules
