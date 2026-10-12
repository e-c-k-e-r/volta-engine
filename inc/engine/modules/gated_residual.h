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

	// mix() with the norm pass already done: `scratch` holds normed [rows, hc * hidden] (written by CollectiveRing::all_reduce_gated_norm) and the room for t;
	// decode-sized calls only (rows <= 8, the streaming kernels)
	static void mix_prenormed(
		const backend::half_t* down, const backend::half_t* up, const backend::half_t* inject, backend::half_t* mixed, float* post, float* scratch,
		int rows, int hidden, int hc, int rank, backend::stream_t stream
	);

	// mix() through the row-tile kernels whatever the row count (the CUDA decode path streams the weights with other kernels; the test compares the two)
	static void mix_reference(
		const float* x, const backend::half_t* norm_w, const backend::half_t* down, const backend::half_t* up, const backend::half_t* inject,
		backend::half_t* mixed, float* post, float* scratch, int rows, int hidden, int hc, int rank, float eps, backend::stream_t stream,
		const float* prev_post = nullptr, const backend::half_t* prev_y = nullptr
	);

	// Tensor-core variant of mix() for many rows, composed by the caller. The fp32 operands are split into fp16 hi / lo halves and every product is two fp16 GEMMs
	// accumulated in fp32 (fp32-class accuracy):
	//   norm():        the norm pass of mix() (with the folded deferred apply), also writing normed as fp16 hi / lo when normed_hi is set
	//   gemm_f32 x2:   z = normed_hi @ [down; inject]^T, then z += normed_lo @ [down; inject]^T   (fp32 [rows, ld])
	//   gate_act():    t = silu( z / hc ) over `rank` columns (as hi / lo), post = 2 * sigmoid( z / hc ) over the hc columns after them
	//   gemm_f32 x2:   u = t_hi @ up^T, then u += t_lo @ up^T   (fp32 [rows, hc * hidden])
	//   up_epilogue(): mixed = mean_s( sigmoid( u[s] ) * normed[s] )
	static void norm( float* x, const backend::half_t* norm_w, float* normed, backend::half_t* normed_hi, backend::half_t* normed_lo, int rows, int hidden, int hc, float eps,
		const float* prev_post, const backend::half_t* prev_y, backend::stream_t stream );
	// out[m, n] (fp32) (+)= a[m, k] (fp16) @ w[n, k]^T (fp16), fp32 accumulation on the tensor cores
	static void gemm_f32( const backend::half_t* a, const backend::half_t* w, float* out, int m, int n, int k, bool accumulate, backend::stream_t stream );
	static void gate_act( const float* z, int ld, backend::half_t* t_hi, backend::half_t* t_lo, float* post, int rows, int rank, int hc, bool has_inject, backend::stream_t stream );
	static void up_epilogue( const float* u, const float* normed, backend::half_t* mixed, int rows, int hidden, int hc, backend::stream_t stream );

	// y[r, :] += sigmoid( x[r] . w ) * s[r, :]  (the shared expert's output gate; w [hidden] fp16)
	// gate[r] = sigmoid( x[r] . w ): the shared expert's scalar gate, for a consumer that applies y += gate * s itself (CUDA only)
	static void shared_gate( const backend::half_t* x, const backend::half_t* w, float* gate, int rows, int hidden, backend::stream_t stream );
	static void add_sigmoid_gated( backend::half_t* y, const backend::half_t* s, const backend::half_t* x, const backend::half_t* w, int rows, int hidden, backend::stream_t stream );

	// out[r * hc + s, :] = fp16( x[r, s, :] * rsqrt(mean(x^2) + eps) * (1 + w[s, :]) ): the per-stream grouped RMS norm (the MTP head's hidden input)
	static void stream_norm( const float* x, const backend::half_t* w, backend::half_t* out, int rows, int hidden, int hc, float eps, backend::stream_t stream );

	// streams[r, s, :] = h[r * hc + s, :] + e[r, :]  (the embedding branch broadcast into every stream)
	static void add_broadcast( const backend::half_t* h, const backend::half_t* e, float* streams, int rows, int hidden, int hc, backend::stream_t stream );

	static void apply( float* x, const float* post, const backend::half_t* y, int rows, int hidden, int hc, backend::stream_t stream );
};

} // namespace engine::modules
