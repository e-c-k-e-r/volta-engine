#pragma once

#include "engine/config.h"
#include "engine/backend/types.h"

namespace engine::modules {

// Operators of the vision tower (ViT) that the language-model modules do not cover. Activations are FP16; the residual stream is
// FP32. All pointers are device pointers; rows are contiguous.
struct Vision {
	// out[m, n] = patches[m, k] @ w[n, k]^T + bias[n] + add[m, n] (FP32 result)
	static void patch_embed( const backend::half_t* patches, const backend::half_t* w, const backend::half_t* bias,
		const backend::half_t* add, float* out, int m, int n, int k, backend::stream_t stream = nullptr );

	// out = LayerNorm( x ) * w + b over rows of n (FP32 in, FP16 out)
	static void layer_norm( const float* x, const backend::half_t* w, const backend::half_t* b, backend::half_t* out,
		int m, int n, float eps, backend::stream_t stream = nullptr );

	// Adds the projection biases to q, k and v ([m, heads * head_dim]), then rotates q and k with the 2-D rotary tables
	// ([m, head_dim] FP32 cos / sin; rotate_half layout: channel i pairs with i + head_dim / 2).
	static void qkv_bias_rope( backend::half_t* q, backend::half_t* k, backend::half_t* v, const backend::half_t* bq,
		const backend::half_t* bk, const backend::half_t* bv, const float* cos_table, const float* sin_table,
		int m, int heads, int head_dim, backend::stream_t stream = nullptr );

	// Bidirectional softmax attention over all m rows (one image): out = softmax( q k^T * scale ) v, per head.
	static void attention( const backend::half_t* q, const backend::half_t* k, const backend::half_t* v, backend::half_t* out,
		int m, int heads, int head_dim, float scale, backend::stream_t stream = nullptr );

	// The same attention on the CUDA cores (reference for the tensor-core kernel)
	static void attention_simt( const backend::half_t* q, const backend::half_t* k, const backend::half_t* v, backend::half_t* out,
		int m, int heads, int head_dim, float scale, backend::stream_t stream = nullptr );

	enum Act : int { ACT_NONE = 0, ACT_GELU_TANH = 1, ACT_GELU_ERF = 2 };
	// x[m, n] = act( x + bias ), in place
	static void bias_act( backend::half_t* x, const backend::half_t* bias, int m, int n, int act, backend::stream_t stream = nullptr );

	// hidden[m, n] += y + bias (FP32 residual stream)
	static void bias_residual( float* hidden, const backend::half_t* y, const backend::half_t* bias, int m, int n,
		backend::stream_t stream = nullptr );

	// out[m / group, group * n] = LayerNorm( x ) with the rows of each group of `group` consecutive rows concatenated (the patch
	// merger; the norm acts on the n channels of every row)
	static void layer_norm_merge( const float* x, const backend::half_t* w, const backend::half_t* b, backend::half_t* out,
		int m, int n, int group, float eps, backend::stream_t stream = nullptr );
};

} // namespace engine::modules
