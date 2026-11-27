#pragma once

#include "engine/config.h"
#include "engine/backend/types.h"

namespace engine::modules {

// Operators of the vision tower (ViT) that the language-model modules do not cover. Activations are FP16; the residual stream is
// FP32. All pointers are device pointers; rows are contiguous.
struct Vision {
	// out[m, n] = patches[m, k] @ w[n, k]^T + bias[n] + add[m, n] (FP32 result; add may be null)
	static void patch_embed( const backend::half_t* patches, const backend::half_t* w, const backend::half_t* bias,
		const backend::half_t* add, float* out, int m, int n, int k, backend::stream_t stream = nullptr );

	// out[i] = half( in[i] ) for n values
	static void f32_to_f16( const float* in, backend::half_t* out, size_t n, backend::stream_t stream = nullptr );

	// Qwen3-VL learned position embeddings, bilinearly interpolated from a side x side table to a gh x gw grid; rows come out in
	// merge-block order (gh / merge, gw / merge, merge, merge). out is [gh * gw, dim] FP16.
	static void pos_interp( const backend::half_t* table, backend::half_t* out, int gh, int gw, int merge, int side, int dim, backend::stream_t stream = nullptr );

	// out = x / rms( x ) * w over rows of n (FP32 in, FP16 out; w is the plain gain, no +1)
	static void rms_norm( const float* x, const backend::half_t* w, backend::half_t* out, int m, int n, float eps, backend::stream_t stream = nullptr );

	// out = LayerNorm( x ) * w + b over rows of n (FP16 in and out)
	static void layer_norm_half( const backend::half_t* x, const backend::half_t* w, const backend::half_t* b, backend::half_t* out,
		int m, int n, float eps, backend::stream_t stream = nullptr );

	// Adds the projection biases to q, k and v, RMS-normalizes every head of q and k with the per-head gains qn / kn (head_dim
	// values), then applies the 2-D rotary; head_dim must be 64.
	static void qkv_bias_norm_rope( backend::half_t* q, backend::half_t* k, backend::half_t* v, const backend::half_t* bq, const backend::half_t* bk,
		const backend::half_t* bv, const backend::half_t* qn, const backend::half_t* kn, const float* cos_table, const float* sin_table,
		int m, int heads, int head_dim, float eps, backend::stream_t stream = nullptr );

	// Gemma 4: RMS-normalizes every head of q and k with the gains qn / kn and every head of v without a gain, then rotates q and k in two
	// halves: the first head_dim / 2 channels by the patch's x position, the rest by its y (pairs i, i + head_dim / 4 inside each half,
	// frequency theta^( -4 i / head_dim )). pos_xy is [m, 2] (x, y) on the device; head_dim <= 128 and a multiple of 4.
	static void qkv_norm_rope2d( backend::half_t* q, backend::half_t* k, backend::half_t* v, const backend::half_t* qn, const backend::half_t* kn,
		const int* pos_xy, int m, int heads, int head_dim, float theta, float eps, backend::stream_t stream = nullptr );

	// Gemma 4's pooler: soft token t of a ( gh / k ) x ( gw / k ) grid is the mean of its k x k patches (rows of x in raster order of the
	// gh x gw patch grid), times sqrt( n ), minus bias, times scale (both [n]; may be null). FP32 in, FP16 out [tokens, n]
	static void pool_std( const float* x, const backend::half_t* bias, const backend::half_t* scale, backend::half_t* out, int gh, int gw, int k, int n,
		backend::stream_t stream = nullptr );

	// out[t, c * group + j] = h[t * group + j, c]: the group's rows laid out channel-major (the input of a stride-`group` conv)
	static void merge_gather( const backend::half_t* h, backend::half_t* out, int tokens, int n, int group, backend::stream_t stream = nullptr );

	// out[m, n] = a[m, k] @ w[n, k]^T (FP16, tensor cores)
	static void dense_gemm( const backend::half_t* a, const backend::half_t* w, backend::half_t* out, int m, int n, int k, backend::stream_t stream = nullptr );

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

	enum Act : int { ACT_NONE = 0, ACT_GELU_TANH = 1, ACT_GELU_ERF = 2, ACT_SILU = 3 };
	// x[m, n] = act( x + bias ), in place (bias may be null)
	static void bias_act( backend::half_t* x, const backend::half_t* bias, int m, int n, int act, backend::stream_t stream = nullptr );

	// x[n] = min( max( x, lo ), hi ), in place (Gemma 4's clipped linears)
	static void clamp( backend::half_t* x, size_t n, float lo, float hi, backend::stream_t stream = nullptr );

	// Gemma 4's audio subsampler layer: a 3x3 stride-2 convolution (zero padding 1) of in [T, F, C] (FP32, channels last) by w [O, C, 3, 3]
	// (FP32) into out [( T + 1 ) / 2, ( F + 1 ) / 2, O], then a LayerNorm over the O channels (gain g, no bias) and ReLU; O <= 256
	static void conv_ln_relu( const float* in, const float* w, const float* g, float* out, int T, int F, int C, int O, float eps,
		backend::stream_t stream = nullptr );

	// Gemma 4's audio light conv: the GLU of x [T, 2D] (g = x[:, :D] * sigmoid( x[:, D:] )) through a causal depthwise convolution of K taps,
	// out[t][c] = sum_k w[c][k] * g[t - K + 1 + k][c] (zero before the first row), out [T, D]
	static void glu_dwconv( const backend::half_t* x, const backend::half_t* w, backend::half_t* out, int T, int D, int K, backend::stream_t stream = nullptr );

	// Gemma 4's audio attention, per head: row t attends to rows t - window + 1 .. t; score = softcap( q'·k_j * k_scale + q'·rel[t - j] ) with
	// q' = q * q_scale (per dimension), softcap( s ) = cap * tanh( s / cap ); rel [window, heads * head_dim] FP32 (the relative position keys);
	// q / k / v / out [T, heads * head_dim]
	static void local_rel_attention( const backend::half_t* q, const backend::half_t* k, const backend::half_t* v, const float* q_scale, float k_scale,
		const float* rel, backend::half_t* out, int T, int heads, int head_dim, int window, float cap, backend::stream_t stream = nullptr );

	// hidden[m, n] += y + bias (FP32 residual stream)
	static void bias_residual( float* hidden, const backend::half_t* y, const backend::half_t* bias, int m, int n,
		backend::stream_t stream = nullptr );

	// out[m / group, group * n] = LayerNorm( x ) with the rows of each group of `group` consecutive rows concatenated (the patch
	// merger; the norm acts on the n channels of every row)
	static void layer_norm_merge( const float* x, const backend::half_t* w, const backend::half_t* b, backend::half_t* out,
		int m, int n, int group, float eps, backend::stream_t stream = nullptr );
};

} // namespace engine::modules
