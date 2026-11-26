#pragma once

#include "engine/backend/types.h"
#include <cstddef>
#include <cstdint>

namespace engine::modules {

// Operators of the vall-e TTS model (src/tts: the backbone, the flow head and its guided sampling). All pointers are device pointers;
// the residual stream and the flow head's state are FP32, the GEMM operands FP16, FSQ digits bytes. Every call is ordered on `stream`.
struct TtsOps {
	// C [M, N] = A [M, K] @ W [N, K]^T (row-major, FP32 accumulation), FP16 or (c32) FP32 out
	static void gemm( const backend::half_t* A, const backend::half_t* W, void* C, int M, int N, int K, bool c32, backend::stream_t stream = nullptr );

	// FSQ digits (level indices; groups of 4 scalars at [8, 5, 5, 5] levels) -> values in [-1, 1], n of them
	static void digits_values( const uint8_t* digits, backend::half_t* vals, int n, backend::stream_t stream = nullptr );

	// The input embedding of L positions (vall_e/model.py V3.hidden): the text embedding text_emb [., d] at ids where kind is 1, else the audio
	// projection a [L, d]; the mask embedding at masked positions, the length query len_q (may be null) where kind is 5; plus the segment
	// embedding of kind - 1 (clamped to [0, 2]) and the row's mode embedding mode_vec [d]. x is [L, d]
	static void embed_mix( float* x, const float* a, const backend::half_t* text_emb, const int* ids, const uint8_t* kind, const uint8_t* masked,
		const float* mask_emb, const float* len_q, const float* segment, const float* mode_vec, int L, int d, backend::stream_t stream = nullptr );

	// The rotary angles' cos / sin of positions 0..n-1 and pairs 0..rot/2-1, [n, rot / 2] each (rotate-half pairs, base theta)
	static void rope_table( float* cs, float* sn, int n, int rot, float theta, backend::stream_t stream = nullptr );

	// x [rows, d] += branch (when given; x keeps the sum), then RMSNorm( x ) * w into out16 (FP16) or out32 (FP32): one of them
	static void add_rms_norm( float* x, const float* branch, const float* w, int rows, int d, float eps, backend::half_t* out16, float* out32,
		backend::stream_t stream = nullptr );

	// y [rows, cols] += bias [cols]
	static void add_bias( float* y, const float* bias, int rows, int cols, backend::stream_t stream = nullptr );

	// The layout qk_norm_rope writes and attention / heads_gate read: [H, L, hd] (true) or [L, H, hd]
	static bool head_major( int head_dim );
	// Device bytes attention needs as scratch for rows of up to max_len positions (0: none)
	static size_t attention_scratch( int heads, int head_dim, int max_len );

	// qkv [L, 3, H, hd] -> q, k, v (layout as head_major): q and k RMS-normalised per head (gains qn / kn, may be null; eps) and rotated
	// (rotate-half RoPE on the first rot dims, the angles of rope_table at each position's place in its row, pos [L]), v copied. hd a
	// multiple of 32 up to 128
	static void qk_norm_rope( const backend::half_t* qkv, const float* qn, const float* kn, backend::half_t* q, backend::half_t* k, backend::half_t* v,
		const int* pos, int L, int H, int hd, int rot, const float* cs, const float* sn, float eps, bool head_major, backend::stream_t stream = nullptr );

	// Bidirectional softmax attention (scale) within one row: positions [s, s + Lr) of a pass of L, every query against every key of the row;
	// q, k, v and o in the head_major layout of the whole pass; scratch as attention_scratch
	static void attention( const backend::half_t* q, const backend::half_t* k, const backend::half_t* v, backend::half_t* o, int s, int Lr, int L, int H, int hd,
		float scale, void* scratch, size_t scratch_bytes, backend::stream_t stream = nullptr );

	// o (head_major layout) -> y [L, H * hd], times sigmoid( g [L, H * hd] ) when g is given (the attention's output gate)
	static void heads_gate( const backend::half_t* o, const backend::half_t* g, backend::half_t* y, int L, int H, int hd, bool head_major,
		backend::stream_t stream = nullptr );

	// up [rows, 2F] (a | b) -> silu( a ) * b [rows, F]
	static void swiglu( const backend::half_t* up, backend::half_t* act, int rows, int F, backend::stream_t stream = nullptr );

	// dst [n, d] (FP16) = src rows offset + pos[i] (FP32)
	static void gather_rows( const float* src, const int* pos, int offset, backend::half_t* dst, int n, int d, backend::stream_t stream = nullptr );

	// timestep_embedding( t, 256 ): [cos( 1000 t f_i ) | sin( ... )], f_i = exp( -ln( 1e4 ) i / 128 ), FP16 [256]
	static void time_embedding( float t, backend::half_t* out, backend::stream_t stream = nullptr );

	// y [rows, cols] = x + bias (may be null), through SiLU when silu, as FP16
	static void bias_act( const float* x, const float* bias, backend::half_t* y, int rows, int cols, bool silu, backend::stream_t stream = nullptr );

	// sc [rows, W] = silu( cond + temb ) as FP16 (temb [W] broadcast over the rows)
	static void cond_silu( const float* cond, const float* temb, backend::half_t* sc, int rows, int W, backend::stream_t stream = nullptr );

	// out [rows, W] = LayerNorm( x ) (no affine, eps 1e-6) * ( 1 + scale ) + shift as FP16; mod rows of mstride, shift at 0, scale at W. W <= 2048
	static void ln_mod( const float* x, const float* mod, int mstride, backend::half_t* out, int rows, int W, backend::stream_t stream = nullptr );

	// x [rows, W] += gate * ( branch + bias ), the gate at mod[row * mstride + 2 W + c]
	static void gate_add( float* x, const float* branch, const float* bias, const float* mod, int mstride, int rows, int W, backend::stream_t stream = nullptr );

	// y [R, n] (FP16) = x [n] for each of R rows
	static void replicate( const float* x, backend::half_t* y, int n, int R, backend::stream_t stream = nullptr );

	// One guided Euler step: v [R, n] (the velocities of the R guidance rows; + bias [V], V the period) -> the guided velocity (cond; with R >= 2
	// null + ( cond - null ) cfg; with R == 3 also + ( cond - promptless ) cfg_prompt), x [n] += it * dt, and vt (may be null) = it
	static void guide_step( const float* v, const float* bias, float* x, float* vt, int n, int V, int R, float cfg, float cfg_prompt, float dt,
		backend::stream_t stream = nullptr );

	// values -> FSQ level indices (round half to even), levels [8, 5, 5, 5] per group of 4
	static void snap( const float* x, uint8_t* out, int n, backend::stream_t stream = nullptr );
};

// Operators of the NeMo codec (src/tts/codec.cpp), FP32 throughout; activations are channel-major [C, T].
struct CodecOps {
	// Conv1d: y [co, T_out] = w [co, ci * k] times the columns of x [ci, T] (tap j of channel c at t stride + j dil - pad, reflected at the ends)
	// + b [co]; the GEMMs go over chunks of time whose columns fit `cols` (chunk floats)
	static void conv( const float* x, const float* w, const float* b, float* y, int ci, int co, int k, int dil, int stride, int pad, int T, int T_out,
		float* cols, size_t chunk, backend::stream_t stream = nullptr );

	// The transposed conv (stride s, kernel 2s) as regrouped at load, w [s * co, 3 * ci] against the input at q - 1, q, q + 1 (zeros past the
	// ends): y [co, T_in * s] (+ b [co]), output q s + r from phase r; cols and ph are chunk floats each
	static void up_conv( const float* x, const float* w, const float* b, float* y, int ci, int co, int s, int T_in, float* cols, float* ph, size_t chunk,
		backend::stream_t stream = nullptr );

	// HalfSnake: x + sin( a x )^2 / ( a + 1e-9 ) on the first C / 2 channels (alpha [C / 2]), leaky ReLU (0.01) on the rest; alpha null: leaky
	// ReLU everywhere. In place when y == x
	static void act( const float* x, float* y, const float* alpha, int C, int T, backend::stream_t stream = nullptr );

	// y = a y + b x over n values
	static void axpy( float* y, const float* x, float a, float b, size_t n, backend::stream_t stream = nullptr );

	// y = clamp( y, -1, 1 )
	static void clamp( float* y, size_t n, backend::stream_t stream = nullptr );

	// digits [T, 32] (level indices) -> x [32, T] = ( index - half ) / half, half = levels / 2 of the scalar's place in its group of 4
	static void dequant( const uint8_t* digits, float* x, int T, const int* levels, backend::stream_t stream = nullptr );

	// FiniteScalarQuantizer.encode: x [32, T] -> digits [T, 32] (levels per place in a group of 4, eps), rounding half to even
	static void quant( const float* x, uint8_t* digits, int T, const int* levels, float eps, backend::stream_t stream = nullptr );
};

} // namespace engine::modules
