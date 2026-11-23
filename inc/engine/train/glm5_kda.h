#pragma once

// Training forms of GLM-5.3-Flash (glm5_next) operations: kda (FP32 on the caller's stream; see .agents/GLM_STAGEC_PLAN.md)
//
// The KDA (Kimi Delta Attention) linear-attention pieces: the per-head l2 norm of q and k, the gates (beta and the per-channel log decay) from
// the caller's projections, and the gated delta rule with a per-key-channel decay. Each forward keeps what its backward needs; each backward
// adds the input gradients into the caller's buffers (+=) unless it says otherwise. The output's gated norm is qwen4_exp's gated_rms and the
// short convolution qwen4_exp's conv4_silu (dilation 1).
#include "engine/backend/types.h"
#include <cstddef>
#include <cstdint>

namespace engine::train::glm5_next {

// KDA's head dim (keys and values alike; key heads = value heads)
constexpr int kKdaDim = 128;

// l2 norm over segments of n values: y = x * inv * scale with inv = rsqrt( sum( x^2 ) + eps ) (the eps on the sum of squares; scale 1 / sqrt( 128 )
// for q, 1 for k); inv[segments] kept
void l2norm_forward( const float* x, int segments, int n, float eps, float scale, float* inv, float* y, backend::stream_t stream );
// dx += the gradient through the norm at the same x and inv
void l2norm_backward( const float* x, int segments, int n, const float* inv, float scale, const float* dy, float* dx, backend::stream_t stream );

// The gates per row: beta = sigmoid( b ) [rows, heads] from b = b_proj a, and the log decay g [rows, heads * D] = lower * sigmoid( exp( A_log[h] ) f )
// with f = fpre + dt_bias (fpre = f_b( f_a a ) [rows, heads * D], dt_bias [heads * D], A_log [heads], channel j = h * D + i; lower -5: g in
// ( -5, 0 )). The projections are the caller's
void kda_gates_forward( const float* b, const float* fpre, const float* dt_bias, const float* A_log, int rows, int heads, int D, float lower, float* beta,
	float* g, backend::stream_t stream );
// From dbeta [rows, heads] and dg [rows, heads * D]: db and dfpre (overwrite), d_dt_bias [heads * D] and dA_log [heads] (+=, null: not wanted)
void kda_gates_backward( const float* b, const float* fpre, const float* dt_bias, const float* A_log, int rows, int heads, int D, float lower, const float* dbeta,
	const float* dg, float* db, float* dfpre, float* d_dt_bias, float* dA_log, backend::stream_t stream );

// The gated delta rule with a per-key-channel decay, per head with state S [kKdaDim x kKdaDim] (rows: key channel i, columns: value channel c),
// zero at the start of each of the seqs sequences of rows / seqs; per row t:
//   S <- diag( e^g_t ) S, m = S^T k_t, d = beta_t ( v_t - m ), S <- S + k_t d^T, o_t = S^T q_t
// (S_t = ( I - beta k k^T ) diag( e^g ) S_{t-1} + beta k v^T; q already carries its 1 / sqrt( 128 )). q, k, v, g and o [rows, heads, kKdaDim], beta
// [rows, heads], row-major; q, k and g 16-byte aligned. The forward keeps the state before every chunk-th row of a sequence in ckpt
// (kda_ckpt_floats); the backward recomputes each chunk from its checkpoint, last chunk first, and adds into dq, dk, dv, dg, dbeta (scratch:
// kda_scratch_floats)
size_t kda_ckpt_floats( int rows, int heads, int chunk, int seqs = 1 );
size_t kda_scratch_floats( int heads, int chunk, int seqs = 1 );
void kda_forward( const float* q, const float* k, const float* v, const float* g, const float* beta, int rows, int heads, int chunk, float* ckpt, float* o,
	backend::stream_t stream, int seqs = 1 );
void kda_backward( const float* q, const float* k, const float* v, const float* g, const float* beta, int rows, int heads, int chunk, const float* ckpt,
	const float* dout, float* dq, float* dk, float* dv, float* dg, float* dbeta, float* scratch, backend::stream_t stream, int seqs = 1 );

} // namespace engine::train::glm5_next
