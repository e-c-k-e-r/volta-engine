#pragma once

// Training primitives on the GPU (FP32, row-major, on the caller's stream): the pieces a gradient pass is built from. The forwards here are
// reference forms that keep what their backward needs; a model's own fast forward produces the layer inputs they start from, and a layer is
// recomputed from its input when its backward runs.
#include "engine/backend/types.h"
#include <cstddef>

namespace engine::train {

// cuBLAS on one stream, row-major: C (m x n) = alpha op( A ) op( B ) + beta C, with op( A ) m x p and op( B ) p x n (ta / tb: A / B stored
// transposed)
class Blas {
public:
	explicit Blas( backend::stream_t stream );
	~Blas();
	Blas( const Blas& ) = delete;
	Blas& operator=( const Blas& ) = delete;
	void gemm( bool ta, bool tb, int m, int n, int p, float alpha, const float* A, const float* B, float beta, float* C );
	// The same with explicit row strides (leading dimensions) of A, B and C: views into wider rows, such as one head of a multi-head row
	void gemm( bool ta, bool tb, int m, int n, int p, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc );
	// batch products of the same shapes at element strides sa, sb, sc (dense matrices)
	void gemm_strided( bool ta, bool tb, int m, int n, int p, float alpha, const float* A, long long sa, const float* B, long long sb, float beta, float* C,
		long long sc, int batch );
	// The same two with FP16 A and B on the tensor cores (FP32 accumulation and C, alpha 1)
	void gemm_h( bool ta, bool tb, int m, int n, int p, const void* A, const void* B, float beta, float* C );
	void gemm_h_strided( bool ta, bool tb, int m, int n, int p, const void* A, long long sa, const void* B, long long sb, float beta, float* C, long long sc,
		int batch );
	backend::stream_t stream() const { return stream_; }

private:
	void* handle_ = nullptr;
	backend::stream_t stream_ = nullptr;
};

// a = act( g ) * u, elementwise: SwiGLU's SiLU, or with limit > 0 GLM's silu( min( g, limit ) ) * clamp( u, -limit, limit )
void swiglu_forward( const float* g, const float* u, float* a, size_t n, float limit, backend::stream_t stream );
// dg, du from da at the same g, u
void swiglu_backward( const float* g, const float* u, const float* da, float* dg, float* du, size_t n, float limit, backend::stream_t stream );

// Rows of in [rows, cols] (times col_scale per column when given) as FP16 at a power-of-two scale per row, the row's largest magnitude
// brought to [2^13, 2^14) (so that small gradients keep FP16's precision), and rinv[r] the scale's inverse
void rows_to_half( const float* in, const float* col_scale, int rows, int cols, void* out, float* rinv, backend::stream_t stream );
// x[r, c] *= s[r]
void scale_rows( float* x, const float* s, int rows, int cols, backend::stream_t stream );

// A weight given as a core between per-channel scales, W = diag( su ) T diag( sv ) (k x n, input-major): an EXL3 tensor's trellis core
// between its input and output scales, or any factored linear. The forward keeps Z = ( X diag( su ) ) T for the backward. With T16 (the
// core in FP16) the two products run on the tensor cores: X diag( su ) and dY diag( sv ) rounded to FP16 per row (rows_to_half), FP32
// accumulation; everything else stays FP32
struct ScaledLinear {
	int k = 0, n = 0;
	const float* T = nullptr;   // k x n
	const float* su = nullptr;  // k
	const float* sv = nullptr;  // n
	float* dsu = nullptr;       // k: the backward adds its gradient here (null: not wanted)
	float* dsv = nullptr;       // n: likewise
	const void* T16 = nullptr;  // k x n FP16, instead of T (last: the fields above are given in order by brace initializers)
};
// Y = X W and Z (rows x n each) from X (rows x k); scratch: rows x k floats
void scaled_linear_forward( Blas& b, const ScaledLinear& l, const float* X, int rows, float* Z, float* Y, float* scratch );
// From dY (rows x n): dX (rows x k, null: not wanted; add_dx: added into dX) and the scale gradients; scratch_n: rows x n floats, scratch_k:
// rows x k floats
void scaled_linear_backward( Blas& b, const ScaledLinear& l, const float* X, const float* Z, const float* dY, int rows, float* dX, float* scratch_n,
	float* scratch_k, bool add_dx = false );

// The row-weighted squared error sum_r w_r |y_r - t_r|^2 (w null: weights of 1), returned; dy (null: not wanted) = scale * 2 w_r ( y_r - t_r )
double weighted_sq_error( const float* y, const float* t, const float* w, int rows, int n, float* dy, float scale, backend::stream_t stream );
// sum_r w_r |t_r|^2
double weighted_sq_norm( const float* t, const float* w, int rows, int n, backend::stream_t stream );

// One Adam step on p (n values) from its gradient g; m, v: the moments, zero before the first step; step counts from 1
struct AdamOptions {
	float lr = 1e-3f;
	float beta1 = 0.9f;
	float beta2 = 0.999f;
	float eps = 1e-8f;
};
void adam_step( float* p, const float* g, float* m, float* v, size_t n, int step, const AdamOptions& o, backend::stream_t stream );
// Adam on log-scale parameters, s = s0 e^a: the step on a from g (the gradient with respect to s, times grad_scale: d / da = s d / ds), then
// s from the new a
void adam_log_scales( const float* s0, float* a, float* s, const float* g, float* m, float* v, size_t n, int step, float grad_scale, const AdamOptions& o,
	backend::stream_t stream );

// Rows by index (a routed expert's rows of a batch): dst[i] = src[idx[i]] (FP16 source, half_src; FP32 otherwise), times w[i] when w is given
void gather_rows( const void* src, bool half_src, const int* idx, const float* w, int n, int cols, float* dst, backend::stream_t stream );
// dst[idx[i]] += w[i] * src[i] (w null: 1); idx without repeats within one call
void scatter_add_rows( const float* src, const int* idx, const float* w, int n, int cols, float* dst, backend::stream_t stream );

// Multiplicative parameters: out = base * exp( a ); the gradient of a is out * ( the gradient of out ), mul
void exp_scale( const float* base, const float* a, float* out, size_t n, backend::stream_t stream );
void mul( const float* x, const float* y, float* out, size_t n, backend::stream_t stream );

} // namespace engine::train
