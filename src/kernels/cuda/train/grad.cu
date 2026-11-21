// Training primitives (engine/train/grad.h): FP32 elementwise kernels, reductions, the scaled linear on cuBLAS, Adam.
#include "engine/train/grad.h"
#include "engine/config.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>

namespace {

#define TRAIN_BLAS( x ) \
	do { \
		const cublasStatus_t s_ = ( x ); \
		ENGINE_ASSERT( s_ == CUBLAS_STATUS_SUCCESS, "engine::train: cuBLAS call failed" ); \
	} while ( 0 )

#define TRAIN_LOOP( i, n ) for ( size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < ( n ); i += (size_t) gridDim.x * blockDim.x )

unsigned blocks_for( size_t n ) { return static_cast<unsigned>( ( n + 255 ) / 256 < 65535u * 8u ? ( n + 255 ) / 256 : 65535u * 8u ); }

cudaStream_t cu( engine::backend::stream_t s ) { return static_cast<cudaStream_t>( s ); }

__device__ __forceinline__ float sigmoid( float x ) { return 1.0f / ( 1.0f + expf( -x ) ); }

__global__ void k_swiglu_fwd( const float* g, const float* u, float* a, size_t n, float limit ) {
	TRAIN_LOOP( i, n ) {
		float x = g[i], y = u[i];
		if ( limit > 0.0f ) {
			x = fminf( x, limit );
			y = fminf( fmaxf( y, -limit ), limit );
		}
		a[i] = x * sigmoid( x ) * y;
	}
}

__global__ void k_swiglu_bwd( const float* g, const float* u, const float* da, float* dg, float* du, size_t n, float limit ) {
	TRAIN_LOOP( i, n ) {
		float x = g[i], y = u[i];
		float gx = 1.0f, gy = 1.0f;   // derivatives of the clamps
		if ( limit > 0.0f ) {
			if ( x >= limit ) x = limit, gx = 0.0f;
			if ( y >= limit ) y = limit, gy = 0.0f;
			if ( y <= -limit ) y = -limit, gy = 0.0f;
		}
		const float s = sigmoid( x );
		dg[i] = da[i] * y * s * ( 1.0f + x * ( 1.0f - s ) ) * gx;
		du[i] = da[i] * x * s * gy;
	}
}

// out[r, c] = A[r, c] * s[c]
__global__ void k_scale_cols( const float* A, const float* s, int rows, int cols, float* out ) {
	TRAIN_LOOP( i, (size_t) rows * cols ) out[i] = A[i] * s[i % cols];
}

// out[r, c] += A[r, c] * s[c]
__global__ void k_scale_cols_add( const float* A, const float* s, int rows, int cols, float* out ) {
	TRAIN_LOOP( i, (size_t) rows * cols ) out[i] += A[i] * s[i % cols];
}

// One block of 256 threads per row: the row (times cs) as FP16 at 2^e, e bringing its largest magnitude to [2^13, 2^14); rinv = 2^-e
__global__ void k_rows_to_half( const float* in, const float* cs, int cols, __half* out, float* rinv ) {
	__shared__ float red[8];
	const size_t r = blockIdx.x;
	const float* row = in + r * cols;
	float mx = 0.0f;
	for ( int c = threadIdx.x; c < cols; c += blockDim.x ) mx = fmaxf( mx, fabsf( cs ? row[c] * cs[c] : row[c] ) );
	for ( int o = 16; o > 0; o >>= 1 ) mx = fmaxf( mx, __shfl_xor_sync( 0xffffffffu, mx, o ) );
	if ( ( threadIdx.x & 31 ) == 0 ) red[threadIdx.x >> 5] = mx;
	__syncthreads();
	mx = 0.0f;
	for ( int w = 0; w < ( blockDim.x >> 5 ); ++w ) mx = fmaxf( mx, red[w] );
	int e = 0;
	if ( mx > 0.0f && isfinite( mx ) ) {
		int ex = 0;
		frexpf( mx, &ex );   // mx < 2^ex <= 2 mx
		e = min( 120, max( -120, 14 - ex ) );
	}
	const float s = ldexpf( 1.0f, e );
	for ( int c = threadIdx.x; c < cols; c += blockDim.x ) out[r * cols + c] = __float2half_rn( ( cs ? row[c] * cs[c] : row[c] ) * s );
	if ( threadIdx.x == 0 ) rinv[r] = ldexpf( 1.0f, -e );
}

__global__ void k_scale_rows( float* x, const float* s, int rows, int cols ) {
	TRAIN_LOOP( i, (size_t) rows * cols ) x[i] *= s[i / cols];
}

// Z[r, c] *= rinv[r], then Y = Z sv[c]
__global__ void k_unscale_rows_cols( float* Z, const float* rinv, const float* sv, int rows, int cols, float* Y ) {
	TRAIN_LOOP( i, (size_t) rows * cols ) {
		const float z = Z[i] * rinv[i / cols];
		Z[i] = z;
		Y[i] = z * sv[i % cols];
	}
}

// acc[c] += sum_r A[r, c] B[r, c]: a block of 256 columns x 256 rows per block, atomically
__global__ void k_colsum_prod( const float* A, const float* B, int rows, int cols, float* acc ) {
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	if ( c >= cols ) return;
	const int r0 = blockIdx.y * 256, r1 = min( rows, r0 + 256 );
	float s = 0.0f;
	for ( int r = r0; r < r1; ++r ) s += A[(size_t) r * cols + c] * B[(size_t) r * cols + c];
	atomicAdd( &acc[c], s );
}

void colsum_prod( const float* A, const float* B, int rows, int cols, float* acc, cudaStream_t st ) {
	const dim3 grid( ( cols + 255 ) / 256, ( rows + 255 ) / 256 );
	k_colsum_prod<<<grid, 256, 0, st>>>( A, B, rows, cols, acc );
}

// Per row r: w_r |y_r - t_r|^2 into part[r]; dy = scale * 2 w_r ( y - t )
__global__ void k_wsq( const float* y, const float* t, const float* w, int n, float* dy, float scale, float* part ) {
	const int r = blockIdx.x;
	const float wr = w ? w[r] : 1.0f;
	float s = 0.0f;
	for ( int c = threadIdx.x; c < n; c += blockDim.x ) {
		const size_t i = (size_t) r * n + c;
		const float d = y ? y[i] - t[i] : t[i];
		s += d * d;
		if ( dy ) dy[i] = scale * 2.0f * wr * d;
	}
	__shared__ float red[256];
	red[threadIdx.x] = s;
	__syncthreads();
	for ( int o = blockDim.x / 2; o > 0; o >>= 1 ) {
		if ( threadIdx.x < o ) red[threadIdx.x] += red[threadIdx.x + o];
		__syncthreads();
	}
	if ( threadIdx.x == 0 ) part[r] = wr * red[0];
}

__global__ void k_adam( float* p, const float* g, float* m, float* v, size_t n, float lr, float b1, float b2, float eps, float c1, float c2 ) {
	TRAIN_LOOP( i, n ) {
		const float gi = g[i];
		const float mi = b1 * m[i] + ( 1.0f - b1 ) * gi;
		const float vi = b2 * v[i] + ( 1.0f - b2 ) * gi * gi;
		m[i] = mi;
		v[i] = vi;
		p[i] -= lr * ( mi / c1 ) / ( sqrtf( vi / c2 ) + eps );
	}
}

__global__ void k_adam_log( const float* s0, float* a, float* s, const float* g, float* m, float* v, size_t n, float gs, float lr, float b1, float b2, float eps,
	float c1, float c2 ) {
	TRAIN_LOOP( i, n ) {
		const float gi = g[i] * s[i] * gs;
		const float mi = b1 * m[i] + ( 1.0f - b1 ) * gi;
		const float vi = b2 * v[i] + ( 1.0f - b2 ) * gi * gi;
		m[i] = mi;
		v[i] = vi;
		const float ai = a[i] - lr * ( mi / c1 ) / ( sqrtf( vi / c2 ) + eps );
		a[i] = ai;
		s[i] = s0[i] * expf( ai );
	}
}

__global__ void k_adam_add( const float* w0, float* a, float* w, const float* g, float* m, float* v, size_t n, float gs, float lr, float b1, float b2, float eps,
	float c1, float c2 ) {
	TRAIN_LOOP( i, n ) {
		const float gi = g[i] * gs;
		const float mi = b1 * m[i] + ( 1.0f - b1 ) * gi;
		const float vi = b2 * v[i] + ( 1.0f - b2 ) * gi * gi;
		m[i] = mi;
		v[i] = vi;
		const float ai = a[i] - lr * ( mi / c1 ) / ( sqrtf( vi / c2 ) + eps );
		a[i] = ai;
		w[i] = w0[i] + ai;
	}
}

__global__ void k_exp_scale( const float* base, const float* a, float* out, size_t n ) {
	TRAIN_LOOP( i, n ) out[i] = base[i] * expf( a[i] );
}

__global__ void k_gather_rows( const void* src, bool half_src, const int* idx, const float* w, int n, int cols, float* dst ) {
	TRAIN_LOOP( i, (size_t) n * cols ) {
		const size_t r = i / cols, c = i % cols;
		const size_t s = (size_t) idx[r] * cols + c;
		const float v = half_src ? __half2float( static_cast<const __half*>( src )[s] ) : static_cast<const float*>( src )[s];
		dst[i] = w ? v * w[r] : v;
	}
}

__global__ void k_scatter_add_rows( const float* src, const int* idx, const float* w, int n, int cols, float* dst ) {
	TRAIN_LOOP( i, (size_t) n * cols ) {
		const size_t r = i / cols;
		dst[(size_t) idx[r] * cols + i % cols] += w ? w[r] * src[i] : src[i];
	}
}

__global__ void k_mul( const float* x, const float* y, float* out, size_t n ) {
	TRAIN_LOOP( i, n ) out[i] = x[i] * y[i];
}

double row_sum( const float* y, const float* t, const float* w, int rows, int n, float* dy, float scale, cudaStream_t st ) {
	float* part = nullptr;
	CUDA_CHECK( cudaMallocAsync( &part, static_cast<size_t>( rows ) * sizeof( float ), st ) );
	k_wsq<<<rows, 256, 0, st>>>( y, t, w, n, dy, scale, part );
	std::vector<float> h( rows );
	CUDA_CHECK( cudaMemcpyAsync( h.data(), part, h.size() * sizeof( float ), cudaMemcpyDeviceToHost, st ) );
	CUDA_CHECK( cudaFreeAsync( part, st ) );
	CUDA_CHECK( cudaStreamSynchronize( st ) );
	double s = 0.0;
	for ( float v : h ) s += v;
	return s;
}

} // namespace

namespace engine::train {

Blas::Blas( backend::stream_t stream ) : stream_( stream ) {
	cublasHandle_t h = nullptr;
	TRAIN_BLAS( cublasCreate( &h ) );
	TRAIN_BLAS( cublasSetStream( h, cu( stream ) ) );
	handle_ = h;
}

Blas::~Blas() {
	if ( handle_ ) cublasDestroy( static_cast<cublasHandle_t>( handle_ ) );
}

void Blas::gemm( bool ta, bool tb, int m, int n, int p, float alpha, const float* A, const float* B, float beta, float* C ) {
	gemm( ta, tb, m, n, p, alpha, A, ta ? m : p, B, tb ? p : n, beta, C, n );
}

void Blas::gemm( bool ta, bool tb, int m, int n, int p, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc ) {
	// Row-major C = op( A ) op( B ) is column-major C^T = op( B )^T op( A )^T on the same storage
	TRAIN_BLAS( cublasSgemm( static_cast<cublasHandle_t>( handle_ ), tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, p, &alpha,
		B, ldb, A, lda, &beta, C, ldc ) );
}

void Blas::gemm_strided( bool ta, bool tb, int m, int n, int p, float alpha, const float* A, long long sa, const float* B, long long sb, float beta, float* C,
	long long sc, int batch ) {
	if ( batch <= 0 || m <= 0 ) return;
	TRAIN_BLAS( cublasSgemmStridedBatched( static_cast<cublasHandle_t>( handle_ ), tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, p, &alpha,
		B, tb ? p : n, sb, A, ta ? m : p, sa, &beta, C, n, sc, batch ) );
}

void Blas::gemm_h( bool ta, bool tb, int m, int n, int p, const void* A, const void* B, float beta, float* C ) {
	const float one = 1.0f;
	TRAIN_BLAS( cublasGemmEx( static_cast<cublasHandle_t>( handle_ ), tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, p, &one, B, CUDA_R_16F,
		tb ? p : n, A, CUDA_R_16F, ta ? m : p, &beta, C, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP ) );
}

void Blas::gemm_h_strided( bool ta, bool tb, int m, int n, int p, const void* A, long long sa, const void* B, long long sb, float beta, float* C, long long sc,
	int batch ) {
	if ( batch <= 0 || m <= 0 ) return;
	const float one = 1.0f;
	TRAIN_BLAS( cublasGemmStridedBatchedEx( static_cast<cublasHandle_t>( handle_ ), tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, p, &one, B,
		CUDA_R_16F, tb ? p : n, sb, A, CUDA_R_16F, ta ? m : p, sa, &beta, C, CUDA_R_32F, n, sc, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP ) );
}

void rows_to_half( const float* in, const float* col_scale, int rows, int cols, void* out, float* rinv, backend::stream_t stream ) {
	if ( rows > 0 ) k_rows_to_half<<<rows, 256, 0, cu( stream )>>>( in, col_scale, cols, static_cast<__half*>( out ), rinv );
}

void scale_rows( float* x, const float* s, int rows, int cols, backend::stream_t stream ) {
	if ( rows > 0 ) k_scale_rows<<<blocks_for( (size_t) rows * cols ), 256, 0, cu( stream )>>>( x, s, rows, cols );
}

void swiglu_forward( const float* g, const float* u, float* a, size_t n, float limit, backend::stream_t stream ) {
	k_swiglu_fwd<<<blocks_for( n ), 256, 0, cu( stream )>>>( g, u, a, n, limit );
}

void swiglu_backward( const float* g, const float* u, const float* da, float* dg, float* du, size_t n, float limit, backend::stream_t stream ) {
	k_swiglu_bwd<<<blocks_for( n ), 256, 0, cu( stream )>>>( g, u, da, dg, du, n, limit );
}

// A scratch of rows * cols floats as an FP16 [rows, cols] block and rows floats after it
static float* half_rows_rinv( float* scratch, int rows, int cols ) {
	return scratch + ( static_cast<size_t>( rows ) * cols * 2 + 15 ) / 16 * 4;
}

void scaled_linear_forward( Blas& b, const ScaledLinear& l, const float* X, int rows, float* Z, float* Y, float* scratch ) {
	const auto st = cu( b.stream() );
	if ( l.T16 ) {
		float* rinv = half_rows_rinv( scratch, rows, l.k );
		rows_to_half( X, l.su, rows, l.k, scratch, rinv, b.stream() );
		b.gemm_h( false, false, rows, l.n, l.k, scratch, l.T16, 0.0f, Z );
		k_unscale_rows_cols<<<blocks_for( (size_t) rows * l.n ), 256, 0, st>>>( Z, rinv, l.sv, rows, l.n, Y );
		return;
	}
	k_scale_cols<<<blocks_for( (size_t) rows * l.k ), 256, 0, st>>>( X, l.su, rows, l.k, scratch );
	b.gemm( false, false, rows, l.n, l.k, 1.0f, scratch, l.T, 0.0f, Z );
	k_scale_cols<<<blocks_for( (size_t) rows * l.n ), 256, 0, st>>>( Z, l.sv, rows, l.n, Y );
}

void scaled_linear_backward( Blas& b, const ScaledLinear& l, const float* X, const float* Z, const float* dY, int rows, float* dX, float* scratch_n,
	float* scratch_k, bool add_dx ) {
	const auto st = cu( b.stream() );
	// dsv = colsum( dY o Z ); dXs = ( dY diag( sv ) ) T^T; dsu = colsum( dXs o X ); dX = dXs diag( su )
	if ( l.dsv ) colsum_prod( dY, Z, rows, l.n, l.dsv, st );
	if ( l.T16 ) {
		float* rinv = half_rows_rinv( scratch_n, rows, l.n );
		rows_to_half( dY, l.sv, rows, l.n, scratch_n, rinv, b.stream() );
		b.gemm_h( false, true, rows, l.k, l.n, scratch_n, l.T16, 0.0f, scratch_k );
		k_scale_rows<<<blocks_for( (size_t) rows * l.k ), 256, 0, st>>>( scratch_k, rinv, rows, l.k );
	}
	else {
		k_scale_cols<<<blocks_for( (size_t) rows * l.n ), 256, 0, st>>>( dY, l.sv, rows, l.n, scratch_n );
		b.gemm( false, true, rows, l.k, l.n, 1.0f, scratch_n, l.T, 0.0f, scratch_k );
	}
	if ( l.dsu ) colsum_prod( scratch_k, X, rows, l.k, l.dsu, st );
	if ( dX && add_dx ) k_scale_cols_add<<<blocks_for( (size_t) rows * l.k ), 256, 0, st>>>( scratch_k, l.su, rows, l.k, dX );
	else if ( dX ) k_scale_cols<<<blocks_for( (size_t) rows * l.k ), 256, 0, st>>>( scratch_k, l.su, rows, l.k, dX );
}

double weighted_sq_error( const float* y, const float* t, const float* w, int rows, int n, float* dy, float scale, backend::stream_t stream ) {
	return row_sum( y, t, w, rows, n, dy, scale, cu( stream ) );
}

double weighted_sq_norm( const float* t, const float* w, int rows, int n, backend::stream_t stream ) {
	return row_sum( nullptr, t, w, rows, n, nullptr, 0.0f, cu( stream ) );
}

void adam_step( float* p, const float* g, float* m, float* v, size_t n, int step, const AdamOptions& o, backend::stream_t stream ) {
	const float c1 = 1.0f - std::pow( o.beta1, static_cast<float>( step ) ), c2 = 1.0f - std::pow( o.beta2, static_cast<float>( step ) );
	k_adam<<<blocks_for( n ), 256, 0, cu( stream )>>>( p, g, m, v, n, o.lr, o.beta1, o.beta2, o.eps, c1, c2 );
}

void adam_log_scales( const float* s0, float* a, float* s, const float* g, float* m, float* v, size_t n, int step, float grad_scale, const AdamOptions& o,
	backend::stream_t stream ) {
	const float c1 = 1.0f - std::pow( o.beta1, static_cast<float>( step ) ), c2 = 1.0f - std::pow( o.beta2, static_cast<float>( step ) );
	if ( n > 0 ) k_adam_log<<<blocks_for( n ), 256, 0, cu( stream )>>>( s0, a, s, g, m, v, n, grad_scale, o.lr, o.beta1, o.beta2, o.eps, c1, c2 );
}

void adam_deltas( const float* w0, float* a, float* w, const float* g, float* m, float* v, size_t n, int step, float grad_scale, const AdamOptions& o,
	backend::stream_t stream ) {
	const float c1 = 1.0f - std::pow( o.beta1, static_cast<float>( step ) ), c2 = 1.0f - std::pow( o.beta2, static_cast<float>( step ) );
	if ( n > 0 ) k_adam_add<<<blocks_for( n ), 256, 0, cu( stream )>>>( w0, a, w, g, m, v, n, grad_scale, o.lr, o.beta1, o.beta2, o.eps, c1, c2 );
}

void gather_rows( const void* src, bool half_src, const int* idx, const float* w, int n, int cols, float* dst, backend::stream_t stream ) {
	if ( n > 0 ) k_gather_rows<<<blocks_for( (size_t) n * cols ), 256, 0, cu( stream )>>>( src, half_src, idx, w, n, cols, dst );
}

void scatter_add_rows( const float* src, const int* idx, const float* w, int n, int cols, float* dst, backend::stream_t stream ) {
	if ( n > 0 ) k_scatter_add_rows<<<blocks_for( (size_t) n * cols ), 256, 0, cu( stream )>>>( src, idx, w, n, cols, dst );
}

void exp_scale( const float* base, const float* a, float* out, size_t n, backend::stream_t stream ) {
	k_exp_scale<<<blocks_for( n ), 256, 0, cu( stream )>>>( base, a, out, n );
}

void mul( const float* x, const float* y, float* out, size_t n, backend::stream_t stream ) {
	k_mul<<<blocks_for( n ), 256, 0, cu( stream )>>>( x, y, out, n );
}

} // namespace engine::train
