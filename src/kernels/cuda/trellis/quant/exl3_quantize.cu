// EXL3 quantization on the GPU (engine/quant/exl3_quantize.h), after exllamav3's exl3_lib/quantize.py (MIT, see LICENSE): finalize_capture_H
// and block_ldl (exl3_input_factor), regularize with the global scale search, ldlq, the back rotation, refit_scales and the packing
// (exl3_quantize_group). Matrices are row-major fp32; the GEMMs are cuBLAS SGEMM, the factorizations cuSOLVER.
#include "engine/quant/exl3_quantize.h"
#include "engine/quant/exl3_encode.h"
#include "engine/modules/trellis_bits.h"
#include "engine/config.h"
#include "engine/core/log.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <vector>

namespace {

using engine::modules::trellis_k2;
using engine::modules::trellis_tile_u16;
using engine::modules::trellis_base_bits;

constexpr int HAD = 128;                       // the rotation's block size, both sides
constexpr float CODEBOOK_SCALE = 1.24371088f;  // RMS the input scales bring every row to, before the global scale
constexpr int SPAN = 128;                      // rows per error-feedback span (16-row blocks inside)

#define EXL3Q_BLAS( x ) \
	do { \
		const cublasStatus_t s_ = ( x ); \
		ENGINE_ASSERT( s_ == CUBLAS_STATUS_SUCCESS, "exl3_quantize: cuBLAS call failed" ); \
	} while ( 0 )
#define EXL3Q_SOLVER( x ) \
	do { \
		const cusolverStatus_t s_ = ( x ); \
		ENGINE_ASSERT( s_ == CUSOLVER_STATUS_SUCCESS, "exl3_quantize: cuSOLVER call failed" ); \
	} while ( 0 )

// Device buffer, freed with its scope
template <typename T>
struct DBuf {
	T* p = nullptr;
	size_t n = 0;
	DBuf() = default;
	explicit DBuf( size_t count ) { alloc( count ); }
	DBuf( const DBuf& ) = delete;
	DBuf& operator=( const DBuf& ) = delete;
	~DBuf() { release(); }
	bool alloc( size_t count ) {
		release();
		if ( !count ) return true;
		if ( cudaMalloc( &p, count * sizeof( T ) ) != cudaSuccess ) {
			(void) cudaGetLastError();
			p = nullptr;
			return false;
		}
		n = count;
		return true;
	}
	void release() {
		if ( p ) CUDA_CHECK( cudaFree( p ) );
		p = nullptr;
		n = 0;
	}
};

// Per device and stream (several workers may share a device, each on its own stream): the library handles, the scaled 128-point Hadamard and
// the factorization status word
struct Ctx {
	cublasHandle_t blas = nullptr;
	cusolverDnHandle_t solver = nullptr;
	float* had = nullptr;
	int* info = nullptr;
	cudaStream_t stream = nullptr;
};

Ctx& ctx( cudaStream_t stream ) {
	static std::mutex mu;
	static std::map<std::pair<int, cudaStream_t>, Ctx> per_stream;
	int dev = 0;
	CUDA_CHECK( cudaGetDevice( &dev ) );
	std::lock_guard<std::mutex> lock( mu );
	Ctx& c = per_stream[{ dev, stream }];
	if ( !c.blas ) {
		EXL3Q_BLAS( cublasCreate( &c.blas ) );
		EXL3Q_SOLVER( cusolverDnCreate( &c.solver ) );
		// Sylvester's construction from the order-1 matrix, as exllamav3 builds every power of two
		std::vector<float> h( 1, 1.0f );
		for ( int d = 1; d < HAD; d *= 2 ) {
			std::vector<float> s( 4 * d * d );
			for ( int r = 0; r < d; ++r ) {
				for ( int col = 0; col < d; ++col ) {
					const float v = h[r * d + col];
					s[r * 2 * d + col] = v;
					s[r * 2 * d + col + d] = v;
					s[( r + d ) * 2 * d + col] = v;
					s[( r + d ) * 2 * d + col + d] = -v;
				}
			}
			h.swap( s );
		}
		const float scale = 1.0f / std::sqrt( static_cast<float>( HAD ) );
		for ( auto& v : h ) v *= scale;
		CUDA_CHECK( cudaMalloc( &c.had, h.size() * sizeof( float ) ) );
		CUDA_CHECK( cudaMemcpy( c.had, h.data(), h.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMalloc( &c.info, sizeof( int ) ) );
	}
	c.stream = stream;
	EXL3Q_BLAS( cublasSetStream( c.blas, stream ) );
	EXL3Q_SOLVER( cusolverDnSetStream( c.solver, stream ) );
	return c;
}

dim3 grid_for( size_t n, int threads = 256 ) {
	const size_t blocks = ( n + threads - 1 ) / threads;
	return dim3( static_cast<unsigned>( blocks < 65535u * 16u ? blocks : 65535u * 16u ) );
}

// ---- kernels -------------------------------------------------------------------------------------------------------------------------------

#define EXL3Q_LOOP( i, n ) for ( size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < ( n ); i += (size_t) gridDim.x * blockDim.x )

__global__ void k_scale( float* x, size_t n, float s ) {
	EXL3Q_LOOP( i, n ) x[i] *= s;
}

__global__ void k_scale_sym( float* H, int k, const float* s ) {
	EXL3Q_LOOP( i, (size_t) k * k ) H[i] *= s[i / k] * s[i % k];
}

__global__ void k_add_diag( float* H, int k, float v ) {
	EXL3Q_LOOP( i, (size_t) k ) H[i * k + i] += v;
}

__global__ void k_get_diag( const float* H, int k, float* d ) {
	EXL3Q_LOOP( i, (size_t) k ) d[i] = H[i * k + i];
}

__global__ void k_floor_diag( float* H, int k, float v ) {
	EXL3Q_LOOP( i, (size_t) k ) H[i * k + i] = fmaxf( H[i * k + i], v );
}

// X[r, c] scaled by v[r] (rows) or v[c] (cols), multiplied or divided
__global__ void k_scale_rows( float* X, int rows, int cols, const float* v, bool divide ) {
	EXL3Q_LOOP( i, (size_t) rows * cols ) {
		const float s = v[i / cols];
		X[i] = divide ? X[i] / s : X[i] * s;
	}
}

__global__ void k_scale_cols( float* X, int rows, int cols, const float* v, bool divide ) {
	EXL3Q_LOOP( i, (size_t) rows * cols ) {
		const float s = v[i % cols];
		X[i] = divide ? X[i] / s : X[i] * s;
	}
}

__global__ void k_sub( float* out, const float* a, const float* b, size_t n ) {
	EXL3Q_LOOP( i, n ) out[i] = a[i] - b[i];
}

__global__ void k_mul( float* a, const float* b, size_t n ) {
	EXL3Q_LOOP( i, n ) a[i] *= b[i];
}

// out[c] = sum over rows of A[r, c] * B[r, c] (B = nullptr: A[r, c]^2), one thread per column
__global__ void k_colsum_prod( const float* A, const float* B, int rows, int cols, double* out ) {
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	if ( c >= cols ) return;
	double s = 0.0;
	for ( int r = 0; r < rows; ++r ) {
		const float a = A[(size_t) r * cols + c];
		s += (double) a * ( B ? B[(size_t) r * cols + c] : a );
	}
	out[c] = s;
}

// out[r] = sum over columns of A[r, c] * B[r, c] (B = nullptr: A^2), one block per row
__global__ void k_rowsum_prod( const float* A, const float* B, int rows, int cols, double* out ) {
	__shared__ double red[256];
	const int r = blockIdx.x;
	double s = 0.0;
	for ( int c = threadIdx.x; c < cols; c += blockDim.x ) {
		const float a = A[(size_t) r * cols + c];
		s += (double) a * ( B ? B[(size_t) r * cols + c] : a );
	}
	red[threadIdx.x] = s;
	__syncthreads();
	for ( int o = blockDim.x / 2; o; o >>= 1 ) {
		if ( (int) threadIdx.x < o ) red[threadIdx.x] += red[threadIdx.x + o];
		__syncthreads();
	}
	if ( threadIdx.x == 0 ) out[r] = red[0];
}

// Inverse of each lower-triangular 16 x 16 diagonal block of L by forward substitution (in double): thread c solves column c
__global__ void k_block_inv16( const float* L, int k, float* inv ) {
	const int j = blockIdx.x, c = threadIdx.x;
	if ( c >= 16 ) return;
	const float* D = L + (size_t) 16 * j * k + 16 * j;
	double x[16];
	for ( int r = 0; r < 16; ++r ) {
		double s = r == c ? 1.0 : 0.0;
		for ( int t = 0; t < r; ++t ) s -= (double) D[(size_t) r * k + t] * x[t];
		x[r] = s / (double) D[(size_t) r * k + r];
	}
	for ( int r = 0; r < 16; ++r ) inv[(size_t) j * 256 + r * 16 + c] = (float) x[r];
}

// The metric of a 16-row block's error under the feedback is C_j C_j^T (C_j: the lower Cholesky factor's diagonal block j): w receives its
// diagonal, the squared norm of each row of C_j, relative to the block's mean (only relative weights inside a tile change the Viterbi's path)
__global__ void k_block_metric( const float* L, int k, float* w ) {
	__shared__ float s[16];
	const int j = blockIdx.x, r = threadIdx.x;
	const float* D = L + (size_t) 16 * j * k + 16 * j;
	float v = 0.0f;
	if ( r < 16 ) {
		for ( int t = 0; t <= r; ++t ) v += D[(size_t) r * k + t] * D[(size_t) r * k + t];
		s[r] = v;
	}
	__syncthreads();
	if ( r < 16 ) {
		float mean = 0.0f;
		for ( int t = 0; t < 16; ++t ) mean += s[t];
		mean /= 16.0f;
		w[16 * j + r] = mean > 0.0f ? fmaxf( v / mean, 1e-3f ) : 1.0f;
	}
}

// open_trellis: start[t] = window 0 of tile t >> d0 (its bits before the stream)
__global__ void k_tile_starts( const uint16_t* codes, int tiles, int d0, uint16_t* start ) {
	EXL3Q_LOOP( t, (size_t) tiles ) start[t] = (uint16_t) ( codes[t * 256] >> d0 );
}

// tile_gain: per tile (one block of 256 threads) the least-squares gain of the decoded tile to its target (weighted by w when given), as an
// 8-bit code g = 1 + (c - 128) / 1024, applied to the decoded tile
__global__ void k_tile_gain( const float* target, float* dec, const float* w, uint8_t* codes ) {
	__shared__ float num[256], den[256];
	const size_t i = (size_t) blockIdx.x * 256 + threadIdx.x;
	const float wi = w ? w[i] : 1.0f, d = dec[i];
	num[threadIdx.x] = wi * target[i] * d;
	den[threadIdx.x] = wi * d * d;
	__syncthreads();
	for ( int o = 128; o > 0; o >>= 1 ) {
		if ( threadIdx.x < o ) {
			num[threadIdx.x] += num[threadIdx.x + o];
			den[threadIdx.x] += den[threadIdx.x + o];
		}
		__syncthreads();
	}
	const float g = den[0] > 0.0f ? num[0] / den[0] : 1.0f;
	const int c = min( 255, max( 0, __float2int_rn( ( g - 1.0f ) * 1024.0f ) + 128 ) );
	if ( threadIdx.x == 0 ) codes[blockIdx.x] = (uint8_t) c;
	dec[i] = d * ( 1.0f + ( c - 128 ) * ( 1.0f / 1024.0f ) );
}

__global__ void k_axpy( float* x, const float* y, float a, size_t n ) {
	EXL3Q_LOOP( i, n ) x[i] += a * y[i];
}

// Held tile gains: a block row's tn tiles (stream order) divided (div) or multiplied by their gains g = ( 896 + c ) / 1024
__global__ void k_tile_fixed_gain( float* x, const uint8_t* codes, int tn, bool div ) {
	EXL3Q_LOOP( i, (size_t) tn * 256 ) {
		const float g = ( 896.0f + codes[i >> 8] ) * ( 1.0f / 1024.0f );
		x[i] = div ? x[i] / g : x[i] * g;
	}
}

// The metric weights of a block row's tiles in stream order: tile t takes those of its factor f[t] (k apart) at rows row .. row + 16
__global__ void k_gather_weights( const float* row_w, const int* f, int k, int row, int tn, bool volta, float* out ) {
	EXL3Q_LOOP( i, (size_t) tn * 256 ) {
		const int t = (int) ( i >> 8 ), p = (int) ( i & 255 );
		int r = 0, c = 0;
		engine::quant::exl3_tile_pos( volta, p, &r, &c );
		out[i] = row_w[(size_t) f[t] * k + row + r];
	}
}

// out[r, 16 j + c] = sum_t L[r, 16 j + t] inv_j[t, c] below the diagonal blocks, 0 on and above them (the diagonal blocks come out as identities,
// which the feedback does not use). Reads only the strict lower block triangle of L
__global__ void k_apply_block_inv( const float* L, int k, const float* inv, float* out ) {
	EXL3Q_LOOP( i, (size_t) k * k ) {
		const int r = (int) ( i / k ), col = (int) ( i % k );
		const int j = col >> 4, c = col & 15;
		if ( r < 16 * ( j + 1 ) ) {
			out[i] = 0.0f;
			continue;
		}
		const float* Lr = L + (size_t) r * k + 16 * j;
		const float* Ij = inv + (size_t) j * 256 + c;
		float s = 0.0f;
		#pragma unroll
		for ( int t = 0; t < 16; ++t ) s += Lr[t] * Ij[t * 16];
		out[i] = s;
	}
}

// Mean square of every 16 x 16 tile of X (rows x cols)
__global__ void k_tile_ms( const float* X, int rows, int cols, float* out ) {
	const int tn = cols / 16;
	EXL3Q_LOOP( t, (size_t) ( rows / 16 ) * tn ) {
		const int kt = (int) ( t / tn ), nt = (int) ( t % tn );
		float s = 0.0f;
		for ( int r = 0; r < 16; ++r ) {
			const float* row = X + (size_t) ( kt * 16 + r ) * cols + nt * 16;
			for ( int c = 0; c < 16; ++c ) s += row[c] * row[c];
		}
		out[t] = s / 256.0f;
	}
}

// Tiles in stream order at the given (tile row, tile column) pairs of X, times s
__global__ void k_gather_tiles( const float* X, int cols, const int2* tiles, int count, bool volta, float s, float* out ) {
	EXL3Q_LOOP( i, (size_t) count * 256 ) {
		const int t = (int) ( i >> 8 ), p = (int) ( i & 255 );
		int r = 0, c = 0;
		engine::quant::exl3_tile_pos( volta, p, &r, &c );
		out[i] = s * X[(size_t) ( tiles[t].x * 16 + r ) * cols + tiles[t].y * 16 + c];
	}
}

// The N / 16 tiles of 16 rows (rows: 16 x N, plus comp when given) in stream order
__global__ void k_gather_block( const float* rows, const float* comp, int N, bool volta, float* tiles ) {
	EXL3Q_LOOP( i, (size_t) N * 16 ) {
		const int t = (int) ( i >> 8 ), p = (int) ( i & 255 );
		int r = 0, c = 0;
		engine::quant::exl3_tile_pos( volta, p, &r, &c );
		const size_t e = (size_t) r * N + t * 16 + c;
		tiles[i] = rows[e] + ( comp ? comp[e] : 0.0f );
	}
}

// Decoded tiles back into the 16 rows of Q, windows into the block row's tile codes
__global__ void k_scatter_block( const float* dec, const uint16_t* win, int N, bool volta, float* q_rows, uint16_t* enc ) {
	EXL3Q_LOOP( i, (size_t) N * 16 ) {
		const int t = (int) ( i >> 8 ), p = (int) ( i & 255 );
		int r = 0, c = 0;
		engine::quant::exl3_tile_pos( volta, p, &r, &c );
		q_rows[(size_t) r * N + t * 16 + c] = dec[i];
		enc[i] = win[i];
	}
}

// The tile codes of columns [c0, c0 + n) out of a tiles_k x (N / 16) x 256 array, contiguous
__global__ void k_slice_codes( const uint16_t* enc, int tiles_k, int tn_all, int t0, int tn, uint16_t* out ) {
	EXL3Q_LOOP( i, (size_t) tiles_k * tn * 256 ) {
		const size_t tile = i >> 8, p = i & 255;
		const size_t kt = tile / tn, nt = tile % tn;
		out[i] = enc[( ( kt * tn_all ) + t0 + nt ) * 256 + p];
	}
}

__global__ void k_to_half( const float* in, uint16_t* out, int n ) {
	EXL3Q_LOOP( i, (size_t) n ) out[i] = __half_as_ushort( __float2half_rn( in[i] ) );
}

// ---- dense algebra --------------------------------------------------------------------------------------------------------------------------

// Row-major C (m x n) = alpha op(A) op(B) + beta C with op(A) m x p, op(B) p x n: the column-major call on the transposes
void gemm( Ctx& c, bool ta, bool tb, int m, int n, int p, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C,
	int ldc ) {
	EXL3Q_BLAS( cublasSgemm( c.blas, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, p, &alpha, B, ldb, A, lda, &beta, C,
		ldc ) );
}

// The same over `count` problems whose operands lie sa / sb / sc elements apart
void gemm_batched( Ctx& c, bool ta, bool tb, int m, int n, int p, float alpha, const float* A, int lda, long long sa, const float* B, int ldb,
	long long sb, float beta, float* C, int ldc, long long sc, int count ) {
	EXL3Q_BLAS( cublasSgemmStridedBatched( c.blas, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, p, &alpha, B, ldb, sb, A,
		lda, sa, &beta, C, ldc, sc, count ) );
}

// Every 128-column block of X (rows x cols) times the Hadamard from the right; tmp holds rows x cols
void had_right( Ctx& c, float* X, float* tmp, int rows, int cols ) {
	const int m = (int) ( (int64_t) rows * cols / HAD );
	gemm( c, false, false, m, HAD, HAD, 1.0f, X, HAD, c.had, HAD, 0.0f, tmp, HAD );
	CUDA_CHECK( cudaMemcpyAsync( X, tmp, (size_t) rows * cols * sizeof( float ), cudaMemcpyDeviceToDevice, c.stream ) );
}

// Every 128-row block of X times the Hadamard from the left
void had_left( Ctx& c, float* X, float* tmp, int rows, int cols ) {
	gemm_batched( c, false, false, HAD, cols, HAD, 1.0f, c.had, HAD, 0, X, cols, (long long) HAD * cols, 0.0f, tmp, cols, (long long) HAD * cols,
		rows / HAD );
	CUDA_CHECK( cudaMemcpyAsync( X, tmp, (size_t) rows * cols * sizeof( float ), cudaMemcpyDeviceToDevice, c.stream ) );
}

int read_info( Ctx& c ) {
	int info = 0;
	CUDA_CHECK( cudaMemcpyAsync( &info, c.info, sizeof( int ), cudaMemcpyDeviceToHost, c.stream ) );
	CUDA_CHECK( cudaStreamSynchronize( c.stream ) );
	return info;
}

// A (k x k, row-major, symmetric) to its Cholesky factor: L with A = L L^T in the lower triangle (the upper one keeps A). The column-major view is
// the same matrix, whose upper factor U (A = U^T U) is L's transpose
bool cholesky( Ctx& c, float* A, int k ) {
	int lwork = 0;
	EXL3Q_SOLVER( cusolverDnSpotrf_bufferSize( c.solver, CUBLAS_FILL_MODE_UPPER, k, A, k, &lwork ) );
	DBuf<float> work( lwork > 0 ? lwork : 1 );
	ENGINE_ASSERT( work.p, "exl3_quantize: no memory for the factorization workspace" );
	EXL3Q_SOLVER( cusolverDnSpotrf( c.solver, CUBLAS_FILL_MODE_UPPER, k, A, k, work.p, lwork, c.info ) );
	return read_info( c ) == 0;
}

// A r = b for symmetric A (k x k), b overwritten by r; A destroyed. Cholesky, then LU when A is not numerically positive definite
bool solve_sym( Ctx& c, float* A, float* b, int k ) {
	DBuf<float> A2( (size_t) k * k );
	ENGINE_ASSERT( A2.p, "exl3_quantize: no memory for the refit system" );
	CUDA_CHECK( cudaMemcpyAsync( A2.p, A, (size_t) k * k * sizeof( float ), cudaMemcpyDeviceToDevice, c.stream ) );
	if ( cholesky( c, A, k ) ) {
		EXL3Q_SOLVER( cusolverDnSpotrs( c.solver, CUBLAS_FILL_MODE_UPPER, k, 1, A, k, b, k, c.info ) );
		return read_info( c ) == 0;
	}
	int lwork = 0;
	EXL3Q_SOLVER( cusolverDnSgetrf_bufferSize( c.solver, k, k, A2.p, k, &lwork ) );
	DBuf<float> work( lwork > 0 ? lwork : 1 );
	DBuf<int> piv( k );
	EXL3Q_SOLVER( cusolverDnSgetrf( c.solver, k, k, A2.p, k, work.p, piv.p, c.info ) );
	if ( read_info( c ) != 0 ) return false;
	// The column-major view is A^T = A
	EXL3Q_SOLVER( cusolverDnSgetrs( c.solver, CUBLAS_OP_N, k, 1, A2.p, k, piv.p, b, k, c.info ) );
	return read_info( c ) == 0;
}

std::vector<float> to_host( const float* d, size_t n, cudaStream_t st ) {
	std::vector<float> h( n );
	CUDA_CHECK( cudaMemcpyAsync( h.data(), d, n * sizeof( float ), cudaMemcpyDeviceToHost, st ) );
	CUDA_CHECK( cudaStreamSynchronize( st ) );
	return h;
}

std::vector<double> to_host( const double* d, size_t n, cudaStream_t st ) {
	std::vector<double> h( n );
	CUDA_CHECK( cudaMemcpyAsync( h.data(), d, n * sizeof( double ), cudaMemcpyDeviceToHost, st ) );
	CUDA_CHECK( cudaStreamSynchronize( st ) );
	return h;
}

void to_device( float* d, const std::vector<float>& h, cudaStream_t st ) {
	CUDA_CHECK( cudaMemcpyAsync( d, h.data(), h.size() * sizeof( float ), cudaMemcpyHostToDevice, st ) );
}

std::vector<float> random_signs( int n, uint64_t seed ) {
	std::mt19937_64 rng( seed );
	std::vector<float> s( n );
	for ( auto& v : s ) v = ( rng() >> 63 ) ? -1.0f : 1.0f;
	return s;
}

double sum_sq( Ctx& c, const float* X, int rows, int cols ) {
	DBuf<double> part( rows );
	k_rowsum_prod<<<rows, 256, 0, c.stream>>>( X, nullptr, rows, cols, part.p );
	const auto h = to_host( part.p, rows, c.stream );
	return std::accumulate( h.begin(), h.end(), 0.0 );
}

double sum_prod( Ctx& c, const float* A, const float* B, int rows, int cols ) {
	DBuf<double> part( rows );
	k_rowsum_prod<<<rows, 256, 0, c.stream>>>( A, B, rows, cols, part.p );
	const auto h = to_host( part.p, rows, c.stream );
	return std::accumulate( h.begin(), h.end(), 0.0 );
}

// ---- regularization and the global scale ----------------------------------------------------------------------------------------------------

// The error feedback inflates what the encoder sees relative to the regularized weight; the scale search runs on the weight before it, so its
// sample is inflated by the expected drift (exllamav3's ldlq_drift, measured on Qwen3-8B)
float ldlq_drift( int code ) {
	switch ( trellis_k2( code ) ) {
		case 2: { return 1.08f; } break;
		case 3: { return 1.035f; } break;
		case 4: { return 1.018f; } break;
		case 5: { return 1.009f; } break;
		case 6: { return 1.004f; } break;
		default: break;
	}
	return 1.0f;
}

// The global scale that minimizes the sampled tile error: a wrapped diagonal of tiles three wide (every tile row and column at least once) plus
// the tiles of highest and lowest RMS; a coarse grid on a third of them, a fine grid around its minimum on all, refined by a parabola. From a
// hint the fine grid goes around it first, and the coarse grid runs only when that grid's minimum is at an edge (scale_search 1: three
// scales on a lighter sample before that; 0: the hint itself)
float scale_search( Ctx& c, const float* W, int k, int n, const engine::quant::Exl3QuantOptions& o, const engine::quant::Exl3EncodeScratch& sc,
	float hint ) {
	const int tk = k / 16, tn = n / 16, width = 3;
	const int diag_len = std::max( tk, tn );
	std::vector<int2> picks;
	for ( int ii = 0; ii < diag_len; ++ii ) {
		for ( int w = 0; w < width; ++w ) picks.push_back( make_int2( ii % tk, ( ii + w ) % tn ) );
	}
	DBuf<float> ms( (size_t) tk * tn );
	k_tile_ms<<<grid_for( (size_t) tk * tn ), 256, 0, c.stream>>>( W, k, n, ms.p );
	const auto h_ms = to_host( ms.p, (size_t) tk * tn, c.stream );
	const int num_tiles = tk * tn;
	int num_x = std::max( 8, diag_len * width / 16 );
	num_x = std::min( num_x, ( num_tiles + 1 ) / 2 );
	std::vector<int> order( num_tiles );
	std::iota( order.begin(), order.end(), 0 );
	std::stable_sort( order.begin(), order.end(), [&]( int a, int b ) { return h_ms[a] > h_ms[b]; } );
	for ( int i = 0; i < num_x; ++i ) picks.push_back( make_int2( order[i] / tn, order[i] % tn ) );
	for ( int i = 0; i < num_x; ++i ) {
		const int t = order[num_tiles - 1 - i];
		picks.push_back( make_int2( t / tn, t % tn ) );
	}
	const int S = static_cast<int>( picks.size() );
	std::vector<int2> subs;
	for ( int i = 0; i < S; i += 3 ) subs.push_back( picks[i] );
	const int Ssub = static_cast<int>( subs.size() );

	const float drift = ldlq_drift( o.code );
	DBuf<int2> d_picks( S ), d_subs( Ssub );
	CUDA_CHECK( cudaMemcpyAsync( d_picks.p, picks.data(), S * sizeof( int2 ), cudaMemcpyHostToDevice, c.stream ) );
	CUDA_CHECK( cudaMemcpyAsync( d_subs.p, subs.data(), Ssub * sizeof( int2 ), cudaMemcpyHostToDevice, c.stream ) );

	// Encode the sample at each scale in one batch; mean squared error of decoded / scale against the (drift-scaled) sample
	auto eval = [&]( const std::vector<float>& scales, const int2* d_list, int count ) {
		const int nsc = static_cast<int>( scales.size() );
		const size_t per = (size_t) count * 256;
		DBuf<float> in( per * nsc ), dec( per * nsc );
		DBuf<uint16_t> win( per * nsc );
		for ( int s = 0; s < nsc; ++s ) {
			k_gather_tiles<<<grid_for( per ), 256, 0, c.stream>>>( W, n, d_list, count, o.volta, drift * scales[s], in.p + per * s );
		}
		engine::quant::exl3_encode_tiles( in.p, dec.p, win.p, count * nsc, o.code, o.cb, sc, c.stream, nullptr, o.open_trellis );
		const auto h_in = to_host( in.p, per * nsc, c.stream );
		const auto h_dec = to_host( dec.p, per * nsc, c.stream );
		std::vector<double> mse( nsc, 0.0 );
		for ( int s = 0; s < nsc; ++s ) {
			double e = 0.0;
			for ( size_t i = 0; i < per; ++i ) {
				const double d = ( h_dec[per * s + i] - h_in[per * s + i] ) / scales[s];
				e += d * d;
			}
			mse[s] = e / per;
		}
		return mse;
	};

	// The fine grid around a centre: its minimum (an index of 0 or 4 is an edge) and the parabola's scale through it and its neighbours
	const float step = 0.075f;
	auto fine_search = [&]( float centre, int& best ) {
		std::vector<float> fine;
		for ( int i = 0; i < 5; ++i ) fine.push_back( centre + step * ( i - 2 ) );
		const auto m2 = eval( fine, d_picks.p, S );
		best = static_cast<int>( std::min_element( m2.begin(), m2.end() ) - m2.begin() );
		double offset = 0.0;
		if ( best > 0 && best < 4 ) {
			const double y0 = m2[best - 1], y1 = m2[best], y2 = m2[best + 1];
			const double denom = y0 - 2.0 * y1 + y2;
			offset = denom > 0.0 ? 0.5 * ( y0 - y2 ) / denom : 0.0;
			offset = std::max( -0.5, std::min( 0.5, offset ) );
		}
		return std::max( fine[best] + static_cast<float>( offset ) * step, 0.01f );
	};
	int best = 0;
	if ( hint >= 0.2f && o.scale_search == 0 ) return hint;
	if ( hint >= 0.2f && o.scale_search == 1 ) {
		// three scales around the hint on the one-wide diagonal and the extreme tiles; the parabola through them unless the minimum is at an edge
		std::vector<int2> light;
		for ( int ii = 0; ii < diag_len; ++ii ) light.push_back( picks[static_cast<size_t>( ii ) * width] );
		light.insert( light.end(), picks.begin() + static_cast<size_t>( diag_len ) * width, picks.end() );
		DBuf<int2> d_light( light.size() );
		CUDA_CHECK( cudaMemcpyAsync( d_light.p, light.data(), light.size() * sizeof( int2 ), cudaMemcpyHostToDevice, c.stream ) );
		const std::vector<float> three = { hint - step, hint, hint + step };
		const auto m3 = eval( three, d_light.p, static_cast<int>( light.size() ) );
		const double denom = m3[0] - 2.0 * m3[1] + m3[2];
		if ( m3[1] <= m3[0] && m3[1] <= m3[2] && denom > 0.0 ) {
			const double offset = std::max( -0.5, std::min( 0.5, 0.5 * ( m3[0] - m3[2] ) / denom ) );
			return std::max( hint + static_cast<float>( offset ) * step, 0.01f );
		}
	}
	if ( hint >= 0.2f ) {
		const float s = fine_search( hint, best );
		if ( best > 0 && best < 4 ) return s;
	}
	std::vector<float> coarse;
	for ( int i = 0; i < 10; ++i ) coarse.push_back( 0.1f + 0.2f * i );
	const auto m1 = eval( coarse, d_subs.p, Ssub );
	return fine_search( coarse[std::min_element( m1.begin(), m1.end() ) - m1.begin()], best );
}

// The weight (k x n, in place) to the encoder's domain: output scales and signs, the right rotation, input scales and signs, the left rotation,
// the global scale (its search from `hint`, see scale_search). su / sv receive the factors the stored suh / svh carry
void regularize( Ctx& c, float* W, float* tmp, int k, int n, const std::vector<float>& su_signs, uint64_t seed,
	const engine::quant::Exl3QuantOptions& o, const engine::quant::Exl3EncodeScratch& sc, float hint, std::vector<float>& su, std::vector<float>& sv,
	float& g_scale ) {
	sv = random_signs( n, seed );
	DBuf<double> acc( std::max( k, n ) );
	DBuf<float> vec( std::max( k, n ) );

	// Output channel RMS, relative to its mean
	k_colsum_prod<<<( n + 255 ) / 256, 256, 0, c.stream>>>( W, nullptr, k, n, acc.p );
	const auto colsq = to_host( acc.p, n, c.stream );
	std::vector<float> out_scales( n );
	double mean = 0.0;
	for ( int i = 0; i < n; ++i ) mean += out_scales[i] = static_cast<float>( std::sqrt( colsq[i] / k ) );
	mean /= n;
	if ( mean > 1e-30 ) {
		for ( auto& v : out_scales ) v = static_cast<float>( v / mean );
	}
	const bool apply_out_scales = o.out_scales != 0;
	std::vector<bool> zero( n );
	for ( int i = 0; i < n; ++i ) zero[i] = std::fabs( out_scales[i] ) < 1e-30f;
	if ( apply_out_scales ) {
		for ( int i = 0; i < n; ++i ) {
			if ( zero[i] ) out_scales[i] = 0.1f;
			sv[i] = sv[i] * out_scales[i] + 1e-10f;
		}
	}
	to_device( vec.p, sv, c.stream );
	k_scale_cols<<<grid_for( (size_t) k * n ), 256, 0, c.stream>>>( W, k, n, vec.p, true );
	for ( int i = 0; i < n; ++i ) {
		if ( zero[i] ) sv[i] = 0.0f;
	}
	had_right( c, W, tmp, k, n );

	// Input channel RMS, brought to the codebook's scale
	k_rowsum_prod<<<k, 256, 0, c.stream>>>( W, nullptr, k, n, acc.p );
	const auto rowsq = to_host( acc.p, k, c.stream );
	su.resize( k );
	for ( int i = 0; i < k; ++i ) {
		float s = static_cast<float>( std::sqrt( rowsq[i] / n ) );
		if ( std::fabs( s ) < 1e-30f ) s = 0.1f;
		su[i] = su_signs[i] * s / -CODEBOOK_SCALE + 1e-10f;
	}
	to_device( vec.p, su, c.stream );
	k_scale_rows<<<grid_for( (size_t) k * n ), 256, 0, c.stream>>>( W, k, n, vec.p, true );
	had_left( c, W, tmp, k, n );

	g_scale = scale_search( c, W, k, n, o, sc, hint );
	k_scale<<<grid_for( (size_t) k * n ), 256, 0, c.stream>>>( W, (size_t) k * n, g_scale );
	for ( auto& v : su ) v /= g_scale;
}

// ---- error feedback -------------------------------------------------------------------------------------------------------------------------

// The error feedback of the columns of a k x N matrix: none, one factor for all of them, or one per column range (the projections side by side,
// each with its own scaled factor, `count` factors k * k apart)
struct Feedback {
	const float* shared = nullptr;
	const float* each = nullptr;
	const float* row_w = nullptr; // the weighted Viterbi metric: k row weights per factor of `each`
	std::vector<int> off, n;      // the column ranges of `each`
	bool any() const { return shared || each; }
	bool uniform() const {
		for ( size_t t = 1; t < n.size(); ++t ) {
			if ( n[t] != n[0] || off[t] != off[t - 1] + n[0] ) return false;
		}
		return !n.empty() && off[0] == 0;
	}
};

// C[:, cols] += L^T B over every column range: rows r0.. of the factors and columns c0.. of them (m rows of C, p rows of B), all ranges at once
// when they are of one width
void feedback_gemm( Ctx& c, const Feedback& fb, int k, int N, int m, int p, int r0, int c0, const float* B, float* C ) {
	if ( fb.shared ) {
		gemm( c, true, false, m, N, p, 1.0f, fb.shared + (size_t) r0 * k + c0, k, B, N, 1.0f, C, N );
		return;
	}
	if ( fb.uniform() ) {
		const int w = fb.n[0];
		gemm_batched( c, true, false, m, w, p, 1.0f, fb.each + (size_t) r0 * k + c0, k, (long long) k * k, B, N, w, 1.0f, C, N, w,
			static_cast<int>( fb.n.size() ) );
		return;
	}
	for ( size_t t = 0; t < fb.n.size(); ++t ) {
		gemm( c, true, false, m, fb.n[t], p, 1.0f, fb.each + t * (size_t) k * k + (size_t) r0 * k + c0, k, B + fb.off[t], N, 1.0f, C + fb.off[t],
			N );
	}
}

// The 16-row blocks of W (k x N) from the last to the first: each one's rows plus the LDL feedback of the error left in the rows after it through
// the encoder, every column of every projection in one launch per block; Q receives the decoded weight, enc the tile codes (k / 16 x N / 16 x 256).
// gains: the tiles' least-squares gains fitted and applied (out); held: each tile encoded at its given gain instead (in)
void ldlq( Ctx& c, const float* W, const Feedback& fb, int k, int N, const engine::quant::Exl3QuantOptions& o,
	const engine::quant::Exl3EncodeScratch& sc, float* Q, uint16_t* enc, uint8_t* gains, const uint8_t* held = nullptr ) {
	const int tn = N / 16;
	const bool feedback = fb.any();
	DBuf<float> prod( feedback ? (size_t) k * N : 0 ), comp( (size_t) 16 * N ), err( (size_t) SPAN * N ), tiles( (size_t) tn * 256 ),
		dec( (size_t) tn * 256 );
	DBuf<uint16_t> win( (size_t) tn * 256 );
	ENGINE_ASSERT( tiles.p && dec.p && win.p && comp.p && err.p && ( !feedback || prod.p ), "exl3_quantize: no memory for the feedback buffers" );
	if ( feedback ) CUDA_CHECK( cudaMemsetAsync( prod.p, 0, (size_t) k * N * sizeof( float ), c.stream ) );
	// The weighted metric: the factor of each tile column, and a block row's weights gathered like its tiles
	const bool weighted = fb.row_w && fb.each;
	DBuf<float> tw( weighted ? (size_t) tn * 256 : 0 );
	DBuf<int> tile_f( weighted ? tn : 0 );
	if ( weighted ) {
		std::vector<int> h( tn, 0 );
		for ( size_t t = 0; t < fb.n.size(); ++t ) {
			for ( int col = fb.off[t] / 16; col < ( fb.off[t] + fb.n[t] ) / 16; ++col ) h[col] = static_cast<int>( t );
		}
		CUDA_CHECK( cudaMemcpyAsync( tile_f.p, h.data(), tn * sizeof( int ), cudaMemcpyHostToDevice, c.stream ) );
	}
	for ( int j = k; j > 0; j -= SPAN ) {
		const int i = j - SPAN;
		for ( int bj = SPAN; bj > 0; bj -= 16 ) {
			const int bi = bj - 16;
			const int row = i + bi;
			const float* cp = nullptr;
			if ( feedback ) {
				// comp = prod[row .. row + 16] + L[i + bj .. j, row .. row + 16]^T (W - Q)[i + bj .. j]
				CUDA_CHECK( cudaMemcpyAsync( comp.p, prod.p + (size_t) row * N, (size_t) 16 * N * sizeof( float ), cudaMemcpyDeviceToDevice,
					c.stream ) );
				const int after = SPAN - bj;
				if ( after > 0 ) {
					const size_t off = (size_t) ( i + bj ) * N;
					k_sub<<<grid_for( (size_t) after * N ), 256, 0, c.stream>>>( err.p, W + off, Q + off, (size_t) after * N );
					feedback_gemm( c, fb, k, N, 16, after, i + bj, row, err.p, comp.p );
				}
				cp = comp.p;
			}
			k_gather_block<<<grid_for( (size_t) N * 16 ), 256, 0, c.stream>>>( W + (size_t) row * N, cp, N, o.volta, tiles.p );
			if ( held ) k_tile_fixed_gain<<<grid_for( (size_t) tn * 256 ), 256, 0, c.stream>>>( tiles.p, held + (size_t) ( row / 16 ) * tn, tn, true );
			if ( weighted ) k_gather_weights<<<grid_for( (size_t) tn * 256 ), 256, 0, c.stream>>>( fb.row_w, tile_f.p, k, row, tn, o.volta, tw.p );
			engine::quant::exl3_encode_tiles( tiles.p, dec.p, win.p, tn, o.code, o.cb, sc, c.stream, weighted ? tw.p : nullptr, o.open_trellis );
			if ( held ) k_tile_fixed_gain<<<grid_for( (size_t) tn * 256 ), 256, 0, c.stream>>>( dec.p, held + (size_t) ( row / 16 ) * tn, tn, false );
			else if ( gains ) k_tile_gain<<<tn, 256, 0, c.stream>>>( tiles.p, dec.p, weighted ? tw.p : nullptr, gains + (size_t) ( row / 16 ) * tn );
			k_scatter_block<<<grid_for( (size_t) N * 16 ), 256, 0, c.stream>>>( dec.p, win.p, N, o.volta, Q + (size_t) row * N,
				enc + (size_t) ( row / 16 ) * tn * 256 );
		}
		// prod[0 .. i] += L[i .. j, 0 .. i]^T (W - Q)[i .. j] (rows from i on are not read again)
		if ( feedback && i > 0 ) {
			const size_t off = (size_t) i * N;
			k_sub<<<grid_for( (size_t) SPAN * N ), 256, 0, c.stream>>>( err.p, W + off, Q + off, (size_t) SPAN * N );
			feedback_gemm( c, fb, k, N, i, SPAN, i, 0, err.p, prod.p );
		}
	}
}

// ---- scale refit ----------------------------------------------------------------------------------------------------------------------------

// tr(E^T H_err E) / tr(W^T H_err W) with E = W - Q and H_err = H - d_off I, from H E and the plain sums
double proxy_error( Ctx& c, const float* W, const float* Q, const float* H, int k, int n, double d_off, double den_err, float* tmp_e,
	float* tmp_he ) {
	k_sub<<<grid_for( (size_t) k * n ), 256, 0, c.stream>>>( tmp_e, W, Q, (size_t) k * n );
	gemm( c, false, false, k, n, k, 1.0f, H, k, tmp_e, n, 0.0f, tmp_he, n );
	const double num = sum_prod( c, tmp_e, tmp_he, k, n ) - d_off * sum_sq( c, tmp_e, k, n );
	return num / std::max( den_err, 1e-30 );
}

// The per-channel scales refitted to the quantized weight Q (original basis, in place) in the Hessian metric, the trellis held fixed: per output
// column the closed form, per input row the linear system ((Q Q^T) o H) r = rowsum(Q o (H W)); two alternating rounds (exllamav3's
// refit_scales). Returns the proxy error after
double refit( Ctx& c, const float* W, float* Q, const float* H, int k, int n, float sigma_reg, std::vector<float>& su, std::vector<float>& sv ) {
	DBuf<float> HW( (size_t) k * n ), HQ( (size_t) k * n ), A( (size_t) k * k ), rhs( k ), vec( std::max( k, n ) ), dg( k );
	DBuf<double> acc( std::max( k, n ) ), acc2( n );
	ENGINE_ASSERT( HW.p && HQ.p && A.p, "exl3_quantize: no memory for the scale refit" );
	gemm( c, false, false, k, n, k, 1.0f, H, k, W, n, 0.0f, HW.p, n );
	const double den = sum_prod( c, W, HW.p, k, n );
	k_get_diag<<<grid_for( k ), 256, 0, c.stream>>>( H, k, dg.p );
	const auto hd = to_host( dg.p, k, c.stream );
	const double hmean = std::accumulate( hd.begin(), hd.end(), 0.0 ) / k;
	const double d_off = sigma_reg / ( 1.0 + sigma_reg ) * hmean;
	const double den_err = den - d_off * sum_sq( c, W, k, n );

	for ( int round = 0; round < 2; ++round ) {
		// Output side
		gemm( c, false, false, k, n, k, 1.0f, H, k, Q, n, 0.0f, HQ.p, n );
		k_colsum_prod<<<( n + 255 ) / 256, 256, 0, c.stream>>>( Q, HW.p, k, n, acc.p );
		k_colsum_prod<<<( n + 255 ) / 256, 256, 0, c.stream>>>( Q, HQ.p, k, n, acc2.p );
		const auto num = to_host( acc.p, n, c.stream );
		const auto dn = to_host( acc2.p, n, c.stream );
		std::vector<float> cs( n );
		for ( int i = 0; i < n; ++i ) cs[i] = dn[i] > 1e-30 ? static_cast<float>( num[i] / dn[i] ) : 1.0f;
		to_device( vec.p, cs, c.stream );
		k_scale_cols<<<grid_for( (size_t) k * n ), 256, 0, c.stream>>>( Q, k, n, vec.p, false );
		for ( int i = 0; i < n; ++i ) sv[i] *= cs[i];

		// Input side
		gemm( c, false, true, k, k, n, 1.0f, Q, n, Q, n, 0.0f, A.p, k );
		k_mul<<<grid_for( (size_t) k * k ), 256, 0, c.stream>>>( A.p, H, (size_t) k * k );
		k_get_diag<<<grid_for( k ), 256, 0, c.stream>>>( A.p, k, dg.p );
		const auto ad = to_host( dg.p, k, c.stream );
		const double amean = std::accumulate( ad.begin(), ad.end(), 0.0 ) / k;
		k_add_diag<<<grid_for( k ), 256, 0, c.stream>>>( A.p, k, static_cast<float>( 1e-6 * amean ) );
		k_rowsum_prod<<<k, 256, 0, c.stream>>>( Q, HW.p, k, n, acc.p );
		const auto b = to_host( acc.p, k, c.stream );
		std::vector<float> r( b.begin(), b.end() );
		to_device( rhs.p, r, c.stream );
		if ( solve_sym( c, A.p, rhs.p, k ) ) {
			r = to_host( rhs.p, k, c.stream );
		} else {
			LOG_WARN( "[exl3_quantize] input scale refit: the system has no solution, input scales kept" );
			std::fill( r.begin(), r.end(), 1.0f );
		}
		for ( auto& v : r ) {
			if ( !std::isfinite( v ) || v <= 0.0f ) v = 1.0f;
		}
		to_device( vec.p, r, c.stream );
		k_scale_rows<<<grid_for( (size_t) k * n ), 256, 0, c.stream>>>( Q, k, n, vec.p, false );
		for ( int i = 0; i < k; ++i ) su[i] *= r[i];
	}
	return proxy_error( c, W, Q, H, k, n, d_off, den_err, HQ.p, HW.p );
}

// ---- factorization --------------------------------------------------------------------------------------------------------------------------

// Floor of a scaled Hessian's diagonal, as a share of its mean (exllamav3 1.6.7+ scaled_ldl): an input channel the weight barely reads (its
// scale orders of magnitude under the rest) takes its share of the damping down with it, and once the Hadamard mixes it with live channels
// that falls below what fp32 resolves, so a matrix positive definite on paper stops factoring. Such channels are held far under the live
// ones; an error left in them does not reach the output either way
constexpr float kScaledDiagFloor = 1e-3f;

// R (k x k, damped) times s on both sides, rotated by the Hadamard on both sides: P S R S P (s = the sign flips, or a projection's full input
// scales: then the metric of its regularized weight, its diagonal held at kScaledDiagFloor of its mean before the rotation)
void rotate_hessian( Ctx& c, const float* H, const float* s, int k, float* R, float* tmp, bool scaled = false ) {
	CUDA_CHECK( cudaMemcpyAsync( R, H, (size_t) k * k * sizeof( float ), cudaMemcpyDeviceToDevice, c.stream ) );
	k_scale_sym<<<grid_for( (size_t) k * k ), 256, 0, c.stream>>>( R, k, s );
	if ( scaled ) {
		DBuf<float> dg( k );
		ENGINE_ASSERT( dg.p, "exl3_quantize: no memory for the diagonal" );
		k_get_diag<<<grid_for( k ), 256, 0, c.stream>>>( R, k, dg.p );
		const auto d = to_host( dg.p, k, c.stream );
		const double mean = std::accumulate( d.begin(), d.end(), 0.0 ) / k;
		k_floor_diag<<<grid_for( k ), 256, 0, c.stream>>>( R, k, static_cast<float>( kScaledDiagFloor * mean ) );
	}
	had_right( c, R, tmp, k, k );
	had_left( c, R, tmp, k, k );
}

// The block-LDL factor of R into L: Cholesky, every column block times the inverse of its diagonal block, zero diagonal blocks (the feedback reads
// below them only); row_w (k, optional) receives the rows' metric weights (k_block_metric). The damping on R grows when the factorization fails
bool block_ldl_factor( Ctx& c, float* R, int k, float sigma_reg, float* L, float* row_w = nullptr ) {
	const size_t kk = (size_t) k * k;
	DBuf<float> tmp( kk ), inv( (size_t) k / 16 * 256 ), dg( k );
	ENGINE_ASSERT( tmp.p && inv.p && dg.p, "exl3_quantize: no memory for the factorization" );
	bool ok = false;
	for ( int attempt = 0; attempt <= 10 && !ok; ++attempt ) {
		CUDA_CHECK( cudaMemcpyAsync( L, R, kk * sizeof( float ), cudaMemcpyDeviceToDevice, c.stream ) );
		ok = cholesky( c, L, k );
		if ( !ok ) {
			k_get_diag<<<grid_for( k ), 256, 0, c.stream>>>( R, k, dg.p );
			const auto rd = to_host( dg.p, k, c.stream );
			const double rmean = std::accumulate( rd.begin(), rd.end(), 0.0 ) / k;
			LOG_WARN( "[exl3_quantize] Cholesky factorization failed, more diagonal damping (attempt %d of 10)", attempt + 1 );
			k_add_diag<<<grid_for( k ), 256, 0, c.stream>>>( R, k, static_cast<float>( 2.0 * sigma_reg * rmean ) );
		}
	}
	if ( !ok ) {
		LOG_ERROR( "[exl3_quantize] Cholesky factorization failed after 10 retries" );
		return false;
	}
	if ( row_w ) k_block_metric<<<k / 16, 16, 0, c.stream>>>( L, k, row_w );
	k_block_inv16<<<k / 16, 16, 0, c.stream>>>( L, k, inv.p );
	k_apply_block_inv<<<grid_for( kk ), 256, 0, c.stream>>>( L, k, inv.p, tmp.p );
	CUDA_CHECK( cudaMemcpyAsync( L, tmp.p, kk * sizeof( float ), cudaMemcpyDeviceToDevice, c.stream ) );
	CUDA_CHECK( cudaStreamSynchronize( c.stream ) );
	return true;
}

} // namespace

namespace engine::quant {

bool exl3_input_factor( Exl3InputFactor& f, const float* h_sum, int k, int64_t rows, float sigma_reg, uint64_t seed, backend::stream_t stream,
	bool shared_ldl ) {
	exl3_input_factor_free( f );
	auto st = static_cast<cudaStream_t>( stream );
	Ctx& c = ctx( st );
	ENGINE_ASSERT( k > 0 && k % HAD == 0, "exl3_input_factor: the input size must be a multiple of 128" );
	f.k = k;
	const auto signs = random_signs( k, seed );
	CUDA_CHECK( cudaMalloc( &f.su_signs, k * sizeof( float ) ) );
	to_device( f.su_signs, signs, st );
	f.fallback = true;
	if ( !h_sum || rows <= 0 ) return true;

	const size_t kk = (size_t) k * k;
	if ( cudaMalloc( &f.H, kk * sizeof( float ) ) != cudaSuccess || ( shared_ldl && cudaMalloc( &f.L, kk * sizeof( float ) ) != cudaSuccess ) ) {
		(void) cudaGetLastError();
		exl3_input_factor_free( f );
		return false;
	}
	// Mean over the rows, damped by sigma_reg times the mean diagonal
	CUDA_CHECK( cudaMemcpyAsync( f.H, h_sum, kk * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
	k_scale<<<grid_for( kk ), 256, 0, st>>>( f.H, kk, static_cast<float>( 1.0 / static_cast<double>( rows ) ) );
	DBuf<float> dg( k );
	k_get_diag<<<grid_for( k ), 256, 0, st>>>( f.H, k, dg.p );
	const auto d = to_host( dg.p, k, st );
	const double diag_mean = std::accumulate( d.begin(), d.end(), 0.0 ) / k;
	if ( !std::isfinite( diag_mean ) || diag_mean < 1e-20 ) {
		if ( !std::isfinite( diag_mean ) ) LOG_WARN( "[exl3_quantize] non-finite Hessian: quantizing without calibration" );
		CUDA_CHECK( cudaFree( f.H ) );
		if ( f.L ) CUDA_CHECK( cudaFree( f.L ) );
		f.H = f.L = nullptr;
		return true;
	}
	k_add_diag<<<grid_for( k ), 256, 0, st>>>( f.H, k, static_cast<float>( sigma_reg * diag_mean ) );

	// The shared factor of P S H S P (sign flips only)
	if ( shared_ldl ) {
		DBuf<float> R( kk ), tmp( kk );
		ENGINE_ASSERT( R.p && tmp.p, "exl3_input_factor: no memory for the factorization" );
		rotate_hessian( c, f.H, f.su_signs, k, R.p, tmp.p );
		if ( !block_ldl_factor( c, R.p, k, sigma_reg, f.L ) ) {
			exl3_input_factor_free( f );
			return false;
		}
	}
	CUDA_CHECK( cudaStreamSynchronize( st ) );
	f.fallback = false;
	return true;
}

void exl3_input_factor_free( Exl3InputFactor& f ) {
	if ( f.H ) CUDA_CHECK( cudaFree( f.H ) );
	if ( f.L ) CUDA_CHECK( cudaFree( f.L ) );
	if ( f.su_signs ) CUDA_CHECK( cudaFree( f.su_signs ) );
	f = Exl3InputFactor{};
}

bool exl3_quantize_group( const Exl3InputFactor& f, const std::vector<const float*>& weights, const std::vector<int>& ns,
	const std::vector<uint64_t>& seeds, const Exl3QuantOptions& o, std::vector<Exl3Tensor>& out, backend::stream_t stream, Exl3QuantTimes* times ) {
	auto st = static_cast<cudaStream_t>( stream );
	Ctx& c = ctx( st );
	// the time since the last mark into *slot (with times)
	auto t_mark = std::chrono::steady_clock::now();
	auto lap = [&]( double Exl3QuantTimes::*slot ) {
		if ( !times ) return;
		CUDA_CHECK( cudaStreamSynchronize( st ) );
		const auto now = std::chrono::steady_clock::now();
		times->*slot += std::chrono::duration<double>( now - t_mark ).count();
		t_mark = now;
	};
	if ( times ) CUDA_CHECK( cudaStreamSynchronize( st ) );
	t_mark = std::chrono::steady_clock::now();
	const int k = f.k;
	const int count = static_cast<int>( weights.size() );
	ENGINE_ASSERT( count > 0 && ns.size() == weights.size() && seeds.size() == weights.size(), "exl3_quantize_group: one size and seed per weight" );
	std::vector<int> off( count + 1, 0 );
	int n_max = 0;
	for ( int b = 0; b < count; ++b ) {
		ENGINE_ASSERT( ns[b] > 0 && ns[b] % HAD == 0, "exl3_quantize_group: output sizes must be multiples of 128" );
		off[b + 1] = off[b] + ns[b];
		n_max = std::max( n_max, ns[b] );
	}
	const int N = off[count];
	const int tiles_k = k / 16, tn_all = N / 16;
	const int words = trellis_tile_u16( o.code );

	Exl3EncodeScratch sc;
	if ( !exl3_encode_scratch_alloc( sc, o.code, o.cb ) ) return false;
	const auto su_signs = to_host( f.su_signs, k, st );

	// Regularize each weight on its own, then side by side into one k x N matrix
	DBuf<float> Wcat( (size_t) k * N ), Qcat( (size_t) k * N ), Wr( (size_t) k * n_max ), tmp( (size_t) k * std::max( n_max, k ) );
	DBuf<uint16_t> enc( (size_t) tiles_k * tn_all * 256 );
	if ( !Wcat.p || !Qcat.p || !Wr.p || !tmp.p || !enc.p ) {
		exl3_encode_scratch_free( sc );
		return false;
	}
	// The feedback: per projection, the factor of its scaled Hessian (made right after its input scales are known), or the input's shared one
	const bool scaled = !f.fallback && o.scaled_hessian;
	ENGINE_ASSERT( f.fallback || scaled || f.L, "exl3_quantize_group: the input factor has no shared block LDL (exl3_input_factor shared_ldl)" );
	const size_t kk = (size_t) k * k;
	// The weighted Viterbi metric needs each projection's own factor
	if ( o.ldl_metric && !scaled && !f.fallback ) LOG_WARN( "[exl3_quantize] the LDL metric needs per-projection factors (scaled_hessian): plain metric" );
	const bool metric = o.ldl_metric && scaled;
	DBuf<float> Ls( scaled ? kk * count : 0 ), R( scaled ? kk : 0 ), Rtmp( scaled ? kk : 0 ), dsu( scaled ? k : 0 ), row_w( metric ? (size_t) k * count : 0 );
	if ( scaled && ( !Ls.p || !R.p || !Rtmp.p || !dsu.p || ( metric && !row_w.p ) ) ) {
		exl3_encode_scratch_free( sc );
		return false;
	}
	Feedback fb;
	if ( scaled ) {
		fb.each = Ls.p;
		fb.row_w = metric ? row_w.p : nullptr;
		fb.off.assign( off.begin(), off.end() - 1 );
		fb.n = ns;
	} else if ( !f.fallback ) {
		fb.shared = f.L;
	}

	std::vector<std::vector<float>> su( count ), sv( count );
	out.assign( count, Exl3Tensor{} );
	for ( int b = 0; b < count; ++b ) {
		const int n = ns[b];
		CUDA_CHECK( cudaMemcpyAsync( Wr.p, weights[b], (size_t) k * n * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
		regularize( c, Wr.p, tmp.p, k, n, su_signs, seeds[b], o, sc, b ? out[b - 1].g_scale : o.scale_hint, su[b], sv[b], out[b].g_scale );
		CUDA_CHECK( cudaMemcpy2DAsync( Wcat.p + off[b], (size_t) N * sizeof( float ), Wr.p, (size_t) n * sizeof( float ), (size_t) n * sizeof( float ),
			k, cudaMemcpyDeviceToDevice, st ) );
		lap( &Exl3QuantTimes::regularize );
		if ( scaled ) {
			// P diag(su) H diag(su) P with the full input scales (the global scale cancels in the block-normalized factor)
			to_device( dsu.p, su[b], st );
			rotate_hessian( c, f.H, dsu.p, k, R.p, Rtmp.p, true );
			if ( !block_ldl_factor( c, R.p, k, o.sigma_reg, Ls.p + kk * b, metric ? row_w.p + (size_t) k * b : nullptr ) ) {
				exl3_encode_scratch_free( sc );
				return false;
			}
			lap( &Exl3QuantTimes::factor );
		}
	}

	DBuf<uint8_t> gains( o.tile_gain ? (size_t) tiles_k * tn_all : 0 );
	ldlq( c, Wcat.p, fb, k, N, o, sc, Qcat.p, enc.p, o.tile_gain ? gains.p : nullptr );
	exl3_encode_scratch_free( sc );
	lap( &Exl3QuantTimes::ldlq );

	DBuf<float> vec( std::max( k, n_max ) ), e( (size_t) k * n_max );
	DBuf<uint16_t> codes( (size_t) tiles_k * ( n_max / 16 ) * 256 ), packed( (size_t) tiles_k * ( n_max / 16 ) * words ), halfs( std::max( k, n_max ) );
	ENGINE_ASSERT( vec.p && e.p && codes.p && packed.p && halfs.p, "exl3_quantize_group: no memory for the outputs" );
	for ( int b = 0; b < count; ++b ) {
		const int n = ns[b], tn = n / 16;
		Exl3Tensor& t = out[b];
		t.k = k;
		t.n = n;
		// Back to the original basis: the left rotation, input scales, the right rotation, output scales
		float* Q = Wr.p;
		CUDA_CHECK( cudaMemcpy2DAsync( Q, (size_t) n * sizeof( float ), Qcat.p + off[b], (size_t) N * sizeof( float ), (size_t) n * sizeof( float ), k,
			cudaMemcpyDeviceToDevice, st ) );
		had_left( c, Q, tmp.p, k, n );
		to_device( vec.p, su[b], st );
		k_scale_rows<<<grid_for( (size_t) k * n ), 256, 0, st>>>( Q, k, n, vec.p, false );
		had_right( c, Q, tmp.p, k, n );
		to_device( vec.p, sv[b], st );
		k_scale_cols<<<grid_for( (size_t) k * n ), 256, 0, st>>>( Q, k, n, vec.p, false );

		if ( !f.fallback && o.refit ) {
			t.proxy_err = refit( c, weights[b], Q, f.H, k, n, o.sigma_reg, su[b], sv[b] );
		} else if ( !f.fallback ) {
			DBuf<float> he( (size_t) k * n ), hw( (size_t) k * n );
			gemm( c, false, false, k, n, k, 1.0f, f.H, k, weights[b], n, 0.0f, hw.p, n );
			DBuf<float> dg( k );
			k_get_diag<<<grid_for( k ), 256, 0, st>>>( f.H, k, dg.p );
			const auto hd = to_host( dg.p, k, st );
			const double d_off = o.sigma_reg / ( 1.0 + o.sigma_reg ) * std::accumulate( hd.begin(), hd.end(), 0.0 ) / k;
			const double den_err = sum_prod( c, weights[b], hw.p, k, n ) - d_off * sum_sq( c, weights[b], k, n );
			t.proxy_err = proxy_error( c, weights[b], Q, f.H, k, n, d_off, den_err, e.p, he.p );
		} else {
			k_sub<<<grid_for( (size_t) k * n ), 256, 0, st>>>( e.p, weights[b], Q, (size_t) k * n );
			t.proxy_err = sum_sq( c, e.p, k, n ) / std::max( sum_sq( c, weights[b], k, n ), 1e-30 );
		}

		// open_trellis: the weight itself (its trellis does not decode to it)
		if ( o.tile_gain ) {
			// This weight's tile columns of the gain codes
			std::vector<uint8_t> all( (size_t) tiles_k * tn_all );
			CUDA_CHECK( cudaMemcpyAsync( all.data(), gains.p, all.size(), cudaMemcpyDeviceToHost, st ) );
			CUDA_CHECK( cudaStreamSynchronize( st ) );
			t.gains.resize( (size_t) tiles_k * tn );
			for ( int r = 0; r < tiles_k; ++r ) {
				std::memcpy( &t.gains[(size_t) r * tn], &all[(size_t) r * tn_all + off[b] / 16], tn );
			}
		}
		// Codes of this weight's tile columns, packed
		k_slice_codes<<<grid_for( (size_t) tiles_k * tn * 256 ), 256, 0, st>>>( enc.p, tiles_k, tn_all, off[b] / 16, tn, codes.p );
		exl3_pack_tiles( packed.p, codes.p, tiles_k * tn, o.code, st );
		// open_trellis: each tile's start state, the bits of its first window before the first code (D(0) = K, or KA at the half rates)
		if ( o.open_trellis ) {
			DBuf<uint16_t> st16( (size_t) tiles_k * tn );
			k_tile_starts<<<grid_for( (size_t) tiles_k * tn ), 256, 0, st>>>( codes.p, tiles_k * tn, trellis_base_bits( o.code ), st16.p );
			t.starts.resize( (size_t) tiles_k * tn );
			CUDA_CHECK( cudaMemcpyAsync( t.starts.data(), st16.p, t.starts.size() * sizeof( uint16_t ), cudaMemcpyDeviceToHost, st ) );
			CUDA_CHECK( cudaStreamSynchronize( st ) );
		}
		t.trellis.resize( (size_t) tiles_k * tn * words );
		CUDA_CHECK( cudaMemcpyAsync( t.trellis.data(), packed.p, t.trellis.size() * sizeof( uint16_t ), cudaMemcpyDeviceToHost, st ) );
		t.suh.resize( k );
		t.svh.resize( n );
		to_device( vec.p, su[b], st );
		k_to_half<<<grid_for( k ), 256, 0, st>>>( vec.p, halfs.p, k );
		CUDA_CHECK( cudaMemcpyAsync( t.suh.data(), halfs.p, k * sizeof( uint16_t ), cudaMemcpyDeviceToHost, st ) );
		CUDA_CHECK( cudaStreamSynchronize( st ) );
		to_device( vec.p, sv[b], st );
		k_to_half<<<grid_for( n ), 256, 0, st>>>( vec.p, halfs.p, n );
		CUDA_CHECK( cudaMemcpyAsync( t.svh.data(), halfs.p, n * sizeof( uint16_t ), cudaMemcpyDeviceToHost, st ) );
		CUDA_CHECK( cudaStreamSynchronize( st ) );
	}
	lap( &Exl3QuantTimes::finish );
	return true;
}

} // namespace engine::quant

namespace {

// A projection's metric P diag( su ) H diag( su ) P (R, k x k) of the damped mean Hessian of h_sum over rows (as exl3_input_factor); tmp: k x k
void projection_metric( Ctx& c, const float* h_sum, int64_t rows, const float* su, int k, float sigma_reg, float* R, float* tmp ) {
	const size_t kk = (size_t) k * k;
	DBuf<float> dg( k );
	CUDA_CHECK( cudaMemcpyAsync( tmp, h_sum, kk * sizeof( float ), cudaMemcpyDeviceToDevice, c.stream ) );
	k_scale<<<grid_for( kk ), 256, 0, c.stream>>>( tmp, kk, static_cast<float>( 1.0 / static_cast<double>( rows ) ) );
	k_get_diag<<<grid_for( k ), 256, 0, c.stream>>>( tmp, k, dg.p );
	const auto d = to_host( dg.p, k, c.stream );
	const double diag_mean = std::accumulate( d.begin(), d.end(), 0.0 ) / k;
	k_add_diag<<<grid_for( k ), 256, 0, c.stream>>>( tmp, k, static_cast<float>( sigma_reg * diag_mean ) );
	DBuf<float> t2( kk );
	ENGINE_ASSERT( t2.p, "exl3_quantize: no memory for the metric" );
	rotate_hessian( c, tmp, su, k, R, t2.p, true );
}

} // namespace

namespace engine::quant {

bool exl3_metric_solve( float* G, const float* T, const float* h_sum, int64_t rows, const float* su, int k, int n, float sigma_reg, double* d_norm2,
	double* t_norm2, backend::stream_t stream ) {
	auto st = static_cast<cudaStream_t>( stream );
	Ctx& c = ctx( st );
	ENGINE_ASSERT( k % HAD == 0 && rows > 0, "exl3_metric_solve: the input size must be a multiple of 128, with rows" );
	const size_t kk = (size_t) k * k;
	DBuf<float> R( kk ), U( kk ), MX( (size_t) k * n );
	if ( !R.p || !U.p || !MX.p ) return false;
	projection_metric( c, h_sum, rows, su, k, sigma_reg, R.p, U.p );
	CUDA_CHECK( cudaMemcpyAsync( U.p, R.p, kk * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
	if ( !cholesky( c, U.p, k ) ) return false;
	// G row-major k x n is G^T column-major (n x k): D^T = G^T M^-1 = G^T U^-1 U^-T with M = U^T U (U: the column-major upper factor)
	const float one = 1.0f;
	EXL3Q_BLAS( cublasStrsm( c.blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, &one, U.p, k, G, n ) );
	EXL3Q_BLAS( cublasStrsm( c.blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n, k, &one, U.p, k, G, n ) );
	if ( d_norm2 ) {
		gemm( c, false, false, k, n, k, 1.0f, R.p, k, G, n, 0.0f, MX.p, n );
		*d_norm2 = sum_prod( c, G, MX.p, k, n );
	}
	if ( t_norm2 ) {
		gemm( c, false, false, k, n, k, 1.0f, R.p, k, T, n, 0.0f, MX.p, n );
		*t_norm2 = sum_prod( c, T, MX.p, k, n );
	}
	CUDA_CHECK( cudaStreamSynchronize( st ) );
	return true;
}

bool exl3_reencode( const float* T, const float* h_sum, int64_t rows, const float* su, const uint8_t* gains, int k, int n, const Exl3QuantOptions& o,
	Exl3Tensor& out, backend::stream_t stream, Exl3ReencodeStep* step ) {
	auto st = static_cast<cudaStream_t>( stream );
	Ctx& c = ctx( st );
	ENGINE_ASSERT( k % HAD == 0 && n % HAD == 0 && rows > 0, "exl3_reencode: sizes must be multiples of 128, with rows" );
	const size_t kk = (size_t) k * k, kn = (size_t) k * n;
	const int tiles_k = k / 16, tn = n / 16, words = trellis_tile_u16( o.code );
	const bool feedback = !step || step->feedback;
	Exl3EncodeScratch sc;
	if ( !exl3_encode_scratch_alloc( sc, o.code, o.cb ) ) return false;
	DBuf<float> R( kk ), tmp( kk ), L( feedback ? kk : 0 ), Q( kn ), row_w( feedback && o.ldl_metric ? k : 0 );
	DBuf<uint16_t> enc( (size_t) tiles_k * tn * 256 ), packed( (size_t) tiles_k * tn * words );
	if ( !R.p || !tmp.p || ( feedback && !L.p ) || !Q.p || !enc.p || !packed.p ) {
		exl3_encode_scratch_free( sc );
		return false;
	}
	// the projection's metric (and with the feedback its factor)
	projection_metric( c, h_sum, rows, su, k, o.sigma_reg, R.p, tmp.p );
	Feedback fb;
	if ( feedback ) {
		DBuf<float> R2( kk );
		ENGINE_ASSERT( R2.p, "exl3_reencode: no memory for the factorization" );
		CUDA_CHECK( cudaMemcpyAsync( R2.p, R.p, kk * sizeof( float ), cudaMemcpyDeviceToDevice, st ) );
		if ( !block_ldl_factor( c, R2.p, k, o.sigma_reg, L.p, o.ldl_metric ? row_w.p : nullptr ) ) {
			exl3_encode_scratch_free( sc );
			return false;
		}
		fb.each = L.p;
		fb.row_w = o.ldl_metric ? row_w.p : nullptr;
		fb.off = { 0 };
		fb.n = { n };
	}
	ldlq( c, T, fb, k, n, o, sc, Q.p, enc.p, nullptr, gains );
	exl3_encode_scratch_free( sc );
	// tr( E^T M E ) / tr( T^T M T ) in the frame
	{
		DBuf<float> he( kn ), ht( kn );
		ENGINE_ASSERT( he.p && ht.p, "exl3_reencode: no memory for the proxy error" );
		gemm( c, false, false, k, n, k, 1.0f, R.p, k, T, n, 0.0f, ht.p, n );
		const double den = sum_prod( c, T, ht.p, k, n );
		out.proxy_err = proxy_error( c, T, Q.p, R.p, k, n, 0.0, den, ht.p, he.p );   // ht reused for E (k x n; tmp is k x k)
		if ( step && step->D ) {
			// dQ = Q - ( T + eta D ), the change from the tiles the target was made from: < G, dQ > and dQ^T M dQ
			k_sub<<<grid_for( kn ), 256, 0, st>>>( ht.p, Q.p, T, kn );
			k_axpy<<<grid_for( kn ), 256, 0, st>>>( ht.p, step->D, -step->eta, kn );
			step->lin = step->G ? sum_prod( c, ht.p, step->G, k, n ) : 0.0;
			gemm( c, false, false, k, n, k, 1.0f, R.p, k, ht.p, n, 0.0f, he.p, n );
			step->quad = sum_prod( c, ht.p, he.p, k, n );
		}
	}
	out.k = k;
	out.n = n;
	exl3_pack_tiles( packed.p, enc.p, tiles_k * tn, o.code, st );
	if ( o.open_trellis ) {
		DBuf<uint16_t> st16( (size_t) tiles_k * tn );
		k_tile_starts<<<grid_for( (size_t) tiles_k * tn ), 256, 0, st>>>( enc.p, tiles_k * tn, trellis_base_bits( o.code ), st16.p );
		out.starts.resize( (size_t) tiles_k * tn );
		CUDA_CHECK( cudaMemcpyAsync( out.starts.data(), st16.p, out.starts.size() * sizeof( uint16_t ), cudaMemcpyDeviceToHost, st ) );
	}
	out.trellis.resize( (size_t) tiles_k * tn * words );
	CUDA_CHECK( cudaMemcpyAsync( out.trellis.data(), packed.p, out.trellis.size() * sizeof( uint16_t ), cudaMemcpyDeviceToHost, st ) );
	CUDA_CHECK( cudaStreamSynchronize( st ) );
	return true;
}

double exl3_proxy_error( const float* W, const float* Q, const float* H, int k, int n, backend::stream_t stream ) {
	auto st = static_cast<cudaStream_t>( stream );
	Ctx& c = ctx( st );
	DBuf<float> e( (size_t) k * n ), he( (size_t) k * n );
	ENGINE_ASSERT( e.p && he.p, "exl3_proxy_error: no memory" );
	gemm( c, false, false, k, n, k, 1.0f, H, k, W, n, 0.0f, he.p, n );
	const double den = sum_prod( c, W, he.p, k, n );
	return proxy_error( c, W, Q, H, k, n, 0.0, den, e.p, he.p );
}

} // namespace engine::quant
