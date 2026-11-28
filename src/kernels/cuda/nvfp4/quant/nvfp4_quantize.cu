// NVFP4 quantization on the GPU (engine/quant/nvfp4_quantize.h): the damped mean Hessian's block LDL (exllamav3's block_ldl, in double), the
// 16-row blocks quantized from the last to the first with the error feedback (exllamav3's ldlq with an NVFP4 block quantizer in place of the
// trellis), the proxy errors and the modelopt packing. Matrices are row-major; the GEMMs are cuBLAS SGEMM, the factorization cuSOLVER.
#include "engine/quant/nvfp4_quantize.h"
#include "engine/config.h"
#include "engine/core/log.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <numeric>
#include <vector>

namespace {

#define NVQ_BLAS( x ) \
	do { \
		const cublasStatus_t s_ = ( x ); \
		ENGINE_ASSERT( s_ == CUBLAS_STATUS_SUCCESS, "nvfp4_quantize: cuBLAS call failed" ); \
	} while ( 0 )
#define NVQ_SOLVER( x ) \
	do { \
		const cusolverStatus_t s_ = ( x ); \
		ENGINE_ASSERT( s_ == CUSOLVER_STATUS_SUCCESS, "nvfp4_quantize: cuSOLVER call failed" ); \
	} while ( 0 )
#define NVQ_LOOP( i, n ) for ( size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < ( n ); i += (size_t) gridDim.x * blockDim.x )

constexpr int BLOCK = 16;   // input values per scale, and the rows of a feedback block

// Device buffer, freed with its scope
template <typename T>
struct DBuf {
	T* p = nullptr;
	DBuf() = default;
	explicit DBuf( size_t count ) { alloc( count ); }
	DBuf( const DBuf& ) = delete;
	DBuf& operator=( const DBuf& ) = delete;
	~DBuf() { if ( p ) CUDA_CHECK( cudaFree( p ) ); }
	bool alloc( size_t count ) {
		if ( p ) CUDA_CHECK( cudaFree( p ) );
		p = nullptr;
		if ( count && cudaMalloc( &p, count * sizeof( T ) ) != cudaSuccess ) {
			(void) cudaGetLastError();
			p = nullptr;
			return false;
		}
		return true;
	}
};

struct Ctx {
	cublasHandle_t blas = nullptr;
	cusolverDnHandle_t solver = nullptr;
	int* info = nullptr;
	cudaStream_t stream = nullptr;
};

// Per device and stream
Ctx& ctx( cudaStream_t stream ) {
	static std::mutex mu;
	static std::map<std::pair<int, cudaStream_t>, Ctx> per_stream;
	int dev = 0;
	CUDA_CHECK( cudaGetDevice( &dev ) );
	std::lock_guard<std::mutex> lock( mu );
	Ctx& c = per_stream[{ dev, stream }];
	if ( !c.blas ) {
		NVQ_BLAS( cublasCreate( &c.blas ) );
		NVQ_SOLVER( cusolverDnCreate( &c.solver ) );
		CUDA_CHECK( cudaMalloc( &c.info, sizeof( int ) ) );
	}
	c.stream = stream;
	NVQ_BLAS( cublasSetStream( c.blas, stream ) );
	NVQ_SOLVER( cusolverDnSetStream( c.solver, stream ) );
	return c;
}

dim3 grid_for( size_t n, int threads = 256 ) {
	const size_t blocks = ( n + threads - 1 ) / threads;
	return dim3( static_cast<unsigned>( blocks < 65535u * 16u ? blocks : 65535u * 16u ) );
}

// Row-major C (m x n) = alpha op(A) op(B) + beta C with op(A) m x p, op(B) p x n: the column-major call on the transposes
void gemm( Ctx& c, bool ta, bool tb, int m, int n, int p, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc ) {
	NVQ_BLAS( cublasSgemm( c.blas, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, p, &alpha, B, ldb, A, lda, &beta, C, ldc ) );
}

// ---- the block quantizer --------------------------------------------------------------------------------------------------------------------

// E4M3 (no sign) code 0..126 -> value: subnormals m * 2^-9, normals (1 + m / 8) * 2^(e - 7), up to 448
__device__ __forceinline__ float e4m3_value( int c ) {
	const int e = c >> 3, m = c & 7;
	return e == 0 ? m * 0.001953125f : ( 1.0f + m * 0.125f ) * exp2f( (float) ( e - 7 ) );
}

// E2M1 magnitude index of a >= 0: the thresholds strictly below it (midpoints, ties to the lower magnitude)
__device__ __forceinline__ int e2m1_index( float a ) {
	return ( a > 0.25f ) + ( a > 0.75f ) + ( a > 1.25f ) + ( a > 1.75f ) + ( a > 2.5f ) + ( a > 3.5f ) + ( a > 5.0f );
}

__device__ __forceinline__ float e2m1_value( int i ) {
	constexpr float grid[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
	return grid[i];
}

// One 16-row block (rows x n, ld n) quantized column by column: codes (4-bit, sign in bit 3) into q (16 x n, ld n), the scale code into sc[col],
// the error W - Q into err (16 x n, ld n) against the block's own weight rows w (the feedback target may differ from w)
__global__ void k_quantize_block( const float* __restrict__ rows, const float* __restrict__ w, int n, float gs, uint8_t* __restrict__ q, uint8_t* __restrict__ sc,
	float* __restrict__ err ) {
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	if ( col >= n ) return;
	float v[BLOCK];
	float amax = 0.0f;
	#pragma unroll
	for ( int r = 0; r < BLOCK; ++r ) {
		v[r] = rows[(size_t) r * n + col];
		amax = fmaxf( amax, fabsf( v[r] ) );
	}
	// the first code whose value is >= the ideal scale, or the one below it when that is nearer
	const float ideal = amax / ( 6.0f * gs );
	int lo = 0, hi = 126;
	while ( lo < hi ) {
		const int mid = ( lo + hi ) >> 1;
		if ( e4m3_value( mid ) >= ideal ) hi = mid;
		else lo = mid + 1;
	}
	int c0 = lo;
	const int below = c0 > 0 ? c0 - 1 : 0;
	if ( ideal - e4m3_value( below ) < e4m3_value( c0 ) - ideal ) c0 = below;
	c0 = min( max( c0, 1 ), 126 );
	float best = 0.0f;
	int best_c = c0;
	for ( int d = -2; d <= 2; ++d ) {
		const int c = min( max( c0 + d, 1 ), 126 );
		const float s = e4m3_value( c ) * gs;
		float e = 0.0f;
		#pragma unroll
		for ( int r = 0; r < BLOCK; ++r ) {
			const float x = v[r] / s;
			const float dq = copysignf( e2m1_value( e2m1_index( fabsf( x ) ) ), x < 0.0f ? -1.0f : 1.0f ) * s;
			e += ( dq - v[r] ) * ( dq - v[r] );
		}
		if ( d == -2 || e < best ) {
			best = e;
			best_c = c;
		}
	}
	const float s = e4m3_value( best_c ) * gs;
	sc[col] = (uint8_t) best_c;
	#pragma unroll
	for ( int r = 0; r < BLOCK; ++r ) {
		const float x = v[r] / s;
		const int i = e2m1_index( fabsf( x ) );
		const bool neg = x < 0.0f;
		q[(size_t) r * n + col] = (uint8_t) ( i | ( neg ? 8 : 0 ) );
		err[(size_t) r * n + col] = w[(size_t) r * n + col] - ( neg ? -e2m1_value( i ) : e2m1_value( i ) ) * s;
	}
}

// ---- the block LDL ---------------------------------------------------------------------------------------------------------------------------

__global__ void k_mean_hessian( const float* h, size_t n, double inv_rows, double* out ) {
	NVQ_LOOP( i, n ) out[i] = (double) h[i] * inv_rows;
}

__global__ void k_add_diag( double* A, int k, double v ) {
	NVQ_LOOP( i, (size_t) k ) A[i * k + i] += v;
}

__global__ void k_diag( const double* A, int k, double* d ) {
	NVQ_LOOP( i, (size_t) k ) d[i] = A[i * k + i];
}

// Inverse of each lower-triangular 16 x 16 diagonal block of L: thread c solves column c
__global__ void k_block_inv16( const double* L, int k, double* inv ) {
	const int j = blockIdx.x, c = threadIdx.x;
	if ( c >= BLOCK ) return;
	const double* D = L + (size_t) BLOCK * j * k + BLOCK * j;
	double x[BLOCK];
	for ( int r = 0; r < BLOCK; ++r ) {
		double s = r == c ? 1.0 : 0.0;
		for ( int t = 0; t < r; ++t ) s -= D[(size_t) r * k + t] * x[t];
		x[r] = s / D[(size_t) r * k + r];
	}
	for ( int r = 0; r < BLOCK; ++r ) inv[(size_t) j * 256 + r * BLOCK + c] = x[r];
}

// out[r, 16 j + c] = sum_t L[r, 16 j + t] inv_j[t, c] below the diagonal blocks, 0 on and above them. Reads the strict lower block triangle of L
__global__ void k_apply_block_inv( const double* L, int k, const double* inv, float* out ) {
	NVQ_LOOP( i, (size_t) k * k ) {
		const int r = (int) ( i / k ), col = (int) ( i % k );
		const int j = col >> 4, c = col & 15;
		if ( r < BLOCK * ( j + 1 ) ) {
			out[i] = 0.0f;
			continue;
		}
		const double* Lr = L + (size_t) r * k + BLOCK * j;
		const double* Ij = inv + (size_t) j * 256 + c;
		double s = 0.0;
		#pragma unroll
		for ( int t = 0; t < BLOCK; ++t ) s += Lr[t] * Ij[t * BLOCK];
		out[i] = (float) s;
	}
}

// The feedback factor of the damped mean Hessian (float, k x k), or false when the factorization fails after the damping retries
bool block_ldl( Ctx& c, const float* h_sum, int k, int64_t rows, float sigma_reg, float* Lf ) {
	const size_t kk = (size_t) k * k;
	DBuf<double> H( kk ), d( k ), inv( (size_t) ( k / BLOCK ) * 256 );
	if ( !H.p || !d.p || !inv.p ) return false;
	k_mean_hessian<<<grid_for( kk ), 256, 0, c.stream>>>( h_sum, kk, 1.0 / (double) rows, H.p );
	auto diag_mean = [&]() {
		k_diag<<<grid_for( k ), 256, 0, c.stream>>>( H.p, k, d.p );
		std::vector<double> h( k );
		CUDA_CHECK( cudaMemcpyAsync( h.data(), d.p, k * sizeof( double ), cudaMemcpyDeviceToHost, c.stream ) );
		CUDA_CHECK( cudaStreamSynchronize( c.stream ) );
		return std::accumulate( h.begin(), h.end(), 0.0 ) / k;
	};
	const double mean = diag_mean();
	if ( !std::isfinite( mean ) || mean <= 0.0 ) return false;
	k_add_diag<<<grid_for( k ), 256, 0, c.stream>>>( H.p, k, sigma_reg * mean );
	DBuf<double> A( kk );
	if ( !A.p ) return false;
	int lwork = 0;
	NVQ_SOLVER( cusolverDnDpotrf_bufferSize( c.solver, CUBLAS_FILL_MODE_UPPER, k, A.p, k, &lwork ) );
	DBuf<double> work( lwork > 0 ? lwork : 1 );
	if ( !work.p ) return false;
	for ( int attempt = 0; attempt < 10; ++attempt ) {
		// row-major H is its own column-major view: the upper factor U (H = U^T U) of that view is the row-major lower factor L
		CUDA_CHECK( cudaMemcpyAsync( A.p, H.p, kk * sizeof( double ), cudaMemcpyDeviceToDevice, c.stream ) );
		NVQ_SOLVER( cusolverDnDpotrf( c.solver, CUBLAS_FILL_MODE_UPPER, k, A.p, k, work.p, lwork, c.info ) );
		int info = 0;
		CUDA_CHECK( cudaMemcpyAsync( &info, c.info, sizeof( int ), cudaMemcpyDeviceToHost, c.stream ) );
		CUDA_CHECK( cudaStreamSynchronize( c.stream ) );
		if ( info == 0 ) {
			k_block_inv16<<<k / BLOCK, 32, 0, c.stream>>>( A.p, k, inv.p );
			k_apply_block_inv<<<grid_for( kk ), 256, 0, c.stream>>>( A.p, k, inv.p, Lf );
			return true;
		}
		k_add_diag<<<grid_for( k ), 256, 0, c.stream>>>( H.p, k, 2.0 * sigma_reg * diag_mean() );
	}
	return false;
}

// ---- proxy error and packing ------------------------------------------------------------------------------------------------------------------

// out[r] = sum over columns of A[r, c] * B[r, c] (B = nullptr: A^2), one block per row
__global__ void k_rowsum_prod( const float* A, const float* B, int cols, double* out ) {
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

double sum_prod( Ctx& c, const float* A, const float* B, int rows, int cols ) {
	DBuf<double> part( rows );
	ENGINE_ASSERT( part.p, "nvfp4_quantize: no memory for a reduction" );
	k_rowsum_prod<<<rows, 256, 0, c.stream>>>( A, B, cols, part.p );
	std::vector<double> h( rows );
	CUDA_CHECK( cudaMemcpyAsync( h.data(), part.p, rows * sizeof( double ), cudaMemcpyDeviceToHost, c.stream ) );
	CUDA_CHECK( cudaStreamSynchronize( c.stream ) );
	return std::accumulate( h.begin(), h.end(), 0.0 );
}

// tr( X^T Hm X ) for X k x n, Hm = h_sum / rows (tmp: k x n); h_sum null: |X|^2
double metric( Ctx& c, const float* X, const float* h_sum, int64_t rows, int k, int n, float* tmp ) {
	if ( !h_sum ) return sum_prod( c, X, nullptr, k, n );
	gemm( c, false, false, k, n, k, static_cast<float>( 1.0 / (double) rows ), h_sum, k, X, n, 0.0f, tmp, n );
	return sum_prod( c, tmp, X, k, n );
}

// codes (k x n, a 4-bit code a byte) -> [n, k / 2] bytes, input 2i in the low nibble
__global__ void k_pack_codes( const uint8_t* q, int k, int n, uint8_t* out ) {
	NVQ_LOOP( i, (size_t) n * ( k / 2 ) ) {
		const int col = (int) ( i / ( k / 2 ) ), b = (int) ( i % ( k / 2 ) );
		out[i] = (uint8_t) ( q[(size_t) ( 2 * b ) * n + col] | ( q[(size_t) ( 2 * b + 1 ) * n + col] << 4 ) );
	}
}

// scale codes (k / 16 x n) -> [n, k / 16]
__global__ void k_transpose_u8( const uint8_t* in, int rows, int cols, uint8_t* out ) {
	NVQ_LOOP( i, (size_t) rows * cols ) {
		const int c = (int) ( i / rows ), r = (int) ( i % rows );
		out[i] = in[(size_t) r * cols + c];
	}
}

__global__ void k_absmax( const float* x, size_t n, unsigned* out ) {
	float m = 0.0f;
	NVQ_LOOP( i, n ) m = fmaxf( m, fabsf( x[i] ) );
	for ( int o = 16; o; o >>= 1 ) m = fmaxf( m, __shfl_xor_sync( 0xffffffffu, m, o ) );
	if ( ( threadIdx.x & 31 ) == 0 ) atomicMax( out, __float_as_uint( m ) );   // non-negative floats order as their bits
}

// The 16-row blocks of w (k x n) from the last to the first; L (null: none) feeds back the error of the blocks after each. err receives W - Q
void ldlq( Ctx& c, const float* w, int k, int n, const float* L, float gs, uint8_t* q, uint8_t* sc, float* err, float* rows ) {
	for ( int i = k / BLOCK - 1; i >= 0; --i ) {
		const int r0 = i * BLOCK, r1 = r0 + BLOCK;
		CUDA_CHECK( cudaMemcpyAsync( rows, w + (size_t) r0 * n, (size_t) BLOCK * n * sizeof( float ), cudaMemcpyDeviceToDevice, c.stream ) );
		// rows += L[r1:, r0:r1]^T err[r1:]
		if ( L && r1 < k ) gemm( c, true, false, BLOCK, n, k - r1, 1.0f, L + (size_t) r1 * k + r0, k, err + (size_t) r1 * n, n, 1.0f, rows, n );
		k_quantize_block<<<( n + 127 ) / 128, 128, 0, c.stream>>>( rows, w + (size_t) r0 * n, n, gs, q + (size_t) r0 * n, sc + (size_t) i * n, err + (size_t) r0 * n );
	}
}

} // namespace

namespace engine::quant {

float nvfp4_global_scale( const float* w, size_t count, backend::stream_t stream ) {
	cudaStream_t st = static_cast<cudaStream_t>( stream );
	DBuf<unsigned> m( 1 );
	ENGINE_ASSERT( m.p, "nvfp4_global_scale: no memory" );
	CUDA_CHECK( cudaMemsetAsync( m.p, 0, sizeof( unsigned ), st ) );
	k_absmax<<<grid_for( count ), 256, 0, st>>>( w, count, m.p );
	unsigned bits = 0;
	CUDA_CHECK( cudaMemcpyAsync( &bits, m.p, sizeof( unsigned ), cudaMemcpyDeviceToHost, st ) );
	CUDA_CHECK( cudaStreamSynchronize( st ) );
	float amax = 0.0f;
	std::memcpy( &amax, &bits, sizeof( float ) );
	return amax / ( 6.0f * 448.0f );
}

bool nvfp4_quantize( const float* w, int k, int n, const float* h_sum, int64_t rows, float global_scale, const Nvfp4QuantOptions& o, Nvfp4Tensor& out,
	backend::stream_t stream ) {
	ENGINE_ASSERT( k % BLOCK == 0 && k > 0 && n > 0, "nvfp4_quantize: k must be a multiple of 16" );
	Ctx& c = ctx( static_cast<cudaStream_t>( stream ) );
	const size_t kn = (size_t) k * n;
	const float gs = global_scale > 0.0f ? global_scale : nvfp4_global_scale( w, kn, stream );
	const float* H = rows > 0 ? h_sum : nullptr;
	DBuf<float> L, err( kn ), tmp( kn ), blk( (size_t) BLOCK * n );
	DBuf<uint8_t> q( kn ), sc( (size_t) ( k / BLOCK ) * n );
	if ( !err.p || !tmp.p || !blk.p || !q.p || !sc.p ) return false;
	out = Nvfp4Tensor{};
	out.k = k;
	out.n = n;
	out.global_scale = gs;
	out.fallback = true;
	if ( H ) {
		if ( !L.alloc( (size_t) k * k ) ) return false;
		out.fallback = !block_ldl( c, H, k, rows, o.sigma_reg, L.p );
		if ( out.fallback ) LOG_INFO( "  nvfp4_quantize: the Hessian did not factor, round to nearest for this matrix\n" );
	}
	const double ref = metric( c, w, H, rows, k, n, tmp.p );
	if ( o.rtn_proxy && !out.fallback ) {
		ldlq( c, w, k, n, nullptr, gs, q.p, sc.p, err.p, blk.p );
		out.proxy_rtn = metric( c, err.p, H, rows, k, n, tmp.p ) / ( ref + 1e-30 );
	}
	ldlq( c, w, k, n, out.fallback ? nullptr : L.p, gs, q.p, sc.p, err.p, blk.p );
	out.proxy_err = metric( c, err.p, H, rows, k, n, tmp.p ) / ( ref + 1e-30 );
	if ( out.fallback && o.rtn_proxy ) out.proxy_rtn = out.proxy_err;
	// the packed tensors, through the error buffer's bytes
	uint8_t* packed = reinterpret_cast<uint8_t*>( tmp.p );
	uint8_t* scales = packed + kn / 2;
	k_pack_codes<<<grid_for( kn / 2 ), 256, 0, c.stream>>>( q.p, k, n, packed );
	k_transpose_u8<<<grid_for( (size_t) ( k / BLOCK ) * n ), 256, 0, c.stream>>>( sc.p, k / BLOCK, n, scales );
	out.codes.resize( kn / 2 );
	out.scales.resize( (size_t) ( k / BLOCK ) * n );
	CUDA_CHECK( cudaMemcpyAsync( out.codes.data(), packed, out.codes.size(), cudaMemcpyDeviceToHost, c.stream ) );
	CUDA_CHECK( cudaMemcpyAsync( out.scales.data(), scales, out.scales.size(), cudaMemcpyDeviceToHost, c.stream ) );
	CUDA_CHECK( cudaStreamSynchronize( c.stream ) );
	return true;
}

} // namespace engine::quant
