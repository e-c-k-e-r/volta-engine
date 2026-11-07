// GGUF linear on CUDA. Up to kGemvRows rows (decode, verify blocks): activations quantized to q8_1 once per call, then one GEMV kernel per
// block type built on llama.cpp's vec_dot functions (ggml/vecdotq.cuh). More rows (prefill chunks): the weight dequantized to FP16 by
// llama.cpp's own converters (ggml/convert.cu) and one tensor-core GEMM. The ggml files are vendored unchanged.
#include "engine/modules/gguf.h"
#include "engine/core/batch.h"
#include "engine/backend/runtime.h"
#include <algorithm>
#include "ggml/vecdotq.cuh"
#include "ggml/convert.cuh"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mutex>

namespace engine::modules {

namespace {

// The block geometry table of gguf_types.cpp against the vendored layouts
static_assert( sizeof( block_q8_0 ) == 34 && sizeof( block_q4_K ) == 144 && sizeof( block_q5_K ) == 176 && sizeof( block_q6_K ) == 210, "block sizes" );
static_assert( sizeof( block_q2_K ) == 84 && sizeof( block_q3_K ) == 110 && sizeof( block_iq4_xs ) == 136 && sizeof( block_iq4_nl ) == 18, "block sizes" );
static_assert( sizeof( block_iq2_xxs ) == 66 && sizeof( block_iq2_xs ) == 74 && sizeof( block_iq2_s ) == 82 && sizeof( block_iq3_xxs ) == 98, "block sizes" );
static_assert( sizeof( block_iq3_s ) == 110 && sizeof( block_iq1_s ) == 50 && sizeof( block_iq1_m ) == 56 && sizeof( block_q8_1 ) == 36, "block sizes" );
static_assert( sizeof( block_q4_0 ) == 18 && sizeof( block_q4_1 ) == 20 && sizeof( block_q5_0 ) == 22 && sizeof( block_q5_1 ) == 24, "block sizes" );

// Values per block, quant ints per block and ints per vec_dot call of each type (llama.cpp's type traits and MMVQ VDR)
template <GgmlType T> struct Traits;
#define GGUF_TRAITS( T, QK, QI, VDR, FN ) \
	template <> struct Traits<GgmlType::T> { \
		static constexpr int qk = QK, qi = QI, vdr = VDR; \
		static __device__ __forceinline__ float dot( const void* x, const block_q8_1* y, int kbx, int kqs ) { return FN( x, y, kbx, kqs ); } \
	};
GGUF_TRAITS( Q4_0, QK4_0, QI4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1 )
GGUF_TRAITS( Q4_1, QK4_1, QI4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1 )
GGUF_TRAITS( Q5_0, QK5_0, QI5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1 )
GGUF_TRAITS( Q5_1, QK5_1, QI5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1 )
GGUF_TRAITS( Q8_0, QK8_0, QI8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1 )
GGUF_TRAITS( Q2_K, QK_K, QI2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1 )
GGUF_TRAITS( Q3_K, QK_K, QI3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1 )
GGUF_TRAITS( Q4_K, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1 )
GGUF_TRAITS( Q5_K, QK_K, QI5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1 )
GGUF_TRAITS( Q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1 )
GGUF_TRAITS( IQ2_XXS, QK_K, QI2_XXS, VDR_IQ2_XXS_Q8_1_MMVQ, vec_dot_iq2_xxs_q8_1 )
GGUF_TRAITS( IQ2_XS, QK_K, QI2_XS, VDR_IQ2_XS_Q8_1_MMVQ, vec_dot_iq2_xs_q8_1 )
GGUF_TRAITS( IQ2_S, QK_K, QI2_S, VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1 )
GGUF_TRAITS( IQ3_XXS, QK_K, QI3_XXS, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1 )
GGUF_TRAITS( IQ3_S, QK_K, QI3_S, VDR_IQ3_S_Q8_1_MMVQ, vec_dot_iq3_s_q8_1 )
GGUF_TRAITS( IQ1_S, QK_K, QI1_S, VDR_IQ1_S_Q8_1_MMVQ, vec_dot_iq1_s_q8_1 )
GGUF_TRAITS( IQ1_M, QK_K, QI1_M, VDR_IQ1_M_Q8_1_MMVQ, vec_dot_iq1_m_q8_1 )
GGUF_TRAITS( IQ4_NL, QK4_NL, QI4_NL, VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1 )
GGUF_TRAITS( IQ4_XS, QK_K, QI4_XS, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1 )
#undef GGUF_TRAITS

constexpr int kWarps = 4;
constexpr int kGemvRows = 16;   // above this many rows, dequantize once and run a GEMM

// The activation column a weight column reads: tiled [v][head][dim] against grouped [head][v][dim]
__device__ __forceinline__ int tiled_source( int i, GgufColTiling t, int k ) {
	if ( t.heads == 0 ) return i;
	const int per_v = t.heads * t.dim, v = i / per_v, head = i % per_v / t.dim, d = i % t.dim;
	return ( head * ( k / per_v ) + v ) * t.dim + d;
}

// x [m, k] FP16 -> y [m][k / 32] q8_1 (llama.cpp's quantize_q8_1: scale amax / 127, the block sum of the unquantized values), in the
// weight's column order
__global__ void quantize_q8_1_kernel( const half* __restrict__ x, block_q8_1* __restrict__ y, int k, GgufColTiling t ) {
	const int row = blockIdx.y;
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= k ) return;
	const float xi = __half2float( x[static_cast<size_t>( row ) * k + tiled_source( i, t, k )] );
	float amax = fabsf( xi ), sum = xi;
	#pragma unroll
	for ( int o = 16; o > 0; o >>= 1 ) {
		amax = fmaxf( amax, __shfl_xor_sync( 0xffffffff, amax, o ) );
		sum += __shfl_xor_sync( 0xffffffff, sum, o );
	}
	const float d = amax / 127.0f;
	const int8_t q = amax == 0.0f ? 0 : static_cast<int8_t>( roundf( xi / d ) );
	block_q8_1& b = y[static_cast<size_t>( row ) * ( k / QK8_1 ) + i / QK8_1];
	b.qs[i % QK8_1] = q;
	if ( i % QK8_1 == 0 ) b.ds = make_half2( d, sum );
}

// x [m, k] in the weight's column order (the prefill GEMM's operand)
__global__ void tile_columns_kernel( const half* __restrict__ x, half* __restrict__ y, int k, GgufColTiling t ) {
	const int row = blockIdx.y;
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < k ) y[static_cast<size_t>( row ) * k + i] = x[static_cast<size_t>( row ) * k + tiled_source( i, t, k )];
}

// One block of kWarps warps per output row; every row of the batch in the same pass over the weight row
template <GgmlType T, int M>
__global__ __launch_bounds__( 32 * kWarps )
void gemv_kernel( const void* __restrict__ w, const block_q8_1* __restrict__ y, half* __restrict__ out, int n, int k ) {
	using Tr = Traits<T>;
	const int row = blockIdx.x;
	const int tid = threadIdx.y * 32 + threadIdx.x;
	const int blocks_per_row = k / Tr::qk, y_blocks = k / QK8_1;
	constexpr int per_iter = Tr::vdr * kWarps * 32 / Tr::qi;
	float acc[M] = {};
	for ( int kbx = tid / ( Tr::qi / Tr::vdr ); kbx < blocks_per_row; kbx += per_iter ) {
		const int kby = kbx * ( Tr::qk / QK8_1 );
		const int kqs = Tr::vdr * ( tid % ( Tr::qi / Tr::vdr ) );
		#pragma unroll
		for ( int j = 0; j < M; ++j ) acc[j] += Tr::dot( w, y + static_cast<size_t>( j ) * y_blocks + kby, row * blocks_per_row + kbx, kqs );
	}
	__shared__ float part[kWarps][M];
	#pragma unroll
	for ( int j = 0; j < M; ++j ) {
		float v = acc[j];
		#pragma unroll
		for ( int o = 16; o > 0; o >>= 1 ) v += __shfl_xor_sync( 0xffffffff, v, o );
		if ( threadIdx.x == 0 ) part[threadIdx.y][j] = v;
	}
	__syncthreads();
	if ( threadIdx.y == 0 && threadIdx.x < M ) {
		float v = 0.0f;
		#pragma unroll
		for ( int wi = 0; wi < kWarps; ++wi ) v += part[wi][threadIdx.x];
		out[static_cast<size_t>( threadIdx.x ) * n + row] = __float2half_rn( v );
	}
}

template <GgmlType T>
void launch_rows( const void* w, const block_q8_1* y, half* out, int n, int k, int m, cudaStream_t stream ) {
	const dim3 grid( n ), block( 32, kWarps );
	switch ( m ) {
		case 1: gemv_kernel<T, 1><<<grid, block, 0, stream>>>( w, y, out, n, k ); break;
		case 2: gemv_kernel<T, 2><<<grid, block, 0, stream>>>( w, y, out, n, k ); break;
		case 3: gemv_kernel<T, 3><<<grid, block, 0, stream>>>( w, y, out, n, k ); break;
		case 4: gemv_kernel<T, 4><<<grid, block, 0, stream>>>( w, y, out, n, k ); break;
		case 5: gemv_kernel<T, 5><<<grid, block, 0, stream>>>( w, y, out, n, k ); break;
		case 6: gemv_kernel<T, 6><<<grid, block, 0, stream>>>( w, y, out, n, k ); break;
		case 7: gemv_kernel<T, 7><<<grid, block, 0, stream>>>( w, y, out, n, k ); break;
		default: gemv_kernel<T, 8><<<grid, block, 0, stream>>>( w, y, out, n, k ); break;
	}
}

template <GgmlType T>
void gemv_tiles( const void* w, const block_q8_1* y, half* out, int n, int k, int m, cudaStream_t stream ) {
	for ( int r = 0; r < m; r += 8 ) launch_rows<T>( w, y + static_cast<size_t>( r ) * ( k / QK8_1 ), out + static_cast<size_t>( r ) * n, n, k, std::min( 8, m - r ), stream );
}

// Per-device state, each behind its own lock: a cudaMalloc waits for its device, whose queue may hold a collective the other ranks
// have to join, so one rank must never hold a lock another rank's device is waiting behind
std::mutex& device_mutex( int dev ) {
	static std::mutex mu[64];
	return mu[dev];
}

// The q8_1 activations of one call, per device, kept between calls (decode graphs capture the pointer): allocated at load
// (LinearGguf::prepare) to cover a prefill chunk of 32K-wide rows, so it does not grow inside a forward
block_q8_1* q8_workspace( size_t blocks ) {
	static block_q8_1* buf[64] = {};
	static size_t cap[64] = {};
	int dev = 0;
	cudaGetDevice( &dev );
	std::lock_guard<std::mutex> lk( device_mutex( dev ) );
	if ( blocks > cap[dev] ) {
		if ( buf[dev] ) cudaFree( buf[dev] );
		const size_t want = std::max( blocks, static_cast<size_t>( pod::MAX_ROWS ) * 1024 );   // a prefill chunk of 32K-wide rows without regrowing
		ENGINE_CHECK( cudaMalloc( &buf[dev], want * sizeof( block_q8_1 ) ) );
		cap[dev] = want;
	}
	return buf[dev];
}

// The FP16 copy of one weight for the prefill GEMM, per device; grows to the largest weight (prefill runs eagerly, never inside a graph)
half* recon_workspace( size_t elems, cudaStream_t stream ) {
	static half* buf[64] = {};
	static size_t cap[64] = {};
	int dev = 0;
	cudaGetDevice( &dev );
	std::lock_guard<std::mutex> lk( device_mutex( dev ) );
	if ( elems > cap[dev] ) {
		ENGINE_CHECK( cudaStreamSynchronize( stream ) );   // the previous copy may still be read by queued work
		if ( buf[dev] ) cudaFree( buf[dev] );
		ENGINE_CHECK( cudaMalloc( &buf[dev], elems * sizeof( half ) ) );
		cap[dev] = elems;
	}
	return buf[dev];
}

cublasHandle_t cublas_handle() {
	static cublasHandle_t h[64] = {};
	int dev = 0;
	cudaGetDevice( &dev );
	std::lock_guard<std::mutex> lk( device_mutex( dev ) );
	if ( !h[dev] ) ENGINE_ASSERT( cublasCreate( &h[dev] ) == CUBLAS_STATUS_SUCCESS, "LinearGguf: cublasCreate failed" );
	return h[dev];
}

} // namespace

void LinearGguf::prepare() {
	q8_workspace( 1 );
	cublas_handle();
}

void LinearGguf::dequant( const void* w, GgmlType type, int64_t elems, half* out, cudaStream_t stream ) {
	const to_fp16_cuda_t fn = ggml_get_to_fp16_cuda( static_cast<ggml_type>( type ) );
	ENGINE_ASSERT( fn != nullptr, "LinearGguf::dequant: no converter for this block type" );
	fn( w, out, elems, stream );
}

bool gguf_linear_supported( GgmlType type ) {
	switch ( type ) {
		case GgmlType::Q4_0: case GgmlType::Q4_1: case GgmlType::Q5_0: case GgmlType::Q5_1: case GgmlType::Q8_0:
		case GgmlType::Q2_K: case GgmlType::Q3_K: case GgmlType::Q4_K: case GgmlType::Q5_K: case GgmlType::Q6_K:
		case GgmlType::IQ2_XXS: case GgmlType::IQ2_XS: case GgmlType::IQ2_S: case GgmlType::IQ3_XXS: case GgmlType::IQ3_S:
		case GgmlType::IQ1_S: case GgmlType::IQ1_M: case GgmlType::IQ4_NL: case GgmlType::IQ4_XS: return true;
		default: return false;
	}
}

void LinearGguf::forward( const pod::TensorView& x, const void* w, GgmlType type, int n, int k, pod::TensorView& out, cudaStream_t stream,
	GgufColTiling tiling ) {
	const int m = x.shape[0];
	ENGINE_ASSERT( x.dtype == pod::DType::FP16 && out.dtype == pod::DType::FP16, "LinearGguf: FP16 activations and output" );
	ENGINE_ASSERT( x.shape[1] == k && k % ggml_block( type ).elems == 0 && k % QK8_1 == 0, "LinearGguf: k does not match the weight's blocks" );
	ENGINE_ASSERT( tiling.heads == 0 || ( tiling.dim > 0 && k % ( tiling.heads * tiling.dim ) == 0 ), "LinearGguf: tiling does not divide k" );
	if ( m > kGemvRows ) {
		// W [n][k] row-major is W^T in column-major terms: C (column-major [n][m]) = op(W)^T [n, k] times x^T [k, m]
		half* wf = recon_workspace( static_cast<size_t>( n ) * k + ( tiling.heads ? static_cast<size_t>( m ) * k : 0 ), stream );
		dequant( w, type, static_cast<int64_t>( n ) * k, wf, stream );
		const half* xs = x.ptr<const half>();
		if ( tiling.heads ) {
			half* xt = wf + static_cast<size_t>( n ) * k;
			tile_columns_kernel<<<dim3( ( k + 255 ) / 256, m ), 256, 0, stream>>>( xs, xt, k, tiling );
			xs = xt;
		}
		cublasHandle_t h = cublas_handle();
		cublasSetStream( h, stream );
		const float alpha = 1.0f, beta = 0.0f;
		const cublasStatus_t st = cublasGemmEx( h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, wf, CUDA_R_16F, k, xs, CUDA_R_16F, k,
			&beta, out.data, CUDA_R_16F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP );
		ENGINE_ASSERT( st == CUBLAS_STATUS_SUCCESS, "LinearGguf: cublasGemmEx failed" );
		return;
	}
	block_q8_1* y = q8_workspace( static_cast<size_t>( m ) * ( k / QK8_1 ) );
	quantize_q8_1_kernel<<<dim3( ( k + 255 ) / 256, m ), 256, 0, stream>>>( x.ptr<const half>(), y, k, tiling );
	half* o = out.ptr<half>();
	switch ( type ) {
#define CASE( T ) case GgmlType::T: gemv_tiles<GgmlType::T>( w, y, o, n, k, m, stream ); break;
		CASE( Q4_0 ) CASE( Q4_1 ) CASE( Q5_0 ) CASE( Q5_1 ) CASE( Q8_0 ) CASE( Q2_K ) CASE( Q3_K ) CASE( Q4_K ) CASE( Q5_K ) CASE( Q6_K )
		CASE( IQ2_XXS ) CASE( IQ2_XS ) CASE( IQ2_S ) CASE( IQ3_XXS ) CASE( IQ3_S ) CASE( IQ1_S ) CASE( IQ1_M ) CASE( IQ4_NL ) CASE( IQ4_XS )
#undef CASE
		default: ENGINE_ASSERT( false, "LinearGguf: unsupported block type" );
	}
}

} // namespace engine::modules
