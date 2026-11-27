// Dense FP16 GEMM on Volta's tensor cores (mma.sync.m8n8k4, FP32 accumulation): out[m, n] = beta * out + a[m, k] * w[n, k]^T (tn) or
// a[m, k] * w[k, n] (nn). A block computes a BM x BN tile over k in steps of BK: both operands' tiles are staged in shared memory (16-byte
// chunks XOR-swizzled by row, so fragment reads and tile stores hit every bank once), two stages: the step after next loads into registers
// right after the barrier, fragments are double-buffered across the mma k steps. Warps tile the block 2 x 2 (64 x 64 each at 128 x 128);
// quad pairs 0 / 1 and 2 / 3 split a 16 x 16 fragment's rows and columns (frag_line). The tile size follows the row count, a split along k
// fills the GPU when the tiles do not (split_reduce adds the partials in order), and the epilogue goes through shared memory to 16-byte stores.
// Fragment layouts of m8n8k4 (measured on sm_70): a lane holds row p of A and column p of B (p = lane % 4 + 4 * (lane >= 16)), 4 consecutive
// k each (B row-major: k row lane % 4, columns 4 (lane >= 16) + 0..3); its 8 accumulators are rows (lane & 1) + 4 (lane >= 16) and + 2,
// columns (lane & 2) + {0, 1, 4, 5}, in the order (r, c) (r, c + 1) (r + 2, c) (r + 2, c + 1) and the same 4 columns on.
#include "engine/core/scratch.h"
#include "engine/modules/gemm.h"
#include "engine/config.h"
#if !defined(USE_ROCM)
#include "gemm_f16_kernel.cuh"
#endif
#include <cuda_fp16.h>
#include <type_traits>
#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>

#if defined(USE_ROCM)
#include <cublas_v2.h>   // hipBLAS through the ROCm stand-in
#endif

#if !defined(USE_ROCM)
namespace engine::modules {

namespace {


// The kernels over the device code of gemm_f16_kernel.cuh
template <int BM, int BN, int BK, int WM, int WN, bool F32, bool WKN>
__global__ __launch_bounds__( WM * WN * 32 )
void gemm_tn_kernel( const half* __restrict__ a, int lda, const half* __restrict__ w, int ldw, void* __restrict__ out, int ldc, int m, int n, int k, float beta,
	float* __restrict__ part, int k_per_split ) {
	gemm_f16::gemm_tn_body<BM, BN, BK, WM, WN, F32, WKN>( a, lda, w, ldw, out, ldc, m, n, k, beta, part, k_per_split );
}

template <bool F32>
__global__ void split_reduce_kernel( const float* __restrict__ part, void* __restrict__ out, int ldc, int m, int n, int splits, float beta ) {
	gemm_f16::split_reduce_body<F32>( part, out, ldc, m, n, splits, beta );
}

// The split-k partials, per stream (GEMMs on one stream run in order; two streams may overlap), grown on demand. A buffer is never freed (a
// captured graph may still use it), and growing only allocates, which a graph capture allows: whether a GEMM splits depends on its shape
// alone, so a captured product sums in the same order as the eager one
constexpr size_t kSplitFloats = size_t( 16 ) << 20;
// Bumped by reset_caches: each launch() configuration then sets its shared-memory attribute and reads its occupancy again
std::atomic<int> g_launch_epoch{ 1 };

// The split-K partial sums of a stream's launches, from the device's arena: powers of two from 64K floats, so a few growths cover every
// shape (a growth leaves the smaller block behind, which a captured graph may still read)
float* split_workspace( size_t floats, cudaStream_t stream ) {
	size_t cap = 65536;
	while ( cap < floats ) cap *= 2;
	cap = std::max( floats, std::min( cap, kSplitFloats ) );
	return static_cast<float*>( engine::core::scratch( "gemm.split_partials", cap * sizeof( float ), reinterpret_cast<uintptr_t>( stream ) ) );
}

int sm_count() {
	static int sms[64] = {};
	int dev = 0;
	cudaGetDevice( &dev );
	if ( sms[dev] == 0 ) CUDA_CHECK( cudaDeviceGetAttribute( &sms[dev], cudaDevAttrMultiProcessorCount, dev ) );
	return sms[dev];
}

// The split-k model, measured on a V100 at the 200 W cap: the multiply rate of 80 SMs with two 128 x 128 blocks each (TFLOPS), the time of
// a partial wave's block alone on its SM and of two sharing one (in tile times), and the bandwidth the split partials move at (GB/s)
constexpr double kSmRate = 70.0, kLone = 1.17, kShared = 1.56, kPartBandwidth = 700.0;

template <int BM, int BN, int BK, int WM, int WN, bool WKN>
void launch( const half* a, int lda, const half* w, int ldw, void* out, int ldc, bool out_f32, int m, int n, int k, float beta, cudaStream_t stream ) {
	constexpr int threads = WM * WN * 32, smem = 2 * ( BM * BK + gemm_f16::w_stage_halves<WKN>( BN, BK ) ) * static_cast<int>( sizeof( half ) );
	auto kernel = out_f32 ? gemm_tn_kernel<BM, BN, BK, WM, WN, true, WKN> : gemm_tn_kernel<BM, BN, BK, WM, WN, false, WKN>;
	static int per_sm[2][64] = {}, epoch[2][64] = {};
	int dev = 0;
	cudaGetDevice( &dev );
	if ( epoch[out_f32][dev] != g_launch_epoch.load() ) {
		epoch[out_f32][dev] = g_launch_epoch.load();
		CUDA_CHECK( cudaFuncSetAttribute( kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem ) );
		CUDA_CHECK( cudaOccupancyMaxActiveBlocksPerMultiprocessor( &per_sm[out_f32][dev], kernel, threads, smem ) );
	}
	const int tiles = ( ( n + BN - 1 ) / BN ) * ( ( m + BM - 1 ) / BM ), slots = per_sm[out_f32][dev] * sm_count();
	// split along k when it spreads the work over the SMs more evenly. Cost of a launch in tile times (one tile on an SM to itself, with the
	// whole GPU busy): full waves of block slots, then a partial wave whose blocks run alone on their SMs or share them; each block is
	// 1 / splits of a tile's work; plus the partials' round trip (written and read back in FP32); at least 256 values of k per split.
	// Small tiles stream the weights and count every slot as a full share
	const int sms = sm_count(), per = per_sm[out_f32][dev];
	const bool big = BM * BN >= 128 * 128;
	// rows rounded up to whole tiles: every row count with the same tiles splits the same way, so an output row does not depend on how many
	// rows ride along (a decode step and a verify of the same token agree)
	const size_t mt = static_cast<size_t>( ( m + BM - 1 ) / BM ) * BM;
	const double tile_s = 2.0 * BM * BN * k / ( kSmRate * 1e12 / 80 ), part_s = 2.0 * 4.0 * mt * n / ( kPartBandwidth * 1e9 );
	auto cost = [&]( int sp ) {
		const int units = tiles * sp, full = units / slots, rest = units % slots;
		const double tail = rest == 0 ? 0.0 : !big ? per : rest <= sms ? kLone : kShared;
		return ( full * per + tail ) * tile_s / sp + ( sp > 1 ? part_s * sp : 0.0 );
	};
	int splits = 1;
	double best = cost( 1 );
	for ( int sp = 2; sp <= std::min( 16, k / 256 ) && sp * mt * n <= kSplitFloats; ++sp ) {
		const double t = cost( sp );
		if ( t < best ) { best = t; splits = sp; }
	}
	int k_per_split = ( ( k + splits - 1 ) / splits + 7 ) / 8 * 8, used = ( k + k_per_split - 1 ) / k_per_split;
	float* part = used > 1 ? split_workspace( static_cast<size_t>( used ) * m * n, stream ) : nullptr;
	const dim3 grid( ( n + BN - 1 ) / BN, ( m + BM - 1 ) / BM, used );
	kernel<<<grid, threads, smem, stream>>>( a, lda, w, ldw, out, ldc, m, n, k, beta, part, k_per_split );
	if ( used > 1 ) {
		const int quads = m * ( ( n + 3 ) / 4 );
		if ( out_f32 ) split_reduce_kernel<true><<<( quads + 255 ) / 256, 256, 0, stream>>>( part, out, ldc, m, n, used, beta );
		else split_reduce_kernel<false><<<( quads + 255 ) / 256, 256, 0, stream>>>( part, out, ldc, m, n, used, beta );
	}
}

template <bool WKN>
void dispatch( const half* a, int lda, const half* w, int ldw, void* out, int ldc, bool out_f32, int m, int n, int k, float beta, cudaStream_t stream ) {
	// tile by row count, from a sweep over the language models' projection shapes (20 - 2048 rows)
	if ( m <= 32 ) launch<32, 64, 64, 1, 2, WKN>( a, lda, w, ldw, out, ldc, out_f32, m, n, k, beta, stream );
	else if ( m <= 64 || ( m > 128 && m <= 192 ) ) launch<64, 64, 64, 2, 2, WKN>( a, lda, w, ldw, out, ldc, out_f32, m, n, k, beta, stream );
	else if ( m <= 128 ) launch<64, 128, 64, 2, 2, WKN>( a, lda, w, ldw, out, ldc, out_f32, m, n, k, beta, stream );
	else launch<128, 128, 32, 2, 2, WKN>( a, lda, w, ldw, out, ldc, out_f32, m, n, k, beta, stream );
}

} // namespace

void Gemm::tn( const half* a, int lda, const half* w, int ldw, void* out, int ldc, bool out_f32, int m, int n, int k, float beta, cudaStream_t stream ) {
	ENGINE_ASSERT( k % 8 == 0 && lda % 8 == 0 && ldw % 8 == 0, "Gemm::tn: k and the operands' leading dimensions in 16-byte rows" );
	ENGINE_ASSERT( reinterpret_cast<uintptr_t>( a ) % 16 == 0 && reinterpret_cast<uintptr_t>( w ) % 16 == 0, "Gemm::tn: 16-byte aligned operands" );
	dispatch<false>( a, lda, w, ldw, out, ldc, out_f32, m, n, k, beta, stream );
}

void Gemm::nn( const half* a, int lda, const half* w, int ldw, void* out, int ldc, bool out_f32, int m, int n, int k, float beta, cudaStream_t stream ) {
	ENGINE_ASSERT( k % 8 == 0 && n % 8 == 0 && lda % 8 == 0 && ldw % 8 == 0, "Gemm::nn: k, n and the operands' leading dimensions in 16-byte rows" );
	ENGINE_ASSERT( reinterpret_cast<uintptr_t>( a ) % 16 == 0 && reinterpret_cast<uintptr_t>( w ) % 16 == 0, "Gemm::nn: 16-byte aligned operands" );
	dispatch<true>( a, lda, w, ldw, out, ldc, out_f32, m, n, k, beta, stream );
}

} // namespace engine::modules
namespace engine::modules {

void Gemm::reset_caches() {
	g_launch_epoch.fetch_add( 1 );
}

} // namespace engine::modules

#else

// ROCm: the kernel above is Volta PTX; hipBLAS does the products (one handle per device)
namespace engine::modules {

namespace {

cublasHandle_t blas() {
	static std::mutex mu;
	static cublasHandle_t h[64] = {};
	int dev = 0;
	cudaGetDevice( &dev );
	std::lock_guard<std::mutex> lk( mu );
	if ( !h[dev] ) ENGINE_ASSERT( cublasCreate( &h[dev] ) == CUBLAS_STATUS_SUCCESS, "Gemm: hipblasCreate failed" );
	return h[dev];
}

void gemm( bool wkn, const half* a, int lda, const half* w, int ldw, void* out, int ldc, bool out_f32, int m, int n, int k, float beta, cudaStream_t stream ) {
	cublasHandle_t h = blas();
	cublasSetStream( h, stream );
	const float one = 1.0f;
	// row-major out[m, n] is column-major [n, m]: out^T = op(w) a^T
	const cublasStatus_t st = cublasGemmEx( h, wkn ? CUBLAS_OP_N : CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one, w, CUDA_R_16F, ldw, a, CUDA_R_16F, lda, &beta, out,
		out_f32 ? CUDA_R_32F : CUDA_R_16F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP );
	ENGINE_ASSERT( st == CUBLAS_STATUS_SUCCESS, "Gemm: hipblasGemmEx failed" );
}

} // namespace

void Gemm::tn( const half* a, int lda, const half* w, int ldw, void* out, int ldc, bool out_f32, int m, int n, int k, float beta, cudaStream_t stream ) {
	gemm( false, a, lda, w, ldw, out, ldc, out_f32, m, n, k, beta, stream );
}

void Gemm::reset_caches() {}

void Gemm::nn( const half* a, int lda, const half* w, int ldw, void* out, int ldc, bool out_f32, int m, int n, int k, float beta, cudaStream_t stream ) {
	gemm( true, a, lda, w, ldw, out, ldc, out_f32, m, n, k, beta, stream );
}

} // namespace engine::modules

#endif
