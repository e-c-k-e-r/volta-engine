// Vulkan backend: the tensor-core path of NVIDIA GPUs without cooperative matrices (Volta). The FP16 GEMMs run the CUDA backend's kernels
// (src/kernels/cuda/gemm/gemm_f16_kernel.cuh; the PTX entry points of gemm_f16_vk.cu, <kernel dir>/cuda_gemm.ptx) launched from the command
// buffers through VK_NV_cuda_kernel_launch, with gemm_f16.cu's tiles by row count and its split along k. Devices with cooperative matrices
// would take shaders of their own beside this (tensor_path); the others keep the shader-core matmuls.
#include "mod_common.h"
#include <algorithm>
#include <cstdlib>
#include <string>

#ifndef SLANG_SPV_DIR
	#error "SLANG_SPV_DIR must point at the generated kernel directory"
#endif

namespace engine::backend::impl {

TensorPath tensor_path() {
	return vk::cuda_launch_available() ? TensorPath::CudaLaunch : TensorPath::None;
}

namespace {

// gemm_f16.cu's tile configurations (every one within a launch's 48 KiB of shared memory); per_sm: the blocks an SM holds (the sm_70 build's
// registers and shared memory)
struct TileCfg { int bm, bn, bk, threads, per_sm; const char* h; const char* f; };
constexpr TileCfg kTileCfg[4] = {
	{ 32, 64, 64, 64, 4, "vk_gemm_tn_32x64x64_h", "vk_gemm_tn_32x64x64_f" },
	{ 64, 64, 64, 128, 3, "vk_gemm_tn_64x64x64_h", "vk_gemm_tn_64x64x64_f" },
	{ 64, 128, 64, 128, 2, "vk_gemm_tn_64x128x64_h", "vk_gemm_tn_64x128x64_f" },
	{ 128, 128, 32, 128, 2, "vk_gemm_tn_128x128x32_h", "vk_gemm_tn_128x128x32_f" },
};
// gemm_f16.cu's split-k model (V100: the multiply rate of 80 SMs, a partial wave's block alone and shared, the partials' bandwidth)
constexpr double kSmRate = 70.0, kLone = 1.17, kShared = 1.56, kPartBandwidth = 700.0;
constexpr size_t kSplitFloats = size_t( 16 ) << 20;

vk::CudaFunction* function( const char* name ) {
	const char* dir = std::getenv( "ENGINE_SPV_DIR" );
	const std::string path = std::string( dir ? dir : SLANG_SPV_DIR ) + "/cuda_gemm.ptx";
	return vk::cuda_function( path.c_str(), name );
}

// The splits along k that spread a GEMM of `tiles` output tiles over the SMs best (gemm_f16.cu's launch)
int model_splits( const TileCfg& t, int m, int n, int k ) {
	const int sms = vk::sm_count() > 0 ? static_cast<int>( vk::sm_count() ) : 80;
	const int tiles = ( ( n + t.bn - 1 ) / t.bn ) * ( ( m + t.bm - 1 ) / t.bm ), slots = t.per_sm * sms;
	const bool big = t.bm * t.bn >= 128 * 128;
	const size_t mt = static_cast<size_t>( ( m + t.bm - 1 ) / t.bm ) * t.bm;
	const double tile_s = 2.0 * t.bm * t.bn * k / ( kSmRate * 1e12 / 80 ), part_s = 2.0 * 4.0 * mt * n / ( kPartBandwidth * 1e9 );
	auto cost = [&]( int sp ) {
		const int units = tiles * sp, full = units / slots, rest = units % slots;
		const double tail = rest == 0 ? 0.0 : !big ? t.per_sm : rest <= sms ? kLone : kShared;
		return ( full * t.per_sm + tail ) * tile_s / sp + ( sp > 1 ? part_s * sp : 0.0 );
	};
	int splits = 1;
	double best = cost( 1 );
	for ( int sp = 2; sp <= std::min( 16, k / 256 ) && sp * mt * n <= kSplitFloats; ++sp ) {
		const double c = cost( sp );
		if ( c < best ) {
			best = c;
			splits = sp;
		}
	}
	return splits;
}

} // namespace

bool cuda_gemm_tn( const void* w, const void* x, void* out, int m, int n, int k, bool f32_out, int k_split, stream_t stream ) {
	if ( tensor_path() != TensorPath::CudaLaunch || m <= 0 || n <= 0 || k <= 0 || k % 8 != 0 ) return false;
	auto aligned = []( const void* p ) { return reinterpret_cast<uintptr_t>( p ) % 16 == 0; };
	if ( !aligned( w ) || !aligned( x ) ) return false;
	// gemm_f16.cu's tile by row count
	const TileCfg& t = kTileCfg[m <= 32 ? 0 : ( m <= 64 || ( m > 128 && m <= 192 ) ) ? 1 : m <= 128 ? 2 : 3];
	vk::CudaFunction* gemm = function( f32_out ? t.f : t.h );
	if ( gemm == nullptr ) return false;
	int k_per_split = k, splits = 1;
	float* part = nullptr;
	if ( k_split > 0 ) {
		// the caller's slabs: slab s's FP32 partial into out + s * m * n
		k_per_split = k_split;
		splits = ( k + k_split - 1 ) / k_split;
		if ( splits > 1 ) {
			if ( !f32_out ) return false;
			part = static_cast<float*>( out );
		}
	}
	else {
		const int sp = model_splits( t, m, n, k );
		k_per_split = ( ( k + sp - 1 ) / sp + 7 ) / 8 * 8;
		splits = ( k + k_per_split - 1 ) / k_per_split;
		if ( splits > 1 ) part = static_cast<float*>( scratch( 42, static_cast<size_t>( splits ) * m * n * sizeof( float ), stream ) );
	}
	vk::CudaFunction* reduce = nullptr;
	if ( k_split <= 0 && splits > 1 ) {
		reduce = function( f32_out ? "vk_split_reduce_f" : "vk_split_reduce_h" );
		if ( reduce == nullptr ) return false;
	}
	const void* a = x;
	const void* wp = w;
	void* o = out;
	int lda = k, ldw = k, ldc = n, mm = m, nn = n, kk = k;
	float beta = 0.0f;
	const void* params[12] = { &a, &lda, &wp, &ldw, &o, &ldc, &mm, &nn, &kk, &beta, &part, &k_per_split };
	const uint32_t smem = static_cast<uint32_t>( 2 * ( t.bm * t.bk + t.bn * t.bk ) * 2 );
	vk::cuda_launch( stream, gemm, static_cast<uint32_t>( ( n + t.bn - 1 ) / t.bn ), static_cast<uint32_t>( ( m + t.bm - 1 ) / t.bm ), static_cast<uint32_t>( splits ),
		static_cast<uint32_t>( t.threads ), 1, 1, smem, params, 12 );
	if ( reduce != nullptr ) {
		const float* pp = part;
		int ss = splits;
		const void* rparams[7] = { &pp, &o, &ldc, &mm, &nn, &ss, &beta };
		const int quads = m * ( ( n + 3 ) / 4 );
		vk::cuda_launch( stream, reduce, static_cast<uint32_t>( ( quads + 255 ) / 256 ), 1, 1, 256, 1, 1, 0, rparams, 7 );
	}
	return true;
}

} // namespace engine::backend::impl
