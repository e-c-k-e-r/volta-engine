// Launcher for the H3 D128 non-causal FlashAttention (SM70 CUTLASS). Kernel headers in this directory are derived from NVIDIA
// CUTLASS example 41 (BSD-3-Clause, notices retained); see kernel/h3/UPSTREAM.md in 1Cat's flash-attention-v100.
#include "engine/video/h3_fmha.h"

#if defined( ENGINE_HAVE_CUTLASS )

#include "engine/config.h"
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include "default_fmha.h"

namespace engine::video {
namespace {

using Half = cutlass::half_t;
constexpr int kHeadDim = 128;

template <int Queries, int Keys>
using KernelFor = typename cutlass::gemm::kernel::H3FMHA<Half, cutlass::arch::Sm70, true, Queries, Keys, kHeadDim>::FMHAKernel;

template <int Queries, int Keys>
__global__ __launch_bounds__( Queries * 2, 1 ) void h3_flash_v100_d128( typename KernelFor<Queries, Keys>::DirectParams params ) {
	extern __shared__ __align__( 16 ) unsigned char storage[];
	KernelFor<Queries, Keys> kernel;
	kernel( params, *reinterpret_cast<typename KernelFor<Queries, Keys>::SharedStorage*>( storage ) );
}

template <int Queries, int Keys>
void launch( const half* q, const half* k, const half* v, half* out, int q_rows, int k_rows, int heads, float scale, bool causal, cudaStream_t stream ) {
	using Kernel = KernelFor<Queries, Keys>;
	typename Kernel::DirectParams params{ reinterpret_cast<Half*>( const_cast<half*>( q ) ), reinterpret_cast<Half*>( const_cast<half*>( k ) ),
		reinterpret_cast<Half*>( const_cast<half*>( v ) ), reinterpret_cast<Half*>( out ), q_rows, k_rows, heads, scale };
	params.causal = causal;
	if constexpr ( Keys == 128 ) {
		CUDA_CHECK( cudaFuncSetAttribute( h3_flash_v100_d128<Queries, Keys>, cudaFuncAttributePreferredSharedMemoryCarveout, 100 ) );
	}
	if constexpr ( Queries == 128 ) {
		CUDA_CHECK( cudaFuncSetAttribute( h3_flash_v100_d128<Queries, Keys>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof( typename Kernel::SharedStorage ) ) );
	}
	h3_flash_v100_d128<Queries, Keys><<<dim3( ( q_rows + Queries - 1 ) / Queries, heads ), Kernel::kThreadCount, sizeof( typename Kernel::SharedStorage ), stream>>>( params );
}

} // namespace

bool h3_fmha_available() { return true; }

bool h3_flash_attention_qk( const half* q, const half* k, const half* v, half* out, int q_rows, int k_rows, int heads, float scale, int key_tile, int query_tile, cudaStream_t stream, bool causal ) {
	if ( q_rows <= 0 ) { return true; }
	if ( key_tile == 0 ) { key_tile = k_rows >= 1024 ? 128 : 64; }
	if ( query_tile == 128 ) {
		if ( key_tile == 128 ) { launch<128, 128>( q, k, v, out, q_rows, k_rows, heads, scale, causal, stream ); } else { launch<128, 64>( q, k, v, out, q_rows, k_rows, heads, scale, causal, stream ); }
	} else {
		if ( key_tile == 128 ) { launch<64, 128>( q, k, v, out, q_rows, k_rows, heads, scale, causal, stream ); } else { launch<64, 64>( q, k, v, out, q_rows, k_rows, heads, scale, causal, stream ); }
	}
	CUDA_CHECK( cudaGetLastError() );
	return true;
}

bool h3_flash_attention( const half* q, const half* k, const half* v, half* out, int rows, int heads, float scale, int key_tile, int query_tile, cudaStream_t stream, bool causal ) {
	return h3_flash_attention_qk( q, k, v, out, rows, rows, heads, scale, key_tile, query_tile, stream, causal );
}

} // namespace engine::video

#else

namespace engine::video {
bool h3_fmha_available() { return false; }
bool h3_flash_attention( const half*, const half*, const half*, half*, int, int, float, int, int, cudaStream_t, bool ) { return false; }
bool h3_flash_attention_qk( const half*, const half*, const half*, half*, int, int, int, float, int, int, cudaStream_t, bool ) { return false; }
} // namespace engine::video

#endif
