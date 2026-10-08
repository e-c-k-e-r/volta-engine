// Vulkan backend: dense FP16 GEMM (the Linear FP16 path) through the portable reference kernel ref_gemm_nt_f16 (16x16 tiles, FP32 accumulation, slow but correct). Quantized formats stay weak stubs
#include "mod_common.h"
#include "engine/modules/linear.h"

namespace engine::modules {

namespace {
namespace vki = ::engine::backend::impl;
}

// out[m][n] = sum_k x[m][k] * w[n][k]; w is [n, k] row-major in memory (the loader layout, as dflash_f16_gemv_nt)
void LinearPlainGEMV::forward(
	const backend::half_t* w,
	const backend::half_t* x,
	backend::half_t* out,
	int m,
	int n,
	int k,
	backend::stream_t stream
) {
	vki::RefGemmParams p{ vki::dp( x ), vki::dp( w ), vki::dp( out ), m, k, n };
	vki::launch( "ref_gemm_nt_f16", p, stream, ( n + 15 ) / 16, ( m + 15 ) / 16 );
}

} // namespace engine::modules
