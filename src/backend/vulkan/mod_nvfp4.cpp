// Vulkan backend: NVFP4 W4A16 linear on the Slang kernel nvfp4_gemm (src/kernels/slang/nvfp4.slang), the prepacked fragment layout of
// src/loader/nvfp4.cpp. One thread per (output column, row): slow but correct, same math and rounding points as src/backend/cpu/linear_nvfp4.cpp.
#include "mod_common.h"
#include "engine/backend/numeric.h"
#include "engine/modules/nvfp4.h"

namespace engine::modules {

namespace {
namespace vki = ::engine::backend::impl;
}

void LinearNVFP4::forward( const pod::TensorView& x, const pod::TensorView& codes, const pod::TensorView& scales, float gscale, pod::TensorView& out, backend::stream_t stream ) {
	const int m = x.shape[0], k = x.shape[1], n = out.shape[1];
	ENGINE_ASSERT( k % 64 == 0 && n % 32 == 0, "NVFP4 (Vulkan): K must be a multiple of 64 and N of 32" );
	// The CUDA kernels dequantize to FP16: block scale = fp16( e4m3 * fp16( gscale * 2^14 ) ), weight = fp16( e2m1 * scale ), both carried at 2^14
	const float gm = backend::to_float( backend::from_float_half( gscale * 16384.0f ) );
	vki::Nvfp4Params p{ vki::dp( codes.data ), vki::dp( scales.data ), vki::dp( x.data ), vki::dp( out.data ), m, k, n, gm };
	vki::launch( "nvfp4_gemm", p, stream, ( n + 63 ) / 64, m );
}

} // namespace engine::modules
