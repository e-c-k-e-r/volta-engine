// Vulkan backend: NVFP4 W4A16 linear on src/kernels/slang/nvfp4.slang, the prepacked fragment layout of src/loader/nvfp4.cpp. Up to 8 rows
// nvfp4_gemv1 / 2 / 4 / nvfp4_gemv (a workgroup per 32-column tile, a lane per column, the warps over contiguous slices of K, the weights decoded
// once per K group for every row); more the weight decoded
// to FP16 [n, k] (nvfp4_dequant, into a scratch) for llama.cpp's FP16 matmul. The weights are the CUDA kernels' FP16 ones (block scale
// fp16( e4m3 * fp16( gscale * 2^14 ) ), weight fp16( e2m1 * scale ), both carried at 2^14), the sums FP32.
#include "mod_common.h"
#include "engine/backend/numeric.h"
#include "engine/modules/nvfp4.h"
#include <algorithm>

namespace engine::modules {

namespace {
namespace vki = ::engine::backend::impl;
struct Nvfp4GemvParams { uint64_t codes, scales, x, out_; int m, k, n; float gm; };
constexpr int kGemvRows = 8;
}

void LinearNVFP4::forward( const pod::TensorView& x, const pod::TensorView& codes, const pod::TensorView& scales, float gscale, pod::TensorView& out, backend::stream_t stream ) {
	const int m = x.shape[0], k = x.shape[1], n = out.shape[1];
	ENGINE_ASSERT( k % 64 == 0 && n % 32 == 0, "NVFP4 (Vulkan): K must be a multiple of 64 and N of 32" );
	const float gm = backend::to_float( backend::from_float_half( gscale * 16384.0f ) );
	if ( m <= kGemvRows ) {
		const Nvfp4GemvParams p{ vki::dp( codes.data ), vki::dp( scales.data ), vki::dp( x.data ), vki::dp( out.data ), m, k, n, gm };
		vki::launch( m == 1 ? "nvfp4_gemv1" : m == 2 ? "nvfp4_gemv2" : m <= 4 ? "nvfp4_gemv4" : "nvfp4_gemv", p, stream, static_cast<uint32_t>( n / 32 ) );
		return;
	}
	void* w = vki::scratch( 4, static_cast<size_t>( k ) * n * 2, stream );
	const Nvfp4GemvParams d{ vki::dp( codes.data ), vki::dp( scales.data ), 0, vki::dp( w ), m, k, n, gm };
	const int64_t lanes = static_cast<int64_t>( n / 32 ) * ( k / 16 ) * 32;
	const int64_t groups = ( lanes + 255 ) / 256;
	vki::launch( "nvfp4_dequant", d, stream, static_cast<uint32_t>( std::min<int64_t>( groups, 65535 ) ), static_cast<uint32_t>( ( groups + 65534 ) / 65535 ) );
	ENGINE_ASSERT( vki::ggml_f16_matmul_split( w, x.data, out.data, m, n, k, stream ), "NVFP4 (Vulkan): the FP16 matmul" );
}

} // namespace engine::modules
