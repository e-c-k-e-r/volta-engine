// Vulkan backend: EXL3 trellis linear (rotate, dequantize, GEMM, rotate) on the portable reference kernels ref_had128,
// ref_trellis_reconstruct and ref_gemm_f16. The dequantized [k, n] weight of the call goes to a device scratch buffer, so the
// cost per call is one reconstruction pass plus a tiled GEMM: correct, not fast. Same math as src/backend/cpu/linear_trellis.cpp.
#include "mod_common.h"
#include "engine/modules/trellis.h"

namespace engine::modules {

namespace {
namespace vki = ::engine::backend::impl;

constexpr float kRScale = 0.08838834764831845f; // 1 / sqrt( 128 )

void trellis_linear( const pod::TensorView& x, const pod::TensorView& trellis, const pod::TensorView& suh, const pod::TensorView& svh, pod::TensorView& out, int bits, bool mcg, bool mul1, backend::stream_t stream ) {
	const int m = x.shape[0];
	const int k = trellis.shape[0] * 16;
	const int n = trellis.shape[1] * 16;
	ENGINE_ASSERT( k % 128 == 0 && n % 128 == 0, "LinearTrellis (Vulkan): dims must be multiples of 128" );
	ENGINE_ASSERT( x.dtype == pod::DType::FP16 && x.shape[1] == k, "LinearTrellis (Vulkan): x must be FP16 [m, k]" );
	ENGINE_ASSERT( out.dtype == pod::DType::FP16, "LinearTrellis (Vulkan): FP16 output only" );
	const int cb = mcg ? 1 : ( mul1 ? 2 : 0 );
	const size_t kn = static_cast<size_t>( k ) * n;
	// Scratch slots: 0 = dequantized weight, 1 = rotated input, 2 = GEMM output
	void* w = vki::scratch( 0, kn * 2, stream );
	void* xh = vki::scratch( 1, static_cast<size_t>( m ) * k * 2, stream );
	void* y = vki::scratch( 2, static_cast<size_t>( m ) * n * 2, stream );
	vki::RefHadParams h1{ vki::dp( x.data ), vki::dp( xh ), vki::dp( suh.data ), 0, k, kRScale };
	vki::launch( "ref_had128", h1, stream, k / 128, m );
	vki::RefReconParams rp{ vki::dp( trellis.data ), vki::dp( w ), k, n, bits, cb };
	vki::launch( "ref_trellis_reconstruct", rp, stream, static_cast<uint32_t>( ( kn + 255 ) / 256 ) );
	vki::RefGemmParams gp{ vki::dp( xh ), vki::dp( w ), vki::dp( y ), m, k, n };
	vki::launch( "ref_gemm_f16", gp, stream, ( n + 15 ) / 16, ( m + 15 ) / 16 );
	vki::RefHadParams h2{ vki::dp( y ), vki::dp( out.data ), 0, vki::dp( svh.data ), n, kRScale };
	vki::launch( "ref_had128", h2, stream, n / 128, m );
}

}

void LinearTrellis::forward( const pod::TensorView& x, const pod::TensorView& trellis, const pod::TensorView& suh, const pod::TensorView& svh, pod::TensorView& out, int bits, bool mcg, bool mul1, backend::stream_t stream ) {
	trellis_linear( x, trellis, suh, svh, out, bits, mcg, mul1, stream );
}

void LinearTrellis::forward_recon( const pod::TensorView& x, const pod::TensorView& trellis, const pod::TensorView& suh, const pod::TensorView& svh, pod::TensorView& out, int bits, bool mcg, bool mul1, backend::stream_t stream ) {
	trellis_linear( x, trellis, suh, svh, out, bits, mcg, mul1, stream );
}

} // namespace engine::modules
