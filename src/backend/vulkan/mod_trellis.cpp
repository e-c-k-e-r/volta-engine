// Vulkan backend: EXL3 trellis linear on src/kernels/slang/trellis.slang. The input rotated by had128 (x * suh); up to 16 rows a GEMV over the
// packed tiles (trellis_gemv1 / _gemv / _gemv16), more the weight decoded to FP16 [n, k] (trellis_dequant, into a scratch) for llama.cpp's FP16
// matmul (split over K when few output tiles would run); the output rotated back (had128, then * svh). Same math as
// src/backend/cpu/linear_trellis.cpp; every width code, both tile orders.
#include "mod_common.h"
#include "engine/modules/trellis.h"
#include "engine/modules/trellis_bits.h"
#include <algorithm>
#include <cstdio>

namespace engine::modules {

namespace {
namespace vki = ::engine::backend::impl;

constexpr float kRScale = 0.08838834764831845f; // 1 / sqrt( 128 )
constexpr int kGemvRows = 16;

struct TrHadParams { uint64_t src, dst, pre, post; int cols; float r_scale; };
struct TrGemvParams { uint64_t codes, x, y; int m, k, n, cb, volta; };
struct TrDequantParams { uint64_t codes, out_; int k, n, cb, volta; };

// The kernel of width code `code` (one SPIR-V per code): trellis_gemv_c4 and the like, the _dot form (the codebook's byte sum as a packed dot
// product) where the device has shaderIntegerDotProduct
const char* per_code( const char* stem, int code ) {
	static thread_local char name[48];
	std::snprintf( name, sizeof( name ), "%s_c%d%s", stem, code, backend::vk::int_dot() ? "_dot" : "" );
	return name;
}

void trellis_linear( const pod::TensorView& x, const pod::TensorView& trellis, const pod::TensorView& suh, const pod::TensorView& svh, pod::TensorView& out, int code, bool mcg,
	bool mul1, bool volta, bool recon, backend::stream_t stream ) {
	const int m = x.shape[0];
	const int k = trellis.shape[0] * 16;
	const int n = trellis.shape[1] * 16;
	ENGINE_ASSERT( k % 128 == 0 && n % 128 == 0, "LinearTrellis (Vulkan): dims must be multiples of 128" );
	ENGINE_ASSERT( x.dtype == pod::DType::FP16 && x.shape[1] == k, "LinearTrellis (Vulkan): x must be FP16 [m, k]" );
	ENGINE_ASSERT( out.dtype == pod::DType::FP16, "LinearTrellis (Vulkan): FP16 output only" );
	ENGINE_ASSERT( trellis.ndim < 3 || trellis.shape[2] == trellis_tile_u16( code ), "LinearTrellis (Vulkan): the tiles' width does not match the code" );
	ENGINE_ASSERT( code >= 1 && code <= 11, "LinearTrellis (Vulkan): width code out of range" );
	const int cb = mcg ? 1 : ( mul1 ? 2 : 0 );
	// Scratch slots: 0 = dequantized weight, 1 = rotated input, 2 = product
	void* xh = vki::scratch( 1, static_cast<size_t>( m ) * k * 2, stream );
	void* y = vki::scratch( 2, static_cast<size_t>( m ) * n * 2, stream );
	const TrHadParams h1{ vki::dp( x.data ), vki::dp( xh ), vki::dp( suh.data ), 0, k, kRScale };
	vki::launch( "trellis_had128", h1, stream, k / 128, m );
	if ( m <= kGemvRows && !recon ) {
		const TrGemvParams g{ vki::dp( trellis.data ), vki::dp( xh ), vki::dp( y ), m, k, n, cb, volta ? 1 : 0 };
		vki::launch( per_code( m == 1 ? "trellis_gemv1" : m <= 8 ? "trellis_gemv" : "trellis_gemv16", code ), g, stream, n / 16 );
	}
	else {
		void* w = vki::scratch( 0, static_cast<size_t>( k ) * n * 2, stream );
		const TrDequantParams d{ vki::dp( trellis.data ), vki::dp( w ), k, n, cb, volta ? 1 : 0 };
		// a workgroup per four K tiles of a column tile (k a multiple of 128: whole groups of four)
		vki::launch( per_code( "trellis_dequant", code ), d, stream, static_cast<uint32_t>( ( k / 64 ) * ( n / 16 ) ) );
		ENGINE_ASSERT( vki::ggml_f16_matmul_split( w, xh, y, m, n, k, stream ), "LinearTrellis (Vulkan): the FP16 matmul" );
	}
	const TrHadParams h2{ vki::dp( y ), vki::dp( out.data ), 0, vki::dp( svh.data ), n, kRScale };
	vki::launch( "trellis_had128", h2, stream, n / 128, m );
}

}

void LinearTrellis::forward( const pod::TensorView& x, const pod::TensorView& trellis, const pod::TensorView& suh, const pod::TensorView& svh, pod::TensorView& out, int bits, bool mcg, bool mul1, backend::stream_t stream, bool volta_tiles ) {
	trellis_linear( x, trellis, suh, svh, out, bits, mcg, mul1, volta_tiles, false, stream );
}

void LinearTrellis::forward_recon( const pod::TensorView& x, const pod::TensorView& trellis, const pod::TensorView& suh, const pod::TensorView& svh, pod::TensorView& out, int bits, bool mcg, bool mul1, backend::stream_t stream, bool volta_tiles ) {
	trellis_linear( x, trellis, suh, svh, out, bits, mcg, mul1, volta_tiles, true, stream );
}

// one projection at a time here
bool trellis_forward_multi( const pod::TensorView*, int, const pod::TensorView*, const pod::TensorView*, const pod::TensorView*, pod::TensorView* const*, int, bool, bool,
	backend::stream_t, bool ) {
	return false;
}

bool trellis_volta_tiles_supported() {
	return true;
}

} // namespace engine::modules
