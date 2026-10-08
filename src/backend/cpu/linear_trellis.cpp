// CPU backend: EXL3 trellis linear (rotate, dequantize, GEMM, rotate) on the E1 reference Slang host kernels
// (src/shaders/ref_gemm.slang, ref_trellis.slang). Bit-exact dequantization vs the engine's reconstruct; the fused CUDA
// kernels and the large-M recon path differ from this only by FP16/FP32 rounding order.
// cpu_kernels.h first: the Slang prelude defines int4, which the backend shim only declares when the prelude is absent
#include "cpu_kernels.h"
#include "engine/modules/trellis.h"
#include "engine/backend/numeric.h"
#include <vector>
#include <cmath>

extern "C" {
void ref_gemm_f16( ComputeVaryingInput*, void*, void* );
void ref_trellis_reconstruct( ComputeVaryingInput*, void*, void* );
void ref_had128( ComputeVaryingInput*, void*, void* );
}

struct RefGemmParams { half* x; half* w; half* out_; int m; int k; int n; };
struct RefReconParams { uint16_t* codes; half* out_; int k_dim; int n_dim; int bits; int cb; };
struct RefHadParams { half* src; half* dst; half* pre; half* post; int cols; float r_scale; };

namespace engine::modules {

namespace {

constexpr float kRScale = 0.08838834764831845f; // 1 / sqrt( 128 )

// Blockwise 128-point Hadamard over the last dim of [rows, cols]: dst = post * ( r_scale * H( pre * src ) ); dst != src
void had_rows( const half* src, half* dst, const half* pre, const half* post, int rows, int cols ) {
	RefHadParams p{ const_cast<half*>( src ), dst, const_cast<half*>( pre ), const_cast<half*>( post ), cols, kRScale };
	impl::cpu_dispatch( ref_had128, &p, static_cast<uint32_t>( cols / 128 ), static_cast<uint32_t>( rows ) );
}

// Codebook index of the reference decoder: 0 = 3INST, 1 = MCG, 2 = MUL1
int codebook( bool mcg, bool mul1 ) {
	return mcg ? 1 : ( mul1 ? 2 : 0 );
}

float half_to_float( half h ) {
	return static_cast<float>( h );
}

// FP32 output (out.dtype == FP32): FP32 GEMM over the dequantized weight and an FP32 Hadamard, no FP16 rounding of the result
void linear_fp32_out( const std::vector<half>& xh, const std::vector<half>& w, const half* svh, float* out, int m, int k, int n ) {
	std::vector<float> acc( static_cast<size_t>( n ) );
	for ( int r = 0; r < m; ++r ) {
		for ( int c = 0; c < n; ++c ) acc[c] = 0.0f;
		for ( int i = 0; i < k; ++i ) {
			const float xv = half_to_float( xh[static_cast<size_t>( r ) * k + i] );
			const half* wr = w.data() + static_cast<size_t>( i ) * n;
			for ( int c = 0; c < n; ++c ) acc[c] += xv * half_to_float( wr[c] );
		}
		for ( int b = 0; b < n / 128; ++b ) {
			float* v = acc.data() + b * 128;
			for ( int len = 1; len < 128; len <<= 1 ) {
				for ( int i = 0; i < 128; i += 2 * len ) {
					for ( int j = i; j < i + len; ++j ) {
						const float a = v[j], c = v[j + len];
						v[j] = a + c;
						v[j + len] = a - c;
					}
				}
			}
		}
		for ( int c = 0; c < n; ++c ) out[static_cast<size_t>( r ) * n + c] = acc[c] * kRScale * half_to_float( svh[c] );
	}
}

void trellis_linear( const pod::TensorView& x, const pod::TensorView& trellis, const pod::TensorView& suh, const pod::TensorView& svh, pod::TensorView& out, int bits, bool mcg, bool mul1 ) {
	const int m = x.shape[0];
	const int k = trellis.shape[0] * 16;
	const int n = trellis.shape[1] * 16;
	ENGINE_ASSERT( k % 128 == 0 && n % 128 == 0, "LinearTrellis (CPU): dims must be multiples of 128" );
	ENGINE_ASSERT( x.dtype == pod::DType::FP16 && x.shape[1] == k, "LinearTrellis (CPU): x must be FP16 [m, k]" );
	ENGINE_ASSERT( out.dtype == pod::DType::FP16 || out.dtype == pod::DType::FP32, "LinearTrellis (CPU): FP16 or FP32 output only" );
	const int cb = codebook( mcg, mul1 );
	std::vector<half> xh( static_cast<size_t>( m ) * k );
	std::vector<half> w( static_cast<size_t>( k ) * n );
	had_rows( x.ptr<const half>(), xh.data(), suh.ptr<const half>(), nullptr, m, k );
	RefReconParams rp{ const_cast<uint16_t*>( trellis.ptr<const uint16_t>() ), w.data(), k, n, bits, cb };
	impl::cpu_dispatch( ref_trellis_reconstruct, &rp, static_cast<uint32_t>( ( static_cast<size_t>( k ) * n + 255 ) / 256 ) );
	if ( out.dtype == pod::DType::FP32 ) {
		linear_fp32_out( xh, w, svh.ptr<const half>(), out.ptr<float>(), m, k, n );
		return;
	}
	std::vector<half> y( static_cast<size_t>( m ) * n );
	RefGemmParams gp{ xh.data(), w.data(), y.data(), m, k, n };
	impl::cpu_dispatch( ref_gemm_f16, &gp, static_cast<uint32_t>( n ), static_cast<uint32_t>( m ) );
	had_rows( y.data(), out.ptr<half>(), nullptr, svh.ptr<const half>(), m, n );
}

} // namespace

void LinearTrellis::forward( const pod::TensorView& x, const pod::TensorView& trellis, const pod::TensorView& suh, const pod::TensorView& svh, pod::TensorView& out, int bits, bool mcg, bool mul1, backend::stream_t ) {
	trellis_linear( x, trellis, suh, svh, out, bits, mcg, mul1 );
}

void LinearTrellis::forward_recon( const pod::TensorView& x, const pod::TensorView& trellis, const pod::TensorView& suh, const pod::TensorView& svh, pod::TensorView& out, int bits, bool mcg, bool mul1, backend::stream_t ) {
	trellis_linear( x, trellis, suh, svh, out, bits, mcg, mul1 );
}

} // namespace engine::modules
