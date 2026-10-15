// CPU backend: NVFP4 W4A16 linear on the prepacked fragment layout of src/loader/nvfp4.cpp (streams [tile][group][lane],
// 32 columns per tile, 16-wide K groups). Each output column is decoded once to FP32 (E2M1 * E4M3 block scale * global scale)
// and dotted with every activation row; FP32 accumulation, FP16 result.
#include "cpu_kernels.h"
#include "engine/modules/nvfp4.h"
#include "engine/quant/fp4_fp8.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace engine::modules {

namespace {


float r16( float f ) {
	return static_cast<float>( static_cast<backend::half_t>( f ) );
}

const float* scale_lut() {
	static const std::vector<float> lut = [] {
		std::vector<float> v( 256 );
		for ( int i = 0; i < 256; ++i ) v[i] = quant::e4m3_to_float( static_cast<unsigned>( i ) );
		return v;
	}();
	return lut.data();
}

} // namespace

void LinearNVFP4::forward(
	const pod::TensorView& x,
	const pod::TensorView& codes,
	const pod::TensorView& scales,
	float gscale,
	pod::TensorView& out,
	backend::stream_t
) {
	const int m = x.shape[0], k = x.shape[1], n = out.shape[1];
	ENGINE_ASSERT( k % 64 == 0 && n % 32 == 0, "NVFP4: K must be a multiple of 64 and N of 32" );
	const int G = k / 16;
	const float* sc_lut = scale_lut();
	// The CUDA kernels dequantize to FP16: block scale = fp16( e4m3 * fp16( gscale * 2^14 ) ), weight = fp16( e2m1 * scale ), both
	// carried at 2^14. Rounding at the same points keeps the logits within the FP16 noise of the device path.
	const float gm = r16( gscale * 16384.0f );

	const uint8_t* cp = codes.ptr<const uint8_t>();
	const uint8_t* sp = scales.ptr<const uint8_t>();
	const backend::half_t* xp = x.ptr<const backend::half_t>();
	backend::half_t* op = out.ptr<backend::half_t>();

	std::vector<float> xf( static_cast<size_t>( m ) * k );
	for ( size_t i = 0; i < xf.size(); ++i ) xf[i] = static_cast<float>( xp[i] );

	impl::cpu_parallel_for( n, 8, [&]( int64_t c0, int64_t c1 ) {
		std::vector<float> w( k );
		for ( int64_t c = c0; c < c1; ++c ) {
			const int t = static_cast<int>( c / 32 ), j = static_cast<int>( c % 32 );
			const int qp = j >> 3, r = j & 7;
			const int lane = qp * 4 + ( r & 3 ) + ( ( r >= 4 ) ? 16 : 0 );
			for ( int g = 0; g < G; ++g ) {
				const size_t at = ( static_cast<size_t>( t ) * G + g ) * 32 + lane;
				uint32_t cw[2];
				std::memcpy( cw, cp + at * 8, 8 );
				const float s = r16( sc_lut[sp[at]] * gm );
				float* wg = &w[g * 16];
				for ( int h = 0; h < 2; ++h ) {
					for ( int i = 0; i < 4; ++i ) {
						wg[h * 8 + 2 * i] = r16( quant::e2m1_to_float( static_cast<int>( ( cw[h] >> ( 4 * i ) ) & 15 ) ) * s ) * ( 1.0f / 16384.0f );
						wg[h * 8 + 2 * i + 1] = r16( quant::e2m1_to_float( static_cast<int>( ( cw[h] >> ( 16 + 4 * i ) ) & 15 ) ) * s ) * ( 1.0f / 16384.0f );
					}
				}
			}
			for ( int r2 = 0; r2 < m; ++r2 ) {
				const float* xr = &xf[static_cast<size_t>( r2 ) * k];
				float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
				for ( int i = 0; i < k; i += 4 ) {
					a0 += xr[i] * w[i];
					a1 += xr[i + 1] * w[i + 1];
					a2 += xr[i + 2] * w[i + 2];
					a3 += xr[i + 3] * w[i + 3];
				}
				op[static_cast<size_t>( r2 ) * n + c] = static_cast<backend::half_t>( ( a0 + a1 ) + ( a2 + a3 ) );
			}
		}
	} );
}

} // namespace engine::modules
