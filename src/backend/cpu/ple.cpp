// CPU backend: the Qwen4Exp n-gram embedding layer (PLE) as plain C++, the same arithmetic as src/backend/cuda/modules/ple.cu
// cpu_kernels.h first: the Slang prelude defines int4, which the backend shim only declares when the prelude is absent
#include "cpu_kernels.h"
#include "engine/modules/ple.h"
#include <cmath>

namespace engine::modules {

namespace {

using impl::h2f;
using impl::f2h;

} // namespace

void PLE::gate_and_norm( const backend::half_t* key, const backend::half_t* value, const float* streams,
	const backend::half_t* nk, const backend::half_t* nq, const backend::half_t* nc, float* gated, backend::half_t* normed,
	int rows, int hidden, int hc, float eps, float gate_scale, backend::stream_t ) {
	impl::cpu_parallel_for( static_cast<int64_t>( rows ) * hc, 4, [&]( int64_t i0, int64_t i1 ) {
		for ( int64_t i = i0; i < i1; ++i ) {
			const int r = static_cast<int>( i / hc ), s = static_cast<int>( i % hc );
			const size_t base = static_cast<size_t>( i ) * hidden;
			const backend::half_t* k = key + base;
			const float* x = streams + base;
			const backend::half_t* v = value + static_cast<size_t>( r ) * hidden;
			const backend::half_t* wks = nk + static_cast<size_t>( s ) * hidden;
			const backend::half_t* wqs = nq + static_cast<size_t>( s ) * hidden;
			const backend::half_t* wcs = nc + static_cast<size_t>( s ) * hidden;
			float sk = 0.0f, sq = 0.0f, sv = 0.0f;
			for ( int d = 0; d < hidden; ++d ) {
				const float kv = h2f( k[d] ), vv = h2f( v[d] );
				sk += kv * kv; sq += x[d] * x[d]; sv += vv * vv;
			}
			const float ik = 1.0f / std::sqrt( sk / hidden + eps ), iq = 1.0f / std::sqrt( sq / hidden + eps );
			float dot = 0.0f;
			for ( int d = 0; d < hidden; ++d ) dot += ( h2f( k[d] ) * ik * ( 1.0f + h2f( wks[d] ) ) ) * ( x[d] * iq * ( 1.0f + h2f( wqs[d] ) ) );
			const float g = dot * gate_scale;
			const float sign = g > 0.0f ? 1.0f : ( g < 0.0f ? -1.0f : 0.0f );
			const float ss = sign * std::sqrt( std::fmax( std::fabs( g ), 1e-6f ) );
			const float sg = 1.0f / ( 1.0f + std::exp( -ss ) );
			// rms of gated = |sg| * rms(value)
			const float ig = 1.0f / std::sqrt( sg * sg * sv / hidden + eps );
			for ( int d = 0; d < hidden; ++d ) {
				const float gv = sg * h2f( v[d] );
				gated[base + d] = gv;
				normed[base + d] = f2h( gv * ig * ( 1.0f + h2f( wcs[d] ) ) );
			}
		}
	} );
}

void PLE::conv_apply( const backend::half_t* normed, const backend::half_t* conv_w, const backend::half_t* ring, const pod::BatchRows* d_rows,
	const float* gated, float* streams, int rows, int channels, int dilation, backend::stream_t ) {
	impl::cpu_parallel_for( channels, 256, [&]( int64_t c0, int64_t c1 ) {
		for ( int r = 0; r < rows; ++r ) {
			const int slot = d_rows->slot[r], pos = d_rows->pos[r];
			for ( int64_t c = c0; c < c1; ++c ) {
				float acc = 0.0f;
				for ( int j = 0; j < 4; ++j ) {
					const int k = ( 3 - j ) * dilation;
					const int sp = pos - k;
					float xv = 0.0f;
					if ( sp >= 0 ) {
						const int rr = r - k;
						if ( k == 0 || ( rr >= 0 && d_rows->slot[rr] == slot && d_rows->pos[rr] == sp ) ) xv = h2f( normed[static_cast<size_t>( k == 0 ? r : rr ) * channels + c] );
						else xv = h2f( ring[( static_cast<size_t>( slot ) * PLE::RING + ( sp % PLE::RING ) ) * channels + c] );
					}
					acc += h2f( conv_w[static_cast<size_t>( c ) * 4 + j] ) * xv;
				}
				const float act = acc / ( 1.0f + std::exp( -acc ) );
				const size_t o = static_cast<size_t>( r ) * channels + c;
				streams[o] += gated[o] + act;
			}
		}
	} );
}

void PLE::ring_update( const backend::half_t* normed, backend::half_t* ring, const pod::BatchRows* d_rows, int rows, int channels, backend::stream_t ) {
	for ( int r = 0; r < rows; ++r ) {
		backend::half_t* dst = ring + ( static_cast<size_t>( d_rows->slot[r] ) * PLE::RING + ( d_rows->pos[r] % PLE::RING ) ) * channels;
		for ( int c = 0; c < channels; ++c ) dst[c] = normed[static_cast<size_t>( r ) * channels + c];
	}
}

} // namespace engine::modules
