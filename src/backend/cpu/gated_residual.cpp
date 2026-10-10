// CPU backend: Qwen4Exp gated residual (fp32 streams) as plain C++, the same arithmetic as the CUDA kernels (src/backend/cuda/modules/gated_residual.cu)
// cpu_kernels.h first: the Slang prelude defines int4, which the backend shim only declares when the prelude is absent
#include "cpu_kernels.h"
#include "engine/modules/gated_residual.h"
#include <cmath>

namespace engine::modules {

namespace {

inline float h2f( backend::half_t v ) { return static_cast<float>( v ); }
inline backend::half_t f2h( float v ) { return static_cast<backend::half_t>( v ); }
inline float sigmoid( float v ) { return 1.0f / ( 1.0f + std::exp( -v ) ); }

} // namespace

void GatedResidual::init_streams( const backend::half_t* in, float* x, int rows, int hidden, int hc, backend::stream_t ) {
	for ( int r = 0; r < rows; ++r ) {
		for ( int s = 0; s < hc; ++s ) {
			float* dst = x + ( static_cast<size_t>( r ) * hc + s ) * hidden;
			for ( int d = 0; d < hidden; ++d ) dst[d] = h2f( in[static_cast<size_t>( r ) * hidden + d] );
		}
	}
}

void GatedResidual::norm( float* x, const backend::half_t* norm_w, float* normed, backend::half_t* normed_hi, backend::half_t* normed_lo, int rows, int hidden, int hc, float eps,
	const float* prev_post, const backend::half_t* prev_y, backend::stream_t ) {
	impl::cpu_parallel_for( static_cast<int64_t>( rows ) * hc, 4, [&]( int64_t i0, int64_t i1 ) {
		for ( int64_t i = i0; i < i1; ++i ) {
			const int r = static_cast<int>( i / hc ), s = static_cast<int>( i % hc );
			const size_t base = static_cast<size_t>( i ) * hidden;
			const float p = prev_y != nullptr ? prev_post[i] : 0.0f;
			float sq = 0.0f;
			for ( int d = 0; d < hidden; ++d ) {
				float v = x[base + d];
				if ( prev_y != nullptr ) {
					v += p * h2f( prev_y[static_cast<size_t>( r ) * hidden + d] );
					x[base + d] = v;
				}
				sq += v * v;
			}
			const float inv = 1.0f / std::sqrt( sq / hidden + eps );
			for ( int d = 0; d < hidden; ++d ) {
				const float nv = x[base + d] * inv * ( 1.0f + h2f( norm_w[static_cast<size_t>( s ) * hidden + d] ) );
				normed[base + d] = nv;
				if ( normed_hi != nullptr ) {
					const backend::half_t hi = f2h( nv );
					normed_hi[base + d] = hi;
					normed_lo[base + d] = f2h( nv - h2f( hi ) );
				}
			}
		}
	} );
}

void GatedResidual::mix(
	const float* x, const backend::half_t* norm_w, const backend::half_t* down, const backend::half_t* up, const backend::half_t* inject,
	backend::half_t* mixed, float* post, float* scratch, int rows, int hidden, int hc, int rank, float eps, backend::stream_t stream,
	const float* prev_post, const backend::half_t* prev_y
) {
	const int flat = hc * hidden;
	float* normed = scratch;
	float* t = scratch + static_cast<size_t>( rows ) * flat;
	norm( const_cast<float*>( x ), norm_w, normed, nullptr, nullptr, rows, hidden, hc, eps, prev_post, prev_y, stream );
	const int outs = rank + ( inject != nullptr ? hc : 0 );
	// t = silu( normed . down / hc ), post = 2 * sigmoid( normed . inject / hc )
	impl::cpu_parallel_for( outs, 8, [&]( int64_t j0, int64_t j1 ) {
		for ( int64_t j = j0; j < j1; ++j ) {
			const backend::half_t* w = j < rank ? down + static_cast<size_t>( j ) * flat : inject + static_cast<size_t>( j - rank ) * flat;
			for ( int r = 0; r < rows; ++r ) {
				const float* nr = normed + static_cast<size_t>( r ) * flat;
				float acc = 0.0f;
				for ( int k = 0; k < flat; ++k ) acc += nr[k] * h2f( w[k] );
				const float z = acc / hc;
				if ( j < rank ) t[static_cast<size_t>( r ) * rank + j] = z / ( 1.0f + std::exp( -z ) );
				else post[static_cast<size_t>( r ) * hc + ( j - rank )] = 2.0f * sigmoid( z );
			}
		}
	} );
	// mixed[r, d] = mean_s( sigmoid( up[s * hidden + d] . t[r] ) * normed[r, s, d] )
	impl::cpu_parallel_for( hidden, 16, [&]( int64_t d0, int64_t d1 ) {
		for ( int64_t d = d0; d < d1; ++d ) {
			for ( int r = 0; r < rows; ++r ) {
				const float* tr = t + static_cast<size_t>( r ) * rank;
				float out = 0.0f;
				for ( int s = 0; s < hc; ++s ) {
					const backend::half_t* u = up + ( static_cast<size_t>( s ) * hidden + d ) * rank;
					float acc = 0.0f;
					for ( int k = 0; k < rank; ++k ) acc += tr[k] * h2f( u[k] );
					out += sigmoid( acc ) * normed[( static_cast<size_t>( r ) * hc + s ) * hidden + d];
				}
				mixed[static_cast<size_t>( r ) * hidden + d] = f2h( out / hc );
			}
		}
	} );
}

void GatedResidual::gemm_f32( const backend::half_t* a, const backend::half_t* w, float* out, int m, int n, int k, bool accumulate, backend::stream_t ) {
	impl::cpu_parallel_for( n, 16, [&]( int64_t c0, int64_t c1 ) {
		for ( int64_t c = c0; c < c1; ++c ) {
			const backend::half_t* wr = w + static_cast<size_t>( c ) * k;
			for ( int r = 0; r < m; ++r ) {
				const backend::half_t* ar = a + static_cast<size_t>( r ) * k;
				float acc = 0.0f;
				for ( int i = 0; i < k; ++i ) acc += h2f( ar[i] ) * h2f( wr[i] );
				float& o = out[static_cast<size_t>( r ) * n + c];
				o = accumulate ? o + acc : acc;
			}
		}
	} );
}

void GatedResidual::gate_act( const float* z, int ld, backend::half_t* t_hi, backend::half_t* t_lo, float* post, int rows, int rank, int hc, bool has_inject, backend::stream_t ) {
	for ( int r = 0; r < rows; ++r ) {
		const float* zr = z + static_cast<size_t>( r ) * ld;
		for ( int j = 0; j < rank + ( has_inject ? hc : 0 ); ++j ) {
			const float v = zr[j] / hc;
			if ( j < rank ) {
				const float tv = v / ( 1.0f + std::exp( -v ) );
				const backend::half_t hi = f2h( tv );
				t_hi[static_cast<size_t>( r ) * rank + j] = hi;
				t_lo[static_cast<size_t>( r ) * rank + j] = f2h( tv - h2f( hi ) );
			}
			else post[static_cast<size_t>( r ) * hc + ( j - rank )] = 2.0f * sigmoid( v );
		}
	}
}

void GatedResidual::up_epilogue( const float* u, const float* normed, backend::half_t* mixed, int rows, int hidden, int hc, backend::stream_t ) {
	for ( int r = 0; r < rows; ++r ) {
		for ( int d = 0; d < hidden; ++d ) {
			float out = 0.0f;
			for ( int s = 0; s < hc; ++s ) out += sigmoid( u[static_cast<size_t>( r ) * hc * hidden + static_cast<size_t>( s ) * hidden + d] ) * normed[( static_cast<size_t>( r ) * hc + s ) * hidden + d];
			mixed[static_cast<size_t>( r ) * hidden + d] = f2h( out / hc );
		}
	}
}

void GatedResidual::add_sigmoid_gated( backend::half_t* y, const backend::half_t* s, const backend::half_t* x, const backend::half_t* w, int rows, int hidden, backend::stream_t ) {
	for ( int r = 0; r < rows; ++r ) {
		const backend::half_t* xr = x + static_cast<size_t>( r ) * hidden;
		float acc = 0.0f;
		for ( int i = 0; i < hidden; ++i ) acc += h2f( xr[i] ) * h2f( w[i] );
		const float gate = sigmoid( acc );
		for ( int i = 0; i < hidden; ++i ) {
			const size_t o = static_cast<size_t>( r ) * hidden + i;
			y[o] = f2h( h2f( y[o] ) + gate * h2f( s[o] ) );
		}
	}
}

void GatedResidual::stream_norm( const float* x, const backend::half_t* w, backend::half_t* out, int rows, int hidden, int hc, float eps, backend::stream_t ) {
	for ( int i = 0; i < rows * hc; ++i ) {
		const int s = i % hc;
		const size_t base = static_cast<size_t>( i ) * hidden;
		float sq = 0.0f;
		for ( int d = 0; d < hidden; ++d ) sq += x[base + d] * x[base + d];
		const float inv = 1.0f / std::sqrt( sq / hidden + eps );
		for ( int d = 0; d < hidden; ++d ) out[base + d] = f2h( x[base + d] * inv * ( 1.0f + h2f( w[static_cast<size_t>( s ) * hidden + d] ) ) );
	}
}

void GatedResidual::add_broadcast( const backend::half_t* h, const backend::half_t* e, float* streams, int rows, int hidden, int hc, backend::stream_t ) {
	for ( int r = 0; r < rows; ++r ) {
		for ( int s = 0; s < hc; ++s ) {
			const size_t o = ( static_cast<size_t>( r ) * hc + s ) * hidden;
			for ( int d = 0; d < hidden; ++d ) streams[o + d] = h2f( h[o + d] ) + h2f( e[static_cast<size_t>( r ) * hidden + d] );
		}
	}
}

void GatedResidual::apply( float* x, const float* post, const backend::half_t* y, int rows, int hidden, int hc, backend::stream_t ) {
	for ( int r = 0; r < rows; ++r ) {
		for ( int s = 0; s < hc; ++s ) {
			const float p = post[static_cast<size_t>( r ) * hc + s];
			float* xs = x + ( static_cast<size_t>( r ) * hc + s ) * hidden;
			for ( int d = 0; d < hidden; ++d ) xs[d] += p * h2f( y[static_cast<size_t>( r ) * hidden + d] );
		}
	}
}

} // namespace engine::modules
