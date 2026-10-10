// CPU backend: the vision-tower operators as plain C++ (FP32 accumulation, FP16 storage), the same arithmetic as src/backend/cuda/modules/vision.cu:
// layer norms, the patch embedding, the interpolated position table, the 2-D rotary with the q / k / v biases, bidirectional attention, the
// bias / activation / residual helpers and the patch merger's gather. The GEMMs go through the plain FP16 GEMV.
// cpu_kernels.h first: the Slang prelude defines int4, which the backend shim only declares when the prelude is absent
#include "cpu_kernels.h"
#include "engine/modules/vision.h"
#include "engine/modules/linear.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace engine::modules {

namespace {

using h16 = backend::half_t;

inline float h2f( h16 v ) { return static_cast<float>( v ); }
inline h16 f2h( float v ) { return static_cast<h16>( v ); }

float gelu_tanh_f( float x ) {
	const float u = 0.7978845608028654f * ( x + 0.044715f * x * x * x );
	return 0.5f * x * ( 1.0f + std::tanh( u ) );
}

float gelu_erf_f( float x ) {
	return 0.5f * x * ( 1.0f + std::erf( x * 0.7071067811865476f ) );
}

// LayerNorm of one row (FP32 in); the output row lands in out row / group at column offset ( row % group ) * n
void layer_norm_row( const float* xr, const h16* w, const h16* b, h16* o, int n, float eps ) {
	float s = 0.0f;
	for ( int i = 0; i < n; ++i ) s += xr[i];
	const float mean = s / n;
	float q = 0.0f;
	for ( int i = 0; i < n; ++i ) q += ( xr[i] - mean ) * ( xr[i] - mean );
	const float inv = 1.0f / std::sqrt( q / n + eps );
	for ( int i = 0; i < n; ++i ) o[i] = f2h( ( xr[i] - mean ) * inv * h2f( w[i] ) + h2f( b[i] ) );
}

void layer_norm_rows( const float* x, const h16* w, const h16* b, h16* out, int m, int n, int group, float eps ) {
	impl::cpu_parallel_for( m, 8, [&]( int64_t r0, int64_t r1 ) {
		for ( int64_t r = r0; r < r1; ++r ) layer_norm_row( x + static_cast<size_t>( r ) * n, w, b, out + static_cast<size_t>( r / group ) * group * n + static_cast<size_t>( r % group ) * n, n, eps );
	} );
}

} // namespace

void Vision::f32_to_f16( const float* in, h16* out, size_t n, backend::stream_t ) {
	for ( size_t i = 0; i < n; ++i ) out[i] = f2h( in[i] );
}

void Vision::pos_interp( const h16* table, h16* out, int gh, int gw, int merge, int side, int dim, backend::stream_t ) {
	const int blocks_w = gw / merge, mm = merge * merge;
	impl::cpu_parallel_for( static_cast<int64_t>( gh ) * gw, 16, [&]( int64_t r0, int64_t r1 ) {
		for ( int64_t row = r0; row < r1; ++row ) {
			const int blk = static_cast<int>( row / mm ), within = static_cast<int>( row % mm );
			const int by = blk / blocks_w, bx = blk % blocks_w;
			const int y = by * merge + within / merge, x = bx * merge + within % merge;
			const float hy = gh == 1 ? 0.0f : static_cast<float>( y ) * ( side - 1 ) / ( gh - 1 );
			const float wx = gw == 1 ? 0.0f : static_cast<float>( x ) * ( side - 1 ) / ( gw - 1 );
			const int hf = static_cast<int>( hy ), wf = static_cast<int>( wx );
			const int hc = std::min( hf + 1, side - 1 ), wc = std::min( wf + 1, side - 1 );
			const float dh = hy - hf, dw = wx - wf;
			const h16* t00 = table + ( static_cast<size_t>( hf ) * side + wf ) * dim;
			const h16* t01 = table + ( static_cast<size_t>( hf ) * side + wc ) * dim;
			const h16* t10 = table + ( static_cast<size_t>( hc ) * side + wf ) * dim;
			const h16* t11 = table + ( static_cast<size_t>( hc ) * side + wc ) * dim;
			for ( int d = 0; d < dim; ++d ) {
				const float val = ( 1 - dh ) * ( 1 - dw ) * h2f( t00[d] ) + ( 1 - dh ) * dw * h2f( t01[d] ) + dh * ( 1 - dw ) * h2f( t10[d] ) + dh * dw * h2f( t11[d] );
				out[static_cast<size_t>( row ) * dim + d] = f2h( val );
			}
		}
	} );
}

void Vision::patch_embed( const h16* patches, const h16* w, const h16* bias, const h16* add, float* out, int m, int n, int k, backend::stream_t ) {
	impl::cpu_parallel_for( n, 16, [&]( int64_t c0, int64_t c1 ) {
		for ( int64_t c = c0; c < c1; ++c ) {
			const h16* wr = w + static_cast<size_t>( c ) * k;
			for ( int r = 0; r < m; ++r ) {
				const h16* pr = patches + static_cast<size_t>( r ) * k;
				float acc = 0.0f;
				for ( int i = 0; i < k; ++i ) acc += h2f( pr[i] ) * h2f( wr[i] );
				out[static_cast<size_t>( r ) * n + c] = acc + h2f( bias[c] ) + ( add != nullptr ? h2f( add[static_cast<size_t>( r ) * n + c] ) : 0.0f );
			}
		}
	} );
}

void Vision::rms_norm( const float* x, const h16* w, h16* out, int m, int n, float eps, backend::stream_t ) {
	for ( int r = 0; r < m; ++r ) {
		const float* xr = x + static_cast<size_t>( r ) * n;
		float q = 0.0f;
		for ( int i = 0; i < n; ++i ) q += xr[i] * xr[i];
		const float inv = 1.0f / std::sqrt( q / n + eps );
		for ( int i = 0; i < n; ++i ) out[static_cast<size_t>( r ) * n + i] = f2h( xr[i] * inv * h2f( w[i] ) );
	}
}

void Vision::layer_norm_half( const h16* x, const h16* w, const h16* b, h16* out, int m, int n, float eps, backend::stream_t ) {
	std::vector<float> row( n );
	for ( int r = 0; r < m; ++r ) {
		for ( int i = 0; i < n; ++i ) row[i] = h2f( x[static_cast<size_t>( r ) * n + i] );
		layer_norm_row( row.data(), w, b, out + static_cast<size_t>( r ) * n, n, eps );
	}
}

void Vision::layer_norm( const float* x, const h16* w, const h16* b, h16* out, int m, int n, float eps, backend::stream_t ) {
	layer_norm_rows( x, w, b, out, m, n, 1, eps );
}

void Vision::layer_norm_merge( const float* x, const h16* w, const h16* b, h16* out, int m, int n, int group, float eps, backend::stream_t ) {
	ENGINE_ASSERT( m % group == 0, "Vision::layer_norm_merge: rows must be a multiple of the group" );
	layer_norm_rows( x, w, b, out, m, n, group, eps );
}

void Vision::qkv_bias_norm_rope( h16* q, h16* k, h16* v, const h16* bq, const h16* bk, const h16* bv, const h16* qn, const h16* kn, const float* cos_table, const float* sin_table,
	int m, int heads, int head_dim, float eps, backend::stream_t ) {
	ENGINE_ASSERT( head_dim == 64 && heads <= 32, "Vision::qkv_bias_norm_rope: 64-wide heads, at most 32" );
	const int width = heads * 64;
	impl::cpu_parallel_for( m, 4, [&]( int64_t r0, int64_t r1 ) {
		for ( int64_t row = r0; row < r1; ++row ) {
			const float* cs = cos_table + static_cast<size_t>( row ) * 64;
			const float* sn = sin_table + static_cast<size_t>( row ) * 64;
			for ( int head = 0; head < heads; ++head ) {
				const size_t base = static_cast<size_t>( row ) * width + head * 64;
				for ( int which = 0; which < 3; ++which ) {
					h16* t = which == 0 ? q : ( which == 1 ? k : v );
					const h16* bias = which == 0 ? bq : ( which == 1 ? bk : bv );
					float a[32], b[32];
					float ss = 0.0f;
					for ( int d = 0; d < 32; ++d ) {
						a[d] = h2f( t[base + d] ) + h2f( bias[head * 64 + d] );
						b[d] = h2f( t[base + d + 32] ) + h2f( bias[head * 64 + d + 32] );
						ss += a[d] * a[d] + b[d] * b[d];
					}
					if ( which < 2 ) {
						const h16* gain = which == 0 ? qn : kn;
						const float inv = 1.0f / std::sqrt( ss / 64.0f + eps );
						for ( int d = 0; d < 32; ++d ) {
							const float na = a[d] * inv * h2f( gain[d] ), nb = b[d] * inv * h2f( gain[d + 32] );
							a[d] = na * cs[d] - nb * sn[d];
							b[d] = nb * cs[d + 32] + na * sn[d + 32];
						}
					}
					for ( int d = 0; d < 32; ++d ) {
						t[base + d] = f2h( a[d] );
						t[base + d + 32] = f2h( b[d] );
					}
				}
			}
		}
	} );
}

void Vision::qkv_bias_rope( h16* q, h16* k, h16* v, const h16* bq, const h16* bk, const h16* bv, const float* cos_table, const float* sin_table, int m, int heads, int head_dim, backend::stream_t ) {
	ENGINE_ASSERT( head_dim % 2 == 0, "Vision::qkv_bias_rope: even head_dim" );
	const int half_dim = head_dim / 2, width = heads * head_dim;
	impl::cpu_parallel_for( m, 4, [&]( int64_t r0, int64_t r1 ) {
		for ( int64_t row = r0; row < r1; ++row ) {
			const size_t base = static_cast<size_t>( row ) * width;
			for ( int h = 0; h < heads; ++h ) {
				for ( int d = 0; d < half_dim; ++d ) {
					const int i1 = h * head_dim + d, i2 = i1 + half_dim;
					const float c1 = cos_table[static_cast<size_t>( row ) * head_dim + d], s1 = sin_table[static_cast<size_t>( row ) * head_dim + d];
					const float c2 = cos_table[static_cast<size_t>( row ) * head_dim + d + half_dim], s2 = sin_table[static_cast<size_t>( row ) * head_dim + d + half_dim];
					{
						const float a = h2f( q[base + i1] ) + h2f( bq[i1] ), b = h2f( q[base + i2] ) + h2f( bq[i2] );
						q[base + i1] = f2h( a * c1 - b * s1 );
						q[base + i2] = f2h( b * c2 + a * s2 );
					}
					{
						const float a = h2f( k[base + i1] ) + h2f( bk[i1] ), b = h2f( k[base + i2] ) + h2f( bk[i2] );
						k[base + i1] = f2h( a * c1 - b * s1 );
						k[base + i2] = f2h( b * c2 + a * s2 );
					}
					v[base + i1] = f2h( h2f( v[base + i1] ) + h2f( bv[i1] ) );
					v[base + i2] = f2h( h2f( v[base + i2] ) + h2f( bv[i2] ) );
				}
			}
		}
	} );
}

void Vision::merge_gather( const h16* h, h16* out, int tokens, int n, int group, backend::stream_t ) {
	for ( int t = 0; t < tokens; ++t ) {
		for ( int idx = 0; idx < n * group; ++idx ) {
			const int c = idx / group, j = idx % group;
			out[static_cast<size_t>( t ) * n * group + idx] = h[( static_cast<size_t>( t ) * group + j ) * n + c];
		}
	}
}

// out[m][n] = a[m][k] . w[n][k]: a plain FP16 GEMM (Flash-Next's n-gram projections use it too, not only the vision tower)
void Vision::dense_gemm( const h16* a, const h16* w, h16* out, int m, int n, int k, backend::stream_t stream ) {
	LinearPlainGEMV::forward( w, a, out, m, n, k, stream );
}

void Vision::attention( const h16* q, const h16* k, const h16* v, h16* out, int m, int heads, int head_dim, float scale, backend::stream_t stream ) {
	attention_simt( q, k, v, out, m, heads, head_dim, scale, stream );
}

void Vision::attention_simt( const h16* q, const h16* k, const h16* v, h16* out, int m, int heads, int head_dim, float scale, backend::stream_t ) {
	const int width = heads * head_dim;
	impl::cpu_parallel_for( static_cast<int64_t>( m ) * heads, 4, [&]( int64_t i0, int64_t i1 ) {
		std::vector<float> qr( head_dim ), sc( m ), acc( head_dim );
		for ( int64_t i = i0; i < i1; ++i ) {
			const int qi = static_cast<int>( i / heads ), head = static_cast<int>( i % heads );
			for ( int d = 0; d < head_dim; ++d ) qr[d] = h2f( q[static_cast<size_t>( qi ) * width + head * head_dim + d] ) * scale;
			float mx = -INFINITY;
			for ( int j = 0; j < m; ++j ) {
				const h16* kr = k + static_cast<size_t>( j ) * width + head * head_dim;
				float dot = 0.0f;
				for ( int d = 0; d < head_dim; ++d ) dot += qr[d] * h2f( kr[d] );
				sc[j] = dot;
				mx = std::fmax( mx, dot );
			}
			float l = 0.0f;
			std::fill( acc.begin(), acc.end(), 0.0f );
			for ( int j = 0; j < m; ++j ) {
				const float p = std::exp( sc[j] - mx );
				l += p;
				const h16* vr = v + static_cast<size_t>( j ) * width + head * head_dim;
				for ( int d = 0; d < head_dim; ++d ) acc[d] += p * h2f( vr[d] );
			}
			const float inv = 1.0f / l;
			for ( int d = 0; d < head_dim; ++d ) out[static_cast<size_t>( qi ) * width + head * head_dim + d] = f2h( acc[d] * inv );
		}
	} );
}

void Vision::bias_act( h16* x, const h16* bias, int m, int n, int act, backend::stream_t ) {
	const size_t total = static_cast<size_t>( m ) * n;
	impl::cpu_parallel_for( static_cast<int64_t>( total ), 4096, [&]( int64_t i0, int64_t i1 ) {
		for ( int64_t i = i0; i < i1; ++i ) {
			float val = h2f( x[i] ) + ( bias != nullptr ? h2f( bias[i % n] ) : 0.0f );
			if ( act == ACT_GELU_TANH ) val = gelu_tanh_f( val );
			else if ( act == ACT_GELU_ERF ) val = gelu_erf_f( val );
			x[i] = f2h( val );
		}
	} );
}

void Vision::bias_residual( float* hidden, const h16* y, const h16* bias, int m, int n, backend::stream_t ) {
	const size_t total = static_cast<size_t>( m ) * n;
	for ( size_t i = 0; i < total; ++i ) hidden[i] += h2f( y[i] ) + h2f( bias[i % n] );
}

} // namespace engine::modules
