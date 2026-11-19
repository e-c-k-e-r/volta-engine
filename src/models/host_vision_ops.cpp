// The operators of the host-CPU vision towers (see host_vision_ops.h)
#include "host_vision_ops.h"
#include "engine/core/log.h"
#include "engine/kernels/cpu/cpu_util.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

namespace engine::models::hv {

namespace cpu = engine::kernels::cpu;

#define HV_AVX2 __attribute__( ( target( "avx2,fma,f16c" ) ) )

// ---- worker pool ----

struct Par::ForTask : cpu::Task {
	const std::function<void( int64_t, int64_t )>* fn = nullptr;
	std::atomic<int64_t> total{ 0 }, chunk{ 1 }, next{ 0 }, done{ 0 };
	bool work() override {
		const int64_t c = chunk.load( std::memory_order_relaxed ), t = total.load( std::memory_order_relaxed );
		const int64_t b = next.fetch_add( c );
		if ( b >= t ) return false;
		const int64_t e = std::min( b + c, t );
		( *fn )( b, e );
		done.fetch_add( e - b, std::memory_order_acq_rel );
		return true;
	}
	bool finished() const override { return done.load( std::memory_order_acquire ) >= total.load( std::memory_order_relaxed ); }
};

namespace {
constexpr size_t kRing = 16;
}

Par::Par( const std::vector<int>& cpus ) : cpus_( cpus ),
	pool_( std::max<int>( 1, static_cast<int>( cpus.size() ) ), cpus.size() > 1 ? std::vector<int>( cpus.begin() + 1, cpus.end() ) : std::vector<int>() ) {
	// A task stays allocated for the pool's whole life: a worker that read the pointer just before the caller cleared it may still touch the refs
	for ( size_t i = 0; i < kRing; ++i ) ring_.push_back( std::make_unique<ForTask>() );
}

Par::~Par() = default;

void Par::run( int64_t total, int64_t chunk, const std::function<void( int64_t, int64_t )>& fn ) {
	if ( total <= 0 ) return;
	chunk = std::max<int64_t>( 1, chunk );
	if ( pool_.size() == 1 || total <= chunk ) {
		fn( 0, total );
		return;
	}
	ForTask& t = *ring_[next_++ % kRing];
	t.fn = &fn;
	t.total.store( total );
	t.chunk.store( chunk );
	t.next.store( 0 );
	t.done.store( 0 );
	pool_.run( t );
	while ( t.refs.load( std::memory_order_acquire ) != 0 ) _mm_pause();
}

Par::Pin::Pin( const Par& par ) {
	static_assert( sizeof( cpu_set_t ) <= sizeof( saved_ ), "affinity mask storage" );
	if ( par.cpus_.empty() ) return;
	cpu_set_t* old = reinterpret_cast<cpu_set_t*>( saved_ );
	if ( pthread_getaffinity_np( pthread_self(), sizeof( cpu_set_t ), old ) != 0 ) return;
	cpu_set_t one;
	CPU_ZERO( &one );
	CPU_SET( par.cpus_[0], &one );
	pinned_ = pthread_setaffinity_np( pthread_self(), sizeof( one ), &one ) == 0;
}

Par::Pin::~Pin() {
	if ( pinned_ ) pthread_setaffinity_np( pthread_self(), sizeof( cpu_set_t ), reinterpret_cast<cpu_set_t*>( saved_ ) );
}

// ---- dense product over panels ----

Dense pack_nk( const uint16_t* w, int n, int k ) {
	Dense d;
	d.k = k;
	d.n = n;
	d.w.resize( static_cast<size_t>( n ) * k );
	for ( int p = 0; p < n / 16; ++p )
		for ( int i = 0; i < k; ++i )
			for ( int c = 0; c < 16; ++c ) d.w[( static_cast<size_t>( p ) * k + i ) * 16 + c] = w[static_cast<size_t>( p * 16 + c ) * k + i];
	return d;
}

namespace {

// R rows of a by one panel of b: FP32 accumulators over the panel's k rows
template <int R>
HV_AVX2 void micro( const float* a, int lda, const uint16_t* b, int k, float* c, int ldc ) {
	__m256 c0[R], c1[R];
#pragma GCC unroll 6
	for ( int r = 0; r < R; ++r ) {
		c0[r] = _mm256_setzero_ps();
		c1[r] = _mm256_setzero_ps();
	}
	for ( int i = 0; i < k; ++i ) {
		const __m256 b0 = _mm256_cvtph_ps( _mm_loadu_si128( reinterpret_cast<const __m128i*>( b + static_cast<size_t>( i ) * 16 ) ) );
		const __m256 b1 = _mm256_cvtph_ps( _mm_loadu_si128( reinterpret_cast<const __m128i*>( b + static_cast<size_t>( i ) * 16 + 8 ) ) );
#pragma GCC unroll 6
		for ( int r = 0; r < R; ++r ) {
			const __m256 av = _mm256_broadcast_ss( a + static_cast<size_t>( r ) * lda + i );
			c0[r] = _mm256_fmadd_ps( av, b0, c0[r] );
			c1[r] = _mm256_fmadd_ps( av, b1, c1[r] );
		}
	}
#pragma GCC unroll 6
	for ( int r = 0; r < R; ++r ) {
		_mm256_storeu_ps( c + static_cast<size_t>( r ) * ldc, c0[r] );
		_mm256_storeu_ps( c + static_cast<size_t>( r ) * ldc + 8, c1[r] );
	}
}

void panel_rows( const float* a, int lda, const uint16_t* b, int k, int rows, float* c, int ldc ) {
	int r = 0;
	for ( ; r + 6 <= rows; r += 6 ) micro<6>( a + static_cast<size_t>( r ) * lda, lda, b, k, c + static_cast<size_t>( r ) * ldc, ldc );
	const float* ar = a + static_cast<size_t>( r ) * lda;
	float* cr = c + static_cast<size_t>( r ) * ldc;
	switch ( rows - r ) {
		case 5: micro<5>( ar, lda, b, k, cr, ldc ); break;
		case 4: micro<4>( ar, lda, b, k, cr, ldc ); break;
		case 3: micro<3>( ar, lda, b, k, cr, ldc ); break;
		case 2: micro<2>( ar, lda, b, k, cr, ldc ); break;
		case 1: micro<1>( ar, lda, b, k, cr, ldc ); break;
		default: break;
	}
}

} // namespace

void gemm( Par& par, const float* a, int m, const Dense& b, float* out ) {
	const int k = b.k, n = b.n, panels = n / 16, rows_per = 48;
	const int chunks = ( m + rows_per - 1 ) / rows_per;
	// Consecutive units are the panels of one row chunk: its rows stay in cache while the panels stream by
	par.run( static_cast<int64_t>( chunks ) * panels, 1, [&]( int64_t u0, int64_t u1 ) {
		for ( int64_t u = u0; u < u1; ++u ) {
			const int ch = static_cast<int>( u / panels ), pn = static_cast<int>( u % panels );
			const int r0 = ch * rows_per, rows = std::min( rows_per, m - r0 );
			panel_rows( a + static_cast<size_t>( r0 ) * k, k, b.w.data() + static_cast<size_t>( pn ) * k * 16, k, rows, out + static_cast<size_t>( r0 ) * n + pn * 16, n );
		}
	} );
}

// ---- quantized projections ----

namespace {

bool find_trellis( const engine::loader::SafeTensorsRepo& repo, const std::string& base, LinSource* src, std::string* error ) {
	const pod::SafeTensorMeta* tm = repo.find( ( base + ".trellis" ).c_str() );
	if ( !tm || tm->ndim < 3 || !repo.find( ( base + ".suh" ).c_str() ) || !repo.find( ( base + ".svh" ).c_str() ) ) {
		*error = "no EXL3 tensor group for " + base;
		return false;
	}
	if ( !repo.find( ( base + ".mul1" ).c_str() ) ) {
		*error = base + " is not a MUL1 codebook matrix (the host decoder reads those)";
		return false;
	}
	src->name = base;
	src->volta = repo.volta_tiles_for( base.c_str() );
	src->kt = tm->shape[0];
	src->nt = tm->shape[1];
	src->bits = tm->shape[2] / 16;
	if ( !cpu::TrellisLinear::supported( src->bits ) || ( src->kt * 16 ) % cpu::kBlock != 0 || ( src->nt * 16 ) % cpu::kBlock != 0 ) {
		*error = base + ": unsupported shape or bit width";
		return false;
	}
	src->codes = static_cast<const uint16_t*>( repo.cpu_data( ( base + ".trellis" ).c_str() ) );
	src->suh = static_cast<const uint16_t*>( repo.cpu_data( ( base + ".suh" ).c_str() ) );
	src->svh = static_cast<const uint16_t*>( repo.cpu_data( ( base + ".svh" ).c_str() ) );
	if ( !src->codes || !src->suh || !src->svh ) {
		*error = "cannot read the tensors of " + base;
		return false;
	}
	return true;
}

} // namespace

bool find_lin( const engine::loader::SafeTensorsRepo& repo, const std::string& base, LinSource* src, std::string* error ) {
	if ( repo.find( ( base + ".trellis" ).c_str() ) ) return find_trellis( repo, base, src, error );
	const pod::SafeTensorMeta* wm = repo.find( ( base + ".weight" ).c_str() );
	if ( !wm || wm->ndim != 2 || ( wm->dtype != pod::DType::FP16 && wm->dtype != pod::DType::BF16 && wm->dtype != pod::DType::FP32 ) || wm->shape[0] % 16 != 0 ) {
		*error = "no usable projection " + base + " (neither EXL3 tensors nor a dense 2-D FP16 / BF16 / FP32 weight with a multiple of 16 rows)";
		return false;
	}
	src->name = base;
	src->dense = repo.cpu_data( ( base + ".weight" ).c_str() );
	src->dtype = wm->dtype;
	src->n = wm->shape[0];
	src->k = wm->shape[1];
	if ( !src->dense ) {
		*error = "cannot read " + base + ".weight";
		return false;
	}
	return true;
}

void decode_all( Par& par, const std::vector<LinSource>& src, const std::vector<Lin*>& dst ) {
	par.run( static_cast<int64_t>( src.size() ), 1, [&]( int64_t b, int64_t e ) {
		for ( int64_t i = b; i < e; ++i ) {
			const LinSource& s = src[i];
			Lin& l = *dst[i];
			if ( s.dense ) {
				// A row of [n, k] at a time to FP16, scattered into the panels
				l.d.k = s.k;
				l.d.n = s.n;
				l.d.w.resize( static_cast<size_t>( s.k ) * s.n );
				std::vector<float> row( s.k );
				std::vector<uint16_t> h( s.k );
				for ( int r = 0; r < s.n; ++r ) {
					if ( s.dtype == pod::DType::FP16 ) std::memcpy( h.data(), static_cast<const uint16_t*>( s.dense ) + static_cast<size_t>( r ) * s.k, s.k * sizeof( uint16_t ) );
					else {
						if ( s.dtype == pod::DType::FP32 ) std::memcpy( row.data(), static_cast<const float*>( s.dense ) + static_cast<size_t>( r ) * s.k, s.k * sizeof( float ) );
						else {
							const uint16_t* bp = static_cast<const uint16_t*>( s.dense ) + static_cast<size_t>( r ) * s.k;
							for ( int c = 0; c < s.k; ++c ) {
								const uint32_t bits = static_cast<uint32_t>( bp[c] ) << 16;
								std::memcpy( &row[c], &bits, 4 );
							}
						}
						cpu::float_to_half( row.data(), h.data(), s.k );
					}
					const int p = r / 16, c16 = r % 16;
					for ( int i2 = 0; i2 < s.k; ++i2 ) l.d.w[( static_cast<size_t>( p ) * s.k + i2 ) * 16 + c16] = h[i2];
				}
				continue;
			}
			l.d.k = s.kt * 16;
			l.d.n = s.nt * 16;
			l.d.w.resize( static_cast<size_t>( l.d.k ) * l.d.n );
			cpu::TrellisLinear::dequantize_panels( s.codes, s.kt, s.nt, s.bits, s.volta, l.d.w.data() );
			l.rot = std::make_unique<cpu::TrellisLinear>( s.kt, s.nt, s.bits, s.suh, s.svh );
		}
	} );
}

void forward( Par& par, Scratch& s, const Lin& lin, const uint16_t* x, int m, uint16_t* out ) {
	const int k = lin.d.k, n = lin.d.n;
	if ( s.xin.size() < static_cast<size_t>( m ) * k ) s.xin.resize( static_cast<size_t>( m ) * k );
	if ( s.raw.size() < static_cast<size_t>( m ) * n ) s.raw.resize( static_cast<size_t>( m ) * n );
	float* xin = s.xin.data();
	float* raw = s.raw.data();
	const bool rotated = lin.rot != nullptr;
	par.run( m, 4, [&]( int64_t r0, int64_t r1 ) {
		thread_local std::vector<float> xf;
		xf.resize( k );
		for ( int64_t r = r0; r < r1; ++r ) {
			float* dst = xin + static_cast<size_t>( r ) * k;
			cpu::half_to_float( x + static_cast<size_t>( r ) * k, rotated ? xf.data() : dst, k );
			if ( rotated ) for ( int b = 0; b < k / cpu::kBlock; ++b ) lin.rot->prepare_block( xf.data() + b * cpu::kBlock, b, dst + b * cpu::kBlock );
		}
	} );
	gemm( par, xin, m, lin.d, raw );
	par.run( m, 4, [&]( int64_t r0, int64_t r1 ) {
		for ( int64_t r = r0; r < r1; ++r ) {
			float* row = raw + static_cast<size_t>( r ) * n;
			if ( rotated ) for ( int b = 0; b < n / cpu::kBlock; ++b ) lin.rot->finish_block( row + b * cpu::kBlock, b );
			cpu::float_to_half( row, out + static_cast<size_t>( r ) * n, n );
		}
	} );
}

// ---- checkpoint vectors ----

std::vector<float> load_f32( const engine::loader::SafeTensorsRepo& repo, const std::string& name, std::string* error ) {
	const pod::SafeTensorMeta* m = repo.find( name.c_str() );
	const void* p = repo.cpu_data( name.c_str() );
	if ( !m || !p ) {
		*error = "no tensor " + name;
		return {};
	}
	// The element count from the byte size: the repo's shapes stop at 4 dims (the patch embedding is 5-D)
	const size_t width = m->dtype == pod::DType::FP32 ? 4 : 2;
	const size_t count = m->size_bytes / width;
	std::vector<float> v( count );
	if ( m->dtype == pod::DType::FP16 ) cpu::half_to_float( static_cast<const uint16_t*>( p ), v.data(), count );
	else if ( m->dtype == pod::DType::BF16 ) {
		const uint16_t* b = static_cast<const uint16_t*>( p );
		for ( size_t i = 0; i < count; ++i ) {
			const uint32_t bits = static_cast<uint32_t>( b[i] ) << 16;
			std::memcpy( &v[i], &bits, 4 );
		}
	}
	else if ( m->dtype == pod::DType::FP32 ) std::memcpy( v.data(), p, count * sizeof( float ) );
	else {
		*error = name + ": unsupported dtype";
		return {};
	}
	return v;
}

std::vector<uint16_t> load_f16( const engine::loader::SafeTensorsRepo& repo, const std::string& name, std::string* error ) {
	const std::vector<float> f = load_f32( repo, name, error );
	std::vector<uint16_t> h( f.size() );
	cpu::float_to_half( f.data(), h.data(), f.size() );
	return h;
}

std::vector<float> load_vec( const engine::loader::SafeTensorsRepo& repo, const std::string& name, std::string* error ) {
	const std::vector<uint16_t> h = load_f16( repo, name, error );
	std::vector<float> f( h.size() );
	cpu::half_to_float( h.data(), f.data(), h.size() );
	return f;
}

// ---- row-wise operators ----

void rms_norm( Par& par, const float* x, const float* w, uint16_t* out, int m, int n, float eps ) {
	par.run( m, 8, [&]( int64_t r0, int64_t r1 ) {
		thread_local std::vector<float> t;
		t.resize( n );
		for ( int64_t r = r0; r < r1; ++r ) {
			const float* xr = x + static_cast<size_t>( r ) * n;
			float q = 0.0f;
			for ( int i = 0; i < n; ++i ) q += xr[i] * xr[i];
			const float inv = 1.0f / std::sqrt( q / n + eps );
			for ( int i = 0; i < n; ++i ) t[i] = xr[i] * inv * w[i];
			cpu::float_to_half( t.data(), out + static_cast<size_t>( r ) * n, n );
		}
	} );
}

void layer_norm( Par& par, const uint16_t* x, const float* w, const float* b, uint16_t* out, int m, int n, float eps ) {
	par.run( m, 8, [&]( int64_t r0, int64_t r1 ) {
		thread_local std::vector<float> t;
		t.resize( n );
		for ( int64_t r = r0; r < r1; ++r ) {
			cpu::half_to_float( x + static_cast<size_t>( r ) * n, t.data(), n );
			float s = 0.0f;
			for ( int i = 0; i < n; ++i ) s += t[i];
			const float mean = s / n;
			float q = 0.0f;
			for ( int i = 0; i < n; ++i ) q += ( t[i] - mean ) * ( t[i] - mean );
			const float inv = 1.0f / std::sqrt( q / n + eps );
			for ( int i = 0; i < n; ++i ) t[i] = ( t[i] - mean ) * inv * w[i] + b[i];
			cpu::float_to_half( t.data(), out + static_cast<size_t>( r ) * n, n );
		}
	} );
}

void layer_norm( Par& par, const float* x, const float* w, const float* b, uint16_t* out, int m, int n, float eps ) {
	par.run( m, 8, [&]( int64_t r0, int64_t r1 ) {
		thread_local std::vector<float> t;
		t.resize( n );
		for ( int64_t r = r0; r < r1; ++r ) {
			const float* xr = x + static_cast<size_t>( r ) * n;
			float s = 0.0f;
			for ( int i = 0; i < n; ++i ) s += xr[i];
			const float mean = s / n;
			float q = 0.0f;
			for ( int i = 0; i < n; ++i ) q += ( xr[i] - mean ) * ( xr[i] - mean );
			const float inv = 1.0f / std::sqrt( q / n + eps );
			for ( int i = 0; i < n; ++i ) t[i] = ( xr[i] - mean ) * inv * w[i] + b[i];
			cpu::float_to_half( t.data(), out + static_cast<size_t>( r ) * n, n );
		}
	} );
}

void bias_residual( Par& par, float* hidden, const uint16_t* y, const float* bias, int m, int n ) {
	par.run( m, 8, [&]( int64_t r0, int64_t r1 ) {
		thread_local std::vector<float> t;
		t.resize( n );
		for ( int64_t r = r0; r < r1; ++r ) {
			cpu::half_to_float( y + static_cast<size_t>( r ) * n, t.data(), n );
			float* h = hidden + static_cast<size_t>( r ) * n;
			for ( int i = 0; i < n; ++i ) h[i] += t[i] + bias[i];
		}
	} );
}

void add_bias( Par& par, uint16_t* x, const float* bias, int m, int n ) {
	if ( !bias ) return;
	par.run( m, 8, [&]( int64_t r0, int64_t r1 ) {
		thread_local std::vector<float> t;
		t.resize( n );
		for ( int64_t r = r0; r < r1; ++r ) {
			uint16_t* row = x + static_cast<size_t>( r ) * n;
			cpu::half_to_float( row, t.data(), n );
			for ( int i = 0; i < n; ++i ) t[i] += bias[i];
			cpu::float_to_half( t.data(), row, n );
		}
	} );
}

void gelu_erf( Par& par, uint16_t* x, const float* bias, int m, int n ) {
	par.run( m, 8, [&]( int64_t r0, int64_t r1 ) {
		thread_local std::vector<float> t;
		t.resize( n );
		for ( int64_t r = r0; r < r1; ++r ) {
			uint16_t* row = x + static_cast<size_t>( r ) * n;
			cpu::half_to_float( row, t.data(), n );
			for ( int i = 0; i < n; ++i ) {
				const float v = bias ? t[i] + bias[i] : t[i];
				t[i] = 0.5f * v * ( 1.0f + std::erf( v * 0.7071067811865476f ) );
			}
			cpu::float_to_half( t.data(), row, n );
		}
	} );
}

void gelu_tanh( Par& par, uint16_t* x, const float* bias, int m, int n ) {
	par.run( m, 8, [&]( int64_t r0, int64_t r1 ) {
		thread_local std::vector<float> t;
		t.resize( n );
		for ( int64_t r = r0; r < r1; ++r ) {
			uint16_t* row = x + static_cast<size_t>( r ) * n;
			cpu::half_to_float( row, t.data(), n );
			for ( int i = 0; i < n; ++i ) {
				const float v = bias ? t[i] + bias[i] : t[i];
				const float u = 0.7978845608028654f * ( v + 0.044715f * v * v * v );
				t[i] = 0.5f * v * ( 1.0f + std::tanh( u ) );
			}
			cpu::float_to_half( t.data(), row, n );
		}
	} );
}

void swiglu( Par& par, const uint16_t* gate, const uint16_t* up, uint16_t* out, int64_t count, float limit ) {
	constexpr int64_t kPiece = 4096;
	par.run( ( count + kPiece - 1 ) / kPiece, 1, [&]( int64_t p0, int64_t p1 ) {
		float g[kPiece], u[kPiece];
		for ( int64_t p = p0; p < p1; ++p ) {
			const int64_t b = p * kPiece, n = std::min( kPiece, count - b );
			cpu::half_to_float( gate + b, g, n );
			cpu::half_to_float( up + b, u, n );
			for ( int64_t i = 0; i < n; ++i ) {
				float gv = g[i], uv = u[i];
				if ( limit > 0.0f ) {
					gv = std::min( gv, limit );
					uv = std::max( -limit, std::min( uv, limit ) );
				}
				g[i] = gv / ( 1.0f + std::exp( -gv ) ) * uv;
			}
			cpu::float_to_half( g, out + b, n );
		}
	} );
}

namespace {

// e^x for x <= 0 (the softmax's shifted scores): 2^n times a degree-5 polynomial of the remainder, relative error about 1e-7
HV_AVX2 __m256 exp_neg( __m256 x ) {
	x = _mm256_max_ps( x, _mm256_set1_ps( -87.0f ) );
	const __m256 t = _mm256_mul_ps( x, _mm256_set1_ps( 1.44269504089f ) );
	const __m256 n = _mm256_round_ps( t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC );
	__m256 r = _mm256_fnmadd_ps( n, _mm256_set1_ps( 0.693359375f ), x );
	r = _mm256_fnmadd_ps( n, _mm256_set1_ps( -2.12194440e-4f ), r );
	__m256 p = _mm256_set1_ps( 1.9875691500e-4f );
	p = _mm256_fmadd_ps( p, r, _mm256_set1_ps( 1.3981999507e-3f ) );
	p = _mm256_fmadd_ps( p, r, _mm256_set1_ps( 8.3334519073e-3f ) );
	p = _mm256_fmadd_ps( p, r, _mm256_set1_ps( 4.1665795894e-2f ) );
	p = _mm256_fmadd_ps( p, r, _mm256_set1_ps( 1.6666665459e-1f ) );
	p = _mm256_fmadd_ps( p, r, _mm256_set1_ps( 5.0000001201e-1f ) );
	p = _mm256_fmadd_ps( p, _mm256_mul_ps( r, r ), _mm256_add_ps( r, _mm256_set1_ps( 1.0f ) ) );
	const __m256i e = _mm256_slli_epi32( _mm256_add_epi32( _mm256_cvtps_epi32( n ), _mm256_set1_epi32( 127 ) ), 23 );
	return _mm256_mul_ps( p, _mm256_castsi256_ps( e ) );
}

// The softmax of the first m of the mp scores of a row (mp a multiple of 16: the rest are padding keys, set to zero), in place
HV_AVX2 void softmax_row( float* s, int m, int mp ) {
	float mx = -INFINITY;
	for ( int j = 0; j < m; ++j ) mx = std::max( mx, s[j] );
	const __m256 vm = _mm256_set1_ps( mx );
	for ( int j = 0; j < mp; j += 8 ) _mm256_storeu_ps( s + j, exp_neg( _mm256_sub_ps( _mm256_loadu_ps( s + j ), vm ) ) );
	for ( int j = m; j < mp; ++j ) s[j] = 0.0f;
	__m256 acc = _mm256_setzero_ps();
	for ( int j = 0; j < mp; j += 8 ) acc = _mm256_add_ps( acc, _mm256_loadu_ps( s + j ) );
	__m128 r = _mm_add_ps( _mm256_castps256_ps128( acc ), _mm256_extractf128_ps( acc, 1 ) );
	r = _mm_add_ps( r, _mm_movehl_ps( r, r ) );
	r = _mm_add_ss( r, _mm_shuffle_ps( r, r, 1 ) );
	const __m256 inv = _mm256_set1_ps( 1.0f / _mm_cvtss_f32( r ) );
	for ( int j = 0; j < mp; j += 8 ) _mm256_storeu_ps( s + j, _mm256_mul_ps( _mm256_loadu_ps( s + j ), inv ) );
}

} // namespace

// Per head: the keys as panels [mp / 16][head_dim][16] (the scores Q K^T are a product over them) and the values transposed into panels
// [hdp / 16][mp][16] (the output P V is a product over those), then blocks of queries: scores by gemm, softmax per row, output by gemm.
void attention( Par& par, Scratch& s, const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* out, int m, int heads, float scale, int head_dim ) {
	const int hd = head_dim, hdp = ( hd + 15 ) / 16 * 16, mp = ( m + 15 ) / 16 * 16, width = heads * hd;
	constexpr int kBlockRows = 384;
	Dense kd, vd;
	kd.k = hd;
	kd.n = mp;
	kd.w.assign( static_cast<size_t>( mp ) * hd, 0 );
	vd.k = mp;
	vd.n = hdp;
	vd.w.assign( static_cast<size_t>( mp ) * hdp, 0 );
	const size_t qb = static_cast<size_t>( std::min( m, kBlockRows ) );
	if ( s.qf.size() < qb * hd ) s.qf.resize( qb * hd );
	if ( s.kf.size() < qb * mp ) s.kf.resize( qb * mp );
	if ( s.vf.size() < qb * hdp ) s.vf.resize( qb * hdp );
	for ( int head = 0; head < heads; ++head ) {
		const int off = head * hd;
		// keys: panel p holds the keys 16p .. 16p+15, the padding keys stay zero
		par.run( mp / 16, 1, [&]( int64_t p0, int64_t p1 ) {
			for ( int64_t p = p0; p < p1; ++p )
				for ( int c = 0; c < 16; ++c ) {
					const int j = static_cast<int>( p ) * 16 + c;
					if ( j >= m ) continue;
					const uint16_t* src = k + static_cast<size_t>( j ) * width + off;
					uint16_t* dst = kd.w.data() + static_cast<size_t>( p ) * hd * 16 + c;
					for ( int i = 0; i < hd; ++i ) dst[static_cast<size_t>( i ) * 16] = src[i];
				}
		} );
		// values: panel p holds the output columns 16p .. 16p+15 for every key j (the padding keys and columns stay zero)
		par.run( ( m + 63 ) / 64, 1, [&]( int64_t c0, int64_t c1 ) {
			for ( int64_t cj = c0; cj < c1; ++cj )
				for ( int j = static_cast<int>( cj ) * 64; j < std::min( m, static_cast<int>( cj ) * 64 + 64 ); ++j ) {
					const uint16_t* src = v + static_cast<size_t>( j ) * width + off;
					for ( int d = 0; d < hd; ++d ) vd.w[( static_cast<size_t>( d / 16 ) * mp + j ) * 16 + d % 16] = src[d];
				}
		} );
		for ( int q0 = 0; q0 < m; q0 += kBlockRows ) {
			const int rows = std::min( kBlockRows, m - q0 );
			par.run( rows, 16, [&]( int64_t r0, int64_t r1 ) {
				for ( int64_t r = r0; r < r1; ++r ) {
					float* dst = s.qf.data() + static_cast<size_t>( r ) * hd;
					cpu::half_to_float( q + static_cast<size_t>( q0 + r ) * width + off, dst, hd );
					for ( int i = 0; i < hd; ++i ) dst[i] *= scale;
				}
			} );
			gemm( par, s.qf.data(), rows, kd, s.kf.data() );                      // scores [rows][mp]
			par.run( rows, 8, [&]( int64_t r0, int64_t r1 ) {
				for ( int64_t r = r0; r < r1; ++r ) softmax_row( s.kf.data() + static_cast<size_t>( r ) * mp, m, mp );
			} );
			gemm( par, s.kf.data(), rows, vd, s.vf.data() );                      // output [rows][hdp]
			par.run( rows, 16, [&]( int64_t r0, int64_t r1 ) {
				for ( int64_t r = r0; r < r1; ++r )
					cpu::float_to_half( s.vf.data() + static_cast<size_t>( r ) * hdp, out + static_cast<size_t>( q0 + r ) * width + off, hd );
			} );
		}
	}
}

} // namespace engine::models::hv
