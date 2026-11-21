// Tier 2: the training primitives (engine/train/grad.h) on a SwiGLU MLP of three scaled linears, y = ( act( x Wg ) * ( x Wu ) ) Wd with
// W = diag( su ) T diag( sv ), against a double-precision host reference:
//   (a) the row-weighted squared-error loss against the host forward
//   (b) the gradients of the six scale vectors and of the input against central differences of the host loss, with SiLU and with GLM's limit
//   (c) Adam steps on multiplicative scale parameters lower the loss
//   (d) gather_rows (FP16 and FP32 sources, with and without per-row weights) and scatter_add_rows against the host, exactly
//   (e) Adam's kernels against a host reference, FP32 and 8-bit block-wise moments, log scales and additive deltas; updated in parts against
//       one update (bit for bit); 8-bit moments optimizing a noisy quadratic as FP32 ones do; the 8-bit codes' round trip
#include "engine/config.h"
#include "engine/train/grad.h"
#include "engine/train/adam8.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr int R = 300, H = 64, I = 32;
int fails = 0;

void check( bool ok, const char* line ) {
	std::printf( "  %-96s %s\n", line, ok ? "OK" : "FAIL" );
	fails += !ok;
}

struct Lin {
	int k, n;
	std::vector<double> T, su, sv;
};

// Host forward of one linear: Y = ( X diag( su ) ) T diag( sv )
std::vector<double> lin( const Lin& l, const std::vector<double>& X, int rows ) {
	std::vector<double> Y( static_cast<size_t>( rows ) * l.n, 0.0 );
	for ( int r = 0; r < rows; ++r ) {
		for ( int i = 0; i < l.k; ++i ) {
			const double x = X[static_cast<size_t>( r ) * l.k + i] * l.su[i];
			for ( int j = 0; j < l.n; ++j ) Y[static_cast<size_t>( r ) * l.n + j] += x * l.T[static_cast<size_t>( i ) * l.n + j];
		}
		for ( int j = 0; j < l.n; ++j ) Y[static_cast<size_t>( r ) * l.n + j] *= l.sv[j];
	}
	return Y;
}

double host_loss( const Lin& g, const Lin& u, const Lin& d, const std::vector<double>& X, const std::vector<double>& Yt, const std::vector<double>& w,
	double limit ) {
	auto G = lin( g, X, R ), U = lin( u, X, R );
	std::vector<double> A( G.size() );
	for ( size_t i = 0; i < A.size(); ++i ) {
		double x = G[i], y = U[i];
		if ( limit > 0.0 ) {
			x = std::min( x, limit );
			y = std::min( std::max( y, -limit ), limit );
		}
		A[i] = x / ( 1.0 + std::exp( -x ) ) * y;
	}
	auto Y = lin( d, A, R );
	double L = 0.0;
	for ( int r = 0; r < R; ++r ) {
		for ( int j = 0; j < H; ++j ) {
			const double e = Y[static_cast<size_t>( r ) * H + j] - Yt[static_cast<size_t>( r ) * H + j];
			L += w[r] * e * e;
		}
	}
	return L;
}

float* up( const std::vector<double>& v ) {
	std::vector<float> f( v.begin(), v.end() );
	float* d = nullptr;
	CUDA_CHECK( cudaMalloc( &d, f.size() * sizeof( float ) ) );
	CUDA_CHECK( cudaMemcpy( d, f.data(), f.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
	return d;
}

float* up( const std::vector<float>& f ) {
	float* d = nullptr;
	CUDA_CHECK( cudaMalloc( &d, std::max<size_t>( f.size(), 1 ) * sizeof( float ) ) );
	CUDA_CHECK( cudaMemcpy( d, f.data(), f.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
	return d;
}

float* zeros( size_t n ) {
	float* d = nullptr;
	CUDA_CHECK( cudaMalloc( &d, n * sizeof( float ) ) );
	CUDA_CHECK( cudaMemset( d, 0, n * sizeof( float ) ) );
	return d;
}

std::vector<float> down( const float* d, size_t n ) {
	std::vector<float> h( n );
	CUDA_CHECK( cudaMemcpy( h.data(), d, n * sizeof( float ), cudaMemcpyDeviceToHost ) );
	return h;
}

// Adam on the host, the reference of the device kernels (adam8.h's steps): log scales (s = s0 e^a) or additive (s = s0 + a); FP32 moments
// m / v or, with m null, the 8-bit ones block by block
void host_adam( bool log_scales, const float* s0, float* a, float* s, const float* g, float* m, float* v, uint8_t* mq, uint8_t* vq, float* mmax,
	float* vmax, size_t n, int step, const engine::train::AdamOptions& o ) {
	namespace q8 = engine::train::adam8;
	const float c1 = 1.0f - std::pow( o.beta1, static_cast<float>( step ) ), c2 = 1.0f - std::pow( o.beta2, static_cast<float>( step ) );
	constexpr size_t B = q8::kBlock;
	float bm[B], bv[B];
	for ( size_t b = 0; b * B < n; ++b ) {
		const size_t i0 = b * B, i1 = std::min( n, i0 + B );
		float am = 0.0f, av = 0.0f;
		for ( size_t i = i0; i < i1; ++i ) {
			float mi = m ? m[i] : q8::dec_m( mq[i], mmax[b] ), vi = m ? v[i] : q8::dec_v( vq[i], vmax[b] );
			const float gi = log_scales ? g[i] * s[i] : g[i];
			float ai = a[i];
			q8::step( gi, mi, vi, ai, o.lr, o.beta1, o.beta2, o.eps, c1, c2, m == nullptr );
			a[i] = ai;
			s[i] = log_scales ? s0[i] * std::exp( ai ) : s0[i] + ai;
			if ( m ) {
				m[i] = mi;
				v[i] = vi;
				continue;
			}
			bm[i - i0] = mi;
			bv[i - i0] = vi;
			am = std::max( am, std::fabs( mi ) );
			av = std::max( av, std::sqrt( vi ) );
		}
		if ( m ) continue;
		for ( size_t i = i0; i < i1; ++i ) {
			mq[i] = q8::enc_m( bm[i - i0], am, q8::draw( i, step, q8::kSaltM ) );
			vq[i] = q8::enc_v( bv[i - i0], av, q8::draw( i, step, q8::kSaltV ) );
		}
		mmax[b] = am;
		vmax[b] = av;
	}
}

} // namespace

int main() {
	std::mt19937 rng( 99 );
	std::normal_distribution<double> gauss( 0.0, 1.0 );
	std::uniform_real_distribution<double> uni( 0.5, 2.0 );
	auto mk = [&]( int k, int n ) {
		Lin l{ k, n, std::vector<double>( static_cast<size_t>( k ) * n ), std::vector<double>( k ), std::vector<double>( n ) };
		for ( auto& v : l.T ) v = gauss( rng ) / std::sqrt( static_cast<double>( k ) );
		for ( auto& v : l.su ) v = uni( rng ) * ( rng() & 1 ? 1.0 : -1.0 );
		for ( auto& v : l.sv ) v = uni( rng ) * ( rng() & 1 ? 1.0 : -1.0 );
		return l;
	};
	Lin g = mk( H, I ), u = mk( H, I ), d = mk( I, H );
	std::vector<double> X( static_cast<size_t>( R ) * H ), Yt( X.size() ), w( R );
	for ( auto& v : X ) v = gauss( rng );
	for ( auto& v : Yt ) v = 0.5 * gauss( rng );
	for ( auto& v : w ) v = uni( rng );
	std::printf( "[*] test_train_grad: %d rows, hidden %d, intermediate %d\n", R, H, I );

	cudaStream_t st = nullptr;
	CUDA_CHECK( cudaStreamCreate( &st ) );
	engine::train::Blas blas( st );
	float *dX = up( X ), *dYt = up( Yt ), *dw = up( w );
	float *Tg = up( g.T ), *Tu = up( u.T ), *Td = up( d.T );
	float *sug = up( g.su ), *svg = up( g.sv ), *suu = up( u.su ), *svu = up( u.sv ), *sud = up( d.su ), *svd = up( d.sv );
	const size_t RI = static_cast<size_t>( R ) * I, RH = static_cast<size_t>( R ) * H;
	float *Zg = zeros( RI ), *G = zeros( RI ), *Zu = zeros( RI ), *U = zeros( RI ), *A = zeros( RI ), *Zd = zeros( RH ), *Y = zeros( RH );
	float *dY = zeros( RH ), *dA = zeros( RI ), *dG = zeros( RI ), *dU = zeros( RI ), *dXg = zeros( RH ), *dXu = zeros( RH );
	float *s_n = zeros( std::max( RI, RH ) ), *s_k = zeros( std::max( RI, RH ) );
	float *gsug = zeros( H ), *gsvg = zeros( I ), *gsuu = zeros( H ), *gsvu = zeros( I ), *gsud = zeros( I ), *gsvd = zeros( H );
	engine::train::ScaledLinear lg{ H, I, Tg, sug, svg, gsug, gsvg }, lu{ H, I, Tu, suu, svu, gsuu, gsvu }, ld{ I, H, Td, sud, svd, gsud, gsvd };

	// One forward + backward; the gradients accumulate, so they are zeroed first
	auto pass = [&]( float limit, bool backward ) {
		engine::train::scaled_linear_forward( blas, lg, dX, R, Zg, G, s_k );
		engine::train::scaled_linear_forward( blas, lu, dX, R, Zu, U, s_k );
		engine::train::swiglu_forward( G, U, A, RI, limit, st );
		engine::train::scaled_linear_forward( blas, ld, A, R, Zd, Y, s_k );
		const double L = engine::train::weighted_sq_error( Y, dYt, dw, R, H, backward ? dY : nullptr, 1.0f, st );
		if ( backward ) {
			for ( float* p : { gsug, gsuu, gsvd } ) CUDA_CHECK( cudaMemsetAsync( p, 0, H * sizeof( float ), st ) );
			for ( float* p : { gsvg, gsvu, gsud } ) CUDA_CHECK( cudaMemsetAsync( p, 0, I * sizeof( float ), st ) );
			engine::train::scaled_linear_backward( blas, ld, A, Zd, dY, R, dA, s_n, s_k );
			engine::train::swiglu_backward( G, U, dA, dG, dU, RI, limit, st );
			engine::train::scaled_linear_backward( blas, lg, dX, Zg, dG, R, dXg, s_n, s_k );
			engine::train::scaled_linear_backward( blas, lu, dX, Zu, dU, R, dXu, s_n, s_k );
			CUDA_CHECK( cudaStreamSynchronize( st ) );
		}
		return L;
	};

	char line[160];
	for ( float limit : { 0.0f, 1.0f } ) {
		const double L = pass( limit, true );
		const double Lh = host_loss( g, u, d, X, Yt, w, limit );
		std::snprintf( line, sizeof( line ), "(a) limit %.1f: loss %.6f, host %.6f", limit, L, Lh );
		check( std::fabs( L - Lh ) <= 1e-5 * std::fabs( Lh ), line );
		// (b) central differences of the host loss at 8 entries of every scale vector and of the input
		struct Param {
			const char* name;
			std::vector<double>* v;
			float* grad;
		};
		const std::vector<float> gx_g = down( dXg, RH ), gx_u = down( dXu, RH );
		std::vector<float> gx( RH );
		for ( size_t i = 0; i < RH; ++i ) gx[i] = gx_g[i] + gx_u[i];
		for ( const Param& p : std::vector<Param>{ { "gate su", &g.su, gsug }, { "gate sv", &g.sv, gsvg }, { "up su", &u.su, gsuu }, { "up sv", &u.sv, gsvu },
				 { "down su", &d.su, gsud }, { "down sv", &d.sv, gsvd }, { "input", &X, nullptr } } ) {
			const std::vector<float> got = p.grad ? down( p.grad, p.v->size() ) : gx;
			double worst = 0.0;
			for ( int s = 0; s < 8; ++s ) {
				const size_t idx = ( static_cast<size_t>( s ) * 7919 ) % p.v->size();
				const double keep = ( *p.v )[idx], h = 1e-5 * std::max( 1.0, std::fabs( keep ) );
				( *p.v )[idx] = keep + h;
				const double lp = host_loss( g, u, d, X, Yt, w, limit );
				( *p.v )[idx] = keep - h;
				const double lm = host_loss( g, u, d, X, Yt, w, limit );
				( *p.v )[idx] = keep;
				const double fd = ( lp - lm ) / ( 2.0 * h );
				worst = std::max( worst, std::fabs( got[idx] - fd ) / std::max( std::fabs( fd ), 1e-3 * std::fabs( Lh ) ) );
			}
			std::snprintf( line, sizeof( line ), "(b) limit %.1f: %-8s gradient against central differences, worst relative error %.2e (max 1e-4)", limit,
				p.name, worst );
			check( worst <= 1e-4, line );
		}
	}

	// (c) Adam on log-scale parameters a (s = s0 exp( a )), gradient s o ds
	{
		struct P {
			float* s;
			float* gs;
			int n;
			float *s0, *a, *ga, *m, *v;
		};
		std::vector<P> ps;
		for ( auto pr : std::vector<std::pair<std::pair<float*, float*>, int>>{ { { sug, gsug }, H }, { { svg, gsvg }, I }, { { suu, gsuu }, H },
				 { { svu, gsvu }, I }, { { sud, gsud }, I }, { { svd, gsvd }, H } } ) {
			P p{ pr.first.first, pr.first.second, pr.second, zeros( pr.second ), zeros( pr.second ), zeros( pr.second ), zeros( pr.second ), zeros( pr.second ) };
			CUDA_CHECK( cudaMemcpy( p.s0, p.s, p.n * sizeof( float ), cudaMemcpyDeviceToDevice ) );
			ps.push_back( p );
		}
		engine::train::AdamOptions o;
		o.lr = 1e-2f;
		const double L0 = pass( 0.0f, false );
		double L = L0;
		for ( int step = 1; step <= 30; ++step ) {
			pass( 0.0f, true );
			for ( auto& p : ps ) {
				engine::train::mul( p.s, p.gs, p.ga, p.n, st );
				engine::train::adam_step( p.a, p.ga, p.m, p.v, p.n, step, o, st );
				engine::train::exp_scale( p.s0, p.a, p.s, p.n, st );
			}
			L = pass( 0.0f, false );
		}
		std::snprintf( line, sizeof( line ), "(c) 30 Adam steps on the scales: loss %.4f -> %.4f", L0, L );
		check( L < 0.95 * L0, line );
	}
	// (d) row gathers and scatters: 40 rows picked out of 300 (no repeats), 64 columns
	{
		const int N = 300, C = 64, M = 40;
		std::vector<int> idx( N );
		for ( int i = 0; i < N; ++i ) idx[i] = i;
		std::shuffle( idx.begin(), idx.end(), rng );
		idx.resize( M );
		std::vector<float> src( static_cast<size_t>( N ) * C ), wr( M ), dst0( static_cast<size_t>( N ) * C );
		std::vector<uint16_t> src16( src.size() );
		for ( size_t i = 0; i < src.size(); ++i ) {
			const half h = __float2half_rn( static_cast<float>( gauss( rng ) ) );
			src[i] = __half2float( h );
			std::memcpy( &src16[i], &h, 2 );
		}
		for ( auto& v : wr ) v = static_cast<float>( uni( rng ) );
		for ( auto& v : dst0 ) v = static_cast<float>( gauss( rng ) );
		float *d_src = up( std::vector<double>( src.begin(), src.end() ) ), *d_w = up( std::vector<double>( wr.begin(), wr.end() ) ),
			  *d_out = zeros( static_cast<size_t>( M ) * C ), *d_dst = up( std::vector<double>( dst0.begin(), dst0.end() ) );
		int* d_idx = nullptr;
		uint16_t* d_src16 = nullptr;
		CUDA_CHECK( cudaMalloc( &d_idx, M * sizeof( int ) ) );
		CUDA_CHECK( cudaMemcpy( d_idx, idx.data(), M * sizeof( int ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMalloc( &d_src16, src16.size() * 2 ) );
		CUDA_CHECK( cudaMemcpy( d_src16, src16.data(), src16.size() * 2, cudaMemcpyHostToDevice ) );
		size_t bad = 0;
		for ( int half_src = 0; half_src < 2; ++half_src ) {
			for ( int weighted = 0; weighted < 2; ++weighted ) {
				engine::train::gather_rows( half_src ? static_cast<const void*>( d_src16 ) : d_src, half_src != 0, d_idx, weighted ? d_w : nullptr, M, C, d_out, st );
				CUDA_CHECK( cudaStreamSynchronize( st ) );
				const std::vector<float> got = down( d_out, static_cast<size_t>( M ) * C );
				for ( int r = 0; r < M; ++r ) {
					for ( int c = 0; c < C; ++c ) {
						const float want = src[static_cast<size_t>( idx[r] ) * C + c] * ( weighted ? wr[r] : 1.0f );
						bad += got[static_cast<size_t>( r ) * C + c] != want;
					}
				}
			}
		}
		std::snprintf( line, sizeof( line ), "(d) gather_rows: %zu of %d values differ from the host's", bad, 4 * M * C );
		check( bad == 0, line );
		// scatter: dst[idx[r]] += w[r] * out[r] with out = the last gather (FP16 source, weighted)
		const std::vector<float> outv = down( d_out, static_cast<size_t>( M ) * C );
		engine::train::scatter_add_rows( d_out, d_idx, d_w, M, C, d_dst, st );
		CUDA_CHECK( cudaStreamSynchronize( st ) );
		const std::vector<float> got = down( d_dst, dst0.size() );
		std::vector<float> want = dst0;
		for ( int r = 0; r < M; ++r ) {
			// the device fuses the multiply into the add (one rounding)
			for ( int c = 0; c < C; ++c ) {
				float& d = want[static_cast<size_t>( idx[r] ) * C + c];
				d = std::fma( wr[r], outv[static_cast<size_t>( r ) * C + c], d );
			}
		}
		size_t sbad = 0;
		for ( size_t i = 0; i < want.size(); ++i ) sbad += got[i] != want[i];
		std::snprintf( line, sizeof( line ), "(d) scatter_add_rows: %zu of %zu values differ from the host's", sbad, want.size() );
		check( sbad == 0, line );
		for ( void* q : { (void*) d_src, (void*) d_w, (void*) d_out, (void*) d_dst, (void*) d_idx, (void*) d_src16 } ) CUDA_CHECK( cudaFree( q ) );
	}
	// (e) Adam's variants: 50 steps on 100000 log-scale parameters (a partial last moment block) with gradients over six decades (a level per
	// 256 values, three decades within), the device and the host each with FP32 and 8-bit moments; then the additive form; and the 8-bit
	// codes' round trip
	{
		namespace q8 = engine::train::adam8;
		const size_t n = 100000, nb = ( n + q8::kBlock - 1 ) / q8::kBlock;
		std::vector<float> s0( n ), mag( n );
		std::uniform_real_distribution<float> us( 0.5f, 2.0f ), ue( -5.0f, 1.0f ), uw( -1.5f, 1.5f );
		std::normal_distribution<float> gn( 0.0f, 1.0f );
		float level = 1.0f;
		for ( size_t i = 0; i < n; ++i ) {
			if ( i % q8::kBlock == 0 ) level = std::pow( 10.0f, ue( rng ) );
			s0[i] = us( rng );
			mag[i] = level * std::pow( 10.0f, uw( rng ) );
		}
		std::vector<std::vector<float>> grads( 50, std::vector<float>( n ) );
		for ( auto& g : grads ) for ( size_t i = 0; i < n; ++i ) g[i] = gn( rng ) * mag[i];
		engine::train::AdamOptions o;
		o.lr = 1e-3f;
		// one variant's deltas a after the steps (log: s = s0 e^a, else s = s0 + a): mode 0 the device, 1 the host reference, 2 the device in
		// parts of 100 moment blocks (as a host-resident state goes through it)
		auto run = [&]( bool log, int mode, int bits ) {
			std::vector<float> a( n, 0.0f ), s = s0, m( n, 0.0f ), v( n, 0.0f ), mmax( nb, 0.0f ), vmax( nb, 0.0f );
			std::vector<uint8_t> mq( n, 0 ), vq( n, 0 );
			if ( mode == 1 ) {
				for ( int k = 1; k <= 50; ++k )
					host_adam( log, s0.data(), a.data(), s.data(), grads[k - 1].data(), bits == 32 ? m.data() : nullptr, v.data(), mq.data(), vq.data(), mmax.data(),
						vmax.data(), n, k, o );
				return a;
			}
			float *d_s0 = up( s0 ), *d_a = up( a ), *d_s = up( s ), *d_g = up( a ), *d_m = up( m ), *d_v = up( v ), *d_mm = up( mmax ), *d_vm = up( vmax );
			uint8_t *d_mq = nullptr, *d_vq = nullptr;
			CUDA_CHECK( cudaMalloc( &d_mq, n ) );
			CUDA_CHECK( cudaMalloc( &d_vq, n ) );
			CUDA_CHECK( cudaMemset( d_mq, 0, n ) );
			CUDA_CHECK( cudaMemset( d_vq, 0, n ) );
			const size_t part = mode == 2 ? 100 * q8::kBlock : n;
			for ( int k = 1; k <= 50; ++k ) {
				CUDA_CHECK( cudaMemcpy( d_g, grads[k - 1].data(), n * sizeof( float ), cudaMemcpyHostToDevice ) );
				for ( size_t o0 = 0; o0 < n; o0 += part ) {
					const size_t c = std::min( part, n - o0 ), b0 = o0 / q8::kBlock;
					float *ps0 = d_s0 + o0, *pa = d_a + o0, *ps = d_s + o0, *pg = d_g + o0;
					if ( bits == 32 && log ) engine::train::adam_log_scales( ps0, pa, ps, pg, d_m + o0, d_v + o0, c, k, 1.0f, o, st );
					else if ( bits == 32 ) engine::train::adam_deltas( ps0, pa, ps, pg, d_m + o0, d_v + o0, c, k, 1.0f, o, st );
					else if ( log ) engine::train::adam_log_scales8( ps0, pa, ps, pg, d_mq + o0, d_vq + o0, d_mm + b0, d_vm + b0, c, k, 1.0f, o, st, o0 );
					else engine::train::adam_deltas8( ps0, pa, ps, pg, d_mq + o0, d_vq + o0, d_mm + b0, d_vm + b0, c, k, 1.0f, o, st, o0 );
				}
			}
			CUDA_CHECK( cudaStreamSynchronize( st ) );
			a = down( d_a, n );
			for ( void* q : { (void*) d_s0, (void*) d_a, (void*) d_s, (void*) d_g, (void*) d_m, (void*) d_v, (void*) d_mm, (void*) d_vm, (void*) d_mq, (void*) d_vq } )
				CUDA_CHECK( cudaFree( q ) );
			return a;
		};
		// the largest difference over the largest delta, and the relative L2 difference
		auto cmp = [&]( const std::vector<float>& x, const std::vector<float>& y, double& worst ) {
			double num = 0.0, den = 0.0, amax = 0.0;
			worst = 0.0;
			for ( size_t i = 0; i < n; ++i ) {
				num += ( double( x[i] ) - y[i] ) * ( double( x[i] ) - y[i] );
				den += double( y[i] ) * y[i];
				amax = std::max( amax, std::fabs( double( y[i] ) ) );
				worst = std::max( worst, std::fabs( double( x[i] ) - y[i] ) );
			}
			worst /= amax;
			return std::sqrt( num / den );
		};
		for ( int log = 1; log >= 0; --log ) {
			const auto d32 = run( log, 0, 32 ), h32 = run( log, 1, 32 ), p32 = run( log, 2, 32 ), d8 = run( log, 0, 8 ), h8 = run( log, 1, 8 ), p8 = run( log, 2, 8 );
			double w1, w2, w3;
			const double r1 = cmp( h32, d32, w1 ), r2 = cmp( h8, d8, w2 ), r3 = cmp( d8, d32, w3 );
			const char* form = log ? "log scales" : "additive";
			std::snprintf( line, sizeof( line ), "(e) Adam, %s: device against the host reference, FP32 moments: largest difference %.2e of the largest delta", form, w1 );
			check( w1 < 1e-4, line );
			std::snprintf( line, sizeof( line ), "(e) Adam, %s: device against the host reference, 8-bit moments: relative L2 %.2e, largest %.2e", form, r2, w2 );
			check( r2 < 1e-3, line );
			const bool same = std::memcmp( p32.data(), d32.data(), n * sizeof( float ) ) == 0 && std::memcmp( p8.data(), d8.data(), n * sizeof( float ) ) == 0;
			std::snprintf( line, sizeof( line ), "(e) Adam, %s: the state updated in parts equals one update, FP32 and 8-bit: %s", form, same ? "bit for bit" : "no" );
			check( same, line );
			std::snprintf( line, sizeof( line ), "(e) Adam, %s: 8-bit against FP32 moments after 50 steps: relative L2 %.4f, largest %.4f (no outliers)", form, r3, w3 );
			check( w3 < 0.5, line );
			(void) r1;
		}
		// optimization: L = sum c ( a - t )^2 with c over six decades and noisy gradients, 300 additive steps from 0; the 8-bit moments reach
		// the FP32 ones' loss
		{
			std::vector<float> t( n ), c( n );
			std::normal_distribution<float> tn( 0.0f, 0.05f );
			for ( size_t i = 0; i < n; ++i ) {
				t[i] = tn( rng );
				c[i] = mag[i];
			}
			std::vector<float> zero( n, 0.0f );
			auto loss = [&]( const std::vector<float>& a ) {
				double L = 0.0;
				for ( size_t i = 0; i < n; ++i ) L += double( c[i] ) * ( a[i] - t[i] ) * ( a[i] - t[i] );
				return L;
			};
			auto quad = [&]( int bits ) {
				std::mt19937 nr( 5 );
				std::vector<float> a( n, 0.0f ), w( n, 0.0f ), m( n, 0.0f ), v( n, 0.0f ), mmax( nb, 0.0f ), vmax( nb, 0.0f ), g( n );
				std::vector<uint8_t> mq( n, 0 ), vq( n, 0 );
				float *d_w0 = up( zero ), *d_a = up( a ), *d_w = up( a ), *d_g = up( a ), *d_m = up( m ), *d_v = up( v ), *d_mm = up( mmax ), *d_vm = up( vmax );
				uint8_t *d_mq = nullptr, *d_vq = nullptr;
				CUDA_CHECK( cudaMalloc( &d_mq, n ) );
				CUDA_CHECK( cudaMalloc( &d_vq, n ) );
				CUDA_CHECK( cudaMemset( d_mq, 0, n ) );
				CUDA_CHECK( cudaMemset( d_vq, 0, n ) );
				engine::train::AdamOptions qo;
				qo.lr = 2e-3f;
				for ( int k = 1; k <= 300; ++k ) {
					for ( size_t i = 0; i < n; ++i ) g[i] = 2.0f * c[i] * ( a[i] - t[i] ) + 0.02f * c[i] * gn( nr );
					CUDA_CHECK( cudaMemcpy( d_g, g.data(), n * sizeof( float ), cudaMemcpyHostToDevice ) );
					if ( bits == 32 ) engine::train::adam_deltas( d_w0, d_a, d_w, d_g, d_m, d_v, n, k, 1.0f, qo, st );
					else engine::train::adam_deltas8( d_w0, d_a, d_w, d_g, d_mq, d_vq, d_mm, d_vm, n, k, 1.0f, qo, st );
					CUDA_CHECK( cudaStreamSynchronize( st ) );
					a = down( d_a, n );
				}
				for ( void* q : { (void*) d_w0, (void*) d_a, (void*) d_w, (void*) d_g, (void*) d_m, (void*) d_v, (void*) d_mm, (void*) d_vm, (void*) d_mq, (void*) d_vq } )
					CUDA_CHECK( cudaFree( q ) );
				return loss( a );
			};
			const double L0 = loss( zero ), l32 = quad( 32 ), l8 = quad( 8 );
			std::snprintf( line, sizeof( line ), "(e) Adam on a noisy quadratic, 300 steps: loss %.4g -> %.4g with FP32 moments, %.4g with 8-bit", L0, l32, l8 );
			check( l32 < 0.05 * L0 && l8 < 1.1 * l32, line );
		}
		// codes: |decode( encode( y ) ) - y| within half a step (1/16 of y) above the subnormal range when rounded to the nearest, within a step
		// when stochastic and unbiased over the draws (v: its root); 0 and the absmax exact
		double worst = 0.0, worst_sr = 0.0, bias = 0.0;
		int counted = 0;
		for ( int i = 0; i < 20000; ++i ) {
			const float y = std::pow( 10.0f, ue( rng ) - 1.0f ), m = ( i & 1 ) ? -y : y;
			const float bm = q8::dec_m( q8::enc_m( m, 1.0f ), 1.0f ), br = std::sqrt( q8::dec_v( q8::enc_v( y * y, 1.0f ), 1.0f ) );
			if ( y > std::ldexp( 1.0f, -13 ) ) worst = std::max( worst, std::fabs( double( bm ) - m ) / y );
			worst = std::max( worst, std::fabs( double( br ) - y ) / y );
			if ( y <= std::ldexp( 1.0f, -13 ) ) continue;
			double mean = 0.0;
			for ( int k = 0; k < 256; ++k ) {
				const float sm = q8::dec_m( q8::enc_m( m, 1.0f, q8::draw( i, k, q8::kSaltM ) ), 1.0f );
				worst_sr = std::max( worst_sr, std::fabs( double( sm ) - m ) / y );
				mean += sm;
			}
			bias += ( mean / 256.0 - m ) / m;   // signed, relative: averages to 0 when unbiased
			++counted;
		}
		bias = std::fabs( bias / std::max( counted, 1 ) );
		const bool ends = q8::dec_m( q8::enc_m( 0.0f, 1.0f ), 1.0f ) == 0.0f && q8::dec_m( q8::enc_m( -1.0f, 1.0f ), 1.0f ) == -1.0f &&
			q8::dec_v( q8::enc_v( 1.0f, 1.0f ), 1.0f ) == 1.0f;
		std::snprintf( line, sizeof( line ), "(e) 8-bit codes: worst relative round trip %.4f nearest (half a step 0.0625), %.4f stochastic (a step 0.125)", worst, worst_sr );
		check( worst <= 0.0625 + 1e-6 && worst_sr <= 0.125 + 1e-6, line );
		std::snprintf( line, sizeof( line ), "(e) 8-bit codes: stochastic rounding's mean relative bias %.5f; 0 and the absmax exact: %s", bias, ends ? "yes" : "no" );
		check( bias < 1e-3 && ends, line );
	}
	std::printf( fails ? "=== TRAIN GRAD: %d FAILURE(S) ===\n" : "=== TRAIN GRAD: PASS ===\n", fails );
	return fails ? 1 : 0;
}
