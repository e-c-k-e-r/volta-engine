// Tier 2: the qwen4_exp training operations (engine/train/qwen4_exp.h) against double-precision host references: each forward against the
// host forward, each backward (from a random output gradient R) against central differences of the host loss sum( R * output ) on a
// sample of input entries. Small shapes; no checkpoint.
#include "engine/config.h"
#include "engine/train/qwen4_exp.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

namespace {

namespace q = engine::train::qwen4_exp;
using Vec = std::vector<double>;
int fails = 0;
std::mt19937 rng( 1234 );

void check( bool ok, const char* line ) {
	std::printf( "  %-100s %s\n", line, ok ? "OK" : "FAIL" );
	fails += !ok;
}

Vec randv( size_t n, double scale = 1.0 ) {
	std::normal_distribution<double> d( 0.0, scale );
	Vec v( n );
	for ( auto& x : v ) x = d( rng );
	return v;
}

float* up( const Vec& v ) {
	std::vector<float> f( v.begin(), v.end() );
	float* d = nullptr;
	CUDA_CHECK( cudaMalloc( &d, std::max<size_t>( f.size(), 1 ) * sizeof( float ) ) );
	if ( !f.empty() ) CUDA_CHECK( cudaMemcpy( d, f.data(), f.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
	return d;
}

float* zeros( size_t n ) {
	float* d = nullptr;
	CUDA_CHECK( cudaMalloc( &d, std::max<size_t>( n, 1 ) * sizeof( float ) ) );
	CUDA_CHECK( cudaMemset( d, 0, std::max<size_t>( n, 1 ) * sizeof( float ) ) );
	return d;
}

Vec down( const float* d, size_t n ) {
	std::vector<float> f( n );
	CUDA_CHECK( cudaDeviceSynchronize() );
	CUDA_CHECK( cudaMemcpy( f.data(), d, n * sizeof( float ), cudaMemcpyDeviceToHost ) );
	return Vec( f.begin(), f.end() );
}

double dot( const Vec& a, const Vec& b ) {
	double s = 0.0;
	for ( size_t i = 0; i < a.size(); ++i ) s += a[i] * b[i];
	return s;
}

// Largest |a - b| relative to the largest |b|
double rel( const Vec& a, const Vec& b ) {
	double m = 0.0, d = 0.0;
	for ( size_t i = 0; i < b.size(); ++i ) {
		m = std::max( m, std::fabs( b[i] ) );
		d = std::max( d, std::fabs( a[i] - b[i] ) );
	}
	return d / ( m + 1e-12 );
}

// The gradient g of loss( x ) against central differences at up to 48 sampled entries
double fd_rel( const std::function<double( const Vec& )>& loss, Vec x, const Vec& g, double h = 1e-5 ) {
	std::vector<size_t> idx( x.size() );
	for ( size_t i = 0; i < idx.size(); ++i ) idx[i] = i;
	std::shuffle( idx.begin(), idx.end(), rng );
	if ( idx.size() > 48 ) idx.resize( 48 );
	double m = 0.0, d = 0.0;
	for ( size_t i : idx ) {
		const double x0 = x[i];
		x[i] = x0 + h;
		const double lp = loss( x );
		x[i] = x0 - h;
		const double lm = loss( x );
		x[i] = x0;
		const double fd = ( lp - lm ) / ( 2 * h );
		m = std::max( m, std::fabs( fd ) );
		d = std::max( d, std::fabs( fd - g[i] ) );
	}
	return d / ( m + 1e-12 );
}

double sig( double x ) { return 1.0 / ( 1.0 + std::exp( -x ) ); }

// Host RMS norm
Vec h_rms( const Vec& x, const Vec& w, int wrows, bool one_plus, bool mean, int segs, int n, double eps ) {
	Vec y( x.size() );
	for ( int s = 0; s < segs; ++s ) {
		double q = 0.0;
		for ( int i = 0; i < n; ++i ) q += x[static_cast<size_t>( s ) * n + i] * x[static_cast<size_t>( s ) * n + i];
		const double inv = 1.0 / std::sqrt( q / ( mean ? n : 1 ) + eps );
		for ( int i = 0; i < n; ++i ) {
			const double wv = w.empty() ? 1.0 : ( one_plus ? 1.0 : 0.0 ) + w[static_cast<size_t>( s % wrows ) * n + i];
			y[static_cast<size_t>( s ) * n + i] = x[static_cast<size_t>( s ) * n + i] * inv * wv;
		}
	}
	return y;
}

void test_rms() {
	std::printf( "(a) RMS norms (stream norm with 1 + w, head norm, l2 norm)\n" );
	struct Case { const char* name; int segs, n, wrows; bool one_plus, mean, weighted; };
	const Case cases[] = { { "streams 6 x 4, 1 + w", 24, 40, 4, true, true, true }, { "heads, w", 10, 32, 1, false, true, true },
		{ "l2, no weight", 10, 32, 1, false, false, false } };
	for ( const Case& c : cases ) {
		const Vec x = randv( static_cast<size_t>( c.segs ) * c.n ), w = c.weighted ? randv( static_cast<size_t>( c.wrows ) * c.n, 0.3 ) : Vec();
		const Vec R = randv( x.size() );
		const double eps = 1e-6;
		float *dx = up( x ), *dw = c.weighted ? up( w ) : nullptr, *dinv = zeros( c.segs ), *dy = zeros( x.size() ), *dR = up( R ), *dgx = zeros( x.size() );
		q::rms_forward( dx, dw, c.wrows, c.one_plus, c.mean, c.segs, c.n, static_cast<float>( eps ), dinv, dy, nullptr );
		q::rms_backward( dx, dw, c.wrows, c.one_plus, c.mean, c.segs, c.n, dinv, dR, dgx, nullptr );
		const Vec y = down( dy, x.size() ), g = down( dgx, x.size() );
		const Vec yh = h_rms( x, w, c.wrows, c.one_plus, c.mean, c.segs, c.n, eps );
		auto loss = [&]( const Vec& xx ) { return dot( R, h_rms( xx, w, c.wrows, c.one_plus, c.mean, c.segs, c.n, eps ) ); };
		const double ef = rel( y, yh ), eg = fd_rel( loss, x, g );
		char line[200];
		std::snprintf( line, sizeof( line ), "%-24s forward %.2e, dx %.2e", c.name, ef, eg );
		check( ef < 1e-5 && eg < 1e-3, line );
		for ( float* p : { dx, dw, dinv, dy, dR, dgx } ) if ( p ) cudaFree( p );
	}
}

// Host hyper-connection site: mixed [rows, H] and post [rows, hc]
void h_hc( const Vec& x, const Vec& norm, const Vec& dn, const Vec& upw, const Vec& inj, int rows, int H, int hc, int rank, Vec& mixed, Vec& post ) {
	const int flat = hc * H;
	const Vec normed = h_rms( x, norm, hc, true, true, rows * hc, H, 1e-6 );
	mixed.assign( static_cast<size_t>( rows ) * H, 0.0 );
	post.assign( static_cast<size_t>( rows ) * hc, 0.0 );
	for ( int r = 0; r < rows; ++r ) {
		Vec t( rank );
		for ( int j = 0; j < rank; ++j ) {
			double z = 0.0;
			for ( int i = 0; i < flat; ++i ) z += normed[static_cast<size_t>( r ) * flat + i] * dn[static_cast<size_t>( j ) * flat + i];
			z /= hc;
			t[j] = z * sig( z );
		}
		for ( int s = 0; s < hc; ++s ) {
			double z = 0.0;
			for ( int i = 0; i < flat; ++i ) z += normed[static_cast<size_t>( r ) * flat + i] * inj[static_cast<size_t>( s ) * flat + i];
			post[static_cast<size_t>( r ) * hc + s] = 2.0 * sig( z / hc );
		}
		for ( int d = 0; d < H; ++d ) {
			double acc = 0.0;
			for ( int s = 0; s < hc; ++s ) {
				double u = 0.0;
				for ( int j = 0; j < rank; ++j ) u += upw[static_cast<size_t>( s * H + d ) * rank + j] * t[j];
				acc += sig( u ) * normed[( static_cast<size_t>( r ) * hc + s ) * H + d];
			}
			mixed[static_cast<size_t>( r ) * H + d] = acc / hc;
		}
	}
}

void test_hc() {
	std::printf( "(b) hyper-connection site (stream norms, low-rank gates, inject gates, mix)\n" );
	const int rows = 5, H = 24, hc = 4, rank = 6, flat = hc * H;
	const Vec x = randv( static_cast<size_t>( rows ) * flat ), norm = randv( flat, 0.2 ), dn = randv( static_cast<size_t>( rank ) * flat, 0.3 ),
		upw = randv( static_cast<size_t>( flat ) * rank, 0.5 ), inj = randv( static_cast<size_t>( hc ) * flat, 0.3 );
	const Vec Rm = randv( static_cast<size_t>( rows ) * H ), Rp = randv( static_cast<size_t>( rows ) * hc );
	engine::train::Blas blas( nullptr );
	q::HcWeights w;
	w.hidden = H; w.hc = hc; w.rank = rank;
	float *d_norm = up( norm ), *d_dn = up( dn ), *d_up = up( upw ), *d_inj = up( inj );
	w.norm = d_norm; w.down = d_dn; w.up = d_up; w.inject = d_inj;
	q::HcCache c;
	c.inv = zeros( rows * hc ); c.normed = zeros( static_cast<size_t>( rows ) * flat ); c.zt = zeros( rows * rank ); c.t = zeros( rows * rank );
	c.u = zeros( static_cast<size_t>( rows ) * flat ); c.zi = zeros( rows * hc );
	float *d_x = up( x ), *d_mixed = zeros( static_cast<size_t>( rows ) * H ), *d_post = zeros( rows * hc ), *d_Rm = up( Rm ), *d_Rp = up( Rp );
	float *d_dx = zeros( x.size() ), *d_scr = zeros( q::hc_scratch_floats( w, rows ) );
	q::hc_forward( blas, w, d_x, rows, c, d_mixed, d_post );
	q::hc_backward( blas, w, d_x, rows, c, d_Rm, d_Rp, d_dx, d_scr );
	Vec mh, ph;
	h_hc( x, norm, dn, upw, inj, rows, H, hc, rank, mh, ph );
	const double em = rel( down( d_mixed, mh.size() ), mh ), ep = rel( down( d_post, ph.size() ), ph );
	auto loss = [&]( const Vec& xx ) { Vec m, p; h_hc( xx, norm, dn, upw, inj, rows, H, hc, rank, m, p ); return dot( Rm, m ) + dot( Rp, p ); };
	const double eg = fd_rel( loss, x, down( d_dx, x.size() ) );
	char line[200];
	std::snprintf( line, sizeof( line ), "forward mixed %.2e, post %.2e, dx %.2e", em, ep, eg );
	check( em < 1e-4 && ep < 1e-4 && eg < 1e-3, line );
	// The final mixer: no inject, no post
	w.inject = nullptr;
	CUDA_CHECK( cudaMemset( d_dx, 0, x.size() * sizeof( float ) ) );
	q::hc_forward( blas, w, d_x, rows, c, d_mixed, nullptr );
	q::hc_backward( blas, w, d_x, rows, c, d_Rm, nullptr, d_dx, d_scr );
	auto loss2 = [&]( const Vec& xx ) { Vec m, p; h_hc( xx, norm, dn, upw, inj, rows, H, hc, rank, m, p ); return dot( Rm, m ); };
	const double eg2 = fd_rel( loss2, x, down( d_dx, x.size() ) );
	std::snprintf( line, sizeof( line ), "final mixer (no inject): dx %.2e", eg2 );
	check( eg2 < 1e-3, line );
	for ( float* p : { d_norm, d_dn, d_up, d_inj, c.inv, c.normed, c.zt, c.t, c.u, c.zi, d_x, d_mixed, d_post, d_Rm, d_Rp, d_dx, d_scr } ) cudaFree( p );
}

void test_apply() {
	std::printf( "(c) gated add x += post y\n" );
	const int rows = 6, H = 40, hc = 4;
	const Vec x = randv( static_cast<size_t>( rows ) * hc * H ), post = randv( rows * hc ), y = randv( static_cast<size_t>( rows ) * H ), R = randv( x.size() );
	float *d_x = up( x ), *d_post = up( post ), *d_y = up( y ), *d_R = up( R ), *d_dy = zeros( y.size() ), *d_dp = zeros( post.size() );
	q::apply_forward( d_x, d_post, d_y, rows, H, hc, nullptr );
	q::apply_backward( d_R, d_post, d_y, rows, H, hc, d_dy, d_dp, nullptr );
	auto fwd = [&]( const Vec& p, const Vec& yy ) {
		Vec o = x;
		for ( int r = 0; r < rows; ++r ) for ( int s = 0; s < hc; ++s ) for ( int d = 0; d < H; ++d )
			o[( static_cast<size_t>( r ) * hc + s ) * H + d] += p[r * hc + s] * yy[static_cast<size_t>( r ) * H + d];
		return o;
	};
	const double ef = rel( down( d_x, x.size() ), fwd( post, y ) );
	const double ey = fd_rel( [&]( const Vec& yy ) { return dot( R, fwd( post, yy ) ); }, y, down( d_dy, y.size() ) );
	const double ep = fd_rel( [&]( const Vec& p ) { return dot( R, fwd( p, y ) ); }, post, down( d_dp, post.size() ) );
	char line[200];
	std::snprintf( line, sizeof( line ), "forward %.2e, dy %.2e, dpost %.2e", ef, ey, ep );
	check( ef < 1e-5 && ey < 1e-3 && ep < 1e-3, line );
	for ( float* p : { d_x, d_post, d_y, d_R, d_dy, d_dp } ) cudaFree( p );
}

void test_conv() {
	std::printf( "(d) causal 4-tap convolution + SiLU (GDN: dilation 1; PLE: dilation 3)\n" );
	for ( int dil : { 1, 3 } ) {
		const int rows = 17, ch = 9;
		const Vec x = randv( static_cast<size_t>( rows ) * ch ), w = randv( static_cast<size_t>( ch ) * 4, 0.5 ), R = randv( x.size() );
		auto fwd = [&]( const Vec& xx ) {
			Vec y( xx.size() );
			for ( int t = 0; t < rows; ++t ) for ( int c = 0; c < ch; ++c ) {
				double s = 0.0;
				for ( int j = 0; j < 4; ++j ) { const int src = t - ( 3 - j ) * dil; if ( src >= 0 ) s += w[c * 4 + j] * xx[static_cast<size_t>( src ) * ch + c]; }
				y[static_cast<size_t>( t ) * ch + c] = s * sig( s );
			}
			return y;
		};
		float *d_x = up( x ), *d_w = up( w ), *d_s = zeros( x.size() ), *d_y = zeros( x.size() ), *d_R = up( R ), *d_dx = zeros( x.size() );
		q::conv4_silu_forward( d_x, d_w, rows, ch, dil, d_s, d_y, nullptr );
		q::conv4_silu_backward( d_s, d_w, rows, ch, dil, d_R, d_dx, nullptr );
		const double ef = rel( down( d_y, x.size() ), fwd( x ) ), eg = fd_rel( [&]( const Vec& xx ) { return dot( R, fwd( xx ) ); }, x, down( d_dx, x.size() ) );
		char line[200];
		std::snprintf( line, sizeof( line ), "dilation %d: forward %.2e, dx %.2e", dil, ef, eg );
		check( ef < 1e-5 && eg < 1e-3, line );
		for ( float* p : { d_x, d_w, d_s, d_y, d_R, d_dx } ) cudaFree( p );
	}
}

void test_gates() {
	std::printf( "(e) sigmoid output gate, GDN gated norm, top-k softmax\n" );
	{
		const size_t n = 300;
		const Vec a = randv( n ), g = randv( n ), R = randv( n );
		auto fwd = [&]( const Vec& aa, const Vec& gg ) { Vec o( n ); for ( size_t i = 0; i < n; ++i ) o[i] = aa[i] * sig( gg[i] ); return o; };
		float *d_a = up( a ), *d_g = up( g ), *d_o = zeros( n ), *d_R = up( R ), *d_da = zeros( n ), *d_dg = zeros( n );
		q::sigmoid_gate_forward( d_a, d_g, d_o, n, nullptr );
		q::sigmoid_gate_backward( d_a, d_g, d_R, d_da, d_dg, n, nullptr );
		const double ef = rel( down( d_o, n ), fwd( a, g ) );
		const double ea = fd_rel( [&]( const Vec& v ) { return dot( R, fwd( v, g ) ); }, a, down( d_da, n ) );
		const double eg = fd_rel( [&]( const Vec& v ) { return dot( R, fwd( a, v ) ); }, g, down( d_dg, n ) );
		char line[200];
		std::snprintf( line, sizeof( line ), "sigmoid gate: forward %.2e, da %.2e, dg %.2e", ef, ea, eg );
		check( ef < 1e-5 && ea < 1e-3 && eg < 1e-3, line );
		for ( float* p : { d_a, d_g, d_o, d_R, d_da, d_dg } ) cudaFree( p );
	}
	{
		const int segs = 12, n = 32;
		const Vec x = randv( static_cast<size_t>( segs ) * n ), z = randv( x.size() ), gamma = randv( n, 0.5 ), R = randv( x.size() );
		auto fwd = [&]( const Vec& xx, const Vec& zz ) {
			Vec y( xx.size() );
			for ( int s = 0; s < segs; ++s ) {
				double q2 = 0.0;
				for ( int i = 0; i < n; ++i ) q2 += xx[static_cast<size_t>( s ) * n + i] * xx[static_cast<size_t>( s ) * n + i];
				const double inv = 1.0 / std::sqrt( q2 / n + 1e-6 );
				for ( int i = 0; i < n; ++i ) { const size_t j = static_cast<size_t>( s ) * n + i; y[j] = xx[j] * inv * gamma[i] * sig( zz[j] ); }
			}
			return y;
		};
		float *d_x = up( x ), *d_z = up( z ), *d_gm = up( gamma ), *d_inv = zeros( segs ), *d_y = zeros( x.size() ), *d_R = up( R ), *d_dx = zeros( x.size() ),
			*d_dz = zeros( x.size() );
		q::gated_rms_forward( d_x, d_z, d_gm, segs, n, 1e-6f, d_inv, d_y, nullptr );
		q::gated_rms_backward( d_x, d_z, d_gm, segs, n, d_inv, d_R, d_dx, d_dz, nullptr );
		const double ef = rel( down( d_y, x.size() ), fwd( x, z ) );
		const double ex = fd_rel( [&]( const Vec& v ) { return dot( R, fwd( v, z ) ); }, x, down( d_dx, x.size() ) );
		const double ez = fd_rel( [&]( const Vec& v ) { return dot( R, fwd( x, v ) ); }, z, down( d_dz, x.size() ) );
		char line[200];
		std::snprintf( line, sizeof( line ), "gated norm: forward %.2e, dx %.2e, dz %.2e", ef, ex, ez );
		check( ef < 1e-5 && ex < 1e-3 && ez < 1e-3, line );
		for ( float* p : { d_x, d_z, d_gm, d_inv, d_y, d_R, d_dx, d_dz } ) cudaFree( p );
	}
	{
		const int rows = 7, k = 10;
		const Vec logit = randv( static_cast<size_t>( rows ) * k, 2.0 ), R = randv( logit.size() );
		auto fwd = [&]( const Vec& l ) {
			Vec w( l.size() );
			for ( int r = 0; r < rows; ++r ) {
				double m = -1e30, s = 0.0;
				for ( int j = 0; j < k; ++j ) m = std::max( m, l[r * k + j] );
				for ( int j = 0; j < k; ++j ) s += std::exp( l[r * k + j] - m );
				for ( int j = 0; j < k; ++j ) w[r * k + j] = std::exp( l[r * k + j] - m ) / s;
			}
			return w;
		};
		const Vec w = fwd( logit );
		float *d_w = up( w ), *d_R = up( R ), *d_dl = zeros( w.size() );
		q::topk_softmax_backward( d_w, d_R, rows, k, d_dl, nullptr );
		const double el = fd_rel( [&]( const Vec& v ) { return dot( R, fwd( v ) ); }, logit, down( d_dl, w.size() ) );
		char line[200];
		std::snprintf( line, sizeof( line ), "top-k softmax: dlogit %.2e", el );
		check( el < 1e-3, line );
		for ( float* p : { d_w, d_R, d_dl } ) cudaFree( p );
	}
}

// Host gated delta rule, the same recurrence as gdn_forward
Vec h_gdn( const Vec& q, const Vec& k, const Vec& v, const Vec& gate, const Vec& beta, int rows, int kh, int vh, double scale ) {
	const int D = 128;
	Vec o( static_cast<size_t>( rows ) * vh * D, 0.0 );
	std::vector<Vec> S( vh, Vec( static_cast<size_t>( D ) * D, 0.0 ) );
	for ( int t = 0; t < rows; ++t ) {
		for ( int h = 0; h < vh; ++h ) {
			const int kk = h / ( vh / kh );
			const double g = std::exp( gate[static_cast<size_t>( t ) * vh + h] ), bt = beta[static_cast<size_t>( t ) * vh + h];
			for ( int c = 0; c < D; ++c ) {
				double* Sc = S[h].data() + static_cast<size_t>( c ) * D;
				double u = 0.0;
				for ( int i = 0; i < D; ++i ) u += Sc[i] * k[( static_cast<size_t>( t ) * kh + kk ) * D + i];
				const double d = bt * ( v[( static_cast<size_t>( t ) * vh + h ) * D + c] - g * u );
				double oc = 0.0;
				for ( int i = 0; i < D; ++i ) {
					Sc[i] = g * Sc[i] + d * k[( static_cast<size_t>( t ) * kh + kk ) * D + i];
					oc += Sc[i] * q[( static_cast<size_t>( t ) * kh + kk ) * D + i];
				}
				o[( static_cast<size_t>( t ) * vh + h ) * D + c] = oc * scale;
			}
		}
	}
	return o;
}

void test_gdn() {
	std::printf( "(f) GDN gated delta rule (chunked checkpoints) and its gates\n" );
	const int rows = 11, kh = 2, vh = 4, D = 128, chunk = 4;
	const double scale = 1.0 / std::sqrt( 128.0 );
	auto unit = [&]( Vec x, int heads ) {   // l2-normed heads, as the layer feeds them
		for ( int r = 0; r < rows * heads; ++r ) {
			double n = 0.0;
			for ( int i = 0; i < D; ++i ) n += x[static_cast<size_t>( r ) * D + i] * x[static_cast<size_t>( r ) * D + i];
			for ( int i = 0; i < D; ++i ) x[static_cast<size_t>( r ) * D + i] /= std::sqrt( n );
		}
		return x;
	};
	const Vec q = unit( randv( static_cast<size_t>( rows ) * kh * D ), kh ), k = unit( randv( q.size() ), kh ), v = randv( static_cast<size_t>( rows ) * vh * D );
	Vec gate = randv( static_cast<size_t>( rows ) * vh, 0.3 ), beta( gate.size() );
	for ( auto& x : gate ) x = -std::fabs( x );
	for ( auto& x : beta ) x = std::uniform_real_distribution<double>( 0.1, 0.9 )( rng );
	const Vec R = randv( v.size() );
	float *d_q = up( q ), *d_k = up( k ), *d_v = up( v ), *d_g = up( gate ), *d_b = up( beta ), *d_o = zeros( v.size() ), *d_R = up( R );
	float *d_ck = zeros( q::gdn_ckpt_floats( rows, vh, chunk ) ), *d_scr = zeros( q::gdn_scratch_floats( vh, chunk ) );
	float *d_dq = zeros( q.size() ), *d_dk = zeros( k.size() ), *d_dv = zeros( v.size() ), *d_dg = zeros( gate.size() ), *d_db = zeros( beta.size() );
	q::gdn_forward( d_q, d_k, d_v, d_g, d_b, rows, kh, vh, static_cast<float>( scale ), chunk, d_ck, d_o, nullptr );
	q::gdn_backward( d_q, d_k, d_v, d_g, d_b, rows, kh, vh, static_cast<float>( scale ), chunk, d_ck, d_R, d_dq, d_dk, d_dv, d_dg, d_db, d_scr, nullptr );
	const double ef = rel( down( d_o, v.size() ), h_gdn( q, k, v, gate, beta, rows, kh, vh, scale ) );
	const double eq = fd_rel( [&]( const Vec& x ) { return dot( R, h_gdn( x, k, v, gate, beta, rows, kh, vh, scale ) ); }, q, down( d_dq, q.size() ) );
	const double ek = fd_rel( [&]( const Vec& x ) { return dot( R, h_gdn( q, x, v, gate, beta, rows, kh, vh, scale ) ); }, k, down( d_dk, k.size() ) );
	const double ev = fd_rel( [&]( const Vec& x ) { return dot( R, h_gdn( q, k, x, gate, beta, rows, kh, vh, scale ) ); }, v, down( d_dv, v.size() ) );
	const double eg = fd_rel( [&]( const Vec& x ) { return dot( R, h_gdn( q, k, v, x, beta, rows, kh, vh, scale ) ); }, gate, down( d_dg, gate.size() ) );
	const double eb = fd_rel( [&]( const Vec& x ) { return dot( R, h_gdn( q, k, v, gate, x, rows, kh, vh, scale ) ); }, beta, down( d_db, beta.size() ) );
	char line[200];
	std::snprintf( line, sizeof( line ), "%d rows, chunks of %d: forward %.2e, dq %.2e, dk %.2e, dv %.2e, dgate %.2e, dbeta %.2e", rows, chunk, ef, eq, ek, ev,
		eg, eb );
	check( ef < 1e-5 && eq < 1e-3 && ek < 1e-3 && ev < 1e-3 && eg < 1e-3 && eb < 1e-3, line );
	for ( float* p : { d_q, d_k, d_v, d_g, d_b, d_o, d_R, d_ck, d_scr, d_dq, d_dk, d_dv, d_dg, d_db } ) cudaFree( p );
	// The gates
	const int gr = 9;
	const Vec a = randv( static_cast<size_t>( gr ) * vh, 2.0 ), b = randv( a.size() ), Alog = randv( vh, 0.5 ), dt = randv( vh, 0.5 );
	const Vec Rg = randv( a.size() ), Rb = randv( a.size() );
	auto gates = [&]( const Vec& aa, const Vec& bb ) {
		double L = 0.0;
		for ( int r = 0; r < gr; ++r ) for ( int h = 0; h < vh; ++h ) {
			const size_t i = static_cast<size_t>( r ) * vh + h;
			const double x = aa[i] + dt[h], sp = x > 20.0 ? x : std::log1p( std::exp( x ) );
			L += Rg[i] * -std::exp( Alog[h] ) * sp + Rb[i] * sig( bb[i] );
		}
		return L;
	};
	float *d_a = up( a ), *d_bb = up( b ), *d_al = up( Alog ), *d_dt = up( dt ), *d_gt = zeros( a.size() ), *d_bt = zeros( a.size() ), *d_Rg = up( Rg ),
		*d_Rb = up( Rb ), *d_da = zeros( a.size() ), *d_dbb = zeros( a.size() );
	q::gdn_gates_forward( d_a, d_bb, d_al, d_dt, gr, vh, d_gt, d_bt, nullptr );
	q::gdn_gates_backward( d_a, d_bb, d_al, d_dt, gr, vh, d_Rg, d_Rb, d_da, d_dbb, nullptr );
	const double ea = fd_rel( [&]( const Vec& x ) { return gates( x, b ); }, a, down( d_da, a.size() ) );
	const double ebb = fd_rel( [&]( const Vec& x ) { return gates( a, x ); }, b, down( d_dbb, b.size() ) );
	std::snprintf( line, sizeof( line ), "gates: da %.2e, db %.2e", ea, ebb );
	check( ea < 1e-3 && ebb < 1e-3, line );
	for ( float* p : { d_a, d_bb, d_al, d_dt, d_gt, d_bt, d_Rg, d_Rb, d_da, d_dbb } ) cudaFree( p );
}

Vec h_attn( const Vec& q, const Vec& k, const Vec& v, int T, int hq, int hk, int d, double scale ) {
	Vec o( static_cast<size_t>( T ) * hq * d, 0.0 );
	for ( int h = 0; h < hq; ++h ) {
		const int kv = h / ( hq / hk );
		for ( int i = 0; i < T; ++i ) {
			Vec sc( i + 1 );
			double m = -1e300;
			for ( int j = 0; j <= i; ++j ) {
				double z = 0.0;
				for ( int c = 0; c < d; ++c ) z += q[( static_cast<size_t>( i ) * hq + h ) * d + c] * k[( static_cast<size_t>( j ) * hk + kv ) * d + c];
				sc[j] = z * scale;
				m = std::max( m, sc[j] );
			}
			double sum = 0.0;
			for ( int j = 0; j <= i; ++j ) { sc[j] = std::exp( sc[j] - m ); sum += sc[j]; }
			for ( int j = 0; j <= i; ++j ) for ( int c = 0; c < d; ++c )
				o[( static_cast<size_t>( i ) * hq + h ) * d + c] += sc[j] / sum * v[( static_cast<size_t>( j ) * hk + kv ) * d + c];
		}
	}
	return o;
}

Vec h_rope( Vec x, int T, int heads, int hd, int rot, double base, double sign ) {
	const int pairs = rot / 2;
	for ( int r = 0; r < T; ++r ) for ( int h = 0; h < heads; ++h ) for ( int p = 0; p < pairs; ++p ) {
		const double th = r * std::pow( base, -2.0 * p / rot ), c = std::cos( th ), s = sign * std::sin( th );
		double* xh = x.data() + ( static_cast<size_t>( r ) * heads + h ) * hd;
		const double a = xh[p], b = xh[p + pairs];
		xh[p] = a * c - b * s;
		xh[p + pairs] = a * s + b * c;
	}
	return x;
}

void test_attention() {
	std::printf( "(g) attention: RoPE, causal GQA softmax attention, q / gate packing\n" );
	const int T = 9, hq = 4, hk = 2, d = 16, rot = 8;
	const double scale = 0.25, base = 10000.0;
	{
		const Vec x = randv( static_cast<size_t>( T ) * hq * d ), R = randv( x.size() );
		float *d_x = up( x ), *d_R = up( R );
		q::rope_forward( d_x, T, hq, d, rot, static_cast<float>( base ), nullptr );
		q::rope_backward( d_R, T, hq, d, rot, static_cast<float>( base ), nullptr );
		const double ef = rel( down( d_x, x.size() ), h_rope( x, T, hq, d, rot, base, 1.0 ) );
		const double eg = fd_rel( [&]( const Vec& xx ) { return dot( R, h_rope( xx, T, hq, d, rot, base, 1.0 ) ); }, x, down( d_R, x.size() ) );
		char line[200];
		std::snprintf( line, sizeof( line ), "RoPE (%d of %d channels): forward %.2e, dx %.2e", rot, d, ef, eg );
		check( ef < 1e-4 && eg < 1e-3, line );
		cudaFree( d_x ); cudaFree( d_R );
	}
	{
		const Vec qv = randv( static_cast<size_t>( T ) * hq * d ), kv = randv( static_cast<size_t>( T ) * hk * d ), vv = randv( kv.size() ), R = randv( qv.size() );
		engine::train::Blas blas( nullptr );
		float *d_q = up( qv ), *d_k = up( kv ), *d_v = up( vv ), *d_o = zeros( qv.size() ), *d_R = up( R ), *d_s = zeros( T * T ), *d_p = zeros( T * T );
		float *d_dq = zeros( qv.size() ), *d_dk = zeros( kv.size() ), *d_dv = zeros( vv.size() );
		q::attention_forward( blas, d_q, d_k, d_v, T, hq, hk, d, static_cast<float>( scale ), d_s, d_o );
		q::attention_backward( blas, d_q, d_k, d_v, T, hq, hk, d, static_cast<float>( scale ), d_R, d_dq, d_dk, d_dv, d_s, d_p );
		const double ef = rel( down( d_o, qv.size() ), h_attn( qv, kv, vv, T, hq, hk, d, scale ) );
		const double eq = fd_rel( [&]( const Vec& x ) { return dot( R, h_attn( x, kv, vv, T, hq, hk, d, scale ) ); }, qv, down( d_dq, qv.size() ) );
		const double ek = fd_rel( [&]( const Vec& x ) { return dot( R, h_attn( qv, x, vv, T, hq, hk, d, scale ) ); }, kv, down( d_dk, kv.size() ) );
		const double ev = fd_rel( [&]( const Vec& x ) { return dot( R, h_attn( qv, kv, x, T, hq, hk, d, scale ) ); }, vv, down( d_dv, vv.size() ) );
		char line[200];
		std::snprintf( line, sizeof( line ), "%d rows, %d q / %d kv heads: forward %.2e, dq %.2e, dk %.2e, dv %.2e", T, hq, hk, ef, eq, ek, ev );
		check( ef < 1e-5 && eq < 1e-3 && ek < 1e-3 && ev < 1e-3, line );
		for ( float* p : { d_q, d_k, d_v, d_o, d_R, d_s, d_p, d_dq, d_dk, d_dv } ) cudaFree( p );
	}
	{
		const Vec qg = randv( static_cast<size_t>( T ) * hq * 2 * d );
		float *d_qg = up( qg ), *d_q = zeros( qg.size() / 2 ), *d_g = zeros( qg.size() / 2 ), *d_back = zeros( qg.size() );
		q::unpack_q_gate( d_qg, T, hq, d, d_q, d_g, nullptr );
		q::pack_q_gate( d_q, d_g, T, hq, d, d_back, nullptr );
		const Vec qh = down( d_q, qg.size() / 2 );
		Vec qgf( qg.size() );
		for ( size_t i = 0; i < qg.size(); ++i ) qgf[i] = static_cast<float>( qg[i] );
		bool ok = rel( down( d_back, qg.size() ), qgf ) == 0.0 && qh[0] == qgf[0] && qh[d] == qgf[2 * d];
		check( ok, "q / gate unpack and pack: exact round trip" );
		for ( float* p : { d_qg, d_q, d_g, d_back } ) cudaFree( p );
	}
}

void test_routing_ple() {
	std::printf( "(h) router top-k, routing backward pieces, PLE gate\n" );
	char line[200];
	{
		const int rows = 6, E = 40, k = 5;
		Vec logit = randv( static_cast<size_t>( rows ) * E );
		logit[3] = logit[7];   // a tie: the lower index first
		float* d_l = up( logit );
		int* d_ids = nullptr;
		CUDA_CHECK( cudaMalloc( &d_ids, rows * k * sizeof( int ) ) );
		float* d_w = zeros( rows * k );
		q::router_topk( d_l, rows, E, k, d_ids, d_w, nullptr );
		std::vector<int> ids( rows * k );
		CUDA_CHECK( cudaMemcpy( ids.data(), d_ids, ids.size() * sizeof( int ), cudaMemcpyDeviceToHost ) );
		const Vec w = down( d_w, rows * k );
		bool ok = true;
		double we = 0.0;
		for ( int r = 0; r < rows; ++r ) {
			std::vector<int> ord( E );
			for ( int e = 0; e < E; ++e ) ord[e] = e;
			std::stable_sort( ord.begin(), ord.end(), [&]( int a, int b ) { return static_cast<float>( logit[r * E + a] ) > static_cast<float>( logit[r * E + b] ); } );
			double sum = 0.0;
			for ( int j = 0; j < k; ++j ) sum += std::exp( logit[r * E + ord[j]] - logit[r * E + ord[0]] );
			for ( int j = 0; j < k; ++j ) {
				ok = ok && ids[r * k + j] == ord[j];
				we = std::max( we, std::fabs( w[r * k + j] - std::exp( logit[r * E + ord[j]] - logit[r * E + ord[0]] ) / sum ) );
			}
		}
		std::snprintf( line, sizeof( line ), "top-%d of %d: selection %s, weights %.2e", k, E, ok ? "exact" : "WRONG", we );
		check( ok && we < 1e-6, line );
		// dX += sum_j dlogit R[ids], and the route dots
		const int H = 24;
		const Vec R = randv( static_cast<size_t>( E ) * H ), dl = randv( rows * k );
		float *d_R = up( R ), *d_dl = up( dl ), *d_dX = zeros( static_cast<size_t>( rows ) * H );
		q::router_input_backward( d_dl, d_ids, d_R, rows, k, H, d_dX, nullptr );
		Vec dX( static_cast<size_t>( rows ) * H, 0.0 );
		for ( int r = 0; r < rows; ++r ) for ( int j = 0; j < k; ++j ) for ( int d = 0; d < H; ++d )
			dX[static_cast<size_t>( r ) * H + d] += static_cast<float>( dl[r * k + j] ) * static_cast<float>( R[static_cast<size_t>( ids[r * k + j] ) * H + d] );
		// route dots over padded routes: out[slot] = dY[row] . Y[p], the padding row skipped
		const Vec dY = randv( static_cast<size_t>( rows ) * H ), Y = randv( static_cast<size_t>( 4 ) * H );
		const std::vector<int> ri = { 5, 0, -1, 3, 2, 0, -1, 1 };   // rows, then slots
		int* d_ri = nullptr;
		CUDA_CHECK( cudaMalloc( &d_ri, ri.size() * sizeof( int ) ) );
		CUDA_CHECK( cudaMemcpy( d_ri, ri.data(), ri.size() * sizeof( int ), cudaMemcpyHostToDevice ) );
		float *d_dY = up( dY ), *d_Y = up( Y ), *d_out = zeros( 4 );
		q::route_dots_pad( d_dY, d_ri, d_ri + 4, d_Y, 4, H, d_out, nullptr );
		Vec outh( 4, 0.0 );
		for ( int i = 0; i < 4; ++i ) {
			if ( ri[i] < 0 ) continue;
			for ( int d = 0; d < H; ++d ) outh[ri[4 + i]] += dY[static_cast<size_t>( ri[i] ) * H + d] * Y[static_cast<size_t>( i ) * H + d];
		}
		const double e1 = rel( down( d_dX, dX.size() ), dX ), e2 = rel( down( d_out, 4 ), outh );
		std::snprintf( line, sizeof( line ), "router input gradient %.2e, route dots %.2e", e1, e2 );
		check( e1 < 1e-5 && e2 < 1e-5, line );
		for ( float* p : { d_l, d_w, d_R, d_dl, d_dX, d_dY, d_Y, d_out } ) cudaFree( p );
		cudaFree( d_ids ); cudaFree( d_ri );
	}
	{
		const size_t n = 200;
		Vec g = randv( n );
		g[0] = 0.0; g[1] = 1e-8;   // the clamp
		const Vec R = randv( n );
		auto fwd = [&]( const Vec& x ) {
			double L = 0.0;
			for ( size_t i = 0; i < n; ++i ) {
				const double sgn = x[i] > 0 ? 1.0 : ( x[i] < 0 ? -1.0 : 0.0 );
				L += R[i] * sig( sgn * std::sqrt( std::max( std::fabs( x[i] ), 1e-6 ) ) );
			}
			return L;
		};
		float *d_g = up( g ), *d_sg = zeros( n ), *d_R = up( R ), *d_dg = zeros( n );
		q::ple_gate_forward( d_g, d_sg, n, nullptr );
		q::ple_gate_backward( d_g, d_R, d_dg, n, nullptr );
		Vec gg = g;
		gg[0] = 0.3; gg[1] = -0.2;   // central differences away from the kink
		Vec dgh = down( d_dg, n );
		const double eg = fd_rel( [&]( const Vec& x ) { Vec y = x; y[0] = g[0]; y[1] = g[1]; return fwd( y ); }, g, dgh, 1e-7 );
		std::snprintf( line, sizeof( line ), "PLE gate: dg %.2e (zero at the clamp: %s)", eg, dgh[0] == 0.0 && dgh[1] == 0.0 ? "yes" : "no" );
		check( eg < 1e-3 && dgh[0] == 0.0 && dgh[1] == 0.0, line );
		(void) gg;
		for ( float* p : { d_g, d_sg, d_R, d_dg } ) cudaFree( p );
	}
}

void test_kl() {
	std::printf( "(i) KL head over vocabulary chunks (top-k teacher with a lumped tail; the stable form of arXiv 2610.31642)\n" );
	const int rows = 4, V = 50, tk = 6, NC = 20;
	for ( int confident = 0; confident < 2; ++confident ) {
		Vec logits = randv( static_cast<size_t>( rows ) * V, 2.0 );
		std::vector<int> ids( rows * tk );
		Vec p( rows * tk ), tail( rows );
		for ( int r = 0; r < rows; ++r ) {
			std::vector<int> perm( V );
			for ( int i = 0; i < V; ++i ) perm[i] = i;
			std::shuffle( perm.begin(), perm.end(), rng );
			double sp = 0.0;
			for ( int j = 0; j < tk; ++j ) {
				ids[r * tk + j] = perm[j];
				p[r * tk + j] = confident ? ( j == 0 ? 1.0 - 1e-5 : 1e-6 ) : std::uniform_real_distribution<double>( 0.02, 0.2 )( rng );
				sp += p[r * tk + j];
			}
			tail[r] = 1.0 - sp;   // a normalized teacher (confident: 5e-6 outside its top)
			// confident: the model puts nearly all its mass on the teacher's first token (margin ~30: 1 - sum( q_top ) is 0 in FP32)
			if ( confident ) logits[static_cast<size_t>( r ) * V + perm[0]] += 30.0;
		}
		// double-precision reference: the KL and its analytic gradient
		auto host = [&]( const Vec& l, Vec* g ) {
			double K = 0.0;
			if ( g ) g->assign( l.size(), 0.0 );
			for ( int r = 0; r < rows; ++r ) {
				double m = -1e300, s = 0.0;
				for ( int i = 0; i < V; ++i ) m = std::max( m, l[static_cast<size_t>( r ) * V + i] );
				for ( int i = 0; i < V; ++i ) s += std::exp( l[static_cast<size_t>( r ) * V + i] - m );
				const double lse = m + std::log( s );
				std::vector<char> top( V, 0 );
				for ( int j = 0; j < tk; ++j ) top[ids[r * tk + j]] = 1;
				double qt = 0.0;
				for ( int i = 0; i < V; ++i ) if ( !top[i] ) qt += std::exp( l[static_cast<size_t>( r ) * V + i] - lse );
				for ( int j = 0; j < tk; ++j ) {
					const double lq = l[static_cast<size_t>( r ) * V + ids[r * tk + j]] - lse;
					K += p[r * tk + j] * ( std::log( p[r * tk + j] ) - lq );
					if ( g ) ( *g )[static_cast<size_t>( r ) * V + ids[r * tk + j]] = std::exp( lq ) - p[r * tk + j];
				}
				K += tail[r] * ( std::log( tail[r] ) - std::log( qt ) );
				if ( g ) for ( int i = 0; i < V; ++i ) if ( !top[i] ) ( *g )[static_cast<size_t>( r ) * V + i] = std::exp( l[static_cast<size_t>( r ) * V + i] - lse ) * ( 1.0 - tail[r] / qt );
			}
			return K;
		};
		int* d_ids = nullptr;
		CUDA_CHECK( cudaMalloc( &d_ids, ids.size() * sizeof( int ) ) );
		CUDA_CHECK( cudaMemcpy( d_ids, ids.data(), ids.size() * sizeof( int ), cudaMemcpyHostToDevice ) );
		float *d_p = up( p ), *d_tail = up( tail ), *d_m = up( Vec( rows, -INFINITY ) ), *d_s = zeros( rows ), *d_sl = zeros( rows * tk ), *d_lt = zeros( rows ),
			*d_gq = zeros( rows * tk ), *d_kl = zeros( rows ), *d_ut = zeros( rows ), *d_dl = zeros( static_cast<size_t>( rows ) * NC );
		auto chunk = [&]( int c0, int nc ) {
			Vec part( static_cast<size_t>( rows ) * nc );
			for ( int r = 0; r < rows; ++r ) for ( int i = 0; i < nc; ++i ) part[static_cast<size_t>( r ) * nc + i] = logits[static_cast<size_t>( r ) * V + c0 + i];
			return up( part );
		};
		for ( int c0 = 0; c0 < V; c0 += NC ) {
			const int nc = std::min( NC, V - c0 );
			float* c = chunk( c0, nc );
			q::kl_gather( c, rows, nc, c0, d_ids, tk, d_sl, nullptr );
			q::kl_lse_update( c, rows, nc, d_m, d_s, nullptr );
			cudaFree( c );
		}
		q::kl_finish( d_m, d_s, d_sl, d_p, d_tail, rows, tk, d_lt, d_gq, d_ut, d_kl, nullptr );
		const Vec kl = down( d_kl, rows );
		double kls = 0.0;
		for ( double v : kl ) kls += v;
		Vec g( logits.size() );
		for ( int c0 = 0; c0 < V; c0 += NC ) {
			const int nc = std::min( NC, V - c0 );
			float* c = chunk( c0, nc );
			q::kl_grad( c, rows, nc, c0, d_lt, d_ut, d_ids, d_gq, tk, d_dl, nullptr );
			const Vec dl = down( d_dl, static_cast<size_t>( rows ) * nc );
			for ( int r = 0; r < rows; ++r ) for ( int i = 0; i < nc; ++i ) g[static_cast<size_t>( r ) * V + c0 + i] = dl[static_cast<size_t>( r ) * nc + i];
			cudaFree( c );
		}
		Vec gh;
		const double Kh = host( logits, &gh );
		const double ek = std::fabs( kls - Kh ) / std::max( Kh, 1e-12 ), eg = rel( g, gh );
		double gsum = 0.0, gmax = 0.0;
		for ( int r = 0; r < rows; ++r ) {
			double sr = 0.0;
			for ( int i = 0; i < V; ++i ) { sr += g[static_cast<size_t>( r ) * V + i]; gmax = std::max( gmax, std::fabs( g[static_cast<size_t>( r ) * V + i] ) ); }
			gsum = std::max( gsum, std::fabs( sr ) );
		}
		char line[200];
		std::snprintf( line, sizeof( line ), "%s rows, vocabulary %d in chunks of %d: KL %.2e, dlogit %.2e, worst row sum %.1e of %.1e", confident ? "confident" : "plain",
			V, NC, ek, eg, gsum, gmax );
		check( ek < 1e-3 && eg < 1e-4 && gsum <= 1e-6 * gmax + 1e-12, line );   // the KL value of a confident row is a small FP32 sum (monitoring only)
		for ( float* x : { d_p, d_tail, d_m, d_s, d_sl, d_lt, d_gq, d_kl, d_ut, d_dl } ) cudaFree( x );
		cudaFree( d_ids );
	}
}

} // namespace

int main() {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	CUDA_CHECK( cudaSetDevice( 0 ) );
	test_rms();
	test_hc();
	test_apply();
	test_conv();
	test_gates();
	test_gdn();
	test_attention();
	test_routing_ple();
	test_kl();
	std::printf( "[%c] test_train_qwen4_exp: %d failure(s)\n", fails ? '-' : '+', fails );
	return fails ? 1 : 0;
}
