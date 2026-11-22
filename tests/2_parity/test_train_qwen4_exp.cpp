// Tier 2: the qwen4_exp training operations (engine/train/qwen4_exp.h) against double-precision host references: each forward against the
// host forward, each backward (from a random output gradient R) against central differences of the host loss sum( R * output ) on a
// sample of input entries. Small shapes; no checkpoint.
#include "engine/config.h"
#include "engine/train/qwen4_exp.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
	{
		// The forward at a layer's shape on one of four ranks (4 key heads, 12 value heads, chunks of 64) over 512 rows
		const int Rb = 512, khb = 4, vhb = 12;
		auto unitb = [&]( Vec x, int heads ) {
			for ( int r = 0; r < Rb * heads; ++r ) {
				double n = 0.0;
				for ( int i = 0; i < D; ++i ) n += x[static_cast<size_t>( r ) * D + i] * x[static_cast<size_t>( r ) * D + i];
				for ( int i = 0; i < D; ++i ) x[static_cast<size_t>( r ) * D + i] /= std::sqrt( n );
			}
			return x;
		};
		const Vec qb = unitb( randv( static_cast<size_t>( Rb ) * khb * D ), khb ), kb = unitb( randv( qb.size() ), khb ), vb = randv( static_cast<size_t>( Rb ) * vhb * D );
		Vec gb = randv( static_cast<size_t>( Rb ) * vhb, 0.05 ), bb( gb.size() );
		for ( auto& x : gb ) x = -std::fabs( x );
		for ( auto& x : bb ) x = std::uniform_real_distribution<double>( 0.1, 0.9 )( rng );
		float *b_q = up( qb ), *b_k = up( kb ), *b_v = up( vb ), *b_g = up( gb ), *b_b = up( bb ), *b_o = zeros( vb.size() );
		float* b_ck = zeros( q::gdn_ckpt_floats( Rb, vhb, 64 ) );
		q::gdn_forward( b_q, b_k, b_v, b_g, b_b, Rb, khb, vhb, static_cast<float>( scale ), 64, b_ck, b_o, nullptr );
		const double eo = rel( down( b_o, vb.size() ), h_gdn( qb, kb, vb, gb, bb, Rb, khb, vhb, scale ) );
		std::snprintf( line, sizeof( line ), "%d rows of %d key and %d value heads, chunks of 64: forward %.2e", Rb, khb, vhb, eo );
		check( eo < 1e-5, line );
		for ( float* p : { b_q, b_k, b_v, b_g, b_b, b_o, b_ck } ) cudaFree( p );
	}
	{
		// Sequences of 37 rows (the forward's tiles of 16; chunks of 24, the backward's sub-chunks of 8: partial ones of each) in one call:
		// each from a zero state, against the host recurrence per sequence; the backward against the single-sequence call on each
		const int L = 37, ns = 3, R3 = ns * L, ch = 24;
		auto unit3 = [&]( Vec x ) {
			for ( int r = 0; r < R3 * kh; ++r ) {
				double n = 0.0;
				for ( int i = 0; i < D; ++i ) n += x[static_cast<size_t>( r ) * D + i] * x[static_cast<size_t>( r ) * D + i];
				for ( int i = 0; i < D; ++i ) x[static_cast<size_t>( r ) * D + i] /= std::sqrt( n );
			}
			return x;
		};
		const Vec q3 = unit3( randv( static_cast<size_t>( R3 ) * kh * D ) ), k3 = unit3( randv( q3.size() ) ), v3 = randv( static_cast<size_t>( R3 ) * vh * D );
		Vec g3 = randv( static_cast<size_t>( R3 ) * vh, 0.3 ), b3( g3.size() );
		for ( auto& x : g3 ) x = -std::fabs( x );
		for ( auto& x : b3 ) x = std::uniform_real_distribution<double>( 0.1, 0.9 )( rng );
		const Vec R3v = randv( v3.size() );
		float *e_q = up( q3 ), *e_k = up( k3 ), *e_v = up( v3 ), *e_g = up( g3 ), *e_b = up( b3 ), *e_o = zeros( v3.size() ), *e_R = up( R3v );
		float *e_ck = zeros( q::gdn_ckpt_floats( R3, vh, ch, ns ) ), *e_scr = zeros( q::gdn_scratch_floats( vh, ch, ns ) );
		float *e_dq = zeros( q3.size() ), *e_dk = zeros( k3.size() ), *e_dv = zeros( v3.size() ), *e_dg = zeros( g3.size() ), *e_db = zeros( b3.size() );
		q::gdn_forward( e_q, e_k, e_v, e_g, e_b, R3, kh, vh, static_cast<float>( scale ), ch, e_ck, e_o, nullptr, ns );
		q::gdn_backward( e_q, e_k, e_v, e_g, e_b, R3, kh, vh, static_cast<float>( scale ), ch, e_ck, e_R, e_dq, e_dk, e_dv, e_dg, e_db, e_scr, nullptr, ns );
		Vec ref;
		for ( int s = 0; s < ns; ++s ) {
			auto part = [&]( const Vec& x, size_t per ) { return Vec( x.begin() + s * L * per, x.begin() + ( s + 1 ) * L * per ); };
			const Vec o_s = h_gdn( part( q3, kh * D ), part( k3, kh * D ), part( v3, vh * D ), part( g3, vh ), part( b3, vh ), L, kh, vh, scale );
			ref.insert( ref.end(), o_s.begin(), o_s.end() );
		}
		const double e3 = rel( down( e_o, v3.size() ), ref );
		// the backward: each sequence alone (one-sequence calls on its rows) gives the same gradients
		std::vector<Vec> g1( 5 );
		for ( int s = 0; s < ns; ++s ) {
			const size_t oq = static_cast<size_t>( s ) * L * kh * D, ov = static_cast<size_t>( s ) * L * vh * D, og = static_cast<size_t>( s ) * L * vh;
			float *f_ck = zeros( q::gdn_ckpt_floats( L, vh, ch ) ), *f_scr = zeros( q::gdn_scratch_floats( vh, ch ) ), *f_o = zeros( static_cast<size_t>( L ) * vh * D );
			float *f_dq = zeros( static_cast<size_t>( L ) * kh * D ), *f_dk = zeros( static_cast<size_t>( L ) * kh * D ), *f_dv = zeros( static_cast<size_t>( L ) * vh * D );
			float *f_dg = zeros( static_cast<size_t>( L ) * vh ), *f_db = zeros( static_cast<size_t>( L ) * vh );
			q::gdn_forward( e_q + oq, e_k + oq, e_v + ov, e_g + og, e_b + og, L, kh, vh, static_cast<float>( scale ), ch, f_ck, f_o, nullptr );
			q::gdn_backward( e_q + oq, e_k + oq, e_v + ov, e_g + og, e_b + og, L, kh, vh, static_cast<float>( scale ), ch, f_ck, e_R + ov, f_dq, f_dk, f_dv, f_dg, f_db,
				f_scr, nullptr );
			const size_t n5[5] = { static_cast<size_t>( L ) * kh * D, static_cast<size_t>( L ) * kh * D, static_cast<size_t>( L ) * vh * D, static_cast<size_t>( L ) * vh,
				static_cast<size_t>( L ) * vh };
			float* f5[5] = { f_dq, f_dk, f_dv, f_dg, f_db };
			for ( int j = 0; j < 5; ++j ) {
				const Vec x = down( f5[j], n5[j] );
				g1[j].insert( g1[j].end(), x.begin(), x.end() );
			}
			for ( float* p : { f_ck, f_scr, f_o, f_dq, f_dk, f_dv, f_dg, f_db } ) cudaFree( p );
		}
		double eb3 = 0.0;
		float* e5[5] = { e_dq, e_dk, e_dv, e_dg, e_db };
		for ( int j = 0; j < 5; ++j ) eb3 = std::max( eb3, rel( down( e5[j], g1[j].size() ), g1[j] ) );
		// and the sums over columns (dq, dk, dgate) against central differences of the host recurrence per sequence
		auto host3 = [&]( const Vec& qq, const Vec& kk, const Vec& gg ) {
			double s = 0.0;
			for ( int sq = 0; sq < ns; ++sq ) {
				auto part = [&]( const Vec& x, size_t per ) { return Vec( x.begin() + sq * L * per, x.begin() + ( sq + 1 ) * L * per ); };
				const Vec o_s = h_gdn( part( qq, kh * D ), part( kk, kh * D ), part( v3, vh * D ), part( gg, vh ), part( b3, vh ), L, kh, vh, scale );
				const Vec r_s = part( R3v, vh * D );
				s += dot( r_s, o_s );
			}
			return s;
		};
		const double fq = fd_rel( [&]( const Vec& x ) { return host3( x, k3, g3 ); }, q3, down( e_dq, q3.size() ) );
		const double fk = fd_rel( [&]( const Vec& x ) { return host3( q3, x, g3 ); }, k3, down( e_dk, k3.size() ) );
		const double fg = fd_rel( [&]( const Vec& x ) { return host3( q3, k3, x ); }, g3, down( e_dg, g3.size() ) );
		std::snprintf( line, sizeof( line ), "%d sequences of %d rows in one call, chunks of %d: forward %.2e, gradients against each alone %.2e, dq %.2e, dk %.2e, dgate %.2e",
			ns, L, ch, e3, eb3, fq, fk, fg );
		check( e3 < 1e-5 && eb3 < 1e-5 && fq < 1e-3 && fk < 1e-3 && fg < 1e-3, line );
		for ( float* p : { e_q, e_k, e_v, e_g, e_b, e_o, e_R, e_ck, e_scr, e_dq, e_dk, e_dv, e_dg, e_db } ) cudaFree( p );
	}
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
		// QSA past its budget, at a budget of 4 pools (16 tokens) over 70 rows: the indexer's queries and pool keys against the host (the
		// same FP16 roundings), the selection against the host's scores of the device's queries and keys (every selected pool scoring at
		// least every other visible one, one FP16 step of slack), and the attention over it in query blocks of 32 against the host's
		// masked attention and central differences
		const int L = 70, budget = 4, np = L / q::kQsaPool, words = ( np + 31 ) / 32, rot_i = 64, D = q::kQsaDim, QH = q::kQsaHeads;
		const double eps = 1e-6, theta = 10000.0;
		auto r16 = []( double x ) { return static_cast<double>( __half2float( __float2half_rn( static_cast<float>( x ) ) ) ); };
		const Vec qk = randv( static_cast<size_t>( L ) * q::kQsaProj ), qw = randv( D, 0.2 ), kw = randv( D, 0.2 );
		auto up_half = [&]( const Vec& v ) {
			std::vector<__half> h( v.size() );
			for ( size_t i = 0; i < v.size(); ++i ) h[i] = __float2half_rn( static_cast<float>( v[i] ) );
			void* p = nullptr;
			CUDA_CHECK( cudaMalloc( &p, h.size() * 2 ) );
			CUDA_CHECK( cudaMemcpy( p, h.data(), h.size() * 2, cudaMemcpyHostToDevice ) );
			return p;
		};
		void *d_qw = up_half( qw ), *d_kw = up_half( kw );
		float *d_qk = up( qk ), *d_qi = zeros( static_cast<size_t>( L ) * QH * D ), *d_rk = zeros( static_cast<size_t>( L ) * D ), *d_pool = zeros( static_cast<size_t>( np ) * D );
		q::qsa_queries( d_qk, L, L, d_qw, rot_i, static_cast<float>( theta ), static_cast<float>( eps ), d_qi, d_rk, nullptr );
		q::qsa_pools( d_rk, 1, L, d_kw, rot_i, static_cast<float>( theta ), static_cast<float>( eps ), d_pool, nullptr );
		// host: rmsnorm * (1 + w) in FP16, rope at a position, FP16
		auto norm_rope = [&]( const double* x, const Vec& w, int pos ) {
			Vec y( D );
			double ms = 0.0;
			for ( int c = 0; c < D; ++c ) ms += x[c] * x[c];
			ms /= D;
			for ( int c = 0; c < D; ++c ) y[c] = r16( x[c] / std::sqrt( ms + eps ) * ( 1.0 + r16( w[c] ) ) );
			Vec o = y;
			for ( int c = 0; c < rot_i; ++c ) {
				const int i = c % ( rot_i / 2 );
				const double a = pos * std::pow( theta, -2.0 * i / rot_i ), cs = std::cos( a ), sn = std::sin( a );
				o[c] = c < rot_i / 2 ? y[c] * cs - y[c + rot_i / 2] * sn : y[c] * cs + y[c - rot_i / 2] * sn;
			}
			for ( double& v : o ) v = r16( v );
			return o;
		};
		Vec hq_( static_cast<size_t>( L ) * QH * D ), hp( static_cast<size_t>( np ) * D );
		for ( int r = 0; r < L; ++r ) for ( int h = 0; h < QH; ++h ) {
			double x[128];
			for ( int c = 0; c < D; ++c ) x[c] = r16( qk[static_cast<size_t>( r ) * q::kQsaProj + h * D + c] );
			const Vec o = norm_rope( x, qw, r );
			std::copy( o.begin(), o.end(), hq_.begin() + ( static_cast<size_t>( r ) * QH + h ) * D );
		}
		for ( int p = 0; p < np; ++p ) {
			double x[128];
			for ( int c = 0; c < D; ++c ) {
				double a = 0.0;
				for ( int j = 0; j < q::kQsaPool; ++j ) a += r16( qk[static_cast<size_t>( p * q::kQsaPool + j ) * q::kQsaProj + QH * D + c] );
				x[c] = r16( a / q::kQsaPool );
			}
			const Vec o = norm_rope( x, kw, p * q::kQsaPool );
			std::copy( o.begin(), o.end(), hp.begin() + static_cast<size_t>( p ) * D );
		}
		const Vec dqi = down( d_qi, hq_.size() ), dpool = down( d_pool, hp.size() );
		const double eqi = rel( dqi, hq_ ), epl = rel( dpool, hp );
		// the selection
		uint32_t* d_bits = nullptr;
		CUDA_CHECK( cudaMalloc( &d_bits, static_cast<size_t>( L ) * words * 4 ) );
		float* d_scr = zeros( q::qsa_scratch_floats( L, 32 ) );
		q::qsa_select( d_qi, d_pool, L, budget, 32, d_scr, d_bits, nullptr );
		std::vector<uint32_t> bits( static_cast<size_t>( L ) * words );
		CUDA_CHECK( cudaMemcpy( bits.data(), d_bits, bits.size() * 4, cudaMemcpyDeviceToHost ) );
		int bad = 0, sparse_rows = 0;
		for ( int p = 0; p < L; ++p ) {
			const int bound = ( p + 1 ) / q::kQsaPool;
			int nsel = 0;
			for ( int x = 0; x < np; ++x ) nsel += ( bits[static_cast<size_t>( p ) * words + x / 32] >> ( x % 32 ) ) & 1;
			if ( bound <= budget ) { bad += nsel != 0; continue; }
			++sparse_rows;
			std::vector<double> sc( bound );
			for ( int x = 0; x < bound; ++x ) {
				double t = 0.0;
				for ( int h = 0; h < QH; ++h ) {
					double z = 0.0;
					for ( int c = 0; c < D; ++c ) z += dqi[( static_cast<size_t>( p ) * QH + h ) * D + c] * dpool[static_cast<size_t>( x ) * D + c];
					t += std::max( z, 0.0 );
				}
				sc[x] = r16( t / std::sqrt( static_cast<double>( D ) ) );
			}
			double lo_sel = 1e300, hi_out = -1e300;
			for ( int x = 0; x < bound; ++x ) {
				const bool s = ( bits[static_cast<size_t>( p ) * words + x / 32] >> ( x % 32 ) ) & 1;
				if ( s ) lo_sel = std::min( lo_sel, sc[x] ); else hi_out = std::max( hi_out, sc[x] );
			}
			bad += nsel != budget || lo_sel < hi_out - std::fabs( hi_out ) * 1e-3;
		}
		// the attention over the selection, query blocks of 32
		const int aq = 2, ak = 1, ad = 16;
		const Vec av = randv( static_cast<size_t>( L ) * aq * ad ), kv2 = randv( static_cast<size_t>( L ) * ak * ad ), vv2 = randv( kv2.size() ), R2 = randv( av.size() );
		auto sees = [&]( int p, int j ) {
			if ( j > p ) return false;
			const int vis = p + 1;
			if ( vis / q::kQsaPool <= budget || j >= vis - vis % q::kQsaPool ) return true;
			return static_cast<bool>( ( bits[static_cast<size_t>( p ) * words + ( j / 4 ) / 32] >> ( ( j / 4 ) % 32 ) ) & 1 );
		};
		auto h_mattn = [&]( const Vec& qq, const Vec& kk, const Vec& vv ) {
			Vec o( qq.size(), 0.0 );
			for ( int h = 0; h < aq; ++h ) for ( int i = 0; i < L; ++i ) {
				Vec sc( i + 1, -1e300 );
				double m = -1e300, sum = 0.0;
				for ( int j = 0; j <= i; ++j ) {
					if ( !sees( i, j ) ) continue;
					double z = 0.0;
					for ( int c = 0; c < ad; ++c ) z += qq[( static_cast<size_t>( i ) * aq + h ) * ad + c] * kk[static_cast<size_t>( j ) * ad + c];
					sc[j] = z * 0.25;
					m = std::max( m, sc[j] );
				}
				for ( int j = 0; j <= i; ++j ) if ( sees( i, j ) ) { sc[j] = std::exp( sc[j] - m ); sum += sc[j]; } else sc[j] = 0.0;
				for ( int j = 0; j <= i; ++j ) for ( int c = 0; c < ad; ++c ) o[( static_cast<size_t>( i ) * aq + h ) * ad + c] += sc[j] / sum * vv[static_cast<size_t>( j ) * ad + c];
			}
			return o;
		};
		engine::train::Blas blas( nullptr );
		float *d_q = up( av ), *d_k = up( kv2 ), *d_v = up( vv2 ), *d_o = zeros( av.size() ), *d_R = up( R2 ), *d_s = zeros( 32 * L ), *d_p = zeros( 32 * L );
		float *d_dq = zeros( av.size() ), *d_dk = zeros( kv2.size() ), *d_dv = zeros( vv2.size() );
		q::QsaSelection sel;
		sel.bits = d_bits;
		sel.words = words;
		sel.budget = budget;
		q::attention_forward( blas, d_q, d_k, d_v, L, aq, ak, ad, 0.25f, d_s, d_o, sel, 32 );
		q::attention_backward( blas, d_q, d_k, d_v, L, aq, ak, ad, 0.25f, d_R, d_dq, d_dk, d_dv, d_s, d_p, sel, 32 );
		const double ea = rel( down( d_o, av.size() ), h_mattn( av, kv2, vv2 ) );
		const double gq = fd_rel( [&]( const Vec& x ) { return dot( R2, h_mattn( x, kv2, vv2 ) ); }, av, down( d_dq, av.size() ) );
		const double gk = fd_rel( [&]( const Vec& x ) { return dot( R2, h_mattn( av, x, vv2 ) ); }, kv2, down( d_dk, kv2.size() ) );
		const double gv = fd_rel( [&]( const Vec& x ) { return dot( R2, h_mattn( av, kv2, x ) ); }, vv2, down( d_dv, vv2.size() ) );
		char line[220];
		std::snprintf( line, sizeof( line ), "QSA (budget %d pools, %d rows): queries %.2e, pool keys %.2e; selection: %d of %d rows past the budget off; attention %.2e, dq %.2e, dk %.2e, dv %.2e",
			budget, L, eqi, epl, bad, sparse_rows, ea, gq, gk, gv );
		check( eqi < 2e-3 && epl < 2e-3 && bad == 0 && sparse_rows > 0 && ea < 1e-5 && gq < 1e-3 && gk < 1e-3 && gv < 1e-3, line );
		for ( float* p : { d_qk, d_qi, d_rk, d_pool, d_scr, d_q, d_k, d_v, d_o, d_R, d_s, d_p, d_dq, d_dk, d_dv } ) cudaFree( p );
		cudaFree( d_qw ); cudaFree( d_kw ); cudaFree( d_bits );
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
