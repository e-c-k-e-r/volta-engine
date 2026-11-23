// Tier 2: the GLM-5.3-Flash mHC training operations (engine/train/glm5_mhc.h) against double-precision host references: each forward against
// the host forward, each backward (from random output gradients R) against central differences of the host loss sum( R * outputs ) on a sample
// of input entries. Small shapes; no checkpoint.
#include "engine/config.h"
#include "engine/train/glm5_mhc.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

namespace {

namespace q = engine::train::glm5_next;
using Vec = std::vector<double>;
int fails = 0;
std::mt19937 rng( 1234 );

void check( bool ok, const char* line ) {
	std::printf( "  %-100s %s\n", line, ok ? "OK" : "FAIL" );
	fails += !ok;
}

Vec randv( size_t n, double scale = 1.0, double mean = 0.0 ) {
	std::normal_distribution<double> d( mean, scale );
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

// What an accumulating backward added: the buffer now minus its start (as uploaded, in FP32)
Vec added( const float* d, const Vec& start ) {
	Vec v = down( d, start.size() );
	for ( size_t i = 0; i < v.size(); ++i ) v[i] -= static_cast<double>( static_cast<float>( start[i] ) );
	return v;
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

constexpr int kIters = 20;
constexpr double kHcEps = 1e-6, kRmsEps = 1e-5, kNormEps = 1e-5;

// A site's weights on the host
struct HSite {
	int H = 0;
	Vec fn, base, scale, w;
};

// Host comb of one row from its 16 logits: the rows' softmax + eps, one column normalization, then 19 x ( row, column )
void h_sinkhorn( const double* e, double* v ) {
	for ( int g = 0; g < 4; ++g ) {
		double m = e[g * 4];
		for ( int k = 1; k < 4; ++k ) m = std::max( m, e[g * 4 + k] );
		double s = 0.0;
		for ( int k = 0; k < 4; ++k ) s += std::exp( e[g * 4 + k] - m );
		for ( int k = 0; k < 4; ++k ) v[g * 4 + k] = std::exp( e[g * 4 + k] - m ) / s + kHcEps;
	}
	auto norm = [&]( bool col ) {
		for ( int a = 0; a < 4; ++a ) {
			double s = 0.0;
			for ( int b = 0; b < 4; ++b ) s += v[col ? b * 4 + a : a * 4 + b];
			for ( int b = 0; b < 4; ++b ) v[col ? b * 4 + a : a * 4 + b] /= s + kHcEps;
		}
	};
	norm( true );
	for ( int it = 1; it < kIters; ++it ) {
		norm( false );
		norm( true );
	}
}

// Host pre step: a [rows, H], post [rows, 4], comb [rows, 16]
void h_pre( const HSite& s, const Vec& X, int rows, Vec& a, Vec& post, Vec& comb ) {
	const int H = s.H, n = 4 * H;
	a.assign( static_cast<size_t>( rows ) * H, 0.0 );
	post.assign( static_cast<size_t>( rows ) * 4, 0.0 );
	comb.assign( static_cast<size_t>( rows ) * 16, 0.0 );
	Vec u( H );
	for ( int r = 0; r < rows; ++r ) {
		const double* x = X.data() + static_cast<size_t>( r ) * n;
		double ss = 0.0;
		for ( int i = 0; i < n; ++i ) ss += x[i] * x[i];
		const double rmr = 1.0 / std::sqrt( ss / n + kRmsEps );
		double l[24];
		for ( int j = 0; j < 24; ++j ) {
			double m = 0.0;
			for ( int i = 0; i < n; ++i ) m += s.fn[static_cast<size_t>( j ) * n + i] * x[i];
			l[j] = m * rmr;
		}
		double pre[4];
		for ( int h = 0; h < 4; ++h ) {
			pre[h] = sig( l[h] * s.scale[0] + s.base[h] ) + kHcEps;
			post[static_cast<size_t>( r ) * 4 + h] = 2.0 * sig( l[4 + h] * s.scale[1] + s.base[4 + h] );
		}
		double e[16];
		for ( int i = 0; i < 16; ++i ) e[i] = l[8 + i] * s.scale[2] + s.base[8 + i];
		h_sinkhorn( e, comb.data() + static_cast<size_t>( r ) * 16 );
		double q2 = 0.0;
		for ( int d = 0; d < H; ++d ) {
			u[d] = 0.0;
			for ( int h = 0; h < 4; ++h ) u[d] += pre[h] * x[h * H + d];
			q2 += u[d] * u[d];
		}
		const double inv = 1.0 / std::sqrt( q2 / H + kNormEps );
		for ( int d = 0; d < H; ++d ) a[static_cast<size_t>( r ) * H + d] = u[d] * inv * s.w[d];
	}
}

void test_pre( int rows, int H ) {
	const int n = 4 * H;
	HSite s;
	s.H = H;
	s.fn = randv( static_cast<size_t>( 24 ) * n, 2.0 / std::sqrt( static_cast<double>( n ) ) );   // l ~ N( 0, 2 )
	s.base = randv( 24, 0.5 );
	s.scale = { 1.0 + 0.3 * randv( 1 )[0], 0.7 + 0.3 * randv( 1 )[0], 1.2 + 0.3 * randv( 1 )[0] };
	s.w = randv( H, 0.3, 1.0 );
	const Vec X = randv( static_cast<size_t>( rows ) * n );
	const Vec Ra = randv( static_cast<size_t>( rows ) * H ), Rp = randv( static_cast<size_t>( rows ) * 4 ), Rc = randv( static_cast<size_t>( rows ) * 16 );
	// the accumulating buffers start from random values
	const Vec D0 = randv( X.size() ), G0fn = randv( s.fn.size() ), G0b = randv( 24 ), G0s = randv( 3 ), G0w = randv( H );
	engine::train::Blas blas( nullptr );
	float *d_fn = up( s.fn ), *d_base = up( s.base ), *d_scale = up( s.scale ), *d_w = up( s.w ), *d_X = up( X );
	float *d_gfn = up( G0fn ), *d_gb = up( G0b ), *d_gs = up( G0s ), *d_gw = up( G0w );
	q::MhcSite ds;
	ds.hidden = H; ds.fn = d_fn; ds.base = d_base; ds.scale = d_scale; ds.norm_w = d_w;
	ds.gfn = d_gfn; ds.gbase = d_gb; ds.gscale = d_gs; ds.gnorm = d_gw;
	float* d_cache = zeros( static_cast<size_t>( rows ) * q::kMhcCacheFloats );
	const q::MhcCache c = q::mhc_cache( d_cache, rows );
	float *d_a = zeros( static_cast<size_t>( rows ) * H ), *d_post = zeros( static_cast<size_t>( rows ) * 4 ), *d_comb = zeros( static_cast<size_t>( rows ) * 16 );
	float *d_Ra = up( Ra ), *d_Rp = up( Rp ), *d_Rc = up( Rc ), *d_dX = up( D0 ), *d_scr = zeros( q::mhc_pre_scratch_floats( ds, rows ) );
	q::mhc_pre_forward( blas, ds, d_X, rows, c, d_a, d_post, d_comb );
	q::mhc_pre_backward( blas, ds, d_X, rows, c, d_Ra, d_Rp, d_Rc, d_dX, d_scr );
	Vec ah, ph, ch;
	h_pre( s, X, rows, ah, ph, ch );
	const Vec comb = down( d_comb, ch.size() );
	const double ea = rel( down( d_a, ah.size() ), ah ), ep = rel( down( d_post, ph.size() ), ph ), ec = rel( comb, ch );
	char line[200];
	std::snprintf( line, sizeof( line ), "rows %d, hidden %d: forward a %.2e, post %.2e, comb %.2e", rows, H, ea, ep, ec );
	check( ea < 1e-5 && ep < 1e-5 && ec < 1e-5, line );
	// the comb is doubly stochastic: a column normalization last, so its column sums are 1 to eps; the row sums of rows with peaked logits
	// (spread ~2.5 here) are still a few 1e-2 off after 20 rounds (the reference's too), so the rows are checked at their median
	Vec rdev;
	double ecol = 0.0;
	for ( int r = 0; r < rows; ++r ) {
		for ( int a = 0; a < 4; ++a ) {
			double rs = 0.0, cs = 0.0;
			for ( int b = 0; b < 4; ++b ) {
				rs += comb[static_cast<size_t>( r ) * 16 + a * 4 + b];
				cs += comb[static_cast<size_t>( r ) * 16 + b * 4 + a];
			}
			rdev.push_back( std::fabs( rs - 1.0 ) );
			ecol = std::max( ecol, std::fabs( cs - 1.0 ) );
		}
	}
	std::sort( rdev.begin(), rdev.end() );
	const double rmed = rdev[rdev.size() / 2], rmax = rdev.back();
	std::snprintf( line, sizeof( line ), "  comb doubly stochastic: |column sum - 1| %.2e, |row sum - 1| median %.2e (max %.2e)", ecol, rmed, rmax );
	check( ecol < 1e-5 && rmed < 1e-2, line );
	auto loss = [&]( const HSite& t, const Vec& XX ) {
		Vec a, p, cb;
		h_pre( t, XX, rows, a, p, cb );
		return dot( Ra, a ) + dot( Rp, p ) + dot( Rc, cb );
	};
	const double ex = fd_rel( [&]( const Vec& v ) { return loss( s, v ); }, X, added( d_dX, D0 ) );
	const double efn = fd_rel( [&]( const Vec& v ) { HSite t = s; t.fn = v; return loss( t, X ); }, s.fn, added( d_gfn, G0fn ) );
	const double eb = fd_rel( [&]( const Vec& v ) { HSite t = s; t.base = v; return loss( t, X ); }, s.base, added( d_gb, G0b ) );
	const double es = fd_rel( [&]( const Vec& v ) { HSite t = s; t.scale = v; return loss( t, X ); }, s.scale, added( d_gs, G0s ) );
	const double ew = fd_rel( [&]( const Vec& v ) { HSite t = s; t.w = v; return loss( t, X ); }, s.w, added( d_gw, G0w ) );
	std::snprintf( line, sizeof( line ), "  backward dX %.2e, gfn %.2e, gbase %.2e, gscale %.2e, gnorm %.2e", ex, efn, eb, es, ew );
	check( ex < 1e-3 && efn < 1e-3 && eb < 1e-3 && es < 1e-3 && ew < 1e-3, line );
	// a alone (dpost, dcomb null) and no site gradients wanted
	q::MhcSite dn = ds;
	dn.gfn = dn.gbase = dn.gscale = dn.gnorm = nullptr;
	CUDA_CHECK( cudaMemset( d_dX, 0, X.size() * sizeof( float ) ) );
	q::mhc_pre_backward( blas, dn, d_X, rows, c, d_Ra, nullptr, nullptr, d_dX, d_scr );
	const double ex2 = fd_rel( [&]( const Vec& v ) { Vec a, p, cb; h_pre( s, v, rows, a, p, cb ); return dot( Ra, a ); }, X, down( d_dX, X.size() ) );
	// comb alone (the Sinkhorn path only)
	const Vec Z = Vec( Ra.size(), 0.0 ), Zp = Vec( Rp.size(), 0.0 );
	float *d_Z = up( Z ), *d_Zp = up( Zp );
	CUDA_CHECK( cudaMemset( d_dX, 0, X.size() * sizeof( float ) ) );
	q::mhc_pre_backward( blas, dn, d_X, rows, c, d_Z, d_Zp, d_Rc, d_dX, d_scr );
	const double ex3 = fd_rel( [&]( const Vec& v ) { Vec a, p, cb; h_pre( s, v, rows, a, p, cb ); return dot( Rc, cb ); }, X, down( d_dX, X.size() ) );
	std::snprintf( line, sizeof( line ), "  backward from a alone: dX %.2e; from comb alone: dX %.2e", ex2, ex3 );
	check( ex2 < 1e-3 && ex3 < 1e-3, line );
	for ( float* p : { d_fn, d_base, d_scale, d_w, d_X, d_gfn, d_gb, d_gs, d_gw, d_cache, d_a, d_post, d_comb, d_Ra, d_Rp, d_Rc, d_dX, d_scr, d_Z, d_Zp } )
		cudaFree( p );
}

// Host post step
Vec h_post( const Vec& Xo, const Vec& post, const Vec& comb, const Vec& y, int rows, int H ) {
	Vec o( Xo.size() );
	for ( int r = 0; r < rows; ++r ) for ( int h = 0; h < 4; ++h ) for ( int d = 0; d < H; ++d ) {
		double v = post[static_cast<size_t>( r ) * 4 + h] * y[static_cast<size_t>( r ) * H + d];
		for ( int g = 0; g < 4; ++g ) v += comb[static_cast<size_t>( r ) * 16 + g * 4 + h] * Xo[( static_cast<size_t>( r ) * 4 + g ) * H + d];
		o[( static_cast<size_t>( r ) * 4 + h ) * H + d] = v;
	}
	return o;
}

void test_post( int rows, int H ) {
	const size_t ns = static_cast<size_t>( rows ) * 4 * H;
	const Vec Xo = randv( ns ), y = randv( static_cast<size_t>( rows ) * H ), R = randv( ns );
	Vec post = randv( static_cast<size_t>( rows ) * 4 ), comb = randv( static_cast<size_t>( rows ) * 16, 0.3, 0.25 );
	for ( auto& p : post ) p = 2.0 * sig( p );
	const Vec D0 = randv( ns );
	float *d_Xo = up( Xo ), *d_post = up( post ), *d_comb = up( comb ), *d_y = up( y ), *d_R = up( R ), *d_Xn = zeros( ns );
	float *d_dXo = up( D0 ), *d_dy = up( randv( y.size() ) ), *d_dp = up( randv( post.size() ) ), *d_dc = up( randv( comb.size() ) );   // overwritten
	q::mhc_post_forward( d_Xo, d_post, d_comb, d_y, rows, H, d_Xn, nullptr );
	q::mhc_post_backward( d_Xo, d_post, d_comb, d_y, d_R, rows, H, d_dXo, d_dy, d_dp, d_dc, nullptr );
	const Vec ref = h_post( Xo, post, comb, y, rows, H );
	const double ef = rel( down( d_Xn, ns ), ref );
	// in place
	float* d_Xi = up( Xo );
	q::mhc_post_forward( d_Xi, d_post, d_comb, d_y, rows, H, d_Xi, nullptr );
	const double ei = rel( down( d_Xi, ns ), ref );
	const double ex = fd_rel( [&]( const Vec& v ) { return dot( R, h_post( v, post, comb, y, rows, H ) ); }, Xo, added( d_dXo, D0 ) );
	const double ey = fd_rel( [&]( const Vec& v ) { return dot( R, h_post( Xo, post, comb, v, rows, H ) ); }, y, down( d_dy, y.size() ) );
	const double ep = fd_rel( [&]( const Vec& v ) { return dot( R, h_post( Xo, v, comb, y, rows, H ) ); }, post, down( d_dp, post.size() ) );
	const double ec = fd_rel( [&]( const Vec& v ) { return dot( R, h_post( Xo, post, v, y, rows, H ) ); }, comb, down( d_dc, comb.size() ) );
	char line[200];
	std::snprintf( line, sizeof( line ), "rows %d, hidden %d: forward %.2e (in place %.2e), dX_old %.2e, dy %.2e, dpost %.2e, dcomb %.2e", rows, H, ef, ei,
		ex, ey, ep, ec );
	check( ef < 1e-5 && ei < 1e-5 && ex < 1e-3 && ey < 1e-3 && ep < 1e-3 && ec < 1e-3, line );
	for ( float* p : { d_Xo, d_post, d_comb, d_y, d_R, d_Xn, d_dXo, d_dy, d_dp, d_dc, d_Xi } ) cudaFree( p );
}

void test_mean( int rows, int H ) {
	const size_t ns = static_cast<size_t>( rows ) * 4 * H;
	const Vec X = randv( ns ), R = randv( static_cast<size_t>( rows ) * H ), D0 = randv( ns );
	auto fwd = [&]( const Vec& xx ) {
		Vec o( static_cast<size_t>( rows ) * H, 0.0 );
		for ( int r = 0; r < rows; ++r ) for ( int h = 0; h < 4; ++h ) for ( int d = 0; d < H; ++d )
			o[static_cast<size_t>( r ) * H + d] += 0.25 * xx[( static_cast<size_t>( r ) * 4 + h ) * H + d];
		return o;
	};
	float *d_X = up( X ), *d_h = zeros( R.size() ), *d_R = up( R ), *d_dX = up( D0 );
	q::hc_mean_forward( d_X, rows, H, d_h, nullptr );
	q::hc_mean_backward( d_R, rows, H, d_dX, nullptr );
	const double ef = rel( down( d_h, R.size() ), fwd( X ) );
	const double eg = fd_rel( [&]( const Vec& v ) { return dot( R, fwd( v ) ); }, X, added( d_dX, D0 ) );
	char line[200];
	std::snprintf( line, sizeof( line ), "rows %d, hidden %d: forward %.2e, dX %.2e", rows, H, ef, eg );
	check( ef < 1e-5 && eg < 1e-3, line );
	for ( float* p : { d_X, d_h, d_R, d_dX } ) cudaFree( p );
}

} // namespace

int main() {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	CUDA_CHECK( cudaSetDevice( 0 ) );
	std::printf( "(a) mHC pre step (mix, gates, Sinkhorn comb, collapse + norm): forward, backward into X, fn, base, scale, norm_w\n" );
	test_pre( 5, 32 );
	test_pre( 3, 520 );
	test_pre( 160, 24 );
	std::printf( "(b) mHC post step X_new = post y + comb^T X_old\n" );
	test_post( 6, 40 );
	test_post( 3, 520 );
	std::printf( "(c) head stream mean\n" );
	test_mean( 5, 48 );
	std::printf( "[%c] test_train_glm5_mhc: %d failure(s)\n", fails ? '-' : '+', fails );
	return fails ? 1 : 0;
}
