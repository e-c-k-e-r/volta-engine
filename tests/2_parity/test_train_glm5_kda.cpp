// Tier 2: GLM-5.3-Flash training operations (engine/train/glm5_kda.h) against double-precision host references and central differences: each
// forward against the host forward, each backward (from a random output gradient R) against central differences of the host loss
// sum( R * output ) on a sample of input entries. The += outputs start at kBase and the overwritten ones at garbage, so that both semantics are
// seen. Small shapes; no checkpoint.
#include "engine/config.h"
#include "engine/train/glm5_kda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

namespace {

namespace g5 = engine::train::glm5_next;
using Vec = std::vector<double>;
int fails = 0;
std::mt19937 rng( 1234 );
constexpr double kBase = 1.0 / 64;   // the += outputs' starting value (small: its FP32 sum with a gradient keeps the gradient's digits)
constexpr int D = g5::kKdaDim;

void check( bool ok, const char* line ) {
	std::printf( "  %-110s %s\n", line, ok ? "OK" : "FAIL" );
	fails += !ok;
}

Vec randv( size_t n, double scale = 1.0 ) {
	std::normal_distribution<double> d( 0.0, scale );
	Vec v( n );
	for ( auto& x : v ) x = d( rng );
	return v;
}

Vec uniform( size_t n, double lo, double hi ) {
	std::uniform_real_distribution<double> d( lo, hi );
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

float* filled( size_t n, double value ) { return up( Vec( n, value ) ); }

Vec down( const float* d, size_t n ) {
	std::vector<float> f( n );
	CUDA_CHECK( cudaDeviceSynchronize() );
	CUDA_CHECK( cudaMemcpy( f.data(), d, n * sizeof( float ), cudaMemcpyDeviceToHost ) );
	return Vec( f.begin(), f.end() );
}

// A += output, its starting kBase taken off
Vec down_added( const float* d, size_t n ) {
	Vec v = down( d, n );
	for ( auto& x : v ) x -= kBase;
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

// Central differences of loss( x ) at up to 48 sampled entries, to compare several gradients against
struct Fd {
	std::vector<size_t> idx;
	Vec fd;
};

Fd fd_sample( const std::function<double( const Vec& )>& loss, Vec x, double h = 1e-5 ) {
	Fd f;
	f.idx.resize( x.size() );
	for ( size_t i = 0; i < f.idx.size(); ++i ) f.idx[i] = i;
	std::shuffle( f.idx.begin(), f.idx.end(), rng );
	if ( f.idx.size() > 48 ) f.idx.resize( 48 );
	for ( size_t i : f.idx ) {
		const double x0 = x[i];
		x[i] = x0 + h;
		const double lp = loss( x );
		x[i] = x0 - h;
		const double lm = loss( x );
		x[i] = x0;
		f.fd.push_back( ( lp - lm ) / ( 2 * h ) );
	}
	return f;
}

double fd_err( const Fd& f, const Vec& g ) {
	double m = 0.0, d = 0.0;
	for ( size_t j = 0; j < f.idx.size(); ++j ) {
		m = std::max( m, std::fabs( f.fd[j] ) );
		d = std::max( d, std::fabs( f.fd[j] - g[f.idx[j]] ) );
	}
	return d / ( m + 1e-12 );
}

double fd_rel( const std::function<double( const Vec& )>& loss, const Vec& x, const Vec& g ) { return fd_err( fd_sample( loss, x ), g ); }

double sig( double x ) { return 1.0 / ( 1.0 + std::exp( -x ) ); }

// Host l2 norm per segment of n
Vec h_l2( const Vec& x, int segs, int n, double eps, double scale ) {
	Vec y( x.size() );
	for ( int s = 0; s < segs; ++s ) {
		double q = 0.0;
		for ( int i = 0; i < n; ++i ) q += x[static_cast<size_t>( s ) * n + i] * x[static_cast<size_t>( s ) * n + i];
		const double m = scale / std::sqrt( q + eps );
		for ( int i = 0; i < n; ++i ) y[static_cast<size_t>( s ) * n + i] = x[static_cast<size_t>( s ) * n + i] * m;
	}
	return y;
}

void test_l2() {
	std::printf( "(a) l2 norm per head (eps on the sum of squares)\n" );
	const int segs = 24, n = D;
	const double eps = 1e-6;
	struct Case { const char* name; double scale; };
	const Case cases[] = { { "q, scale 1 / sqrt( 128 )", 1.0 / std::sqrt( 128.0 ) }, { "k, scale 1", 1.0 } };
	for ( const Case& c : cases ) {
		Vec x = randv( static_cast<size_t>( segs ) * n );
		for ( int i = 0; i < n; ++i ) x[i] *= 0.01;   // one small head
		const Vec R = randv( x.size() );
		float *d_x = up( x ), *d_inv = filled( segs, 7.0 ), *d_y = filled( x.size(), 7.0 ), *d_R = up( R ), *d_dx = filled( x.size(), kBase );
		g5::l2norm_forward( d_x, segs, n, static_cast<float>( eps ), static_cast<float>( c.scale ), d_inv, d_y, nullptr );
		g5::l2norm_backward( d_x, segs, n, d_inv, static_cast<float>( c.scale ), d_R, d_dx, nullptr );
		const double ef = rel( down( d_y, x.size() ), h_l2( x, segs, n, eps, c.scale ) );
		const double eg = fd_rel( [&]( const Vec& xx ) { return dot( R, h_l2( xx, segs, n, eps, c.scale ) ); }, x, down_added( d_dx, x.size() ) );
		char line[200];
		std::snprintf( line, sizeof( line ), "%-26s %d heads of %d: forward %.2e, dx %.2e", c.name, segs, n, ef, eg );
		check( ef < 1e-5 && eg < 1e-3, line );
		for ( float* p : { d_x, d_inv, d_y, d_R, d_dx } ) cudaFree( p );
	}
}

// Host gates: beta [rows, heads], g [rows, heads * Dg]
void h_gates( const Vec& b, const Vec& fpre, const Vec& dt, const Vec& Alog, int rows, int heads, int Dg, double lower, Vec& beta, Vec& g ) {
	const size_t cols = static_cast<size_t>( heads ) * Dg;
	beta.resize( b.size() );
	g.resize( fpre.size() );
	for ( size_t i = 0; i < b.size(); ++i ) beta[i] = sig( b[i] );
	for ( int r = 0; r < rows; ++r ) for ( size_t j = 0; j < cols; ++j ) {
		const size_t i = r * cols + j;
		g[i] = lower * sig( std::exp( Alog[j / Dg] ) * ( fpre[i] + dt[j] ) );
	}
}

void test_gates() {
	std::printf( "(b) KDA gates: beta = sigmoid( b ), g = -5 sigmoid( e^A_log ( fpre + dt_bias ) )\n" );
	const double lower = -5.0;
	struct Case { int rows, heads, Dg; };
	const Case cases[] = { { 70, 3, D }, { 9, 3, 20 } };   // rows over two of the backward's row spans; a D off the warp width
	for ( const Case& c : cases ) {
		const size_t cols = static_cast<size_t>( c.heads ) * c.Dg, nb = static_cast<size_t>( c.rows ) * c.heads, nf = c.rows * cols;
		const Vec b = randv( nb, 2.0 ), fpre = randv( nf, 1.5 ), dt = randv( cols, 0.5 ), Alog = randv( c.heads, 0.5 ), Rb = randv( nb ), Rg = randv( nf );
		auto loss = [&]( const Vec& bb, const Vec& ff, const Vec& dd, const Vec& aa ) {
			Vec be, gg;
			h_gates( bb, ff, dd, aa, c.rows, c.heads, c.Dg, lower, be, gg );
			return dot( Rb, be ) + dot( Rg, gg );
		};
		float *d_b = up( b ), *d_f = up( fpre ), *d_dt = up( dt ), *d_al = up( Alog ), *d_beta = filled( nb, 7.0 ), *d_g = filled( nf, 7.0 ), *d_Rb = up( Rb ),
			*d_Rg = up( Rg );
		float *d_db = filled( nb, 7.0 ), *d_df = filled( nf, 7.0 ), *d_ddt = filled( cols, kBase ), *d_dal = filled( c.heads, kBase );
		g5::kda_gates_forward( d_b, d_f, d_dt, d_al, c.rows, c.heads, c.Dg, static_cast<float>( lower ), d_beta, d_g, nullptr );
		g5::kda_gates_backward( d_b, d_f, d_dt, d_al, c.rows, c.heads, c.Dg, static_cast<float>( lower ), d_Rb, d_Rg, d_db, d_df, d_ddt, d_dal, nullptr );
		Vec beh, gh;
		h_gates( b, fpre, dt, Alog, c.rows, c.heads, c.Dg, lower, beh, gh );
		const double eb = rel( down( d_beta, nb ), beh ), eg = rel( down( d_g, nf ), gh );
		const Vec db = down( d_db, nb ), df = down( d_df, nf );
		const double gb = fd_rel( [&]( const Vec& x ) { return loss( x, fpre, dt, Alog ); }, b, db );
		const double gf = fd_rel( [&]( const Vec& x ) { return loss( b, x, dt, Alog ); }, fpre, df );
		const double gd = fd_rel( [&]( const Vec& x ) { return loss( b, fpre, x, Alog ); }, dt, down_added( d_ddt, cols ) );
		const double ga = fd_rel( [&]( const Vec& x ) { return loss( b, fpre, dt, x ); }, Alog, down_added( d_dal, c.heads ) );
		// the parameter gradients not wanted: the same db and dfpre
		float *d_db2 = filled( nb, 7.0 ), *d_df2 = filled( nf, 7.0 );
		g5::kda_gates_backward( d_b, d_f, d_dt, d_al, c.rows, c.heads, c.Dg, static_cast<float>( lower ), d_Rb, d_Rg, d_db2, d_df2, nullptr, nullptr, nullptr );
		const bool same = down( d_db2, nb ) == db && down( d_df2, nf ) == df;
		char line[200];
		std::snprintf( line, sizeof( line ), "%d rows, %d heads of %d: forward beta %.2e, g %.2e, db %.2e, dfpre %.2e, d_dt_bias %.2e, dA_log %.2e%s", c.rows,
			c.heads, c.Dg, eb, eg, gb, gf, gd, ga, same ? "" : ", null outputs differ" );
		check( eb < 1e-5 && eg < 1e-5 && gb < 1e-3 && gf < 1e-3 && gd < 1e-3 && ga < 1e-3 && same, line );
		for ( float* p : { d_b, d_f, d_dt, d_al, d_beta, d_g, d_Rb, d_Rg, d_db, d_df, d_ddt, d_dal, d_db2, d_df2 } ) cudaFree( p );
	}
}

// Host delta rule with a per-key-channel decay, the same recurrence as kda_forward: o [rows, heads, D], seqs sequences from a zero state
Vec h_kda( const Vec& q, const Vec& k, const Vec& v, const Vec& g, const Vec& beta, int rows, int heads, int seqs ) {
	const int L = rows / seqs;
	Vec o( static_cast<size_t>( rows ) * heads * D, 0.0 ), S( static_cast<size_t>( heads ) * D * D ), e( D );   // S [h][c][i]
	for ( int t = 0; t < rows; ++t ) {
		if ( t % L == 0 ) std::fill( S.begin(), S.end(), 0.0 );
		for ( int h = 0; h < heads; ++h ) {
			const size_t b = ( static_cast<size_t>( t ) * heads + h ) * D;
			for ( int i = 0; i < D; ++i ) e[i] = std::exp( g[b + i] );
			const double bt = beta[static_cast<size_t>( t ) * heads + h];
			for ( int c = 0; c < D; ++c ) {
				double* Sc = S.data() + ( static_cast<size_t>( h ) * D + c ) * D;
				double m = 0.0;
				for ( int i = 0; i < D; ++i ) {
					Sc[i] *= e[i];
					m += Sc[i] * k[b + i];
				}
				const double d = bt * ( v[b + c] - m );
				double oc = 0.0;
				for ( int i = 0; i < D; ++i ) {
					Sc[i] += k[b + i] * d;
					oc += Sc[i] * q[b + i];
				}
				o[b + c] = oc;
			}
		}
	}
	return o;
}

// The recurrence's inputs as the layer feeds them: q and k l2-normed per head (q times 1 / sqrt( 128 )), g in ( -5, 0 ) (weighted toward slow
// decays), beta in ( 0, 1 )
struct KdaIn {
	Vec q, k, v, g, beta;
};

KdaIn kda_inputs( int rows, int heads ) {
	KdaIn in;
	const size_t n = static_cast<size_t>( rows ) * heads * D;
	auto unit = [&]( Vec x, double scale ) {
		for ( size_t r = 0; r < n / D; ++r ) {
			double s = 0.0;
			for ( int i = 0; i < D; ++i ) s += x[r * D + i] * x[r * D + i];
			for ( int i = 0; i < D; ++i ) x[r * D + i] *= scale / std::sqrt( s );
		}
		return x;
	};
	in.q = unit( randv( n ), 1.0 / std::sqrt( 128.0 ) );
	in.k = unit( randv( n ), 1.0 );
	in.v = randv( n );
	in.g = uniform( n, 0.0, 1.0 );
	for ( auto& x : in.g ) x = -5.0 * x * x;
	in.beta = uniform( static_cast<size_t>( rows ) * heads, 0.05, 0.95 );
	return in;
}

// The GPU forward and backward of one chunking: o, and the gradients from R (dq, dk, dv, dg, dbeta; base: the += outputs' start)
struct KdaOut {
	Vec o, grad[5];
};

KdaOut kda_run( const KdaIn& in, const Vec& R, int rows, int heads, int chunk, int seqs, double base ) {
	const size_t n = in.q.size(), nb = in.beta.size();
	float *d_q = up( in.q ), *d_k = up( in.k ), *d_v = up( in.v ), *d_g = up( in.g ), *d_b = up( in.beta ), *d_R = up( R ), *d_o = filled( n, 7.0 );
	float *d_ck = filled( g5::kda_ckpt_floats( rows, heads, chunk, seqs ), 7.0 ), *d_scr = filled( g5::kda_scratch_floats( heads, chunk, seqs ), 7.0 );
	float* d_gr[5] = { filled( n, base ), filled( n, base ), filled( n, base ), filled( n, base ), filled( nb, base ) };
	g5::kda_forward( d_q, d_k, d_v, d_g, d_b, rows, heads, chunk, d_ck, d_o, nullptr, seqs );
	g5::kda_backward( d_q, d_k, d_v, d_g, d_b, rows, heads, chunk, d_ck, d_R, d_gr[0], d_gr[1], d_gr[2], d_gr[3], d_gr[4], d_scr, nullptr, seqs );
	CUDA_CHECK( cudaGetLastError() );
	KdaOut out;
	out.o = down( d_o, n );
	for ( int j = 0; j < 5; ++j ) {
		out.grad[j] = down( d_gr[j], j < 4 ? n : nb );
		for ( auto& x : out.grad[j] ) x -= base;
	}
	for ( float* p : { d_q, d_k, d_v, d_g, d_b, d_R, d_o, d_ck, d_scr, d_gr[0], d_gr[1], d_gr[2], d_gr[3], d_gr[4] } ) cudaFree( p );
	return out;
}

void test_kda() {
	std::printf( "(c) KDA delta rule with per-channel decay (chunked checkpoints)\n" );
	char line[220];
	{
		// Two sequences of 19 rows (the forward's tiles of 16 and the backward's sub-chunks of 8: partial ones of each) under several chunkings:
		// 8 and 5 (one sub-chunk a chunk), 12 (two, the second partial), 24 (one chunk of three sub-chunks, past the sequence's end)
		const int L = 19, seqs = 2, rows = seqs * L, heads = 2;
		const KdaIn in = kda_inputs( rows, heads );
		const Vec R = randv( in.v.size() );
		const Vec oh = h_kda( in.q, in.k, in.v, in.g, in.beta, rows, heads, seqs );
		auto loss = [&]( int which, const Vec& x ) {
			const Vec& q = which == 0 ? x : in.q;
			const Vec& k = which == 1 ? x : in.k;
			const Vec& v = which == 2 ? x : in.v;
			const Vec& g = which == 3 ? x : in.g;
			const Vec& b = which == 4 ? x : in.beta;
			return dot( R, h_kda( q, k, v, g, b, rows, heads, seqs ) );
		};
		const Vec* xs[5] = { &in.q, &in.k, &in.v, &in.g, &in.beta };
		Fd fd[5];
		for ( int j = 0; j < 5; ++j ) fd[j] = fd_sample( [&]( const Vec& x ) { return loss( j, x ); }, *xs[j] );
		for ( int chunk : { 8, 5, 12, 24 } ) {
			const KdaOut r = kda_run( in, R, rows, heads, chunk, seqs, kBase );
			const double ef = rel( r.o, oh );
			double e[5];
			for ( int j = 0; j < 5; ++j ) e[j] = fd_err( fd[j], r.grad[j] );
			std::snprintf( line, sizeof( line ), "%d sequences of %d rows, %d heads, chunks of %2d: forward %.2e, dq %.2e, dk %.2e, dv %.2e, dg %.2e, dbeta %.2e",
				seqs, L, heads, chunk, ef, e[0], e[1], e[2], e[3], e[4] );
			check( ef < 1e-5 && e[0] < 1e-3 && e[1] < 1e-3 && e[2] < 1e-3 && e[3] < 1e-3 && e[4] < 1e-3, line );
		}
	}
	{
		// The production chunk of 64 over 2 sequences of 200 rows (chunks of 64 with a partial last one, eight sub-chunks a chunk): the forward
		// against the host, the gradients against the chunks of 8 above's (the recomputed states are the forward's bit for bit, so the two differ
		// by the order of the atomic column sums only)
		const int L = 200, seqs = 2, rows = seqs * L, heads = 3;
		const KdaIn in = kda_inputs( rows, heads );
		const Vec R = randv( in.v.size() );
		const KdaOut a = kda_run( in, R, rows, heads, 64, seqs, 0.0 ), b = kda_run( in, R, rows, heads, 8, seqs, 0.0 );
		const double ef = rel( a.o, h_kda( in.q, in.k, in.v, in.g, in.beta, rows, heads, seqs ) );
		double eg = 0.0;
		for ( int j = 0; j < 5; ++j ) eg = std::max( eg, rel( a.grad[j], b.grad[j] ) );
		std::snprintf( line, sizeof( line ), "%d sequences of %d rows, %d heads, chunks of 64: forward %.2e, gradients against chunks of 8 %.2e", seqs, L, heads,
			ef, eg );
		check( ef < 1e-5 && eg < 1e-5, line );
	}
}

} // namespace

int main() {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	CUDA_CHECK( cudaSetDevice( 0 ) );
	test_l2();
	test_gates();
	test_kda();
	std::printf( "[%c] test_train_glm5_kda: %d failure(s)\n", fails ? '-' : '+', fails );
	return fails ? 1 : 0;
}
