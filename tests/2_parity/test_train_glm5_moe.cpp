// Tier 2: GLM-5.3-Flash's MoE training operations (engine/train/glm5_moe.h) against double-precision host references: each forward against
// the host forward, each backward (from a random output gradient R) against the host's analytic gradient and central differences of the
// host loss sum( R * output ), the routing selection held fixed. Small shapes; no checkpoint.
#include "engine/config.h"
#include "engine/train/glm5_moe.h"
#include "engine/train/qwen4_exp.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

namespace {

namespace g5 = engine::train::glm5_next;
namespace q = engine::train::qwen4_exp;
using Vec = std::vector<double>;
int fails = 0;
std::mt19937 rng( 1234 );

constexpr int kE = 288, kK = 8;
constexpr double kScale = 2.5, kLimit = 10.0;

void check( bool ok, const char* line ) {
	std::printf( "  %-110s %s\n", line, ok ? "OK" : "FAIL" );
	fails += !ok;
}

// Normal values rounded to FP32, so that the host and the device see the same inputs
Vec randv( size_t n, double scale = 1.0 ) {
	std::normal_distribution<double> d( 0.0, scale );
	Vec v( n );
	for ( auto& x : v ) x = static_cast<float>( d( rng ) );
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

int* up_ints( const std::vector<int>& v ) {
	int* d = nullptr;
	CUDA_CHECK( cudaMalloc( &d, std::max<size_t>( v.size(), 1 ) * sizeof( int ) ) );
	if ( !v.empty() ) CUDA_CHECK( cudaMemcpy( d, v.data(), v.size() * sizeof( int ), cudaMemcpyHostToDevice ) );
	return d;
}

Vec down( const float* d, size_t n ) {
	std::vector<float> f( n );
	CUDA_CHECK( cudaDeviceSynchronize() );
	CUDA_CHECK( cudaMemcpy( f.data(), d, n * sizeof( float ), cudaMemcpyDeviceToHost ) );
	return Vec( f.begin(), f.end() );
}

std::vector<int> down_ints( const int* d, size_t n ) {
	std::vector<int> v( n );
	CUDA_CHECK( cudaDeviceSynchronize() );
	CUDA_CHECK( cudaMemcpy( v.data(), d, n * sizeof( int ), cudaMemcpyDeviceToHost ) );
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

// The gradient g of loss( x ) against central differences at the entries idx
double fd_rel_at( const std::function<double( const Vec& )>& loss, Vec x, const Vec& g, const std::vector<size_t>& idx, double h = 1e-5 ) {
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

// ... at up to 48 sampled entries
double fd_rel( const std::function<double( const Vec& )>& loss, const Vec& x, const Vec& g, double h = 1e-5 ) {
	std::vector<size_t> idx( x.size() );
	std::iota( idx.begin(), idx.end(), size_t( 0 ) );
	std::shuffle( idx.begin(), idx.end(), rng );
	if ( idx.size() > 48 ) idx.resize( 48 );
	return fd_rel_at( loss, x, g, idx, h );
}

// Up to n entries of each list, shuffled
std::vector<size_t> sample( std::vector<std::vector<size_t>> lists, size_t n ) {
	std::vector<size_t> out;
	for ( auto& l : lists ) {
		std::shuffle( l.begin(), l.end(), rng );
		out.insert( out.end(), l.begin(), l.begin() + std::min( n, l.size() ) );
	}
	return out;
}

double sig( double x ) { return 1.0 / ( 1.0 + std::exp( -x ) ); }
double silu( double x ) { return x * sig( x ); }

// ---- host routing ----
// Per row the k largest sigmoid( logit ) + bias (bias empty: 0), best first, ties to the lower index
std::vector<int> h_select( const Vec& logit, const Vec& bias, int rows, int E, int k ) {
	std::vector<int> ids( static_cast<size_t>( rows ) * k ), ord( E );
	std::vector<double> key( E );
	for ( int r = 0; r < rows; ++r ) {
		for ( int e = 0; e < E; ++e ) key[e] = sig( logit[static_cast<size_t>( r ) * E + e] ) + ( bias.empty() ? 0.0 : bias[e] );
		std::iota( ord.begin(), ord.end(), 0 );
		std::stable_sort( ord.begin(), ord.end(), [&]( int a, int b ) { return key[a] > key[b]; } );
		std::copy( ord.begin(), ord.begin() + k, ids.begin() + static_cast<size_t>( r ) * k );
	}
	return ids;
}

// The weights from the selected logits x [rows, k]: scale s_j / ( sum s + 1e-9 )
Vec h_weights( const Vec& x, int rows, int k, double scale ) {
	Vec w( x.size() );
	for ( int r = 0; r < rows; ++r ) {
		double S = 1e-9;
		for ( int j = 0; j < k; ++j ) S += sig( x[static_cast<size_t>( r ) * k + j] );
		for ( int j = 0; j < k; ++j ) w[static_cast<size_t>( r ) * k + j] = scale * sig( x[static_cast<size_t>( r ) * k + j] ) / S;
	}
	return w;
}

// Their gradient at the selected logits from dw
Vec h_weights_grad( const Vec& x, const Vec& dw, int rows, int k, double scale ) {
	const Vec w = h_weights( x, rows, k, scale );
	Vec dl( x.size() );
	for ( int r = 0; r < rows; ++r ) {
		double S = 1e-9, t = 0.0;
		for ( int j = 0; j < k; ++j ) {
			S += sig( x[static_cast<size_t>( r ) * k + j] );
			t += w[static_cast<size_t>( r ) * k + j] * dw[static_cast<size_t>( r ) * k + j];
		}
		for ( int j = 0; j < k; ++j ) {
			const size_t i = static_cast<size_t>( r ) * k + j;
			const double s = sig( x[i] );
			dl[i] = s * ( 1.0 - s ) * ( scale * dw[i] - t ) / S;
		}
	}
	return dl;
}

Vec gather_selected( const Vec& logit, const std::vector<int>& ids, int rows, int E, int k ) {
	Vec x( static_cast<size_t>( rows ) * k );
	for ( int r = 0; r < rows; ++r )
		for ( int j = 0; j < k; ++j ) x[static_cast<size_t>( r ) * k + j] = logit[static_cast<size_t>( r ) * E + ids[static_cast<size_t>( r ) * k + j]];
	return x;
}

void test_routing() {
	std::printf( "(a) sigmoid routing: biased top-%d of %d, weights %.1f s / ( sum s + 1e-9 ), backward at a fixed selection\n", kK, kE, kScale );
	char line[220];
	const int rows = 8192, E = kE, k = kK;
	Vec logit = randv( static_cast<size_t>( rows ) * E, 1.5 ), bias = randv( E, 0.1 );
	for ( int e = 1; e < E; e += 2 ) bias[e] = bias[e - 1];   // pairs ( 2i, 2i + 1 ) share a bias: copying a logit within a pair ties their keys
	// Ties: one row's best expert tied by its partner (lower index first), and two rows' k-th expert tied by a partner below (the partner
	// takes the slot) and above (the expert keeps it)
	int tie_rows[3] = { -1, -1, -1 }, tie_e[3] = { 0, 0, 0 };
	{
		const std::vector<int> pre = h_select( logit, bias, rows, E, k );
		auto selected = [&]( int r, int e ) { return std::find( pre.begin() + static_cast<size_t>( r ) * k, pre.begin() + static_cast<size_t>( r + 1 ) * k, e ) !=
			pre.begin() + static_cast<size_t>( r + 1 ) * k; };
		for ( int r = 0; r < rows && ( tie_rows[0] < 0 || tie_rows[1] < 0 || tie_rows[2] < 0 ); ++r ) {
			const int top = pre[static_cast<size_t>( r ) * k], kth = pre[static_cast<size_t>( r ) * k + k - 1];
			int c = -1, e = 0;
			if ( tie_rows[0] < 0 && !selected( r, top ^ 1 ) ) c = 0, e = top;
			else if ( tie_rows[1] < 0 && ( kth & 1 ) && !selected( r, kth ^ 1 ) ) c = 1, e = kth;
			else if ( tie_rows[2] < 0 && !( kth & 1 ) && !selected( r, kth ^ 1 ) ) c = 2, e = kth;
			if ( c < 0 ) continue;
			tie_rows[c] = r;
			tie_e[c] = e;
			logit[static_cast<size_t>( r ) * E + ( e ^ 1 )] = logit[static_cast<size_t>( r ) * E + e];
		}
	}
	float *d_l = up( logit ), *d_b = up( bias ), *d_w = zeros( static_cast<size_t>( rows ) * k ), *d_wg = zeros( static_cast<size_t>( rows ) * k );
	int* d_ids = up_ints( std::vector<int>( static_cast<size_t>( rows ) * k, -1 ) );
	g5::glm_route_forward( d_l, d_b, rows, E, k, static_cast<float>( kScale ), d_ids, d_w, nullptr );
	g5::glm_route_given( d_l, d_ids, rows, E, k, static_cast<float>( kScale ), d_wg, nullptr );
	const std::vector<int> ids = down_ints( d_ids, static_cast<size_t>( rows ) * k ), hid = h_select( logit, bias, rows, E, k );
	const std::vector<int> nob = h_select( logit, Vec(), rows, E, k );
	const Vec w = down( d_w, ids.size() ), wg = down( d_wg, ids.size() );
	int wrong = 0, moved = 0;
	for ( int r = 0; r < rows; ++r ) {
		bool same = true;
		for ( int j = 0; j < k; ++j ) same = same && ids[static_cast<size_t>( r ) * k + j] == hid[static_cast<size_t>( r ) * k + j];
		wrong += !same;
		std::vector<int> a( hid.begin() + static_cast<size_t>( r ) * k, hid.begin() + static_cast<size_t>( r + 1 ) * k );
		std::vector<int> b( nob.begin() + static_cast<size_t>( r ) * k, nob.begin() + static_cast<size_t>( r + 1 ) * k );
		std::sort( a.begin(), a.end() );
		std::sort( b.begin(), b.end() );
		moved += a != b;
	}
	auto in_row = [&]( int r, int e ) { return std::find( ids.begin() + static_cast<size_t>( r ) * k, ids.begin() + static_cast<size_t>( r + 1 ) * k, e ) !=
		ids.begin() + static_cast<size_t>( r + 1 ) * k; };
	const bool ties_found = tie_rows[0] >= 0 && tie_rows[1] >= 0 && tie_rows[2] >= 0;
	const bool ties_ok = ties_found && ids[static_cast<size_t>( tie_rows[0] ) * k] == std::min( tie_e[0], tie_e[0] ^ 1 ) &&
		ids[static_cast<size_t>( tie_rows[0] ) * k + 1] == std::max( tie_e[0], tie_e[0] ^ 1 ) &&
		ids[static_cast<size_t>( tie_rows[1] ) * k + k - 1] == ( tie_e[1] ^ 1 ) && !in_row( tie_rows[1], tie_e[1] ) &&
		ids[static_cast<size_t>( tie_rows[2] ) * k + k - 1] == tie_e[2] && !in_row( tie_rows[2], tie_e[2] ^ 1 );
	const Vec wh = h_weights( gather_selected( logit, hid, rows, E, k ), rows, k, kScale );
	const double ew = rel( w, wh ), eg = rel( wg, w );
	std::snprintf( line, sizeof( line ), "%d rows: selection %s (%d rows differ; the bias moves %d), ties lower index first: %s", rows, wrong ? "WRONG" : "exact",
		wrong, moved, ties_ok ? "yes" : "no" );
	check( wrong == 0 && moved > 0 && ties_ok, line );
	std::snprintf( line, sizeof( line ), "weights %.2e, given-selection weights against the forward's %.2e", ew, eg );
	check( ew < 1e-5 && eg < 1e-6, line );

	// Backward: dlogit at the selected logits against the host gradient and central differences of sum( R * w ) in them
	const Vec R = randv( ids.size() );
	float *d_R = up( R ), *d_dl = zeros( ids.size() );
	g5::glm_route_backward( d_l, d_ids, rows, E, k, static_cast<float>( kScale ), d_R, d_dl, nullptr );
	const Vec dl = down( d_dl, ids.size() ), xs = gather_selected( logit, ids, rows, E, k );
	auto loss = [&]( const Vec& x ) { return dot( R, h_weights( x, rows, k, kScale ) ); };
	const double ea = rel( dl, h_weights_grad( xs, R, rows, k, kScale ) ), ef = fd_rel( loss, xs, dl );
	std::snprintf( line, sizeof( line ), "backward dlogit: %.2e against the host gradient, %.2e against central differences", ea, ef );
	check( ea < 1e-5 && ef < 1e-3, line );
	for ( float* p : { d_l, d_b, d_w, d_wg, d_R, d_dl } ) cudaFree( p );
	cudaFree( d_ids );
}

// The router's input gradient through qwen4_exp's router_input_backward: logits = a W^T (W [experts, hidden]), dX += sum_j dlogit_j W[ids_j]
void test_router_input() {
	std::printf( "(b) the router's input gradient: glm_route_backward into qwen4_exp::router_input_backward\n" );
	char line[220];
	const int rows = 16, H = 64, E = kE, k = kK;
	const Vec a = randv( static_cast<size_t>( rows ) * H ), W = randv( static_cast<size_t>( E ) * H, 0.2 ), bias = randv( E, 0.1 );
	auto logits_of = [&]( const Vec& x ) {
		Vec l( static_cast<size_t>( rows ) * E );
		for ( int r = 0; r < rows; ++r )
			for ( int e = 0; e < E; ++e ) {
				double s = 0.0;
				for ( int d = 0; d < H; ++d ) s += x[static_cast<size_t>( r ) * H + d] * W[static_cast<size_t>( e ) * H + d];
				l[static_cast<size_t>( r ) * E + e] = s;
			}
		return l;
	};
	const Vec logit = logits_of( a ), R = randv( static_cast<size_t>( rows ) * k ), dX0 = randv( static_cast<size_t>( rows ) * H );
	float *d_l = up( logit ), *d_b = up( bias ), *d_w = zeros( static_cast<size_t>( rows ) * k ), *d_R = up( R ), *d_dl = zeros( static_cast<size_t>( rows ) * k );
	float *d_W = up( W ), *d_dX = up( dX0 );
	int* d_ids = up_ints( std::vector<int>( static_cast<size_t>( rows ) * k, -1 ) );
	g5::glm_route_forward( d_l, d_b, rows, E, k, static_cast<float>( kScale ), d_ids, d_w, nullptr );
	g5::glm_route_backward( d_l, d_ids, rows, E, k, static_cast<float>( kScale ), d_R, d_dl, nullptr );
	q::router_input_backward( d_dl, d_ids, d_W, rows, k, H, d_dX, nullptr );
	const std::vector<int> ids = down_ints( d_ids, static_cast<size_t>( rows ) * k );
	Vec dX = down( d_dX, dX0.size() );
	for ( size_t i = 0; i < dX.size(); ++i ) dX[i] -= dX0[i];   // router_input_backward adds
	// host: the analytic chain and central differences in a, the selection fixed at the device's
	const Vec dlh = h_weights_grad( gather_selected( logit, ids, rows, E, k ), R, rows, k, kScale );
	Vec dXh( dX.size(), 0.0 );
	for ( int r = 0; r < rows; ++r )
		for ( int j = 0; j < k; ++j )
			for ( int d = 0; d < H; ++d )
				dXh[static_cast<size_t>( r ) * H + d] += dlh[static_cast<size_t>( r ) * k + j] * W[static_cast<size_t>( ids[static_cast<size_t>( r ) * k + j] ) * H + d];
	auto loss = [&]( const Vec& x ) { return dot( R, h_weights( gather_selected( logits_of( x ), ids, rows, E, k ), rows, k, kScale ) ); };
	const double ea = rel( dX, dXh ), ef = fd_rel( loss, a, dX );
	std::snprintf( line, sizeof( line ), "dX (+=, from [rows, k] dlogit): %.2e against the host chain, %.2e against central differences in a", ea, ef );
	check( ea < 1e-5 && ef < 1e-3, line );
	for ( float* p : { d_l, d_b, d_w, d_R, d_dl, d_W, d_dX } ) cudaFree( p );
	cudaFree( d_ids );
}

// ---- SwiGLU with the clamp after the SiLU ----
double h_act( double g, double u, double L ) {
	if ( L <= 0.0 ) return silu( g ) * u;
	return std::min( silu( g ), L ) * std::max( -L, std::min( u, L ) );
}

void h_act_grad( double g, double u, double da, double L, double& dg, double& du ) {
	const double s = sig( g ), sl = g * s;
	const double uc = L > 0.0 ? std::max( -L, std::min( u, L ) ) : u, a = L > 0.0 ? std::min( sl, L ) : sl;
	dg = ( L <= 0.0 || sl < L ) ? da * uc * s * ( 1.0 + g * ( 1.0 - s ) ) : 0.0;
	du = ( L <= 0.0 || ( u > -L && u < L ) ) ? da * a : 0.0;
}

// Entries by clamp: silu( g ) at the limit; |u| at the limit (the gate free); neither; each at least `margin` from the kinks
std::vector<std::vector<size_t>> by_clamp( const Vec& g, const Vec& u, double L, double margin, const std::vector<bool>* skip = nullptr ) {
	std::vector<std::vector<size_t>> c( 3 );
	for ( size_t i = 0; i < g.size(); ++i ) {
		if ( skip && ( *skip )[i] ) continue;
		if ( std::fabs( silu( g[i] ) - L ) < margin || std::fabs( std::fabs( u[i] ) - L ) < margin ) continue;
		c[silu( g[i] ) > L ? 0 : ( std::fabs( u[i] ) > L ? 1 : 2 )].push_back( i );
	}
	return c;
}

void test_swiglu_after() {
	std::printf( "(c) SwiGLU with the clamp after the SiLU, elementwise: min( silu( g ), %.0f ) clamp( u, -%.0f, %.0f )\n", kLimit, kLimit, kLimit );
	char line[220];
	const size_t n = 4096;
	const Vec g = randv( n, 7.0 ), u = randv( n, 8.0 ), R = randv( n );
	float *d_g = up( g ), *d_u = up( u ), *d_a = zeros( n ), *d_a0 = zeros( n ), *d_R = up( R ), *d_dg = zeros( n ), *d_du = zeros( n );
	g5::swiglu_after_forward( d_g, d_u, d_a, n, static_cast<float>( kLimit ), nullptr );
	g5::swiglu_after_forward( d_g, d_u, d_a0, n, 0.0f, nullptr );
	g5::swiglu_after_backward( d_g, d_u, d_R, d_dg, d_du, n, static_cast<float>( kLimit ), nullptr );
	Vec ah( n ), a0h( n ), dgh( n ), duh( n );
	int ng = 0, nu = 0;
	for ( size_t i = 0; i < n; ++i ) {
		ah[i] = h_act( g[i], u[i], kLimit );
		a0h[i] = h_act( g[i], u[i], 0.0 );
		h_act_grad( g[i], u[i], R[i], kLimit, dgh[i], duh[i] );
		ng += silu( g[i] ) > kLimit;
		nu += std::fabs( u[i] ) > kLimit;
	}
	const Vec dg = down( d_dg, n ), du = down( d_du, n );
	const double ef = rel( down( d_a, n ), ah ), ef0 = rel( down( d_a0, n ), a0h ), eag = rel( dg, dgh ), eau = rel( du, duh );
	std::snprintf( line, sizeof( line ), "forward %.2e (silu( g ) > 10: %d, |u| > 10: %d of %zu), unclamped (limit 0) %.2e", ef, ng, nu, n, ef0 );
	check( ef < 1e-5 && ef0 < 1e-5 && ng > 0 && nu > 0, line );
	const std::vector<size_t> idx = sample( by_clamp( g, u, kLimit, 1e-3 ), 16 );
	auto loss_g = [&]( const Vec& x ) { double L = 0.0; for ( size_t i = 0; i < n; ++i ) L += R[i] * h_act( x[i], u[i], kLimit ); return L; };
	auto loss_u = [&]( const Vec& x ) { double L = 0.0; for ( size_t i = 0; i < n; ++i ) L += R[i] * h_act( g[i], x[i], kLimit ); return L; };
	const double efg = fd_rel_at( loss_g, g, dg, idx ), efu = fd_rel_at( loss_u, u, du, idx );
	std::snprintf( line, sizeof( line ), "backward dg %.2e, du %.2e against the host; %.2e, %.2e against central differences (%zu off the kinks)", eag, eau,
		efg, efu, idx.size() );
	check( eag < 1e-5 && eau < 1e-5 && efg < 1e-3 && efu < 1e-3 && idx.size() == 48, line );
	for ( float* p : { d_g, d_u, d_a, d_a0, d_R, d_dg, d_du } ) cudaFree( p );
}

// The grouped-route layout: 3 experts of a 5-expert table, 8 padded rows each, 6 / 8 / 3 of them real
void test_route_swiglu() {
	std::printf( "(d) routed SwiGLU in the grouped-route layout (3 experts x 8 padded rows, per-expert scale tables)\n" );
	char line[220];
	const int m = 8, P = 3 * m, cols = 32, tab_n = 5;
	const std::vector<int> gexp = { 4, 0, 2 }, real = { 6, 8, 3 };
	std::vector<bool> pad( static_cast<size_t>( P ) * cols );
	for ( int p = 0; p < P; ++p ) for ( int c = 0; c < cols; ++c ) pad[static_cast<size_t>( p ) * cols + c] = p % m >= real[p / m];
	Vec Zg = randv( pad.size(), 7.0 ), Zu = randv( pad.size(), 8.0 );
	for ( size_t i = 0; i < pad.size(); ++i ) if ( pad[i] ) Zg[i] = Zu[i] = 0.0;
	const Vec dAk = randv( pad.size() );   // padding rows too
	auto pos_vec = [&]() { Vec v = randv( cols, 0.2 ); for ( auto& x : v ) x = static_cast<float>( std::exp( x ) ); return v; };
	std::vector<Vec> svg( tab_n ), svu( tab_n ), sud( tab_n );
	std::vector<const float*> tg( tab_n, nullptr ), tu( tab_n, nullptr ), td( tab_n, nullptr );
	std::vector<float*> owned;
	for ( int e : gexp ) {
		svg[e] = pos_vec(); svu[e] = pos_vec(); sud[e] = pos_vec();
		for ( auto* pr : { &tg, &tu, &td } ) {
			const Vec& v = pr == &tg ? svg[e] : ( pr == &tu ? svu[e] : sud[e] );
			owned.push_back( up( v ) );
			( *pr )[e] = owned.back();
		}
	}
	auto up_tab = [&]( const std::vector<const float*>& t ) {
		const float** d = nullptr;
		CUDA_CHECK( cudaMalloc( &d, t.size() * sizeof( float* ) ) );
		CUDA_CHECK( cudaMemcpy( d, t.data(), t.size() * sizeof( float* ), cudaMemcpyHostToDevice ) );
		return d;
	};
	const float **d_tg = up_tab( tg ), **d_tu = up_tab( tu ), **d_td = up_tab( td );
	int* d_gexp = up_ints( gexp );
	const size_t N = pad.size();
	float *d_Zg = up( Zg ), *d_Zu = up( Zu ), *d_G = zeros( N ), *d_U = zeros( N ), *d_A = zeros( N ), *d_dAk = up( dAk ), *d_dG = zeros( N ), *d_dU = zeros( N );
	const float L = static_cast<float>( kLimit );
	g5::glm_route_swiglu( d_Zg, d_Zu, d_gexp, m, d_tg, d_tu, P, cols, L, d_G, d_U, d_A, nullptr );
	g5::glm_route_swiglu_backward( d_dAk, d_G, d_U, d_gexp, m, d_td, P, cols, L, d_dG, d_dU, nullptr );
	// host
	auto ex = [&]( size_t i ) { return gexp[i / cols / m]; };
	Vec Gh( N ), Uh( N ), Ah( N ), dGh( N ), dUh( N );
	int ng = 0, nu = 0;
	for ( size_t i = 0; i < N; ++i ) {
		const int e = ex( i ), c = static_cast<int>( i % cols );
		Gh[i] = Zg[i] * svg[e][c];
		Uh[i] = Zu[i] * svu[e][c];
		Ah[i] = h_act( Gh[i], Uh[i], kLimit );
		h_act_grad( Gh[i], Uh[i], dAk[i] * sud[e][c], kLimit, dGh[i], dUh[i] );
		ng += silu( Gh[i] ) > kLimit;
		nu += std::fabs( Uh[i] ) > kLimit;
	}
	const Vec G = down( d_G, N ), U = down( d_U, N ), A = down( d_A, N ), dG = down( d_dG, N ), dU = down( d_dU, N );
	bool pad_zero = true;
	for ( size_t i = 0; i < N; ++i ) if ( pad[i] ) pad_zero = pad_zero && G[i] == 0.0 && U[i] == 0.0 && A[i] == 0.0 && dG[i] == 0.0 && dU[i] == 0.0;
	const double eG = rel( G, Gh ), eU = rel( U, Uh ), eA = rel( A, Ah );
	std::snprintf( line, sizeof( line ), "forward G %.2e, U %.2e, A %.2e (silu( G ) > 10: %d, |U| > 10: %d of %d), padding rows zero: %s", eG, eU, eA, ng, nu,
		( real[0] + real[1] + real[2] ) * cols, pad_zero ? "yes" : "no" );
	check( eG < 1e-5 && eU < 1e-5 && eA < 1e-5 && ng > 0 && nu > 0 && pad_zero, line );
	// central differences in Zg, Zu of sum( dAk su_down A ): their gradients dG sv_gate, dU sv_up
	Vec dZg( N ), dZu( N );
	for ( size_t i = 0; i < N; ++i ) {
		dZg[i] = dG[i] * svg[ex( i )][i % cols];
		dZu[i] = dU[i] * svu[ex( i )][i % cols];
	}
	auto loss = [&]( const Vec& zg, const Vec& zu ) {
		double s = 0.0;
		for ( size_t i = 0; i < N; ++i ) {
			const int e = ex( i ), c = static_cast<int>( i % cols );
			s += dAk[i] * sud[e][c] * h_act( zg[i] * svg[e][c], zu[i] * svu[e][c], kLimit );
		}
		return s;
	};
	std::vector<std::vector<size_t>> cats = by_clamp( Gh, Uh, kLimit, 1e-3, &pad );
	std::vector<size_t> pads;
	for ( size_t i = 0; i < N; ++i ) if ( pad[i] ) pads.push_back( i );
	cats.push_back( pads );
	const std::vector<size_t> idx = sample( cats, 12 );
	const double eaG = rel( dG, dGh ), eaU = rel( dU, dUh );
	const double efg = fd_rel_at( [&]( const Vec& x ) { return loss( x, Zu ); }, Zg, dZg, idx ),
		efu = fd_rel_at( [&]( const Vec& x ) { return loss( Zg, x ); }, Zu, dZu, idx );
	std::snprintf( line, sizeof( line ), "backward dG %.2e, dU %.2e against the host; Zg %.2e, Zu %.2e against central differences (%zu, 12 padding)", eaG,
		eaU, efg, efu, idx.size() );
	check( eaG < 1e-5 && eaU < 1e-5 && efg < 1e-3 && efu < 1e-3 && idx.size() == 48, line );
	for ( float* p : { d_Zg, d_Zu, d_G, d_U, d_A, d_dAk, d_dG, d_dU } ) cudaFree( p );
	for ( float* p : owned ) cudaFree( p );
	cudaFree( d_tg ); cudaFree( d_tu ); cudaFree( d_td ); cudaFree( d_gexp );
}

} // namespace

int main() {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	CUDA_CHECK( cudaSetDevice( 0 ) );
	test_routing();
	test_router_input();
	test_swiglu_after();
	test_route_swiglu();
	std::printf( "[%c] test_train_glm5_moe: %d failure(s)\n", fails ? '-' : '+', fails );
	return fails ? 1 : 0;
}
