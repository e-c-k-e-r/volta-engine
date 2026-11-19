// Tier 2: the training primitives (engine/train/grad.h) on a SwiGLU MLP of three scaled linears, y = ( act( x Wg ) * ( x Wu ) ) Wd with
// W = diag( su ) T diag( sv ), against a double-precision host reference:
//   (a) the row-weighted squared-error loss against the host forward
//   (b) the gradients of the six scale vectors and of the input against central differences of the host loss, with SiLU and with GLM's limit
//   (c) Adam steps on multiplicative scale parameters lower the loss
#include "engine/config.h"
#include "engine/train/grad.h"
#include <cuda_runtime.h>
#include <algorithm>
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
	std::printf( fails ? "=== TRAIN GRAD: %d FAILURE(S) ===\n" : "=== TRAIN GRAD: PASS ===\n", fails );
	return fails ? 1 : 0;
}
