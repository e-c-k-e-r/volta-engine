// Tier 2 (checkpoint): one qwen4_exp training layer (engine/train/qwen4_exp.h, Model) of an EXL3 checkpoint against central differences of
// its own forward: random input streams (and n-gram rows for the PLE layer), the loss sum( R * layer( x ) ); the input gradient on sampled
// entries and the scale gradients of sampled channels (the routing held at the unperturbed input's selection) of several tensors (attention or GDN projections, the shared expert, routed experts),
// and the weight gradients of the norms, the small dense tensors and the router (trainable here; the forward checked against their FP16
// copies), the tile gains' gradients by their sum against the scales'; then two sequences through one call against each alone.
//   TRAIN_MODEL_DIR=<EXL3 Flash-Next checkpoint> test_train_qwen4_exp_layer [layer ...]   (default: 1, the PLE + GDN layer, and 3, attention)
#include "engine/config.h"
#include "engine/train/qwen4_exp.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

namespace q = engine::train::qwen4_exp;
int fails = 0;

void check( bool ok, const char* line ) {
	std::printf( "  %-100s %s\n", line, ok ? "OK" : "FAIL" );
	fails += !ok;
}

std::vector<float> down( const float* d, size_t n ) {
	std::vector<float> h( n );
	CUDA_CHECK( cudaDeviceSynchronize() );
	CUDA_CHECK( cudaMemcpy( h.data(), d, n * sizeof( float ), cudaMemcpyDeviceToHost ) );
	return h;
}

float get( const float* d, size_t i ) {
	float v = 0.0f;
	CUDA_CHECK( cudaDeviceSynchronize() );
	CUDA_CHECK( cudaMemcpy( &v, d + i, sizeof( float ), cudaMemcpyDeviceToHost ) );
	return v;
}

void set( float* d, size_t i, float v ) {
	CUDA_CHECK( cudaDeviceSynchronize() );
	CUDA_CHECK( cudaMemcpy( d + i, &v, sizeof( float ), cudaMemcpyHostToDevice ) );
	// a copy from pageable memory may return before its DMA lands, and the model's stream does not wait for the default stream
	CUDA_CHECK( cudaDeviceSynchronize() );
}

} // namespace

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	const char* dir = std::getenv( "TRAIN_MODEL_DIR" );
	if ( !dir ) {
		std::printf( "[-] TRAIN_MODEL_DIR not set\n" );
		return 2;
	}
	std::vector<int> layers;
	for ( int i = 1; i < argc; ++i ) layers.push_back( std::atoi( argv[i] ) );
	if ( layers.empty() ) layers = { 1, 3 };
	CUDA_CHECK( cudaSetDevice( 0 ) );
	std::mt19937 rng( 99 );
	std::normal_distribution<float> nd( 0.0f, 1.0f );
	const int T = 24;
	for ( int L : layers ) {
		q::Model m;
		m.set_train_dense( true );   // the small dense tensors', the router's and the hyper-connection mixes' gradients checked with the norms'
		m.set_train_router( true );
		m.set_train_hc( true );
		m.set_train_gains( true );
		if ( !m.load( dir, L, L + 1, 0, T ) ) {
			std::printf( "[-] cannot load layer %d of %s\n", L, dir );
			return 1;
		}
		// the central differences in FP32 products (FP16 rounding makes the forward step-like at the perturbations' scale); the tensor-core
		// products are compared against them at the end
		m.set_fp16( false );
		const q::Config& c = m.config();
		const size_t flat = static_cast<size_t>( c.hc ) * c.hidden, nx = T * flat;
		std::printf( "(layer %d, %s%s) %d rows\n", L, c.full_attention( L ) ? "attention" : "GDN", L == c.ple_layer ? " + PLE" : "", T );
		std::vector<float> hx( nx ), hr( nx ), hk( T * flat ), hv( static_cast<size_t>( T ) * c.hidden );
		for ( auto& v : hx ) v = nd( rng );
		for ( auto& v : hr ) v = nd( rng );
		for ( auto& v : hk ) v = nd( rng );
		for ( auto& v : hv ) v = 0.3f * nd( rng );
		float *x, *y, *R, *dx, *pk, *pv;
		for ( float** p : { &x, &y, &R, &dx } ) CUDA_CHECK( cudaMalloc( p, nx * sizeof( float ) ) );
		CUDA_CHECK( cudaMalloc( &pk, hk.size() * sizeof( float ) ) );
		CUDA_CHECK( cudaMalloc( &pv, hv.size() * sizeof( float ) ) );
		CUDA_CHECK( cudaMemcpy( x, hx.data(), nx * sizeof( float ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( R, hr.data(), nx * sizeof( float ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( pk, hk.data(), hk.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( pv, hv.data(), hv.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaDeviceSynchronize() );
		auto loss = [&]() {
			m.forward( L, x, y, T, pk, pv );
			const std::vector<float> h = down( y, nx );
			double s = 0.0;
			for ( size_t i = 0; i < nx; ++i ) s += static_cast<double>( hr[i] ) * h[i];
			return s;
		};
		loss();   // the routing of this input, then held for every perturbation
		{
			// the same layer with its small dense tensors as FP16 copies (not trained): the same forward, bit for bit
			const std::vector<float> y_t = down( y, nx );
			q::Model m2;
			if ( !m2.load( dir, L, L + 1, 0, T ) ) return 1;
			m2.set_fp16( false );
			m2.forward( L, x, y, T, pk, pv );
			const bool same = std::memcmp( down( y, nx ).data(), y_t.data(), nx * sizeof( float ) ) == 0;
			check( same, "forward with the small dense tensors trained against as FP16 copies: bit for bit" );
		}
		m.freeze_routing( true );
		// the backward after a kept forward (reusing its sublayer outputs), as the training runs it
		m.forward( L, x, y, T, pk, pv, true );
		const std::vector<float> y32 = down( y, nx );
		m.zero_grads();
		m.backward( L, x, R, dx, T, pk, pv );
		const std::vector<float> g = down( dx, nx );
		// Input gradient on sampled entries
		double worst = 0.0, gmax = 0.0;
		size_t worst_i = 0;
		double worst_fd = 0.0, worst_l0 = 0.0, worst_lp = 0.0, worst_lm = 0.0;
		for ( int s = 0; s < 12; ++s ) {
			const size_t i = std::uniform_int_distribution<size_t>( 0, nx - 1 )( rng );
			const float x0 = hx[i], h = 1e-2f * std::max( 1.0f, std::fabs( x0 ) );
			set( x, i, x0 + h );
			const double lp = loss();
			set( x, i, x0 - h );
			const double lm = loss();
			set( x, i, x0 );
			const double fd = ( lp - lm ) / ( 2 * h );
			if ( std::fabs( fd - g[i] ) > worst ) {
				worst = std::fabs( fd - g[i] );
				worst_i = i;
				worst_fd = fd;
				worst_lp = lp;
				worst_lm = lm;
				worst_l0 = loss();
			}
			gmax = std::max( gmax, std::fabs( fd ) );
		}
		char line[200];
		std::snprintf( line, sizeof( line ), "input gradient, 12 entries: max error %.3g of max %.3g", worst, gmax );
		if ( worst > 2e-2 * gmax + 1e-3 )
			std::printf( "    (entry %zu: differences %.6g, analytic %.6g; loss %.9g, at +h %.9g, at -h %.9g)\n", worst_i, worst_fd, g[worst_i], worst_l0, worst_lp,
				worst_lm );
		check( worst <= 2e-2 * gmax + 1e-3, line );
		// Scale gradients: a few tensors, a few channels each
		std::vector<q::Tuned*> ts = m.tensors();
		std::vector<q::Tuned*> pick;
		for ( q::Tuned* t : ts ) if ( t->name.find( ".experts." ) == std::string::npos ) pick.push_back( t );
		// routed experts: per projection the ones with the largest input-scale gradient (routed by some of the rows)
		for ( const char* kind : { "gate_proj", "up_proj", "down_proj" } ) {
			std::vector<std::pair<double, q::Tuned*>> ex;
			for ( q::Tuned* t : ts ) {
				if ( t->name.find( ".experts." ) == std::string::npos || t->name.find( kind ) == std::string::npos ) continue;
				const std::vector<float> gu = down( t->gu, t->k );
				double n2 = 0.0;
				for ( float v : gu ) n2 += v * v;
				ex.push_back( { n2, t } );
			}
			std::sort( ex.begin(), ex.end(), []( const auto& a, const auto& b ) { return a.first > b.first; } );
			for ( size_t i = 0; i < std::min<size_t>( kind[0] == 'g' ? 2 : 1, ex.size() ); ++i ) pick.push_back( ex[i].second );
		}
		for ( q::Tuned* t : pick ) {
			const std::vector<float> gu = down( t->gu, t->k ), gv = down( t->gv, t->n );
			double w = 0.0, gm = 0.0;
			for ( int s = 0; s < 3; ++s ) {
				const bool in = s < 2;
				const int nch = in ? t->k : t->n;
				// the channel with a large gradient, and random ones
				int ch = std::uniform_int_distribution<int>( 0, nch - 1 )( rng );
				if ( s == 0 ) ch = static_cast<int>( std::max_element( gu.begin(), gu.end(), []( float a, float b ) { return std::fabs( a ) < std::fabs( b ); } ) - gu.begin() );
				float* sc = in ? t->su : t->sv;
				const float s0 = get( sc, ch ), h = 1e-2f * std::max( 1e-3f, std::fabs( s0 ) );
				set( sc, ch, s0 + h );
				const double lp = loss();
				set( sc, ch, s0 - h );
				const double lm = loss();
				set( sc, ch, s0 );
				const double fd = ( lp - lm ) / ( 2 * h ), an = in ? gu[ch] : gv[ch];
				w = std::max( w, std::fabs( fd - an ) );
				gm = std::max( gm, std::fabs( fd ) );
			}
			std::snprintf( line, sizeof( line ), "%-56s scales: error %.3g of %.3g (s %.3g)", t->name.c_str(), w, gm, get( t->su, 0 ) );
			check( w <= 3e-2 * gm + 1e-4, line );
			if ( t->tg ) {
				// the tile gains (8-bit codes: no central differences at their step): scaling every gain scales the weight as scaling every
				// input scale does, so sum_t g_t dL / dg_t = sum_i su_i dL / dsu_i = sum_j sv_j dL / dsv_j (the per-tile sums: test_train_qwen4_exp)
				const size_t nt = static_cast<size_t>( t->k / 16 ) * ( t->n / 16 );
				const std::vector<float> tg = down( t->tg, nt ), gt = down( t->gtg, nt ), su = down( t->su, t->k ), sv = down( t->sv, t->n );
				double sg = 0.0, ss = 0.0, sw = 0.0, mg = 0.0;
				for ( size_t i = 0; i < nt; ++i ) {
					sg += static_cast<double>( tg[i] ) * gt[i];
					mg += std::fabs( static_cast<double>( tg[i] ) * gt[i] );
				}
				for ( int i = 0; i < t->k; ++i ) ss += static_cast<double>( su[i] ) * gu[i];
				for ( int j = 0; j < t->n; ++j ) sw += static_cast<double>( sv[j] ) * gv[j];
				std::snprintf( line, sizeof( line ), "%-56s gains: sum g dg %.4g, sum su dsu %.4g, sum sv dsv %.4g", t->name.c_str(), sg, ss, sw );
				check( std::fabs( sg - ss ) <= 1e-2 * mg + 1e-6 && std::fabs( sg - sw ) <= 1e-2 * mg + 1e-6, line );
			}
		}
		// Norm weights and the small dense tensors: the largest-gradient entry and a random one of each (absolute steps: additive deltas)
		for ( q::Dense* d : m.dense_tensors() ) {
			const std::vector<float> gd = down( d->g, d->n );
			double w = 0.0, gm = 0.0;
			for ( int s = 0; s < 2; ++s ) {
				int ch = std::uniform_int_distribution<int>( 0, d->n - 1 )( rng );
				if ( s == 0 ) ch = static_cast<int>( std::max_element( gd.begin(), gd.end(), []( float a, float b ) { return std::fabs( a ) < std::fabs( b ); } ) - gd.begin() );
				const float w0 = get( d->w, ch ), h = 4e-3f + 1e-2f * std::fabs( w0 );
				set( d->w, ch, w0 + h );
				const double lp = loss();
				set( d->w, ch, w0 - h );
				const double lm = loss();
				set( d->w, ch, w0 );
				const double fd = ( lp - lm ) / ( 2 * h );
				w = std::max( w, std::fabs( fd - gd[ch] ) );
				gm = std::max( gm, std::fabs( fd ) );
			}
			const std::string nm = d->name.substr( d->name.find( "layers." ) == std::string::npos ? 0 : d->name.find( "layers." ) );
			std::snprintf( line, sizeof( line ), "%-56s weights: error %.3g of %.3g", nm.c_str(), w, gm );
			check( w <= 3e-2 * gm + 1e-4, line );
		}
		{
			// Router distillation: a random routing teacher (16 experts per row with probabilities, the rest as the tail) at weight 0.7; the loss
			// sum( R * layer( x ) ) + 0.7 KL( teacher || router ), its input gradient against central differences
			const int TK = 16, E = c.experts;
			std::vector<int> tid( static_cast<size_t>( T ) * TK );
			std::vector<float> tp( tid.size() ), tt( T );
			std::vector<int> perm( E );
			for ( int r = 0; r < T; ++r ) {
				for ( int e = 0; e < E; ++e ) perm[e] = e;
				std::shuffle( perm.begin(), perm.end(), rng );
				float s = 0.0f;
				for ( int j = 0; j < TK; ++j ) {
					tid[static_cast<size_t>( r ) * TK + j] = perm[j];
					s += tp[static_cast<size_t>( r ) * TK + j] = std::exp( nd( rng ) );
				}
				for ( int j = 0; j < TK; ++j ) tp[static_cast<size_t>( r ) * TK + j] *= 0.9f / s;
				tt[r] = 0.1f;
			}
			int* d_tid;
			float *d_tp, *d_tt;
			CUDA_CHECK( cudaMalloc( &d_tid, tid.size() * sizeof( int ) ) );
			CUDA_CHECK( cudaMalloc( &d_tp, tp.size() * sizeof( float ) ) );
			CUDA_CHECK( cudaMalloc( &d_tt, tt.size() * sizeof( float ) ) );
			CUDA_CHECK( cudaMemcpy( d_tid, tid.data(), tid.size() * sizeof( int ), cudaMemcpyHostToDevice ) );
			CUDA_CHECK( cudaMemcpy( d_tp, tp.data(), tp.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
			CUDA_CHECK( cudaMemcpy( d_tt, tt.data(), tt.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
			CUDA_CHECK( cudaDeviceSynchronize() );
			q::Teacher tch;
			tch.ids = d_tid;
			tch.p = d_tp;
			tch.tail = d_tt;
			tch.tk = TK;
			// the term alone (no output gradient: R zero), so that its gradient is what the differences see
			const float w = 0.7f;
			m.set_router_teacher( { tch }, w );
			float* R0;
			CUDA_CHECK( cudaMalloc( &R0, nx * sizeof( float ) ) );
			CUDA_CHECK( cudaMemset( R0, 0, nx * sizeof( float ) ) );
			auto loss_r = [&]() {
				m.forward( L, x, y, T, pk, pv );
				return static_cast<double>( w ) * m.router_kl();
			};
			// the distribution's KL, then the top-k boundary's margins (the selection held, so that the margin terms are smooth)
			for ( const float margin : { 0.0f, 0.5f } ) {
				m.set_router_margin( margin );
				const double kl0 = loss_r() / w / T;
				m.forward( L, x, y, T, pk, pv, true );
				m.zero_grads();
				m.backward( L, x, R0, dx, T, pk, pv );
				m.router_kl();
				const std::vector<float> gr = down( dx, nx );
				double wr = 0.0, gmr = 0.0;
				for ( int s = 0; s < 12; ++s ) {
					// entries with a large gradient and random ones
					size_t i = std::uniform_int_distribution<size_t>( 0, nx - 1 )( rng );
					if ( s < 4 ) i = static_cast<size_t>( std::max_element( gr.begin(), gr.end(), []( float a, float b ) { return std::fabs( a ) < std::fabs( b ); } ) - gr.begin() ) + s;
					i = std::min( i, nx - 1 );
					const float x0 = hx[i], h = 1e-3f * std::max( 1.0f, std::fabs( x0 ) );
					set( x, i, x0 + h );
					const double lp = loss_r();
					set( x, i, x0 - h );
					const double lm = loss_r();
					set( x, i, x0 );
					const double fd = ( lp - lm ) / ( 2 * h );
					wr = std::max( wr, std::fabs( fd - gr[i] ) );
					gmr = std::max( gmr, std::fabs( fd ) );
				}
				std::snprintf( line, sizeof( line ), "router distillation alone (%s %.3g per row), input gradient: max error %.3g of max %.3g",
					margin > 0.0f ? "margins" : "KL", kl0, wr, gmr );
				check( gmr > 0.0 && wr <= 3e-2 * gmr + 1e-6, line );
			}
			m.set_router_margin( 0.0f );
			cudaFree( R0 );
			m.set_router_teacher( {}, 0.0f );
			for ( void* p : { (void*) d_tid, (void*) d_tp, (void*) d_tt } ) cudaFree( p );
		}
		{
			// the backward without a kept forward (the sublayers recomputed) gives the same input gradient
			m.forward( L, x, y, T, pk, pv );
			m.zero_grads();
			m.backward( L, x, R, dx, T, pk, pv );
			const std::vector<float> g2 = down( dx, nx );
			double d2 = 0.0, n2 = 0.0;
			for ( size_t i = 0; i < nx; ++i ) {
				d2 += ( g2[i] - g[i] ) * ( g2[i] - g[i] );
				n2 += g[i] * g[i];
			}
			std::snprintf( line, sizeof( line ), "input gradient, recomputed sublayers against kept ones: relative difference %.3g", std::sqrt( d2 / ( n2 + 1e-30 ) ) );
			check( std::sqrt( d2 / ( n2 + 1e-30 ) ) < 1e-5, line );
		}
		{
			// The tensor-core products (the training default) against the FP32 ones checked above, at the same input: the layer's change, the
			// input gradient and the picked tensors' scale gradients
			auto rel = []( const std::vector<float>& a, const std::vector<float>& b ) {
				double d = 0.0, n = 0.0;
				for ( size_t i = 0; i < a.size(); ++i ) {
					d += ( a[i] - b[i] ) * ( a[i] - b[i] );
					n += static_cast<double>( b[i] ) * b[i];
				}
				return std::sqrt( d / ( n + 1e-30 ) );
			};
			std::vector<std::vector<float>> s32;
			for ( q::Tuned* t : pick ) {
				std::vector<float> v = down( t->gu, t->k ), w = down( t->gv, t->n );
				v.insert( v.end(), w.begin(), w.end() );
				s32.push_back( std::move( v ) );
			}
			m.set_fp16( true );
			m.forward( L, x, y, T, pk, pv, true );
			const std::vector<float> y16 = down( y, nx );
			m.zero_grads();
			m.backward( L, x, R, dx, T, pk, pv );
			const std::vector<float> g16 = down( dx, nx );
			std::vector<float> c16( nx ), c32( nx );   // the layer's change (its output less the input streams)
			for ( size_t i = 0; i < nx; ++i ) {
				c16[i] = y16[i] - hx[i];
				c32[i] = y32[i] - hx[i];
			}
			double sw = 0.0;
			std::string at;
			for ( size_t j = 0; j < pick.size(); ++j ) {
				std::vector<float> v = down( pick[j]->gu, pick[j]->k ), w = down( pick[j]->gv, pick[j]->n );
				v.insert( v.end(), w.begin(), w.end() );
				const double e = rel( v, s32[j] );
				if ( e > sw ) {
					sw = e;
					at = pick[j]->name;
				}
			}
			std::snprintf( line, sizeof( line ), "FP16 products against FP32: the layer's change %.3g, input gradient %.3g, scale gradients up to %.3g", rel( c16, c32 ),
				rel( g16, g ), sw );
			check( rel( c16, c32 ) < 2e-3 && rel( g16, g ) < 1e-2 && sw < 2e-2, line );
			std::printf( "    (the worst scale gradient: %s)\n", at.c_str() );
		}
		for ( float* p : { x, y, R, dx, pk, pv } ) cudaFree( p );
		{
			// Two sequences of 100 rows (GDN chunks of 64: one whole and one part each) through one call against each alone: the layer's change,
			// the input gradient and every gradient of the tensors (the two alone accumulating)
			const int Ts = 100, Tb = 2 * Ts;
			q::Model mb;
			mb.set_train_dense( true );
			mb.set_train_router( true );
			mb.set_train_hc( true );
			mb.set_train_gains( true );
			if ( !mb.load( dir, L, L + 1, 0, Tb ) ) return 1;
			// FP32 products: with FP16 ones the roundings turn the libraries' other summation orders at other row counts into differences
			// of the FP16 path's own size
			mb.set_fp16( false );
			const size_t nb = static_cast<size_t>( Tb ) * flat, ns = static_cast<size_t>( Ts ) * flat, H = static_cast<size_t>( c.hidden );
			std::vector<float> bx( nb ), br( nb ), bk( nb ), bv( Tb * H );
			for ( auto& v : bx ) v = nd( rng );
			for ( auto& v : br ) v = nd( rng );
			for ( auto& v : bk ) v = nd( rng );
			for ( auto& v : bv ) v = 0.3f * nd( rng );
			float *xb, *yb, *Rb, *dxb, *pkb, *pvb;
			for ( float** p : { &xb, &yb, &Rb, &dxb, &pkb } ) CUDA_CHECK( cudaMalloc( p, nb * sizeof( float ) ) );
			CUDA_CHECK( cudaMalloc( &pvb, bv.size() * sizeof( float ) ) );
			CUDA_CHECK( cudaMemcpy( xb, bx.data(), nb * sizeof( float ), cudaMemcpyHostToDevice ) );
			CUDA_CHECK( cudaMemcpy( Rb, br.data(), nb * sizeof( float ), cudaMemcpyHostToDevice ) );
			CUDA_CHECK( cudaMemcpy( pkb, bk.data(), nb * sizeof( float ), cudaMemcpyHostToDevice ) );
			CUDA_CHECK( cudaMemcpy( pvb, bv.data(), bv.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
			CUDA_CHECK( cudaDeviceSynchronize() );
			auto grads = [&]() {
				std::vector<float> all;
				for ( q::Tuned* t : mb.tensors() ) {
					const std::vector<float> u = down( t->gu, t->k ), v = down( t->gv, t->n );
					all.insert( all.end(), u.begin(), u.end() );
					all.insert( all.end(), v.begin(), v.end() );
					if ( t->gtg ) {
						const std::vector<float> g2 = down( t->gtg, static_cast<size_t>( t->k / 16 ) * ( t->n / 16 ) );
						all.insert( all.end(), g2.begin(), g2.end() );
					}
				}
				for ( q::Dense* d : mb.dense_tensors() ) {
					const std::vector<float> g2 = down( d->g, d->n );
					all.insert( all.end(), g2.begin(), g2.end() );
				}
				return all;
			};
			auto rel = []( const std::vector<float>& a, const std::vector<float>& b ) {
				double d = 0.0, n = 0.0;
				for ( size_t i = 0; i < a.size(); ++i ) {
					d += ( a[i] - b[i] ) * ( a[i] - b[i] );
					n += static_cast<double>( b[i] ) * b[i];
				}
				return std::sqrt( d / ( n + 1e-30 ) );
			};
			mb.forward( L, xb, yb, Tb, pkb, pvb, true, 2 );
			std::vector<float> cb = down( yb, nb );
			mb.zero_grads();
			mb.backward( L, xb, Rb, dxb, Tb, pkb, pvb, 2 );
			const std::vector<float> gb = down( dxb, nb ), wb = grads();
			mb.zero_grads();
			for ( int s = 0; s < 2; ++s ) {
				const size_t o = s * ns, ov = static_cast<size_t>( s ) * Ts * H;
				mb.forward( L, xb + o, yb + o, Ts, pkb + o, pvb + ov, true );
				mb.backward( L, xb + o, Rb + o, dxb + o, Ts, pkb + o, pvb + ov );
			}
			std::vector<float> c1 = down( yb, nb );
			const std::vector<float> g1 = down( dxb, nb ), w1 = grads();
			for ( size_t i = 0; i < nb; ++i ) {
				cb[i] -= bx[i];
				c1[i] -= bx[i];
			}
			std::snprintf( line, sizeof( line ), "two sequences in one call against each alone: the layer's change %.3g, input gradient %.3g, weights' %.3g",
				rel( cb, c1 ), rel( gb, g1 ), rel( wb, w1 ) );
			check( rel( cb, c1 ) < 1e-5 && rel( gb, g1 ) < 1e-5 && rel( wb, w1 ) < 1e-5, line );
			for ( float* p : { xb, yb, Rb, dxb, pkb, pvb } ) cudaFree( p );
		}
	}
	std::printf( "[%c] test_train_qwen4_exp_layer: %d failure(s)\n", fails ? '-' : '+', fails );
	return fails ? 1 : 0;
}
