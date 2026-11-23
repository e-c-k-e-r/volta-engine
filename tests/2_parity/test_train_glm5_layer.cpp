// Tier 2 (checkpoint): one glm5_next training layer (engine/train/glm5_next.h, Model) of a GLM-5.3-Flash EXL3 checkpoint against central
// differences of its own forward: random input streams, the loss sum( R * layer( x ) ); the input gradient on sampled entries, the scale
// gradients of sampled channels (the routing held at the unperturbed input's selection) of the attention's, the MLP's or shared expert's and
// routed experts' tensors, and the norm weights' gradients. Then the backward after a kept forward against the recomputing one, the routed
// experts' codes staged from host memory against the codes on the device (bit for bit), the tensor-core products against the FP32 ones, and
// two sequences through one call against each alone.
//   TRAIN_MODEL_DIR=<EXL3 GLM-5.3-Flash checkpoint> test_train_glm5_layer [layer ...]   (default: 0, KDA + dense MLP; 3, MLA + MoE; 4, KDA + MoE)
#include "engine/config.h"
#include "engine/train/glm5_next.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

namespace g5 = engine::train::glm5_next;
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

double rel( const std::vector<float>& a, const std::vector<float>& b ) {
	double d = 0.0, n = 0.0;
	for ( size_t i = 0; i < b.size(); ++i ) {
		d += ( a[i] - b[i] ) * ( a[i] - b[i] );
		n += static_cast<double>( b[i] ) * b[i];
	}
	return std::sqrt( d / ( n + 1e-30 ) );
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
	if ( layers.empty() ) layers = { 0, 3, 4 };
	CUDA_CHECK( cudaSetDevice( 0 ) );
	std::mt19937 rng( 99 );
	std::normal_distribution<float> nd( 0.0f, 1.0f );
	const int T = 96;   // two of KDA's 64-row chunks
	for ( int L : layers ) {
		g5::Model m;
		if ( !m.load( dir, L, L + 1, 0, 2 * T ) ) {
			std::printf( "[-] cannot load layer %d of %s\n", L, dir );
			return 1;
		}
		// the central differences in FP32 products (FP16 rounding makes the forward step-like at the perturbations' scale); the tensor-core
		// products are compared against them at the end
		m.set_fp16( false );
		const g5::Config& c = m.config();
		const size_t flat = static_cast<size_t>( c.hc ) * c.hidden, nx = T * flat;
		std::printf( "(layer %d, %s + %s) %d rows\n", L, c.full_attention( L ) ? "MLA" : "KDA", c.moe( L ) ? "MoE" : "dense MLP", T );
		// small streams: the sublayers see them through norms (their scale does not matter there), and the residual they are added onto
		// then rounds at a scale below the perturbations' effect on the loss
		const float xs = 0.02f;
		std::vector<float> hx( 2 * nx ), hr( nx );
		for ( auto& v : hx ) v = xs * nd( rng );
		for ( auto& v : hr ) v = nd( rng );
		float *x, *y, *R, *dx;
		for ( float** p : { &x, &y, &dx } ) CUDA_CHECK( cudaMalloc( p, 2 * nx * sizeof( float ) ) );
		CUDA_CHECK( cudaMalloc( &R, nx * sizeof( float ) ) );
		CUDA_CHECK( cudaMemcpy( x, hx.data(), 2 * nx * sizeof( float ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( R, hr.data(), nx * sizeof( float ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaDeviceSynchronize() );
		auto loss = [&]() {
			m.forward( L, x, y, T );
			const std::vector<float> h = down( y, nx );
			double s = 0.0;
			for ( size_t i = 0; i < nx; ++i ) s += static_cast<double>( hr[i] ) * h[i];
			return s;
		};
		loss();   // the routing of this input, then held for every perturbation
		m.freeze_routing( true );
		const std::vector<float> y32 = down( y, nx );
		char line[240];
		{
			// the routed experts' codes on the device instead of staged from host memory: the same forward, bit for bit
			g5::Model m2;
			m2.set_expert_host( false );
			if ( !m2.load( dir, L, L + 1, 0, 2 * T ) ) return 1;
			m2.set_fp16( false );
			m2.forward( L, x, y, T );
			const bool same = std::memcmp( down( y, nx ).data(), y32.data(), nx * sizeof( float ) ) == 0;
			check( same, "forward with the experts' codes on the device against staged from host memory: bit for bit" );
		}
		// the backward after a kept forward (reusing its sublayer outputs), as the training runs it, against the recomputing one
		m.zero_grads();
		m.backward( L, x, R, dx, T );
		const std::vector<float> g_re = down( dx, nx );
		m.forward( L, x, y, T, true );
		m.zero_grads();
		m.backward( L, x, R, dx, T );
		const std::vector<float> g = down( dx, nx );
		std::snprintf( line, sizeof( line ), "input gradient after a kept forward against the recomputing backward: relative difference %.3g", rel( g, g_re ) );
		check( rel( g, g_re ) < 1e-6, line );
		// Input gradient on sampled entries
		{
			double worst = 0.0, gmax = 0.0;
			size_t wi = 0;
			// the differences at step h (and h / 4 for the worst entry: a kink within h shows as a change between the two)
			auto diff = [&]( size_t i, float h ) {
				const float x0 = hx[i];
				set( x, i, x0 + h );
				const double lp = loss();
				set( x, i, x0 - h );
				const double lm = loss();
				set( x, i, x0 );
				return ( lp - lm ) / ( 2 * h );
			};
			for ( int s = 0; s < 12; ++s ) {
				const size_t i = std::uniform_int_distribution<size_t>( 0, nx - 1 )( rng );
				const double fd = diff( i, 1e-2f * std::max( xs, std::fabs( hx[i] ) ) );
				if ( std::fabs( fd - g[i] ) > worst ) wi = i;
				worst = std::max( worst, std::fabs( fd - g[i] ) );
				gmax = std::max( gmax, std::fabs( fd ) );
			}
			// (an entry's loss change sits near the FP32 rounding of the deep layers' outputs: 5% of the largest)
			std::snprintf( line, sizeof( line ), "input gradient, 12 entries: max error %.3g of max %.3g", worst, gmax );
			if ( worst > 5e-2 * gmax + 1e-3 )
				std::printf( "    (worst: entry %zu, analytic %.6g, differences %.6g at h, %.6g at h / 4)\n", wi, g[wi],
					diff( wi, 1e-2f * std::max( xs, std::fabs( hx[wi] ) ) ), diff( wi, 2.5e-3f * std::max( xs, std::fabs( hx[wi] ) ) ) );
			check( worst <= 5e-2 * gmax + 1e-3, line );
			// along the gradient (every entry moved, at most 1% of the streams' scale): the loss change against |g|^2 times the step
			double gmx = 0.0, g2 = 0.0;
			for ( float v : g ) {
				gmx = std::max( gmx, static_cast<double>( std::fabs( v ) ) );
				g2 += static_cast<double>( v ) * v;
			}
			std::vector<float> xp( hx.begin(), hx.begin() + nx );
			auto along = [&]( double eps ) {
				auto at = [&]( double sgn ) {
					for ( size_t i = 0; i < nx; ++i ) xp[i] = static_cast<float>( hx[i] + sgn * eps * g[i] );
					CUDA_CHECK( cudaMemcpy( x, xp.data(), nx * sizeof( float ), cudaMemcpyHostToDevice ) );
					CUDA_CHECK( cudaDeviceSynchronize() );
					return loss();
				};
				return ( at( 1.0 ) - at( -1.0 ) ) / ( 2 * eps );
			};
			// (the SwiGLU clamps' kinks crossed along the way bias the differences in proportion to the step: the deep layers' activations
			// reach them often, so the smaller step decides)
			const double eps = 1e-2 * xs / gmx, dl = along( eps ), dl4 = along( eps / 4 );
			CUDA_CHECK( cudaMemcpy( x, hx.data(), nx * sizeof( float ), cudaMemcpyHostToDevice ) );
			CUDA_CHECK( cudaDeviceSynchronize() );
			std::snprintf( line, sizeof( line ), "input gradient along itself: differences %.6g (%.6g at a quarter of the step), |g|^2 %.6g", dl, dl4, g2 );
			check( std::fabs( dl4 - g2 ) <= 1e-2 * g2, line );
		}
		// Scale gradients: every tensor but the routed experts, of those per projection the ones with the largest input-scale gradient
		std::vector<g5::Tuned*> ts = m.tensors(), pick;
		for ( g5::Tuned* t : ts ) if ( t->name.find( ".experts." ) == std::string::npos ) pick.push_back( t );
		for ( const char* kind : { "gate_proj", "up_proj", "down_proj" } ) {
			std::vector<std::pair<double, g5::Tuned*>> ex;
			for ( g5::Tuned* t : ts ) {
				if ( t->name.find( ".experts." ) == std::string::npos || t->name.find( ".shared_experts." ) != std::string::npos ||
					t->name.find( kind ) == std::string::npos )
					continue;
				const std::vector<float> gu = down( t->gu, t->k );
				double n2 = 0.0;
				for ( float v : gu ) n2 += v * v;
				ex.push_back( { n2, t } );
			}
			std::sort( ex.begin(), ex.end(), []( const auto& a, const auto& b ) { return a.first > b.first; } );
			for ( size_t i = 0; i < std::min<size_t>( kind[0] == 'g' ? 2 : 1, ex.size() ); ++i ) pick.push_back( ex[i].second );
		}
		for ( g5::Tuned* t : pick ) {
			const std::vector<float> gu = down( t->gu, t->k ), gv = down( t->gv, t->n );
			double w = 0.0, gm = 0.0;
			char at[160] = "";
			for ( int s = 0; s < 4; ++s ) {
				const bool in = s < 2;
				const std::vector<float>& gg = in ? gu : gv;
				const int nch = in ? t->k : t->n;
				// the channel with the largest gradient, and a random one, of each side
				int ch = std::uniform_int_distribution<int>( 0, nch - 1 )( rng );
				if ( s % 2 == 0 ) ch = static_cast<int>( std::max_element( gg.begin(), gg.end(), []( float a, float b ) { return std::fabs( a ) < std::fabs( b ); } ) - gg.begin() );
				float* sc = in ? t->su : t->sv;
				const float s0 = get( sc, ch ), h = 1e-2f * std::max( 1e-3f, std::fabs( s0 ) );
				set( sc, ch, s0 + h );
				const double lp = loss();
				set( sc, ch, s0 - h );
				const double lm = loss();
				set( sc, ch, s0 );
				const double fd = ( lp - lm ) / ( 2 * h );
				if ( std::fabs( fd - gg[ch] ) > w )
					std::snprintf( at, sizeof( at ), "%s %d: s %.4g, differences %.6g, analytic %.6g", in ? "su" : "sv", ch, s0, fd, gg[ch] );
				w = std::max( w, std::fabs( fd - gg[ch] ) );
				gm = std::max( gm, std::fabs( fd ) );
			}
			std::snprintf( line, sizeof( line ), "%-60s scales: error %.3g of %.3g", t->name.c_str(), w, gm );
			if ( w > 3e-2 * gm + 1e-4 ) std::printf( "    (worst: %s)\n", at );
			check( w <= 3e-2 * gm + 1e-4, line );
		}
		// The norm weights: the entry with the largest gradient and a random one each
		for ( g5::Dense* d : m.dense_tensors() ) {
			const std::vector<float> gd = down( d->g, d->n );
			double w = 0.0, gm = 0.0;
			char at[160] = "";
			for ( int s = 0; s < 2; ++s ) {
				int i = std::uniform_int_distribution<int>( 0, d->n - 1 )( rng );
				if ( s == 0 ) i = static_cast<int>( std::max_element( gd.begin(), gd.end(), []( float a, float b ) { return std::fabs( a ) < std::fabs( b ); } ) - gd.begin() );
				const float w0 = get( d->w, i ), h = 1e-2f * std::max( 1e-2f, std::fabs( w0 ) );
				set( d->w, i, w0 + h );
				const double lp = loss();
				set( d->w, i, w0 - h );
				const double lm = loss();
				set( d->w, i, w0 );
				const double fd = ( lp - lm ) / ( 2 * h );
				if ( std::fabs( fd - gd[i] ) > w ) std::snprintf( at, sizeof( at ), "%d: w %.4g, differences %.6g, analytic %.6g", i, w0, fd, gd[i] );
				w = std::max( w, std::fabs( fd - gd[i] ) );
				gm = std::max( gm, std::fabs( fd ) );
			}
			std::snprintf( line, sizeof( line ), "%-60s weights: error %.3g of %.3g", d->name.c_str(), w, gm );
			if ( w > 3e-2 * gm + 1e-4 ) std::printf( "    (worst: %s)\n", at );
			check( w <= 3e-2 * gm + 1e-4, line );
		}
		// The tensor-core products against the FP32 ones
		{
			m.set_fp16( true );
			m.forward( L, x, y, T );
			const std::vector<float> y16 = down( y, nx );
			m.set_fp16( false );
			std::vector<float> d16( nx ), d32( nx );
			for ( size_t i = 0; i < nx; ++i ) {
				d16[i] = y16[i] - hx[i];
				d32[i] = y32[i] - hx[i];
			}
			std::snprintf( line, sizeof( line ), "tensor-core products: the layer's change against the FP32 products' %.3g", rel( d16, d32 ) );
			check( rel( d16, d32 ) < 2e-2, line );
		}
		// Two sequences through one call against each alone (the second sequence's rows after the first's)
		{
			m.freeze_routing( false );
			m.forward( L, x, y, 2 * T, false, 2 );
			const std::vector<float> both = down( y, 2 * nx );
			m.forward( L, x, y, T );
			std::vector<float> each = down( y, nx );
			m.forward( L, x + nx, y, T );
			const std::vector<float> second = down( y, nx );
			each.insert( each.end(), second.begin(), second.end() );
			std::snprintf( line, sizeof( line ), "two sequences in one call against each alone: relative difference %.3g", rel( both, each ) );
			check( rel( both, each ) < 1e-5, line );
			// and the backward: the input gradient and every tensor's gradients of one call over both against the two alone (their sums)
			std::vector<float> hr2( 2 * nx );
			for ( auto& v : hr2 ) v = nd( rng );
			float* R2;
			CUDA_CHECK( cudaMalloc( &R2, 2 * nx * sizeof( float ) ) );
			CUDA_CHECK( cudaMemcpy( R2, hr2.data(), 2 * nx * sizeof( float ), cudaMemcpyHostToDevice ) );
			CUDA_CHECK( cudaDeviceSynchronize() );
			auto all_grads = [&]() {
				std::vector<float> v;
				for ( g5::Tuned* t : m.tensors() ) {
					const std::vector<float> u = down( t->gu, t->k ), w = down( t->gv, t->n );
					v.insert( v.end(), u.begin(), u.end() );
					v.insert( v.end(), w.begin(), w.end() );
				}
				for ( g5::Dense* d : m.dense_tensors() ) {
					const std::vector<float> u = down( d->g, d->n );
					v.insert( v.end(), u.begin(), u.end() );
				}
				return v;
			};
			m.forward( L, x, y, 2 * T, true, 2 );
			m.zero_grads();
			m.backward( L, x, R2, dx, 2 * T, 2 );
			const std::vector<float> dx_both = down( dx, 2 * nx ), g_both = all_grads();
			m.zero_grads();
			m.forward( L, x, y, T, true );
			m.backward( L, x, R2, dx, T );
			std::vector<float> dx_each = down( dx, nx );
			m.forward( L, x + nx, y, T, true );
			m.backward( L, x + nx, R2 + nx, dx, T );
			const std::vector<float> dx_second = down( dx, nx ), g_each = all_grads();
			dx_each.insert( dx_each.end(), dx_second.begin(), dx_second.end() );
			std::snprintf( line, sizeof( line ), "two sequences in one call against each alone: input gradient %.3g, every tensor's gradients %.3g", rel( dx_both, dx_each ),
				rel( g_both, g_each ) );
			check( rel( dx_both, dx_each ) < 1e-4 && rel( g_both, g_each ) < 1e-4, line );
			cudaFree( R2 );
		}
		if ( m.has_head() ) {
			// The head (the streams' mean, the final norm, lm_head, the KL to a random teacher of the top 16 and a tail) against central
			// differences: the streams' gradient, lm_head's scales and the final norm's weights
			const int tk = 16;
			std::vector<int> ids( static_cast<size_t>( T ) * tk );
			std::vector<float> tp( ids.size() ), tt( T );
			std::uniform_int_distribution<int> ud( 0, c.vocab - 1 );
			for ( int r = 0; r < T; ++r ) {
				double s = 0.0;
				for ( int j = 0; j < tk; ++j ) {
					ids[static_cast<size_t>( r ) * tk + j] = ( ud( rng ) / tk ) * tk + j;   // distinct per row
					s += tp[static_cast<size_t>( r ) * tk + j] = std::exp( nd( rng ) );
				}
				tt[r] = 0.2f;
				for ( int j = 0; j < tk; ++j ) tp[static_cast<size_t>( r ) * tk + j] = static_cast<float>( 0.8 * tp[static_cast<size_t>( r ) * tk + j] / s );
			}
			int* d_ids;
			float *d_p, *d_tail;
			CUDA_CHECK( cudaMalloc( &d_ids, ids.size() * sizeof( int ) ) );
			CUDA_CHECK( cudaMalloc( &d_p, tp.size() * sizeof( float ) ) );
			CUDA_CHECK( cudaMalloc( &d_tail, tt.size() * sizeof( float ) ) );
			CUDA_CHECK( cudaMemcpy( d_ids, ids.data(), ids.size() * sizeof( int ), cudaMemcpyHostToDevice ) );
			CUDA_CHECK( cudaMemcpy( d_p, tp.data(), tp.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
			CUDA_CHECK( cudaMemcpy( d_tail, tt.data(), tt.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
			CUDA_CHECK( cudaDeviceSynchronize() );
			g5::Teacher t;
			t.ids = d_ids;
			t.p = d_p;
			t.tail = d_tail;
			t.tk = tk;
			// streams of the model's own scale (the head's norm makes the KL depend on their direction only)
			std::vector<float> hs( nx );
			for ( auto& v : hs ) v = nd( rng );
			CUDA_CHECK( cudaMemcpy( x, hs.data(), nx * sizeof( float ), cudaMemcpyHostToDevice ) );
			CUDA_CHECK( cudaDeviceSynchronize() );
			m.zero_grads();
			const double kl0 = m.head( x, T, t, dx );
			const std::vector<float> gx = down( dx, nx );
			auto kl = [&]() { return m.head( x, T, t, nullptr ); };
			double w = 0.0, gm = 0.0;
			for ( int s = 0; s < 12; ++s ) {
				const size_t i = std::uniform_int_distribution<size_t>( 0, nx - 1 )( rng );
				const float x0 = hs[i], h = 1e-2f * std::max( 1.0f, std::fabs( x0 ) );
				set( x, i, x0 + h );
				const double lp = kl();
				set( x, i, x0 - h );
				const double lm = kl();
				set( x, i, x0 );
				const double fd = ( lp - lm ) / ( 2 * h );
				w = std::max( w, std::fabs( fd - gx[i] ) );
				gm = std::max( gm, std::fabs( fd ) );
			}
			std::snprintf( line, sizeof( line ), "head (KL %.4g per row): streams' gradient, 12 entries: max error %.3g of max %.3g", kl0 / T, w, gm );
			check( w <= 3e-2 * gm + 1e-4, line );
			for ( g5::Tuned* tl : m.tensors() ) {
				if ( tl->name != "lm_head" ) continue;
				const std::vector<float> gu = down( tl->gu, tl->k ), gv = down( tl->gv, tl->n );
				double ws = 0.0, gms = 0.0;
				for ( int s = 0; s < 4; ++s ) {
					const bool in = s < 2;
					const std::vector<float>& gg = in ? gu : gv;
					int ch = std::uniform_int_distribution<int>( 0, ( in ? tl->k : tl->n ) - 1 )( rng );
					if ( s % 2 == 0 ) ch = static_cast<int>( std::max_element( gg.begin(), gg.end(), []( float a, float b ) { return std::fabs( a ) < std::fabs( b ); } ) - gg.begin() );
					float* sc = in ? tl->su : tl->sv;
					const float s0 = get( sc, ch ), h = 1e-2f * std::max( 1e-3f, std::fabs( s0 ) );
					set( sc, ch, s0 + h );
					const double lp = kl();
					set( sc, ch, s0 - h );
					const double lm = kl();
					set( sc, ch, s0 );
					const double fd = ( lp - lm ) / ( 2 * h );
					ws = std::max( ws, std::fabs( fd - gg[ch] ) );
					gms = std::max( gms, std::fabs( fd ) );
				}
				std::snprintf( line, sizeof( line ), "head: lm_head scales: error %.3g of %.3g", ws, gms );
				check( ws <= 3e-2 * gms + 1e-4, line );
			}
			for ( g5::Dense* d : m.dense_tensors() ) {
				if ( d->name != "model.language_model.norm.weight" ) continue;
				const std::vector<float> gd = down( d->g, d->n );
				double wn = 0.0, gmn = 0.0;
				for ( int s = 0; s < 3; ++s ) {
					int i = std::uniform_int_distribution<int>( 0, d->n - 1 )( rng );
					if ( s == 0 ) i = static_cast<int>( std::max_element( gd.begin(), gd.end(), []( float a, float b ) { return std::fabs( a ) < std::fabs( b ); } ) - gd.begin() );
					const float w0 = get( d->w, i ), h = 1e-2f * std::max( 1e-2f, std::fabs( w0 ) );
					set( d->w, i, w0 + h );
					const double lp = kl();
					set( d->w, i, w0 - h );
					const double lm = kl();
					set( d->w, i, w0 );
					const double fd = ( lp - lm ) / ( 2 * h );
					wn = std::max( wn, std::fabs( fd - gd[i] ) );
					gmn = std::max( gmn, std::fabs( fd ) );
				}
				std::snprintf( line, sizeof( line ), "head: final norm weights: error %.3g of %.3g", wn, gmn );
				check( wn <= 3e-2 * gmn + 1e-4, line );
			}
			for ( void* p : { (void*) d_ids, (void*) d_p, (void*) d_tail } ) cudaFree( p );
		}
		for ( float* p : { x, y, R, dx } ) cudaFree( p );
	}
	std::printf( "[%c] test_train_glm5_layer: %d failure(s)\n", fails ? '-' : '+', fails );
	return fails ? 1 : 0;
}
