// Tier 2 (checkpoint): the tensor-parallel qwen4_exp training model (engine/train/qwen4_exp.h, Model over ranks with NCCL) against the
// single-device one on the same layers and inputs: a layer's output streams, its input gradient from sum( R * layer( x ) ) and the scale
// gradients (the shared tensors' summed over the ranks, the experts' from their rank), the norm weights' and the small dense tensors' (all
// trainable); then the head on the last layer's range: the KL,
// the streams' gradient and lm_head's scale gradients.
//   TRAIN_MODEL_DIR=<EXL3 Flash-Next checkpoint> test_train_qwen4_exp_tp [layer ...]   (default: 1, the PLE + GDN layer, and 3, attention;
//   all visible devices as the ranks)
#include "engine/config.h"
#include "engine/train/qwen4_exp.h"
#include <cuda_runtime.h>
#include <nccl.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
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

float* up( const float* h, size_t n ) {
	float* d = nullptr;
	CUDA_CHECK( cudaMalloc( &d, n * sizeof( float ) ) );
	CUDA_CHECK( cudaMemcpy( d, h, n * sizeof( float ), cudaMemcpyHostToDevice ) );
	// a copy from pageable memory may return before its DMA lands, and the model's stream does not wait for the default stream
	CUDA_CHECK( cudaDeviceSynchronize() );
	return d;
}

double rel( const std::vector<float>& a, const std::vector<float>& b ) {
	double d = 0.0, n = 0.0;
	for ( size_t i = 0; i < b.size(); ++i ) {
		d += ( a[i] - b[i] ) * ( a[i] - b[i] );
		n += static_cast<double>( b[i] ) * b[i];
	}
	return std::sqrt( d / ( n + 1e-30 ) );
}

using Grads = std::map<std::string, std::pair<std::vector<float>, std::vector<float>>>;

// every tensor's gradients (added into g: the shared tensors' partial sums of the ranks add up)
void grads( q::Model& m, Grads& g ) {
	for ( q::Tuned* t : m.tensors() ) {
		const std::vector<float> u = down( t->gu, t->k ), v = down( t->gv, t->n );
		auto& e = g[t->name];
		if ( e.first.empty() ) {
			e.first.assign( t->k, 0.0f );
			e.second.assign( t->n, 0.0f );
		}
		for ( int i = 0; i < t->k; ++i ) e.first[i] += u[i];
		for ( int i = 0; i < t->n; ++i ) e.second[i] += v[i];
		if ( t->gtg ) {
			const size_t nt = static_cast<size_t>( t->k / 16 ) * ( t->n / 16 );
			const std::vector<float> w = down( t->gtg, nt );
			auto& eg = g[t->name + ".tgain"];
			if ( eg.first.empty() ) eg.first.assign( nt, 0.0f );
			for ( size_t i = 0; i < nt; ++i ) eg.first[i] += w[i];
		}
	}
	// the norm weights (every rank holds them all; its partial sum, rank 0 alone for the PLE layer's)
	for ( q::Dense* d : m.dense_tensors() ) {
		const std::vector<float> u = down( d->g, d->n );
		auto& e = g[d->name];
		if ( e.first.empty() ) e.first.assign( d->n, 0.0f );
		for ( int i = 0; i < d->n; ++i ) e.first[i] += u[i];
	}
}

// The worst relative difference over the tensors of a kind (name fragment) with a nonzero reference gradient
void compare_grads( const Grads& ref, const Grads& tp, const char* frag, const char* what ) {
	double worst = 0.0;
	int n = 0;
	std::string at;
	for ( const auto& kv : ref ) {
		if ( kv.first.find( frag ) == std::string::npos ) continue;
		const auto it = tp.find( kv.first );
		if ( it == tp.end() ) {
			worst = 1e30;
			at = kv.first + " (missing)";
			continue;
		}
		double nr = 0.0;
		for ( float v : kv.second.first ) nr += v * v;
		if ( nr == 0.0 ) continue;
		const double e = std::max( rel( it->second.first, kv.second.first ), rel( it->second.second, kv.second.second ) );
		if ( e > worst ) {
			worst = e;
			at = kv.first;
		}
		++n;
	}
	char line[240];
	std::snprintf( line, sizeof( line ), "%s scale gradients (%d tensors): worst relative difference %.3g (%s)", what, n, worst, at.c_str() );
	check( n > 0 && worst < 2e-3, line );
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
	int n = 0;
	CUDA_CHECK( cudaGetDeviceCount( &n ) );
	if ( n < 2 ) {
		std::printf( "[-] needs two devices or more\n" );
		return 2;
	}
	std::vector<int> devs( n );
	for ( int r = 0; r < n; ++r ) devs[r] = r;
	std::vector<ncclComm_t> comms( n );
	if ( ncclCommInitAll( comms.data(), n, devs.data() ) != ncclSuccess ) {
		std::printf( "[-] NCCL init failed\n" );
		return 1;
	}
	q::Config c;
	if ( !c.read( dir ) ) return 1;
	const int T = 32 * n, Tl = T / n, H = c.hidden, flat = c.hc * H;
	std::mt19937 rng( 7 );
	std::normal_distribution<float> nd( 0.0f, 1.0f );
	auto randv = [&]( size_t cnt, float s ) {
		std::vector<float> v( cnt );
		for ( float& x : v ) x = s * nd( rng );
		return v;
	};
	std::printf( "%d ranks, %d rows\n", n, T );
	for ( int L : layers ) {
		std::printf( "(layer %d, %s%s)\n", L, c.full_attention( L ) ? "attention" : "GDN", L == c.ple_layer ? " + PLE" : "" );
		const std::vector<float> hx = randv( static_cast<size_t>( T ) * flat, 1.0f ), hr = randv( static_cast<size_t>( T ) * flat, 1.0f ),
			hk = randv( static_cast<size_t>( T ) * flat, 1.0f ), hv = randv( static_cast<size_t>( T ) * H, 0.3f );
		// the single-device reference
		std::vector<float> y_ref, dx_ref;
		Grads g_ref;
		{
			CUDA_CHECK( cudaSetDevice( 0 ) );
			q::Model m;
			m.set_train_dense( true );
			m.set_train_router( true );
			m.set_train_hc( true );
			m.set_train_gains( true );
			if ( !m.load( dir, L, L + 1, 0, T ) ) return 1;
			float *x = up( hx.data(), hx.size() ), *R = up( hr.data(), hr.size() ), *pk = up( hk.data(), hk.size() ), *pv = up( hv.data(), hv.size() ), *y, *dx;
			CUDA_CHECK( cudaMalloc( &y, hx.size() * sizeof( float ) ) );
			CUDA_CHECK( cudaMalloc( &dx, hx.size() * sizeof( float ) ) );
			m.forward( L, x, y, T, pk, pv, true );
			y_ref = down( y, hx.size() );
			m.zero_grads();
			m.backward( L, x, R, dx, T, pk, pv );
			dx_ref = down( dx, hx.size() );
			grads( m, g_ref );
			for ( float* p : { x, R, pk, pv, y, dx } ) cudaFree( p );
		}
		// the ranks, each its rows of the streams
		std::vector<float> y_tp( hx.size() ), dx_tp( hx.size() );
		Grads g_tp;
		std::vector<Grads> g_r( n );
		std::vector<int> ok( n, 1 );
		std::vector<std::thread> th;
		for ( int r = 0; r < n; ++r ) th.emplace_back( [&, r] {
			CUDA_CHECK( cudaSetDevice( r ) );
			q::Model m;
			m.set_train_dense( true );
			m.set_train_router( true );
			m.set_train_hc( true );
			m.set_train_gains( true );
			ok[r] = m.load( dir, L, L + 1, r, T, r, n, comms[r] );
			if ( !ok[r] ) std::abort();   // the other ranks would wait on it forever
			const size_t o = static_cast<size_t>( r ) * Tl * flat, cnt = static_cast<size_t>( Tl ) * flat;
			float *x = up( hx.data() + o, cnt ), *R = up( hr.data() + o, cnt ), *pk = up( hk.data(), hk.size() ), *pv = up( hv.data(), hv.size() ), *y, *dx;
			CUDA_CHECK( cudaMalloc( &y, cnt * sizeof( float ) ) );
			CUDA_CHECK( cudaMalloc( &dx, cnt * sizeof( float ) ) );
			m.forward( L, x, y, T, pk, pv, true );
			const std::vector<float> yr = down( y, cnt );
			std::copy( yr.begin(), yr.end(), y_tp.begin() + o );
			m.zero_grads();
			m.backward( L, x, R, dx, T, pk, pv );
			const std::vector<float> dr = down( dx, cnt );
			std::copy( dr.begin(), dr.end(), dx_tp.begin() + o );
			grads( m, g_r[r] );
			for ( float* p : { x, R, pk, pv, y, dx } ) cudaFree( p );
			CUDA_CHECK( cudaDeviceSynchronize() );
		} );
		for ( auto& t : th ) t.join();
		for ( const Grads& g : g_r ) for ( const auto& kv : g ) {
			auto& e = g_tp[kv.first];
			if ( e.first.empty() ) e = kv.second;
			else {
				for ( size_t i = 0; i < e.first.size(); ++i ) e.first[i] += kv.second.first[i];
				for ( size_t i = 0; i < e.second.size(); ++i ) e.second[i] += kv.second.second[i];
			}
		}
		char line[200];
		// (the tensor-core products round their inputs to FP16: a last-bit difference of the sharded products can flip one of them)
		std::snprintf( line, sizeof( line ), "output streams: relative difference %.3g", rel( y_tp, y_ref ) );
		check( rel( y_tp, y_ref ) < 1e-4, line );
		std::snprintf( line, sizeof( line ), "input gradient: relative difference %.3g", rel( dx_tp, dx_ref ) );
		check( rel( dx_tp, dx_ref ) < 1e-4, line );
		compare_grads( g_ref, g_tp, c.full_attention( L ) ? ".self_attn." : ".linear_attn.", c.full_attention( L ) ? "attention" : "GDN" );
		compare_grads( g_ref, g_tp, ".shared_expert.", "shared expert" );
		compare_grads( g_ref, g_tp, ".experts.", "routed experts" );
		compare_grads( g_ref, g_tp, ".tgain", "routed experts' tile gain" );
		compare_grads( g_ref, g_tp, "norm", "norm weight" );
		compare_grads( g_ref, g_tp, "block_inject", "inject mix" );
		compare_grads( g_ref, g_tp, "input_mix_weight", "hyper-connection down / up mix" );
		compare_grads( g_ref, g_tp, "shared_expert_gate", "shared expert gate" );
		compare_grads( g_ref, g_tp, "mlp.gate.weight", "router" );
		if ( L == c.ple_layer ) compare_grads( g_ref, g_tp, "ple.conv1d", "PLE convolution" );
	}
	{
		// The head on the last layer's range
		const int L = c.layers - 1;
		std::printf( "(head)\n" );
		const std::vector<float> hx = randv( static_cast<size_t>( T ) * flat, 1.0f );
		const int tk = 16;
		std::vector<int> ids( static_cast<size_t>( T ) * tk );
		std::vector<float> p( ids.size() ), tail( T );
		std::uniform_int_distribution<int> ud( 0, c.vocab - 1 );
		for ( int r = 0; r < T; ++r ) {
			double s = 0.0;
			for ( int j = 0; j < tk; ++j ) {
				ids[static_cast<size_t>( r ) * tk + j] = ( ud( rng ) / tk ) * tk + j;   // distinct per row
				p[static_cast<size_t>( r ) * tk + j] = std::exp( nd( rng ) );
				s += p[static_cast<size_t>( r ) * tk + j];
			}
			tail[r] = 0.2f;
			for ( int j = 0; j < tk; ++j ) p[static_cast<size_t>( r ) * tk + j] = static_cast<float>( 0.8 * p[static_cast<size_t>( r ) * tk + j] / s );
		}
		auto teacher = [&]( q::Teacher& t ) {
			int* di = nullptr;
			CUDA_CHECK( cudaMalloc( &di, ids.size() * sizeof( int ) ) );
			CUDA_CHECK( cudaMemcpy( di, ids.data(), ids.size() * sizeof( int ), cudaMemcpyHostToDevice ) );
			t.ids = di;
			t.p = up( p.data(), p.size() );
			t.tail = up( tail.data(), tail.size() );
			t.tk = tk;
		};
		double kl_ref = 0.0;
		std::vector<float> dx_ref;
		Grads g_ref;
		{
			CUDA_CHECK( cudaSetDevice( 0 ) );
			q::Model m;
			if ( !m.load( dir, L, L + 1, 0, T ) ) return 1;
			q::Teacher t;
			teacher( t );
			float *x = up( hx.data(), hx.size() ), *dx;
			CUDA_CHECK( cudaMalloc( &dx, hx.size() * sizeof( float ) ) );
			m.zero_grads();
			kl_ref = m.head( x, T, t, dx );
			dx_ref = down( dx, hx.size() );
			grads( m, g_ref );
		}
		std::vector<double> kl_r( n );
		std::vector<float> dx_tp( hx.size() );
		std::vector<Grads> g_r( n );
		std::vector<std::thread> th;
		for ( int r = 0; r < n; ++r ) th.emplace_back( [&, r] {
			CUDA_CHECK( cudaSetDevice( r ) );
			q::Model m;
			if ( !m.load( dir, L, L + 1, r, T, r, n, comms[r] ) ) std::abort();
			q::Teacher t;
			teacher( t );
			const size_t o = static_cast<size_t>( r ) * Tl * flat, cnt = static_cast<size_t>( Tl ) * flat;
			float *x = up( hx.data() + o, cnt ), *dx;
			CUDA_CHECK( cudaMalloc( &dx, cnt * sizeof( float ) ) );
			m.zero_grads();
			kl_r[r] = m.head( x, T, t, dx );
			const std::vector<float> dr = down( dx, cnt );
			std::copy( dr.begin(), dr.end(), dx_tp.begin() + o );
			grads( m, g_r[r] );
			CUDA_CHECK( cudaDeviceSynchronize() );
		} );
		for ( auto& t : th ) t.join();
		Grads g_tp;
		for ( const Grads& g : g_r ) for ( const auto& kv : g ) {
			auto& e = g_tp[kv.first];
			if ( e.first.empty() ) e = kv.second;
			else {
				for ( size_t i = 0; i < e.first.size(); ++i ) e.first[i] += kv.second.first[i];
				for ( size_t i = 0; i < e.second.size(); ++i ) e.second[i] += kv.second.second[i];
			}
		}
		char line[200];
		double kd = 0.0;
		for ( double k : kl_r ) kd = std::max( kd, std::fabs( k - kl_ref ) );
		std::snprintf( line, sizeof( line ), "KL %.6f (every rank within %.3g of the single device's %.6f)", kl_r[0], kd, kl_ref );
		check( kd < 1e-4 * std::max( 1.0, std::fabs( kl_ref ) ), line );
		std::snprintf( line, sizeof( line ), "streams' gradient: relative difference %.3g", rel( dx_tp, dx_ref ) );
		check( rel( dx_tp, dx_ref ) < 1e-4, line );
		compare_grads( g_ref, g_tp, "lm_head", "lm_head" );
		compare_grads( g_ref, g_tp, "hyper_connection_mixer.hc_norm", "final mixer norm" );
	}
	for ( ncclComm_t cm : comms ) ncclCommDestroy( cm );
	std::printf( "[%c] test_train_qwen4_exp_tp: %d failure(s)\n", fails ? '-' : '+', fails );
	return fails ? 1 : 0;
}
