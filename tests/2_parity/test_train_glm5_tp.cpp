// Tier 2 (checkpoint): the tensor-parallel glm5_next training model (engine/train/glm5_next.h, Model over ranks with NCCL) against the
// single-device one on the same layers and inputs: a layer's output streams, its input gradient from sum( R * layer( x ) ) and the scale
// gradients (the shared tensors' summed over the ranks, the experts' from their rank) and the norm weights'; then the head on the last
// layer's range: the KL, the streams' gradient, lm_head's scale gradients and the final norm's.
//   TRAIN_MODEL_DIR=<EXL3 GLM-5.3-Flash checkpoint> test_train_glm5_tp [layer ...]   (default: 0, KDA + dense MLP; 3, MLA + MoE; 4, KDA + MoE;
//   all visible devices as the ranks)
#include "engine/config.h"
#include "engine/train/glm5_next.h"
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

// every tensor's gradients and the norm weights' (added into g: the shared tensors' partial sums of the ranks add up)
void grads( g5::Model& m, Grads& g ) {
	for ( g5::Tuned* t : m.tensors() ) {
		const std::vector<float> u = down( t->gu, t->k ), v = down( t->gv, t->n );
		auto& e = g[t->name];
		if ( e.first.empty() ) {
			e.first.assign( t->k, 0.0f );
			e.second.assign( t->n, 0.0f );
		}
		for ( int i = 0; i < t->k; ++i ) e.first[i] += u[i];
		for ( int i = 0; i < t->n; ++i ) e.second[i] += v[i];
	}
	for ( g5::Dense* d : m.dense_tensors() ) {
		const std::vector<float> u = down( d->g, d->n );
		auto& e = g[d->name];
		if ( e.first.empty() ) e.first.assign( d->n, 0.0f );
		for ( int i = 0; i < d->n; ++i ) e.first[i] += u[i];
	}
}

void add( const std::vector<Grads>& parts, Grads& sum ) {
	for ( const Grads& g : parts ) for ( const auto& kv : g ) {
		auto& e = sum[kv.first];
		if ( e.first.empty() ) e = kv.second;
		else {
			for ( size_t i = 0; i < e.first.size(); ++i ) e.first[i] += kv.second.first[i];
			for ( size_t i = 0; i < e.second.size(); ++i ) e.second[i] += kv.second.second[i];
		}
	}
}

// The worst relative difference over the tensors of a kind (name fragment) with a nonzero reference gradient
void compare_grads( const Grads& ref, const Grads& tp, const char* frag, const char* what, const char* skip = nullptr ) {
	double worst = 0.0;
	int n = 0;
	std::string at;
	for ( const auto& kv : ref ) {
		if ( kv.first.find( frag ) == std::string::npos || ( skip && kv.first.find( skip ) != std::string::npos ) ) continue;
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
	std::snprintf( line, sizeof( line ), "%s gradients (%d tensors): worst relative difference %.3g (%s)", what, n, worst, at.c_str() );
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
	if ( layers.empty() ) layers = { 0, 3, 4 };
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
	g5::Config c;
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
		std::printf( "(layer %d, %s + %s)\n", L, c.full_attention( L ) ? "MLA" : "KDA", c.moe( L ) ? "MoE" : "dense MLP" );
		const std::vector<float> hx = randv( static_cast<size_t>( T ) * flat, 1.0f ), hr = randv( static_cast<size_t>( T ) * flat, 1.0f );
		// the single-device reference
		std::vector<float> y_ref, dx_ref;
		Grads g_ref;
		{
			CUDA_CHECK( cudaSetDevice( 0 ) );
			g5::Model m;
			if ( !m.load( dir, L, L + 1, 0, T ) ) return 1;
			float *x = up( hx.data(), hx.size() ), *R = up( hr.data(), hr.size() ), *y, *dx;
			CUDA_CHECK( cudaMalloc( &y, hx.size() * sizeof( float ) ) );
			CUDA_CHECK( cudaMalloc( &dx, hx.size() * sizeof( float ) ) );
			m.forward( L, x, y, T, true );
			y_ref = down( y, hx.size() );
			m.zero_grads();
			m.backward( L, x, R, dx, T );
			dx_ref = down( dx, hx.size() );
			grads( m, g_ref );
			for ( float* p : { x, R, y, dx } ) cudaFree( p );
		}
		// the ranks, each its rows of the streams
		std::vector<float> y_tp( hx.size() ), dx_tp( hx.size() );
		std::vector<Grads> g_r( n );
		std::vector<std::thread> th;
		for ( int r = 0; r < n; ++r ) th.emplace_back( [&, r] {
			CUDA_CHECK( cudaSetDevice( r ) );
			g5::Model m;
			if ( !m.load( dir, L, L + 1, r, T, r, n, comms[r] ) ) std::abort();   // the other ranks would wait on it forever
			const size_t o = static_cast<size_t>( r ) * Tl * flat, cnt = static_cast<size_t>( Tl ) * flat;
			float *x = up( hx.data() + o, cnt ), *R = up( hr.data() + o, cnt ), *y, *dx;
			CUDA_CHECK( cudaMalloc( &y, cnt * sizeof( float ) ) );
			CUDA_CHECK( cudaMalloc( &dx, cnt * sizeof( float ) ) );
			m.forward( L, x, y, T, true );
			const std::vector<float> yr = down( y, cnt );
			std::copy( yr.begin(), yr.end(), y_tp.begin() + o );
			m.zero_grads();
			m.backward( L, x, R, dx, T );
			const std::vector<float> dr = down( dx, cnt );
			std::copy( dr.begin(), dr.end(), dx_tp.begin() + o );
			grads( m, g_r[r] );
			for ( float* p : { x, R, y, dx } ) cudaFree( p );
			CUDA_CHECK( cudaDeviceSynchronize() );
		} );
		for ( auto& t : th ) t.join();
		Grads g_tp;
		add( g_r, g_tp );
		char line[200];
		// (the tensor-core products round their inputs to FP16: a last-bit difference of the sharded products can flip one of them)
		std::snprintf( line, sizeof( line ), "output streams: relative difference %.3g", rel( y_tp, y_ref ) );
		check( rel( y_tp, y_ref ) < 1e-4, line );
		std::snprintf( line, sizeof( line ), "input gradient: relative difference %.3g", rel( dx_tp, dx_ref ) );
		check( rel( dx_tp, dx_ref ) < 1e-4, line );
		compare_grads( g_ref, g_tp, ".self_attn.", c.full_attention( L ) ? "MLA scale and norm" : "KDA scale and norm" );
		if ( c.moe( L ) ) {
			compare_grads( g_ref, g_tp, ".shared_experts.", "shared expert scale" );
			compare_grads( g_ref, g_tp, ".experts.", "routed experts' scale", ".shared_experts." );
		}
		else compare_grads( g_ref, g_tp, ".mlp.", "dense MLP scale" );
		compare_grads( g_ref, g_tp, "layernorm.weight", "site norm weight" );
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
		auto teacher = [&]( g5::Teacher& t ) {
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
			g5::Model m;
			if ( !m.load( dir, L, L + 1, 0, T ) ) return 1;
			g5::Teacher t;
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
			g5::Model m;
			if ( !m.load( dir, L, L + 1, r, T, r, n, comms[r] ) ) std::abort();
			g5::Teacher t;
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
		add( g_r, g_tp );
		char line[200];
		double kd = 0.0;
		for ( double k : kl_r ) kd = std::max( kd, std::fabs( k - kl_ref ) );
		std::snprintf( line, sizeof( line ), "KL %.6f (every rank within %.3g of the single device's %.6f)", kl_r[0], kd, kl_ref );
		check( kd < 1e-4 * std::max( 1.0, std::fabs( kl_ref ) ), line );
		std::snprintf( line, sizeof( line ), "streams' gradient: relative difference %.3g", rel( dx_tp, dx_ref ) );
		check( rel( dx_tp, dx_ref ) < 1e-4, line );
		compare_grads( g_ref, g_tp, "lm_head", "lm_head scale" );
		compare_grads( g_ref, g_tp, "model.language_model.norm.weight", "final norm weight" );
	}
	for ( ncclComm_t cm : comms ) ncclCommDestroy( cm );
	std::printf( "[%c] test_train_glm5_tp: %d failure(s)\n", fails ? '-' : '+', fails );
	return fails ? 1 : 0;
}
