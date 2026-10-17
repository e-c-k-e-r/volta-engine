// Feasibility of piggybacking decode rows on a prefill chunk (GLM): one forward whose rows are a causal chunk of slot 0 followed by single
// decode rows of other slots, against the same rows run as separate forwards. Reports the max |d logit| of every row and the divergence of a
// continued decode afterwards (the state each slot ended with).
//   test_mixed_rows [chunk=128] [decode_slots=1] [order=0]     (order 1 puts the decode rows first)
#include "engine/runtime/test_driver.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace engine::runtime;

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	const int chunk = argc > 1 ? std::atoi( argv[1] ) : 128;
	const int nd = argc > 2 ? std::atoi( argv[2] ) : 1;
	const int order = argc > 3 ? std::atoi( argv[3] ) : 0;
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = "glm5_next";
	cfg.model_dir = test::GLM_DIR;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	mc.warm_graphs( 8 );
	const int V = mc.metadata().vocab_size;
	const int pre = 200;                     // context each decoding slot already holds
	std::vector<int> ctx( pre ), body( chunk );
	uint32_t st = 777u;
	auto rnd = [&]() { st = st * 1664525u + 1013904223u; return 1000 + static_cast<int>( ( st >> 16 ) % 50000 ); };
	for ( int& t : ctx ) t = rnd();
	for ( int& t : body ) t = rnd();
	int dtok[4];
	for ( int i = 0; i < nd; ++i ) dtok[i] = rnd();

	auto setup = [&]() {
		for ( int s = 0; s <= nd; ++s ) mc.reset_slot( s );
		for ( int s = 1; s <= nd; ++s ) mc.prefill( ctx.data(), pre, s );
	};
	auto read = [&]( int rows ) {
		std::vector<half> h( static_cast<size_t>( rows ) * V );
		mc.read_logits( rows, h.data() );
		std::vector<float> f( h.size() );
		for ( size_t i = 0; i < f.size(); ++i ) f[i] = __half2float( h[i] );
		return f;
	};

	// Reference: the chunk alone (all rows' logits), then each decode row alone
	setup();
	std::vector<float> ref_chunk, ref_dec[4];
	{
		pod::BatchRows r = pod::BatchRows::causal( chunk, 0, 0 );
		mc.forward( body.data(), r );
		ref_chunk = read( chunk );
		for ( int i = 0; i < nd; ++i ) {
			mc.forward( &dtok[i], 1, pre, i + 1 );
			ref_dec[i] = read( 1 );
		}
	}
	// Mixed
	setup();
	std::vector<int> toks;
	pod::BatchRows mix;
	mix.count = chunk + nd;
	auto put = [&]( int row, int tok, int slot, int pos ) { toks.push_back( tok ); mix.slot[row] = slot; mix.pos[row] = pos; };
	if ( order == 0 ) {
		for ( int i = 0; i < chunk; ++i ) put( i, body[i], 0, i );
		for ( int i = 0; i < nd; ++i ) put( chunk + i, dtok[i], i + 1, pre );
	}
	else {
		for ( int i = 0; i < nd; ++i ) put( i, dtok[i], i + 1, pre );
		for ( int i = 0; i < chunk; ++i ) put( nd + i, body[i], 0, i );
	}
	mc.forward( toks.data(), mix );
	const std::vector<float> got = read( chunk + nd );
	auto maxd = [&]( const float* a, const float* b ) {
		double m = 0.0;
		for ( int i = 0; i < V; ++i ) m = std::max( m, static_cast<double>( std::fabs( a[i] - b[i] ) ) );
		return m;
	};
	const int c0 = order == 0 ? 0 : nd, d0 = order == 0 ? chunk : 0;
	double mc_max = 0.0;
	for ( int r = 0; r < chunk; ++r ) mc_max = std::max( mc_max, maxd( &got[static_cast<size_t>( c0 + r ) * V], &ref_chunk[static_cast<size_t>( r ) * V] ) );
	std::printf( "chunk rows (%d): max |d logit| vs the chunk alone = %.5f\n", chunk, mc_max );
	for ( int i = 0; i < nd; ++i )
		std::printf( "decode row %d: max |d logit| vs alone = %.5f\n", i, maxd( &got[static_cast<size_t>( d0 + i ) * V], ref_dec[i].data() ) );
	// State check: one more decode step on every slot, mixed-run state against the separately-run state is not available in one process, so
	// compare against a rerun of the separate path
	std::vector<float> cont_mix[5];
	{
		int t1[5];
		for ( int s = 0; s <= nd; ++s ) t1[s] = rnd();
		for ( int s = 0; s <= nd; ++s ) {
			mc.forward( &t1[s], 1, s == 0 ? chunk : pre + 1, s );
			cont_mix[s] = read( 1 );
		}
		setup();
		pod::BatchRows r = pod::BatchRows::causal( chunk, 0, 0 );
		mc.forward( body.data(), r );
		for ( int i = 0; i < nd; ++i ) mc.forward( &dtok[i], 1, pre, i + 1 );
		for ( int s = 0; s <= nd; ++s ) {
			mc.forward( &t1[s], 1, s == 0 ? chunk : pre + 1, s );
			const std::vector<float> f = read( 1 );
			std::printf( "continued decode, slot %d: max |d logit| mixed-state vs separate-state = %.5f\n", s, maxd( f.data(), cont_mix[s].data() ) );
		}
	}
	mc.shutdown();
	return 0;
}
