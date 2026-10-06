// Row parity: an M-row forward (Mode A causal block / Mode B batched slots) must reproduce
// the logits of the equivalent sequence of M=1 forwards.
#include "engine/runtime/test_driver.h"
#include <cmath>
#include <cstring>
#include <vector>

using namespace engine::runtime;

namespace impl {

std::vector<float> logits_row( ModelCluster& mc, int row ) {
	const int V = mc.metadata().vocab_size;
	std::vector<half> h( V );
	CUDA_CHECK( cudaSetDevice( 0 ) );
	CUDA_CHECK( cudaMemcpy( h.data(), mc.model( 0 ).logits().ptr<const half>() + static_cast<size_t>( row ) * V, V * sizeof( half ), cudaMemcpyDeviceToHost ) );
	std::vector<float> f( V );
	for ( int i = 0; i < V; ++i ) f[i] = __half2float( h[i] );
	return f;
}

void compare( const char* tag, const std::vector<float>& a, const std::vector<float>& b, double& worst_cos ) {
	double dot = 0, na = 0, nb = 0, maxd = 0;
	int ia = 0, ib = 0;
	for ( size_t i = 0; i < a.size(); ++i ) {
		dot += a[i] * b[i];
		na += a[i] * a[i];
		nb += b[i] * b[i];
		maxd = std::fmax( maxd, std::fabs( a[i] - b[i] ) );
		if ( a[i] > a[ia] ) ia = static_cast<int>( i );
		if ( b[i] > b[ib] ) ib = static_cast<int>( i );
	}
	const double cos = dot / ( std::sqrt( na ) * std::sqrt( nb ) + 1e-12 );
	worst_cos = std::fmin( worst_cos, cos );
	std::printf( "    %-24s cos=%.6f max_abs=%.3f argmax %d/%d\n", tag, cos, maxd, ia, ib );
}

} // namespace impl

int main() {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = "qwen38";
	cfg.model_dir = test::QWEN_DIR;
	ENGINE_ASSERT( mc.init( cfg ), "init failed" );

	const std::vector<int> prompt = { 248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198 };
	const int M = 8;
	double worst = 1.0;

	// Reference: M=1 forwards; keep logits of the last M positions
	mc.reset_slot( 0 );
	std::vector<std::vector<float>> ref;
	for ( int p = 0; p < static_cast<int>( prompt.size() ); ++p ) {
		mc.forward( &prompt[p], 1, p, 0 );
		if ( p >= static_cast<int>( prompt.size() ) - M ) ref.push_back( impl::logits_row( mc, 0 ) );
	}

	// Mode A: prefix with M=1, then one causal block of M rows
	const int base = static_cast<int>( prompt.size() ) - M;
	mc.reset_slot( 0 );
	for ( int p = 0; p < base; ++p ) mc.forward( &prompt[p], 1, p, 0 );
	mc.forward( &prompt[base], M, base, 0 );
	std::printf( "[*] Mode A (causal block M=%d)\n", M );
	for ( int r = 0; r < M; ++r ) {
		char tag[32];
		std::snprintf( tag, sizeof( tag ), "row %d (pos %d)", r, base + r );
		impl::compare( tag, ref[r], impl::logits_row( mc, r ), worst );
	}

	// Mode B: the same prompt in 4 slots, last token batched across slots
	std::printf( "[*] Mode B (4 slots, 1 row each)\n" );
	const int last = static_cast<int>( prompt.size() ) - 1;
	int slots[4], pos[4], toks[4];
	for ( int s = 0; s < 4; ++s ) {
		mc.reset_slot( s );
		for ( int p = 0; p < last; ++p ) mc.forward( &prompt[p], 1, p, s );
		slots[s] = s;
		pos[s] = last;
		toks[s] = prompt[last];
	}
	mc.forward( toks, pod::BatchRows::batched( 4, slots, pos ) );
	for ( int s = 0; s < 4; ++s ) {
		char tag[32];
		std::snprintf( tag, sizeof( tag ), "slot %d", s );
		impl::compare( tag, ref[M - 1], impl::logits_row( mc, s ), worst );
	}

	mc.shutdown();
	std::printf( "[%c] worst cos %.6f\n", worst >= 0.999 ? '+' : '-', worst );
	ENGINE_ASSERT( worst >= 0.999, "row parity failed" );
	return 0;
}
