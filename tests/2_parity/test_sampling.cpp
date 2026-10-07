// Host-only checks of the sampler: (1) sample_token reproduces the truncated distribution, (2) speculative
// acceptance against a deterministic draft returns tokens distributed exactly as the target, whatever the draft
// token is (including one outside the candidate set), and (3) the same seed gives the same stream.
#include "engine/runtime/sampling.h"
#include <cstdio>
#include <map>

using namespace engine::runtime;

namespace {

int g_fail = 0;

Candidates make_cands() {
	Candidates c;
	c.n = 16;
	for ( int i = 0; i < 16; ++i ) {
		c.ids[i] = 1000 + 7 * i;
		c.logits[i] = 12.0f - 0.9f * i + ( i % 3 ) * 0.2f;
	}
	// sorted descending
	for ( int i = 1; i < 16; ++i ) {
		for ( int j = i; j > 0 && c.logits[j] > c.logits[j - 1]; --j ) {
			std::swap( c.logits[j], c.logits[j - 1] );
			std::swap( c.ids[j], c.ids[j - 1] );
		}
	}
	return c;
}

void check_freq( const char* name, const std::map<int, long>& hist, long n, const Candidates& c, const double* probs ) {
	double worst = 0.0;
	for ( int i = 0; i < c.n; ++i ) {
		const auto it = hist.find( c.ids[i] );
		const double f = ( it == hist.end() ? 0.0 : static_cast<double>( it->second ) ) / n;
		const double sd = std::sqrt( probs[i] * ( 1.0 - probs[i] ) / n ) + 1e-12;
		const double z = std::fabs( f - probs[i] ) / sd;
		worst = z > worst ? z : worst;
		if ( probs[i] == 0.0 && f != 0.0 ) worst = 1e9;
	}
	const bool ok = worst < 4.5;
	g_fail += ok ? 0 : 1;
	std::printf( "  %-44s worst z %.2f  %s\n", name, worst, ok ? "PASS" : "FAIL" );
}

} // namespace

int main() {
	const Candidates c = make_cands();
	const long N = 400000;
	const SamplingParams params[] = {
		{ 1.0f, 1.0f, 16, 1 }, { 0.7f, 0.95f, 16, 1 }, { 1.0f, 0.9f, 8, 1 }, { 1.5f, 1.0f, 4, 1 },
	};
	for ( const auto& p : params ) {
		double probs[kSampleTopK];
		make_dist( c, p, probs );
		char name[96];

		Rng rng( 42 );
		std::map<int, long> h;
		for ( long i = 0; i < N; ++i ) ++h[sample_token( c, p, rng )];
		std::snprintf( name, sizeof( name ), "sample  T=%.1f p=%.2f k=%d", p.temperature, p.top_p, p.top_k );
		check_freq( name, h, N, c, probs );

		// Drafts: the mode, a mid-rank token, a rank the truncation removes, and a token not in the set.
		const int drafts[] = { c.ids[0], c.ids[3], c.ids[12], 424242 };
		for ( int d : drafts ) {
			Rng r2( 7 );
			std::map<int, long> hs;
			long acc = 0;
			for ( long i = 0; i < N; ++i ) {
				bool a = false;
				++hs[spec_step( c, p, d, r2, &a )];
				acc += a ? 1 : 0;
			}
			std::snprintf( name, sizeof( name ), "spec    T=%.1f p=%.2f k=%d draft=%d", p.temperature, p.top_p, p.top_k, d );
			check_freq( name, hs, N, c, probs );
		}
	}

	// Reproducibility.
	{
		Rng a( 123 ), b( 123 );
		bool same = true;
		for ( int i = 0; i < 1000; ++i ) same = same && ( a.uniform() == b.uniform() );
		g_fail += same ? 0 : 1;
		std::printf( "  %-44s %s\n", "same seed, same stream", same ? "PASS" : "FAIL" );
	}
	std::printf( "=== SAMPLING: %s ===\n", g_fail ? "FAIL" : "PASS" );
	return g_fail ? 1 : 0;
}
