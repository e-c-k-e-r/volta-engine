#pragma once

#include "engine/config.h"
#include <cmath>
#include <cstdint>

namespace engine::runtime {

// Token sampling over a per-row candidate set (the top kSampleTopK logits across the vocabulary shards, sorted
// descending). top_k is capped at kSampleTopK and top_p is applied inside the candidate set, so the target
// distribution is "softmax(logits / T) over the top min(top_k, 16) tokens, cut at cumulative top_p, renormalised".
// Qwen's recommended top_k = 20 is therefore approximated by 16; the mass outside the top 16 is negligible at the
// recommended temperatures, and this definition is exact for the speculative acceptance below.
static constexpr int kSampleTopK = 16;

struct SamplingParams {
	float temperature = 0.0f;   // 0 = greedy
	float top_p = 1.0f;
	int top_k = kSampleTopK;
	uint64_t seed = 0;
	bool greedy() const { return temperature <= 0.0f; }
};

struct Candidates {
	int n = 0;                  // valid entries, sorted by logit descending
	int ids[kSampleTopK] = {};
	float logits[kSampleTopK] = {};
};

// splitmix64-seeded xoshiro256++: small, fast, and reproducible across runs for a given seed.
class Rng {
public:
	explicit Rng( uint64_t seed = 0x9E3779B97F4A7C15ULL ) { reseed( seed ); }
	void reseed( uint64_t seed ) {
		for ( auto& w : s_ ) {
			seed += 0x9E3779B97F4A7C15ULL;
			uint64_t z = seed;
			z = ( z ^ ( z >> 30 ) ) * 0xBF58476D1CE4E5B9ULL;
			z = ( z ^ ( z >> 27 ) ) * 0x94D049BB133111EBULL;
			w = z ^ ( z >> 31 );
		}
	}
	// Uniform in [0, 1).
	double uniform() { return ( next() >> 11 ) * ( 1.0 / 9007199254740992.0 ); }

private:
	uint64_t next() {
		const uint64_t r = rotl( s_[0] + s_[3], 23 ) + s_[0];
		const uint64_t t = s_[1] << 17;
		s_[2] ^= s_[0];
		s_[3] ^= s_[1];
		s_[1] ^= s_[2];
		s_[0] ^= s_[3];
		s_[2] ^= t;
		s_[3] = rotl( s_[3], 45 );
		return r;
	}
	static uint64_t rotl( uint64_t x, int k ) { return ( x << k ) | ( x >> ( 64 - k ) ); }
	uint64_t s_[4];
};

// Probability of each candidate under the truncated, temperature-scaled distribution. probs[i] pairs with ids[i];
// entries cut by top_k / top_p are exactly 0. Requires !params.greedy().
inline void make_dist( const Candidates& c, const SamplingParams& p, double probs[kSampleTopK] ) {
	const int k = MAX( 1, MIN( MIN( p.top_k, kSampleTopK ), c.n ) );
	const double inv_t = 1.0 / static_cast<double>( p.temperature );
	double mx = c.logits[0];
	for ( int i = 1; i < k; ++i ) mx = c.logits[i] > mx ? c.logits[i] : mx;
	double sum = 0.0;
	for ( int i = 0; i < k; ++i ) {
		probs[i] = std::exp( ( c.logits[i] - mx ) * inv_t );
		sum += probs[i];
	}
	// Candidates are sorted descending, so the nucleus is a prefix.
	double cum = 0.0;
	int keep = k;
	for ( int i = 0; i < k; ++i ) {
		cum += probs[i] / sum;
		if ( cum >= p.top_p ) {
			keep = i + 1;
			break;
		}
	}
	double kept = 0.0;
	for ( int i = 0; i < keep; ++i ) kept += probs[i];
	for ( int i = 0; i < kSampleTopK; ++i ) probs[i] = ( i < keep ) ? probs[i] / kept : 0.0;
}

inline int sample_dist( const Candidates& c, const double probs[kSampleTopK], Rng& rng ) {
	const double u = rng.uniform();
	double cum = 0.0;
	int last = 0;
	for ( int i = 0; i < c.n; ++i ) {
		if ( probs[i] <= 0.0 ) continue;
		cum += probs[i];
		last = i;
		if ( u < cum ) return c.ids[i];
	}
	return c.ids[last];   // rounding: u landed past the accumulated mass
}

inline int sample_token( const Candidates& c, const SamplingParams& p, Rng& rng ) {
	if ( p.greedy() ) return c.ids[0];
	double probs[kSampleTopK];
	make_dist( c, p, probs );
	return sample_dist( c, probs, rng );
}

// One verification step of speculative sampling against a deterministic draft token `d` (a point-mass proposal,
// which is what the DFlash2 walk produces). Accept d with probability p(d); otherwise draw from p with d removed
// and renormalised. The returned token is distributed exactly as a sample from p. Sets *accepted.
inline int spec_step( const Candidates& c, const SamplingParams& p, int d, Rng& rng, bool* accepted ) {
	double probs[kSampleTopK];
	make_dist( c, p, probs );
	double pd = 0.0;
	for ( int i = 0; i < c.n; ++i ) {
		if ( c.ids[i] == d ) pd = probs[i];
	}
	if ( rng.uniform() < pd ) {
		*accepted = true;
		return d;
	}
	*accepted = false;
	// p(d) < 1 here (u < 1 always accepts when pd == 1), so the residual has positive mass.
	for ( int i = 0; i < c.n; ++i ) {
		if ( c.ids[i] == d ) probs[i] = 0.0;
	}
	double rest = 0.0;
	for ( int i = 0; i < c.n; ++i ) rest += probs[i];
	for ( int i = 0; i < c.n; ++i ) probs[i] /= rest;
	return sample_dist( c, probs, rng );
}

} // namespace engine::runtime
