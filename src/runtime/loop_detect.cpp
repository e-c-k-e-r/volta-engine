#include "engine/runtime/loop_detect.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace engine::runtime {

namespace {

constexpr uint64_t kMod = ( 1ull << 61 ) - 1;    // rolling hash modulus (Mersenne prime)
constexpr uint64_t kBase = 1000003;
constexpr double kExactRate = 0.995;             // recent match rate counted as exact repetition
constexpr double kNearExactRate = 0.97;          // ... as near-exact (tiny-vocabulary text)
constexpr int kDiverseWindow = 256;              // tokens examined for vocabulary diversity (at most)
constexpr int kDiverseMin = 12;                  // distinct tokens for text to count as varied
constexpr int kPromptBloomBits = 16;             // Bloom filter bits per prompt token
constexpr int kPromptSourcedMin = 4;             // prompt-sourced differences before a run counts as transcription

uint64_t mulmod( uint64_t a, uint64_t b ) { return static_cast<uint64_t>( ( static_cast<unsigned __int128>( a ) * b ) % kMod ); }

// h of the k-gram ending at token t, given the token leaving the window (0 before the window is full)
uint64_t roll( uint64_t h, uint64_t out, uint64_t bk, int token ) {
	return ( mulmod( h, kBase ) + kMod - mulmod( out, bk ) + static_cast<uint64_t>( token ) + 1 ) % kMod;
}

uint64_t powmod( uint64_t b, int e ) {
	uint64_t r = 1;
	while ( e-- > 0 ) r = mulmod( r, b );
	return r;
}

} // namespace

LoopDetector::LoopDetector() : LoopDetector( Params() ) {}

LoopDetector::LoopDetector( const Params& p ) : p_( p ) {
	uint64_t slots = 1;
	while ( slots < 2ull * p_.max_period ) slots <<= 1;
	mask_ = slots - 1;
	key_.assign( slots, 0 );
	newest_.assign( slots, -1 );
	pos_.assign( slots * p_.ways_pos, -1 );
	head_.assign( slots, 0 );
	ring_.assign( p_.k, 0 );
	bk_ = powmod( kBase, p_.k );
	hashes_.assign( static_cast<size_t>( p_.max_period ) * p_.max_mult + 1, 0 );
	tokens_.assign( kDiverseWindow, -1 );
	on_.assign( p_.max_runs, 0 );
	exempt_.assign( p_.max_runs, 0 );
	start_.assign( p_.max_runs, 0 );
	last_.assign( p_.max_runs, 0 );
	sourced_last_.assign( p_.max_runs, 0 );
	per_.assign( p_.max_runs, 0 );
	matched_.assign( p_.max_runs, 0 );
	unmatched_.assign( p_.max_runs, 0 );
	sourced_.assign( p_.max_runs, 0 );
	recent_.assign( p_.max_runs, 1.0 );
}

void LoopDetector::prime( const int* prompt, int n ) {
	int bits = 10;
	while ( ( 1ull << bits ) < static_cast<uint64_t>( std::max( n, 1 ) ) * kPromptBloomBits ) ++bits;
	bloom_mask_ = ( 1ull << bits ) - 1;
	bloom_.assign( static_cast<size_t>( 1ull << bits ) >> 3, 0 );
	std::vector<uint64_t> ring( p_.k, 0 );
	uint64_t h = 0;
	for ( int t = 0; t < n; ++t ) {
		const int i = t % p_.k;
		const uint64_t out = t >= p_.k ? ring[i] : 0;
		ring[i] = static_cast<uint64_t>( prompt[t] ) + 1;
		h = roll( h, out, bk_, prompt[t] );
		if ( t < p_.k - 1 ) continue;
		const uint64_t hh = h ? h : 1;
		for ( uint64_t x : { hh, hh >> 21, hh >> 42 } ) {
			const uint64_t b = x & bloom_mask_;
			bloom_[b >> 3] |= static_cast<uint8_t>( 1u << ( b & 7 ) );
		}
	}
}

bool LoopDetector::in_prompt( uint64_t h ) const {
	if ( bloom_.empty() ) return false;
	for ( uint64_t x : { h, h >> 21, h >> 42 } ) {
		const uint64_t b = x & bloom_mask_;
		if ( !( ( bloom_[b >> 3] >> ( b & 7 ) ) & 1 ) ) return false;
	}
	return true;
}

bool LoopDetector::diverse( int64_t t, int p ) const {
	const int64_t n = std::min<int64_t>( std::min( std::max( p, 64 ), kDiverseWindow ), t + 1 );
	if ( n < 32 ) return true;
	std::unordered_set<int64_t> seen;
	for ( int64_t j = 0; j < n; ++j ) seen.insert( tokens_[( t - j ) % kDiverseWindow] );
	return static_cast<int>( seen.size() ) >= kDiverseMin;
}

bool LoopDetector::feed( int token ) {
	const int64_t t = ++t_;
	const int k = p_.k;
	const int i = static_cast<int>( t % k );
	const uint64_t out = t >= k ? ring_[i] : 0;
	ring_[i] = static_cast<uint64_t>( token ) + 1;
	h_ = roll( h_, out, bk_, token );
	tokens_[t % kDiverseWindow] = token;
	if ( t < k - 1 ) return detected();
	const uint64_t h = h_ ? h_ : 1;

	// earlier occurrences of this k-gram within max_period, as distances
	const uint64_t b = h & mask_ & ~1ull;
	int64_t e = -1;
	for ( uint64_t w : { b, b + 1 } ) {
		if ( key_[w] == h && newest_[w] >= t - p_.max_period ) { e = static_cast<int64_t>( w ); break; }
	}
	const int npos = p_.ways_pos;
	int64_t dists[16];
	int nd = 0;
	if ( e >= 0 ) {
		for ( int j = 0; j < npos; ++j ) {
			const int64_t x = pos_[e * npos + j];
			if ( x >= 0 && t - x <= p_.max_period && nd < 16 ) dists[nd++] = t - x;
		}
	}
	else {
		e = static_cast<int64_t>( newest_[b] <= newest_[b + 1] ? b : b + 1 );
		key_[e] = h;
		for ( int j = 0; j < npos; ++j ) pos_[e * npos + j] = -1;
		head_[e] = 0;
	}
	const bool new_prompt_content = nd == 0 && in_prompt( h );

	// update the runs
	const int64_t nh = static_cast<int64_t>( hashes_.size() );
	for ( int r = 0; r < p_.max_runs; ++r ) {
		if ( !on_[r] ) continue;
		const int p = per_[r];
		const int tl = static_cast<int>( p * p_.tol );
		int hit_m = 0;
		int64_t hit_q = 0;
		for ( int m = 1; m <= p_.max_mult && !hit_m; ++m ) {
			const int64_t lo = static_cast<int64_t>( m ) * ( p - tl ), hi = static_cast<int64_t>( m ) * ( p + tl );
			for ( int j = 0; j < nd; ++j )
				if ( lo <= dists[j] && dists[j] <= hi ) { hit_m = m; hit_q = dists[j]; break; }
		}
		// the few stored positions can all lie inside the current copy when the unit itself repeats phrases: compare the hash m periods back
		if ( !hit_m ) {
			for ( int m = 1; m <= p_.max_mult; ++m ) {
				const int64_t s = t - static_cast<int64_t>( m ) * p;
				if ( s >= 0 && t - s < nh && hashes_[s % nh] == h ) { hit_m = m; hit_q = static_cast<int64_t>( m ) * p; break; }
			}
		}
		const double tau = std::max( 32, p );
		if ( hit_m ) {
			if ( hit_m == 1 && hit_q != p ) per_[r] = static_cast<int>( hit_q );   // follow insertions / deletions
			last_[r] = t;
			matched_[r]++;
			recent_[r] += ( 1.0 - recent_[r] ) / tau;
		}
		else if ( t - last_[r] > p_.gap ) {
			on_[r] = 0;
			continue;
		}
		else {
			recent_[r] -= recent_[r] / tau;
			unmatched_[r]++;
			if ( new_prompt_content ) { sourced_[r]++; sourced_last_[r] = t; }
		}
		if ( !detected() && is_loop( r, t ) ) { detected_at_ = t; period_ = per_[r]; }
	}

	// start runs for distances no live run explains
	for ( int j = 0; j < nd; ++j ) {
		const int64_t q = dists[j];
		bool explained = false;
		for ( int r = 0; r < p_.max_runs && !explained; ++r ) {
			if ( !on_[r] ) continue;
			const int p = per_[r], tl = static_cast<int>( p * p_.tol );
			for ( int m = 1; m <= p_.max_mult; ++m )
				if ( std::llabs( q - static_cast<int64_t>( m ) * p ) <= static_cast<int64_t>( m ) * tl ) { explained = true; break; }
		}
		if ( explained ) continue;
		int fr = -1;
		for ( int r = 0; r < p_.max_runs; ++r ) if ( !on_[r] ) { fr = r; break; }
		if ( fr < 0 ) {
			// replace a live run that has not matched beyond its first token, otherwise drop the candidate
			fr = static_cast<int>( std::min_element( matched_.begin(), matched_.end() ) - matched_.begin() );
			if ( matched_[fr] > 1 ) continue;
		}
		on_[fr] = 1;
		start_[fr] = t - q;
		per_[fr] = static_cast<int>( q );
		last_[fr] = t;
		matched_[fr] = 1;
		unmatched_[fr] = sourced_[fr] = 0;
		sourced_last_[fr] = 0;
		exempt_[fr] = 0;
		recent_[fr] = 1.0;
	}

	// record this k-gram
	hashes_[t % nh] = h;
	const int hd = head_[e];
	pos_[e * npos + hd] = t;
	head_[e] = static_cast<int8_t>( ( hd + 1 ) % npos );
	newest_[e] = t;
	return detected();
}

bool LoopDetector::is_loop( int r, int64_t t ) {
	const int p = per_[r];
	const int64_t span = t - start_[r];
	const double recent = recent_[r];
	// transcription of distinct prompt items (sticky once established, lapses two periods after the last new prompt content)
	if ( sourced_[r] >= kPromptSourcedMin && sourced_[r] >= p_.prompt_frac * unmatched_[r] ) exempt_[r] = 1;
	if ( exempt_[r] && t - sourced_last_[r] <= 2ll * p ) return false;
	const bool exact = recent >= kExactRate;
	const int reps = exact ? p_.exact_reps : p_.min_reps, length = exact ? p_.exact_len : p_.min_len;
	if ( span < static_cast<int64_t>( reps ) * p || span < length ) return false;
	if ( matched_[r] < p_.min_frac * std::max<int64_t>( span - p, 1 ) || recent < p_.min_recent ) return false;
	// tiny vocabulary: structure alone makes k-grams recur, so require (near) exact repetition
	if ( !diverse( t, p ) ) return exact || ( recent >= kNearExactRate && span >= p_.lowdiv_len );
	return true;
}

} // namespace engine::runtime
