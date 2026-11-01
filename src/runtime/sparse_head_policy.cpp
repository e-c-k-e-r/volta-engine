#include "engine/runtime/sparse_head_policy.h"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <string>

namespace engine::runtime {

namespace {
constexpr double kMissEma = 1.0 / 512.0;   // time constant of the miss rate, in target tokens
constexpr double kTimeEma = 0.05;          // weight of a new round in the timing averages
constexpr int kMinScored = 64;             // generated tokens scored against the subset before it is used
}

void SparseHeadPolicy::init( int vocab, const Params& p ) {
	p_ = p;
	vocab_ = vocab;
	counts_.assign( vocab, 0.0f );
	in_.assign( vocab, 0 );
}

void SparseHeadPolicy::observe_prompt( const int* tokens, int n ) {
	if ( !active() ) return;
	for ( int i = 0; i < n; ++i ) {
		if ( tokens[i] >= 0 && tokens[i] < vocab_ ) counts_[tokens[i]] += p_.prompt_weight;
	}
	evidence_ += p_.prompt_weight * n;
}

void SparseHeadPolicy::observe_generated( int token ) {
	if ( !active() || token < 0 || token >= vocab_ ) return;
	counts_[token] += 1.0f;
	evidence_ += 1.0;
	++since_save_;
	++since_rebuild_;
	if ( ++since_decay_ >= p_.decay_tokens ) {
		for ( float& c : counts_ ) c *= 0.5f;
		evidence_ *= 0.5;
		since_decay_ = 0;
	}
	if ( built_ ) {
		// a plain mean over the first tokens scored against a subset, then the exponential average
		++scored_;
		miss_ += ( ( in_[token] ? 0.0 : 1.0 ) - miss_ ) * std::max( kMissEma, 1.0 / scored_ );
	}
}

bool SparseHeadPolicy::rebuild_due() {
	if ( !active() || evidence_ < p_.min_evidence ) return false;
	if ( built_ && since_rebuild_ < p_.interval ) return false;
	since_rebuild_ = 0;
	if ( !built_ ) return true;
	// extract again only when the ranking would replace enough of the subset
	const std::vector<int> next = ranked();
	int changed = 0;
	for ( int id : next ) changed += in_[id] ? 0 : 1;
	return changed >= p_.min_change * std::max( subset_n_, 1 );
}

std::vector<int> SparseHeadPolicy::ranked() const {
	// incumbents are scored with the margin, so a challenger must beat an incumbent's count by it to take the place
	std::vector<int> ids;
	ids.reserve( vocab_ );
	for ( int i = 0; i < vocab_; ++i ) if ( counts_[i] > 0.0f || in_[i] ) ids.push_back( i );
	auto score = [&]( int i ) { return counts_[i] * ( in_[i] ? 1.0f + p_.swap_margin : 1.0f ); };
	const int n = std::min( p_.capacity, static_cast<int>( ids.size() ) );
	std::partial_sort( ids.begin(), ids.begin() + n, ids.end(), [&]( int a, int b ) { const float sa = score( a ), sb = score( b ); return sa != sb ? sa > sb : a < b; } );
	ids.resize( n );
	return ids;
}

void SparseHeadPolicy::set_subset( const std::vector<int>& kept ) {
	std::fill( in_.begin(), in_.end(), 0 );
	for ( int id : kept ) if ( id >= 0 && id < vocab_ ) in_[id] = 1;
	subset_n_ = static_cast<int>( kept.size() );
	if ( !built_ ) scored_ = 0;   // the first subset is scored from scratch; later ones continue the average (the subset changes little)
	built_ = subset_n_ > 0;
}

namespace {
constexpr uint32_t kMagic = 0x53504844;   // "SPHD"
}

bool SparseHeadPolicy::load( const char* path ) {
	if ( !active() || !path || !*path ) return false;
	std::FILE* f = std::fopen( path, "rb" );
	if ( !f ) return false;
	uint32_t head[2] = {};
	double ev = 0.0;
	std::vector<float> c( vocab_ );
	const bool ok = std::fread( head, sizeof( head ), 1, f ) == 1 && head[0] == kMagic && static_cast<int>( head[1] ) == vocab_ && std::fread( &ev, sizeof( ev ), 1, f ) == 1 &&
		std::fread( c.data(), sizeof( float ), c.size(), f ) == c.size();
	std::fclose( f );
	if ( !ok ) return false;
	counts_ = std::move( c );
	evidence_ = ev;
	return true;
}

bool SparseHeadPolicy::save( const char* path ) const {
	if ( !active() || !path || !*path || evidence_ <= 0.0 ) return false;
	const std::string tmp = std::string( path ) + ".tmp";
	std::FILE* f = std::fopen( tmp.c_str(), "wb" );
	if ( !f ) return false;
	const uint32_t head[2] = { kMagic, static_cast<uint32_t>( vocab_ ) };
	bool ok = std::fwrite( head, sizeof( head ), 1, f ) == 1 && std::fwrite( &evidence_, sizeof( evidence_ ), 1, f ) == 1 &&
		std::fwrite( counts_.data(), sizeof( float ), counts_.size(), f ) == counts_.size();
	ok = std::fclose( f ) == 0 && ok;
	return ok && std::rename( tmp.c_str(), path ) == 0;
}

bool SparseHeadPolicy::save_due() {
	if ( since_save_ < 4096 ) return false;
	since_save_ = 0;
	return true;
}

void SparseHeadPolicy::draft_step_time( bool sparse, double ms ) {
	double& t = sparse ? t_sparse_ : t_full_;
	t = t > 0.0 ? t + ( ms - t ) * kTimeEma : ms;
}

void SparseHeadPolicy::round_time( double ms, int draft_steps ) {
	t_round_ = t_round_ > 0.0 ? t_round_ + ( ms - t_round_ ) * kTimeEma : ms;
	steps_ = steps_ > 0.0 ? steps_ + ( draft_steps - steps_ ) * kTimeEma : draft_steps;
}

bool SparseHeadPolicy::use_sparse() {
	if ( !built_ || t_full_ <= 0.0 || scored_ < kMinScored ) return on_ = false;
	if ( t_sparse_ <= 0.0 ) return on_ = true;   // measure the sparse steps once
	if ( on_ && ++sparse_rounds_ >= p_.full_probe_rounds ) {
		sparse_rounds_ = 0;
		return false;   // one full round refreshes the full head's step time (the mode stays on)
	}
	const double saving = std::max( 0.0, t_full_ - t_sparse_ ) * std::max( steps_, 1.0 );
	const double breakeven = t_round_ > 0.0 ? 2.0 * saving / t_round_ : 0.0;
	on_ = miss_ < breakeven * ( on_ ? 1.2 : 0.8 );
	return on_;
}

} // namespace engine::runtime
