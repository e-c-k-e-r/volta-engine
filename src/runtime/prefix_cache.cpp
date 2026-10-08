#include "engine/runtime/prefix_cache.h"

#include <algorithm>

namespace engine::runtime {

const PrefixCache::Entry* PrefixCache::find( const std::vector<int>& prompt ) {
	stats_.lookups++;
	Entry* best = nullptr;
	for ( auto& e : entries_ ) {
		const size_t n = e->tokens.size();
		if ( n >= prompt.size() || ( best && n <= best->tokens.size() ) ) continue;
		if ( std::equal( e->tokens.begin(), e->tokens.end(), prompt.begin() ) ) best = e.get();
	}
	if ( best ) {
		best->last_used = ++clock_;
		stats_.hits++;
		stats_.hit_tokens += best->tokens.size();
	}
	return best;
}

bool PrefixCache::contains( const std::vector<int>& tokens ) const {
	for ( const auto& e : entries_ ) {
		if ( e->tokens == tokens ) return true;
	}
	return false;
}

void PrefixCache::evict_for( size_t incoming ) {
	size_t held = stats().bytes;
	while ( held + incoming > budget_ ) {
		size_t victim = entries_.size();
		for ( size_t k = 0; k < entries_.size(); ++k ) {
			if ( entries_[k]->pinned ) continue;
			if ( victim == entries_.size() || entries_[k]->last_used < entries_[victim]->last_used ) victim = k;
		}
		if ( victim == entries_.size() ) return; // only pinned entries left
		held -= cost( entries_[victim]->snap, cluster_.tp() );
		cluster_.free_snapshot( &entries_[victim]->snap );
		entries_.erase( entries_.begin() + static_cast<long>( victim ) );
		stats_.evictions++;
	}
}

bool PrefixCache::insert( std::vector<int> tokens, ModelCluster::SeqSnapshot snap, bool pinned ) {
	const size_t incoming = cost( snap, cluster_.tp() );
	for ( size_t k = 0; k < entries_.size(); ++k ) {
		if ( entries_[k]->tokens == tokens ) {
			pinned = pinned || entries_[k]->pinned;
			cluster_.free_snapshot( &entries_[k]->snap );
			entries_.erase( entries_.begin() + static_cast<long>( k ) );
			break;
		}
	}
	evict_for( incoming );
	if ( stats().bytes + incoming > budget_ ) {
		cluster_.free_snapshot( &snap );
		return false;
	}
	auto e = std::make_unique<Entry>();
	e->tokens = std::move( tokens );
	e->snap = snap;
	e->last_used = ++clock_;
	e->pinned = pinned;
	entries_.push_back( std::move( e ) );
	stats_.inserts++;
	return true;
}

void PrefixCache::clear() {
	for ( auto& e : entries_ ) cluster_.free_snapshot( &e->snap );
	entries_.clear();
}

PrefixCache::Stats PrefixCache::stats() const {
	Stats s = stats_;
	s.entries = entries_.size();
	s.bytes = 0;
	for ( const auto& e : entries_ ) s.bytes += cost( e->snap, cluster_.tp() );
	return s;
}

} // namespace engine::runtime
