#include "engine/runtime/prefix_cache.h"

#include <algorithm>

namespace engine::runtime {

void PrefixCache::touch_chain( Entry* e ) {
	const uint64_t t = ++clock_;
	for ( ; e; e = e->parent ) e->last_used = t;
}

PrefixCache::Entry* PrefixCache::find( const std::vector<int>& prompt ) {
	stats_.lookups++;
	Entry* best = nullptr;
	for ( auto& e : entries_ ) {
		const size_t n = e->tokens.size();
		if ( n >= prompt.size() || ( best && n <= best->tokens.size() ) ) continue;
		if ( std::equal( e->tokens.begin(), e->tokens.end(), prompt.begin() ) ) best = e.get();
	}
	if ( best ) {
		touch_chain( best );
		stats_.hits++;
		stats_.hit_tokens += best->tokens.size();
	}
	return best;
}

PrefixCache::Entry* PrefixCache::exact( const std::vector<int>& tokens ) {
	for ( auto& e : entries_ ) {
		if ( e->tokens == tokens ) return e.get();
	}
	return nullptr;
}

bool PrefixCache::restore( int slot, Entry* entry ) {
	std::vector<Entry*> chain;
	for ( Entry* e = entry; e; e = e->parent ) chain.push_back( e );
	std::reverse( chain.begin(), chain.end() );
	for ( size_t k = 0; k + 1 < chain.size(); ++k ) {
		if ( !cluster_.load_snapshot( slot, chain[k]->snap, true ) ) return false;
	}
	return cluster_.load_snapshot( slot, entry->snap, false );
}

void PrefixCache::remove( size_t index ) {
	Entry* e = entries_[index].get();
	if ( e->parent ) e->parent->children--;
	cluster_.free_snapshot( &e->snap );
	entries_.erase( entries_.begin() + static_cast<long>( index ) );
}

bool PrefixCache::evict_for( size_t incoming, const Entry* protect ) {
	size_t held = stats().bytes;
	while ( held + incoming > budget_ ) {
		size_t victim = entries_.size();
		for ( size_t k = 0; k < entries_.size(); ++k ) {
			const Entry* e = entries_[k].get();
			if ( e->pinned || e->children > 0 ) continue;
			bool guarded = false;
			for ( const Entry* p = protect; p; p = p->parent ) guarded = guarded || p == e;
			if ( guarded ) continue;
			if ( victim == entries_.size() || e->last_used < entries_[victim]->last_used ) victim = k;
		}
		if ( victim == entries_.size() ) return false; // nothing evictable
		held -= cost( entries_[victim]->snap, cluster_.tp() );
		remove( victim );
		stats_.evictions++;
	}
	return true;
}

PrefixCache::Entry* PrefixCache::insert( Entry* parent, std::vector<int> tokens, ModelCluster::SeqSnapshot snap, bool pinned ) {
	const size_t incoming = cost( snap, cluster_.tp() );
	if ( !evict_for( incoming, parent ) ) {
		cluster_.free_snapshot( &snap );
		return nullptr;
	}
	auto e = std::make_unique<Entry>();
	e->tokens = std::move( tokens );
	e->parent = parent;
	e->snap = snap;
	e->pinned = pinned;
	if ( parent ) parent->children++;
	Entry* raw = e.get();
	entries_.push_back( std::move( e ) );
	touch_chain( raw );
	stats_.inserts++;
	return raw;
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
