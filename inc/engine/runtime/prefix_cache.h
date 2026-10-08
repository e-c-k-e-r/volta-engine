#pragma once

#include "engine/runtime/cluster.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::runtime {

// Host-RAM store of sequence snapshots keyed by the exact token prefix they cover, organized as a tree: a node holds the
// KV rows of its own segment [parent.tokens, tokens) plus the recurrent state at its end, so a chain of checkpoints along
// one prompt stores every KV row once. A request resumes from the deepest node whose tokens are a proper prefix of its
// prompt (at least one token must remain to produce logits). Eviction is least-recently-used over leaves; pinned nodes
// (the shared system-prompt root) and nodes with children stay. Not thread safe: scheduler thread only.
class PrefixCache {
public:
	struct Entry {
		std::vector<int> tokens;        // the whole prefix
		Entry* parent = nullptr;
		int children = 0;
		ModelCluster::SeqSnapshot snap; // KV rows [parent->tokens.size() or 0, tokens.size()) + state at the end
		uint64_t last_used = 0;
		bool pinned = false;
	};

	struct Stats {
		uint64_t lookups = 0;
		uint64_t hits = 0;
		uint64_t hit_tokens = 0;     // prompt tokens not prefilled thanks to hits
		uint64_t inserts = 0;
		uint64_t evictions = 0;
		size_t entries = 0;
		size_t bytes = 0;            // pinned host bytes held by entries, all ranks
	};

	// budget_bytes: total pinned host bytes across ranks the entries may hold
	PrefixCache( ModelCluster& cluster, size_t budget_bytes ) : cluster_( cluster ), budget_( budget_bytes ) {}
	~PrefixCache() { clear(); }

	// Deepest entry whose tokens are a proper prefix of prompt; nullptr if none. Counts a lookup, marks the chain used.
	Entry* find( const std::vector<int>& prompt );
	Entry* exact( const std::vector<int>& tokens );
	// Writes the entry's chain into the slot (ancestors' KV first, then the entry's KV, state and taps). False on failure.
	bool restore( int slot, Entry* entry );
	// Adds a node under `parent` (nullptr: a root). Takes ownership of snap (freed on failure). The parent and its ancestors
	// are never evicted to make room. Returns the new node, or nullptr if it did not fit.
	Entry* insert( Entry* parent, std::vector<int> tokens, ModelCluster::SeqSnapshot snap, bool pinned );
	void clear();
	Stats stats() const;
	size_t budget() const { return budget_; }

private:
	static size_t cost( const ModelCluster::SeqSnapshot& s, int tp ) { return s.bytes * static_cast<size_t>( tp ); }
	void remove( size_t index );
	bool evict_for( size_t incoming, const Entry* protect );
	void touch_chain( Entry* e );

	ModelCluster& cluster_;
	size_t budget_;
	std::vector<std::unique_ptr<Entry>> entries_;
	uint64_t clock_ = 0;
	Stats stats_;
};

} // namespace engine::runtime
