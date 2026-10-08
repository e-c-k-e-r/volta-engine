#pragma once

#include "engine/runtime/cluster.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::runtime {

// Host-RAM store of sequence snapshots keyed by the exact token prefix they cover. A request resumes from the longest
// stored prefix of its prompt (strictly shorter than the prompt, so at least one token is left to produce logits).
// Entries are evicted least-recently-used; pinned entries (the shared system-prompt root) are never evicted.
// Not thread safe: used by the scheduler thread only.
class PrefixCache {
public:
	struct Entry {
		std::vector<int> tokens;
		ModelCluster::SeqSnapshot snap;
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

	// Longest entry whose tokens are a proper prefix of prompt; nullptr if none. Counts a lookup and marks the entry used.
	const Entry* find( const std::vector<int>& prompt );
	bool contains( const std::vector<int>& tokens ) const;
	// Takes ownership of snap (freed back to the cluster on failure or eviction). An entry with identical tokens is replaced.
	bool insert( std::vector<int> tokens, ModelCluster::SeqSnapshot snap, bool pinned );
	void clear();
	Stats stats() const;
	size_t budget() const { return budget_; }

private:
	static size_t cost( const ModelCluster::SeqSnapshot& s, int tp ) { return s.bytes * static_cast<size_t>( tp ); }
	void evict_for( size_t incoming );

	ModelCluster& cluster_;
	size_t budget_;
	std::vector<std::unique_ptr<Entry>> entries_;
	uint64_t clock_ = 0;
	Stats stats_;
};

} // namespace engine::runtime
