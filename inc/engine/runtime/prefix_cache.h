#pragma once

#include "engine/runtime/cluster.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace engine::runtime {

// Host-RAM store of sequence snapshots keyed by the exact token prefix they cover, organized as a tree: a node holds the
// KV rows of its own segment [parent.tokens, tokens) plus the recurrent state at its end, so a chain of checkpoints along
// one prompt stores every KV row once. A request resumes from the deepest node whose tokens are a proper prefix of its
// prompt (at least one token must remain to produce logits). Eviction is least-recently-used over leaves; pinned nodes
// (the shared system-prompt root) and nodes with children stay. Not thread safe: scheduler thread only.
//
// Disk tier (optional, a directory): an evicted node is written to a file instead of being dropped (by a writer thread, which frees its host
// buffers afterwards), every node is written when the cache is destroyed, and the files found at start are indexed, so a prefix survives a
// restart. A lookup that matches deeper on disk than in RAM reads that node and its missing ancestors back into RAM. Files live under a
// subdirectory named by a fingerprint of everything the snapshot layout depends on (model, KV format, ranks, sizes), so another
// configuration never reads them. The tier keeps its own byte budget, dropping the least recently used files.
class PrefixCache {
public:
	struct Entry {
		std::vector<int> tokens;        // the whole prefix
		Entry* parent = nullptr;
		int children = 0;
		ModelCluster::SeqSnapshot snap; // KV rows [parent->tokens.size() or 0, tokens.size()) + state at the end
		uint64_t last_used = 0;
		bool pinned = false;
		bool on_disk = false;           // a file holds the same node (an eviction just frees it)
	};

	struct Stats {
		uint64_t lookups = 0;
		uint64_t hits = 0;
		uint64_t hit_tokens = 0;     // prompt tokens not prefilled thanks to hits
		uint64_t inserts = 0;
		uint64_t evictions = 0;
		size_t entries = 0;
		size_t bytes = 0;            // pinned host bytes held by entries, all ranks
		uint64_t disk_hits = 0;      // lookups served (in part) from files
		size_t disk_entries = 0;
		size_t disk_bytes = 0;
	};

	// budget_bytes: total pinned host bytes across ranks the entries may hold
	PrefixCache( ModelCluster& cluster, size_t budget_bytes ) : cluster_( cluster ), budget_( budget_bytes ) {}
	~PrefixCache();
	// Enables the disk tier: files under dir/<fingerprint>/, at most disk_budget bytes; indexes the files already there. False when the
	// directory cannot be created.
	bool enable_disk( const std::string& dir, const std::string& fingerprint, size_t disk_budget );

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

	// disk tier
	struct DiskEntry {
		std::vector<int> tokens;
		int from = 0;
		bool with_taps = false;
		size_t bytes = 0;        // per rank
		std::string path;
		uint64_t last_used = 0;
		bool ready = false;      // the file is complete
	};
	struct WriteJob {
		std::vector<int> tokens;
		ModelCluster::SeqSnapshot snap;
		std::string path;
		bool free_after = false;
	};
	bool disk_on() const { return !disk_dir_.empty(); }
	std::string disk_path( const std::vector<int>& tokens ) const;
	void spill( Entry* e, bool own_buffers );      // queue the node's file (own_buffers: the writer frees them)
	bool write_file( const WriteJob& job );
	void writer_loop();
	void drain_writes();
	void disk_evict_for( size_t incoming );
	Entry* promote( const std::vector<int>& prompt, Entry* ram_best );   // a deeper disk match read back into RAM, or ram_best
	Entry* load_disk( size_t index, Entry* parent );

	ModelCluster& cluster_;
	size_t budget_;
	std::vector<std::unique_ptr<Entry>> entries_;
	uint64_t clock_ = 0;
	Stats stats_;
	std::string disk_dir_;
	size_t disk_budget_ = 0;
	std::vector<DiskEntry> disk_;   // guarded by disk_mtx_ (the writer marks entries ready)
	mutable std::mutex disk_mtx_;
	std::condition_variable disk_cv_;
	std::deque<WriteJob> writes_;
	bool writer_stop_ = false;
	int writes_busy_ = 0;
	std::thread writer_;
};

} // namespace engine::runtime
