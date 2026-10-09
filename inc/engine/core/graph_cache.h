#pragma once

#include "engine/config.h"
#include "engine/core/log.h"
#include "engine/runtime/options.h"
#include "engine/core/batch.h"
#include "engine/backend/runtime.h"
#include <cstdio>
#include <cstdlib>

namespace engine::core {

// Per-row-count CUDA graph cache for a model's decode body (optionally two variants per count, for bodies whose
// launch sequence depends on a host-known row property such as all rows sharing one slot).
// The captured body must take every step-varying value (slots, positions, token ids,
// all-reduce sequence numbers) from device memory so a replay is valid for any step.
//   call 1 for m : eager (lazy one-time setup such as smem attributes happens outside capture)
//   call 2 for m : capture (thread-local mode, the ranks capture concurrently) + instantiate
//   call 3+      : cudaGraphLaunch
// ENGINE_DEBUG_NO_GRAPHS=1 (Options::no_graphs) forces eager execution.
class GraphCache {
public:
	GraphCache() {
		enabled_ = !runtime::options().no_graphs;
	}
	GraphCache( const GraphCache& ) = delete;
	GraphCache& operator=( const GraphCache& ) = delete;
	~GraphCache() { reset(); }

	template <typename Body>
	void run( int m, backend::stream_t stream, Body&& body, int variant = 0 ) {
		ENGINE_ASSERT( variant >= 0 && variant < 2, "GraphCache::run: variant out of range" );
		ENGINE_ASSERT( m >= 1 && m <= pod::MAX_ROWS, "GraphCache::run: row count out of range" );
#if !ENGINE_BACKEND_HAS_GRAPHS
		// No capture on this backend: the body is the execution
		body( stream );
		return;
#else
		// Prefill chunks (above the decode sizes) run eagerly: a graph per chunk size would never be replayed.
		if ( !enabled_ || stream == nullptr || m > pod::MAX_BATCH ) {
			body( stream );
			return;
		}
		backend::graph_exec_t& exec = exec_[m][variant];
		if ( !exec ) {
			if ( !warm_[m][variant] ) {
				warm_[m][variant] = true;
				body( stream );
				return;
			}
			backend::graph_capture_begin( stream );
			body( stream );
			backend::graph_t graph = backend::graph_capture_end( stream );
			const size_t nodes = backend::graph_node_count( graph );
			exec = backend::graph_instantiate( graph );
			backend::graph_destroy( graph );
			if ( backend::get_device() == 0 ) {
				LOG_DEBUG( "[graph] captured m=%d decode body: %zu nodes\n", m, nodes );
			}
		}
		backend::graph_launch( exec, stream );
		++replays_;
#endif
	}

	// Caller must have the owning device current
	void reset() {
		for ( int m = 0; m <= pod::MAX_BATCH; ++m ) {
			for ( int v = 0; v < 2; ++v ) {
				if ( exec_[m][v] ) {
					backend::graph_exec_destroy( exec_[m][v] );
					exec_[m][v] = nullptr;
				}
				warm_[m][v] = false;
			}
		}
	}

	bool enabled() const { return enabled_; }
	unsigned long long replays() const { return replays_; }
	void set_enabled( bool on ) { enabled_ = on; }

private:
	backend::graph_exec_t exec_[pod::MAX_BATCH + 1][2] = {};
	bool warm_[pod::MAX_BATCH + 1][2] = {};
	bool enabled_ = true;
	unsigned long long replays_ = 0;
};

} // namespace engine::core
