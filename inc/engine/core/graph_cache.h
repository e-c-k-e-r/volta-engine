#pragma once

#include "engine/config.h"
#include "engine/core/batch.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace engine::core {

// Per-row-count CUDA graph cache for a model's decode body (optionally two variants per count, for bodies whose
// launch sequence depends on a host-known row property such as all rows sharing one slot).
// The captured body must take every step-varying value (slots, positions, token ids,
// all-reduce sequence numbers) from device memory so a replay is valid for any step.
//   call 1 for m : eager (lazy one-time setup such as smem attributes happens outside capture)
//   call 2 for m : capture (thread-local mode, 4 ranks capture concurrently) + instantiate
//   call 3+      : cudaGraphLaunch
// VOLTA_NO_GRAPHS=1 forces eager execution.
class GraphCache {
public:
	GraphCache() {
		const char* env = std::getenv( "VOLTA_NO_GRAPHS" );
		enabled_ = !( env && env[0] == '1' );
	}
	GraphCache( const GraphCache& ) = delete;
	GraphCache& operator=( const GraphCache& ) = delete;
	~GraphCache() { reset(); }

	template <typename Body>
	void run( int m, cudaStream_t stream, Body&& body, int variant = 0 ) {
		ENGINE_ASSERT( variant >= 0 && variant < 2, "GraphCache::run: variant out of range" );
		ENGINE_ASSERT( m >= 1 && m <= pod::MAX_ROWS, "GraphCache::run: row count out of range" );
		// Prefill chunks (above the decode sizes) run eagerly: a graph per chunk size would never be replayed.
		if ( !enabled_ || stream == nullptr || m > pod::MAX_BATCH ) {
			body( stream );
			return;
		}
		cudaGraphExec_t& exec = exec_[m][variant];
		if ( !exec ) {
			if ( !warm_[m][variant] ) {
				warm_[m][variant] = true;
				body( stream );
				return;
			}
			cudaGraph_t graph = nullptr;
			CUDA_CHECK( cudaStreamBeginCapture( stream, cudaStreamCaptureModeThreadLocal ) );
			body( stream );
			CUDA_CHECK( cudaStreamEndCapture( stream, &graph ) );
			size_t nodes = 0;
			CUDA_CHECK( cudaGraphGetNodes( graph, nullptr, &nodes ) );
			CUDA_CHECK( cudaGraphInstantiate( &exec, graph, nullptr, nullptr, 0 ) );
			CUDA_CHECK( cudaGraphDestroy( graph ) );
			int dev = 0;
			CUDA_CHECK( cudaGetDevice( &dev ) );
			if ( dev == 0 ) {
				std::printf( "    [graph] captured m=%d decode body: %zu nodes\n", m, nodes );
			}
		}
		CUDA_CHECK( cudaGraphLaunch( exec, stream ) );
		++replays_;
	}

	// Caller must have the owning device current
	void reset() {
		for ( int m = 0; m <= pod::MAX_BATCH; ++m ) {
			for ( int v = 0; v < 2; ++v ) {
				if ( exec_[m][v] ) {
					cudaGraphExecDestroy( exec_[m][v] );
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
	cudaGraphExec_t exec_[pod::MAX_BATCH + 1][2] = {};
	bool warm_[pod::MAX_BATCH + 1][2] = {};
	bool enabled_ = true;
	unsigned long long replays_ = 0;
};

} // namespace engine::core
