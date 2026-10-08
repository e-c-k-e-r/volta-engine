#pragma once

#include "engine/config.h"
#include "engine/backend/runtime.h"
#include <pthread.h>
#include <sched.h>
#include <atomic>
#include <functional>
#include <thread>

namespace engine::runtime {

// Persistent per-GPU worker threads (one per rank, pinned to NUMA node 0 core == rank).
// run( fn ) executes fn( rank ) on every rank concurrently and returns once all ranks finish.
class RankPool {
public:
	RankPool() = default;
	RankPool( const RankPool& ) = delete;
	RankPool& operator=( const RankPool& ) = delete;
	~RankPool() { stop(); }

	void start( int num_ranks = ENGINE_NUM_GPUS ) {
		ENGINE_ASSERT( !running_, "RankPool::start: already running" );
		num_ranks_ = num_ranks;
		pthread_barrier_init( &barrier_, nullptr, num_ranks_ + 1 );
		running_ = true;
		for ( int r = 0; r < num_ranks_; ++r ) {
			threads_[r] = std::thread( &RankPool::worker, this, r );
		}
	}

	void run( const std::function<void( int )>& fn ) {
		ENGINE_ASSERT( running_, "RankPool::run: pool not started" );
		task_ = &fn;
		pthread_barrier_wait( &barrier_ ); // release workers
		pthread_barrier_wait( &barrier_ ); // wait for completion
		task_ = nullptr;
	}

	void stop() {
		if ( !running_ ) return;
		task_ = nullptr;
		stopping_ = true;
		pthread_barrier_wait( &barrier_ );
		for ( int r = 0; r < num_ranks_; ++r ) {
			threads_[r].join();
		}
		pthread_barrier_destroy( &barrier_ );
		running_ = false;
		stopping_ = false;
	}

	int num_ranks() const { return num_ranks_; }

private:
	void worker( int rank ) {
		cpu_set_t cpuset;
		CPU_ZERO( &cpuset );
		CPU_SET( rank, &cpuset );
		pthread_setaffinity_np( pthread_self(), sizeof( cpu_set_t ), &cpuset );
		backend::set_device( rank );
		while ( true ) {
			pthread_barrier_wait( &barrier_ );
			if ( stopping_ ) break;
			( *task_ )( rank );
			pthread_barrier_wait( &barrier_ );
		}
	}

	std::thread threads_[ENGINE_NUM_GPUS];
	pthread_barrier_t barrier_{};
	const std::function<void( int )>* task_ = nullptr;
	std::atomic<bool> stopping_{ false };
	bool running_ = false;
	int num_ranks_ = 0;
};

} // namespace engine::runtime
