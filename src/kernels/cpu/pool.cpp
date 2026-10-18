// Thread pool of the CPU kernels (see pool.h)
#include "engine/kernels/cpu/pool.h"
#include <chrono>
#if defined( __x86_64__ )
#include <immintrin.h>
#define CPUX_X86 1
#endif
#if defined( __linux__ )
#include <sched.h>
#include <pthread.h>
#endif

namespace engine::kernels::cpu {

Pool::Pool( int threads, const std::vector<int>& cpus, int spin_us ) : n_( threads < 1 ? 1 : threads ), spin_us_( spin_us ), cpus_( cpus ) {
	for ( int t = 1; t < n_; ++t ) th_.emplace_back( [this, t] { worker( t ); } );
}

Pool::~Pool() {
	stop_ = true;
	{ std::lock_guard<std::mutex> l( mu_ ); gen_.fetch_add( 1 ); }
	cv_.notify_all();
	for ( auto& t : th_ ) t.join();
}

void Pool::worker( int tid ) {
#if defined( __linux__ )
	if ( static_cast<size_t>( tid - 1 ) < cpus_.size() ) {
		cpu_set_t cs; CPU_ZERO( &cs ); CPU_SET( cpus_[tid - 1], &cs );
		pthread_setaffinity_np( pthread_self(), sizeof cs, &cs );
	}
#endif
	uint32_t seen = 0;
	for ( ;; ) {
		const auto t0 = std::chrono::steady_clock::now();
		uint32_t g;
		int spins = 0;
		while ( ( g = gen_.load( std::memory_order_acquire ) ) == seen ) {
#if CPUX_X86
			_mm_pause();
#endif
			if ( ( ++spins & 1023 ) == 0 && std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::steady_clock::now() - t0 ).count() > spin_us_ ) {
				std::unique_lock<std::mutex> l( mu_ );
				parked_.fetch_add( 1 );
				cv_.wait( l, [&] { return gen_.load( std::memory_order_acquire ) != seen; } );
				parked_.fetch_sub( 1 );
			}
		}
		if ( stop_.load() ) return;
		seen = g;
		Task* t = cur_.load( std::memory_order_acquire );
		if ( !t ) continue;
		t->refs.fetch_add( 1, std::memory_order_acq_rel );
		if ( cur_.load( std::memory_order_acquire ) == t ) {
			while ( !t->finished() ) {
				if ( !t->work() ) {
#if CPUX_X86
					_mm_pause();
#endif
				}
			}
		}
		t->refs.fetch_sub( 1, std::memory_order_acq_rel );
	}
}

void Pool::run( Task& task ) {
	if ( n_ > 1 ) {
		cur_.store( &task, std::memory_order_release );
		{ std::lock_guard<std::mutex> l( mu_ ); gen_.fetch_add( 1, std::memory_order_release ); }   // under the lock: a worker deciding to park sees it or is notified
		if ( parked_.load() > 0 ) cv_.notify_all();
	}
	task.refs.fetch_add( 1, std::memory_order_acq_rel );
	while ( !task.finished() ) {
		if ( !task.work() ) {
#if CPUX_X86
			_mm_pause();
#endif
		}
	}
	task.refs.fetch_sub( 1, std::memory_order_acq_rel );
	if ( n_ > 1 ) cur_.store( nullptr, std::memory_order_release );
}

}   // namespace engine::kernels::cpu
