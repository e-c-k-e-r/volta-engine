// Host-to-device upload of page-cache (mmap'd) data through a ring of pinned chunks: the CPU copy of one chunk (several threads, page faults included) overlaps the
// DMA of the previous ones. Used by the weight loaders. An H3Stager owns its ring; with an H3StageGroup the ring is shared by several loader threads (one per GPU)
// that upload the same tensors in the same order: one thread copies each chunk out of the page cache once and every GPU DMAs it.
#pragma once
#include "engine/config.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

namespace engine::video {

constexpr size_t kStageChunk = 16ull << 20;
constexpr int kStageSlots = 4, kStageCopyThreads = 4;

// Copies `bytes` from `src` into the pinned buffer with several threads (one thread reaches only ~5 GB/s of page-cache reads)
inline void h3_parallel_copy( void* dst, const void* src, size_t bytes ) {
	const size_t part = ( bytes + kStageCopyThreads - 1 ) / kStageCopyThreads;
	std::thread helpers[kStageCopyThreads - 1];
	for ( int t = 1; t < kStageCopyThreads; ++t ) {
		const size_t b = std::min( bytes, part * t ), e = std::min( bytes, part * ( t + 1 ) );
		helpers[t - 1] = std::thread( [dst, src, b, e] { std::memcpy( static_cast<char*>( dst ) + b, static_cast<const char*>( src ) + b, e - b ); } );
	}
	std::memcpy( dst, src, std::min( bytes, part ) );
	for ( std::thread& h : helpers ) { h.join(); }
}

// The shared ring of a group of n loader threads. Rank 0 reads the source; all ranks must call H3Stager::upload with the same sizes in the same order.
class H3StageGroup {
public:
	explicit H3StageGroup( int n ) : n_( n ) {
		for ( int i = 0; i < kStageSlots; ++i ) { CUDA_CHECK( cudaHostAlloc( &pinned_[i], kStageChunk, cudaHostAllocPortable ) ); }
	}
	~H3StageGroup() { for ( int i = 0; i < kStageSlots; ++i ) { (void) cudaFreeHost( pinned_[i] ); } }
	H3StageGroup( const H3StageGroup& ) = delete;
	H3StageGroup& operator=( const H3StageGroup& ) = delete;

	int size() const { return n_; }
	void* slot( int i ) const { return pinned_[i]; }
	cudaEvent_t& event( int rank, int slot ) { return events_[rank][slot]; } // set when the rank's DMA of the slot's last chunk has been enqueued
	static constexpr int kMaxRanks = 8;
	// a loader that cannot continue (missing tensor) releases the others so nobody waits forever
	void abort() {
		std::lock_guard<std::mutex> lk( m_ );
		aborted_ = true;
		cv_.notify_all();
	}
	bool aborted() const { return aborted_; }
	void barrier() {
		std::unique_lock<std::mutex> lk( m_ );
		if ( aborted_ ) { return; }
		const int gen = gen_;
		if ( ++count_ == n_ ) {
			count_ = 0;
			++gen_;
			cv_.notify_all();
		} else {
			cv_.wait( lk, [&] { return gen != gen_ || aborted_; } );
		}
	}

private:
	int n_, count_ = 0, gen_ = 0;
	bool aborted_ = false;
	void* pinned_[kStageSlots] = {};
	cudaEvent_t events_[kMaxRanks][kStageSlots] = {};
	std::mutex m_;
	std::condition_variable cv_;
};

class H3Stager {
public:
	explicit H3Stager( cudaStream_t stream, H3StageGroup* group = nullptr, int rank = 0 ) : stream_( stream ), group_( group ), rank_( rank ) {
		for ( int i = 0; i < kStageSlots; ++i ) {
			if ( !group_ ) { CUDA_CHECK( cudaHostAlloc( &own_[i], kStageChunk, cudaHostAllocDefault ) ); }
			CUDA_CHECK( cudaEventCreateWithFlags( &free_[i], cudaEventDisableTiming ) );
			CUDA_CHECK( cudaEventRecord( free_[i], stream_ ) );
			if ( group_ ) { group_->event( rank_, i ) = free_[i]; }
		}
	}
	~H3Stager() {
		for ( int i = 0; i < kStageSlots; ++i ) {
			(void) cudaEventDestroy( free_[i] );
			if ( own_[i] ) { (void) cudaFreeHost( own_[i] ); }
		}
	}
	H3Stager( const H3Stager& ) = delete;
	H3Stager& operator=( const H3Stager& ) = delete;

	// Enqueues the copy of `bytes` from `src` to the device address `dst` on the stream (the data is in pinned memory before this returns). In a group, `src` only
	// matters on rank 0 but every rank must pass the same `bytes`.
	void upload( void* dst, const void* src, size_t bytes ) {
		const char* p = static_cast<const char*>( src );
		for ( size_t off = 0; off < bytes; off += kStageChunk ) {
			const int slot = static_cast<int>( next_++ % kStageSlots );
			const size_t n = std::min( kStageChunk, bytes - off );
			void* pin = group_ ? group_->slot( slot ) : own_[slot];
			if ( !group_ ) {
				CUDA_CHECK( cudaEventSynchronize( free_[slot] ) ); // the DMA that last read this chunk is done
				h3_parallel_copy( pin, p + off, n );
			} else {
				if ( rank_ == 0 ) { // every rank's DMA of this slot's previous chunk has been enqueued before the previous barriers; wait for them to finish
					for ( int r = 0; r < group_->size(); ++r ) { if ( group_->event( r, slot ) ) { CUDA_CHECK( cudaEventSynchronize( group_->event( r, slot ) ) ); } }
					h3_parallel_copy( pin, p + off, n );
				}
				group_->barrier(); // the chunk is in the ring
			}
			CUDA_CHECK( cudaMemcpyAsync( static_cast<char*>( dst ) + off, pin, n, cudaMemcpyHostToDevice, stream_ ) );
			CUDA_CHECK( cudaEventRecord( free_[slot], stream_ ) );
		}
	}

private:
	cudaStream_t stream_;
	H3StageGroup* group_;
	int rank_;
	void* own_[kStageSlots] = {};
	cudaEvent_t free_[kStageSlots] = {};
	size_t next_ = 0;
};

} // namespace engine::video
