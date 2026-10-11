// Host-to-device upload of page-cache (mmap'd) data through a ring of pinned chunks: the CPU copy of one chunk (several threads, page faults included) overlaps the
// DMA of the previous ones. Used by the weight loaders; one instance per loading thread.
#pragma once
#include "engine/config.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <thread>

namespace engine::video {

class H3Stager {
public:
	static constexpr size_t kChunk = 16ull << 20;
	static constexpr int kSlots = 4, kCopyThreads = 4;

	explicit H3Stager( cudaStream_t stream ) : stream_( stream ) {
		for ( int i = 0; i < kSlots; ++i ) {
			CUDA_CHECK( cudaHostAlloc( &pinned_[i], kChunk, cudaHostAllocDefault ) );
			CUDA_CHECK( cudaEventCreateWithFlags( &free_[i], cudaEventDisableTiming ) );
			CUDA_CHECK( cudaEventRecord( free_[i], stream_ ) );
		}
	}
	~H3Stager() {
		for ( int i = 0; i < kSlots; ++i ) {
			cudaEventDestroy( free_[i] );
			cudaFreeHost( pinned_[i] );
		}
	}
	H3Stager( const H3Stager& ) = delete;
	H3Stager& operator=( const H3Stager& ) = delete;

	// Enqueues the copy of `bytes` from `src` to the device address `dst` on the stream; `src` must stay valid until the stream is synchronized (it is copied
	// into pinned memory before this returns, so only `dst` is in flight)
	void upload( void* dst, const void* src, size_t bytes ) {
		const char* p = static_cast<const char*>( src );
		for ( size_t off = 0, k = 0; off < bytes; off += kChunk, ++k ) {
			const int slot = static_cast<int>( next_++ % kSlots );
			const size_t n = std::min( kChunk, bytes - off );
			CUDA_CHECK( cudaEventSynchronize( free_[slot] ) ); // the DMA that last read this chunk is done
			const size_t part = ( n + kCopyThreads - 1 ) / kCopyThreads;
			std::thread helpers[kCopyThreads - 1];
			for ( int t = 1; t < kCopyThreads; ++t ) {
				const size_t b = std::min( n, part * t ), e = std::min( n, part * ( t + 1 ) );
				helpers[t - 1] = std::thread( [this, slot, p, off, b, e] { std::memcpy( static_cast<char*>( pinned_[slot] ) + b, p + off + b, e - b ); } );
			}
			std::memcpy( pinned_[slot], p + off, std::min( n, part ) );
			for ( std::thread& h : helpers ) { h.join(); }
			CUDA_CHECK( cudaMemcpyAsync( static_cast<char*>( dst ) + off, pinned_[slot], n, cudaMemcpyHostToDevice, stream_ ) );
			CUDA_CHECK( cudaEventRecord( free_[slot], stream_ ) );
		}
	}

private:
	cudaStream_t stream_;
	void* pinned_[kSlots] = {};
	cudaEvent_t free_[kSlots] = {};
	size_t next_ = 0;
};

} // namespace engine::video
