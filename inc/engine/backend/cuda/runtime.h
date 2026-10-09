#pragma once

#include "engine/backend/types.h"

// CUDA backend runtime layer (selected by engine/backend/runtime.h): every function is a one-line inline forward to the
// CUDA runtime API, so the generated code is the same as calling cuda* directly.
	#define ENGINE_CHECK(call) CUDA_CHECK(call)

namespace engine::backend {

inline cudaMemcpyKind copy_kind( Copy c ) {
	switch ( c ) {
		case Copy::H2D: return cudaMemcpyHostToDevice;
		case Copy::D2H: return cudaMemcpyDeviceToHost;
		case Copy::D2D: return cudaMemcpyDeviceToDevice;
		default: return cudaMemcpyDefault;
	}
}

inline void set_device( int device ) { ENGINE_CHECK( cudaSetDevice( device ) ); }
inline int get_device() { int d = 0; ENGINE_CHECK( cudaGetDevice( &d ) ); return d; }
inline int device_count() { int n = 0; ENGINE_CHECK( cudaGetDeviceCount( &n ) ); return n; }

inline stream_t stream_create() { stream_t s = nullptr; ENGINE_CHECK( cudaStreamCreate( &s ) ); return s; }
inline void stream_destroy( stream_t s ) { ENGINE_CHECK( cudaStreamDestroy( s ) ); }
inline void stream_sync( stream_t s ) { ENGINE_CHECK( cudaStreamSynchronize( s ) ); }
// Side-stream ordering: after stream_fork( main, side ) work queued on `side` waits for everything queued on `main` so far;
// stream_join( main, side ) makes `main` wait for everything queued on `side`. Both capture as graph edges.
inline cudaEvent_t& fork_event() {
	thread_local cudaEvent_t e = [] { cudaEvent_t x = nullptr; ENGINE_CHECK( cudaEventCreateWithFlags( &x, cudaEventDisableTiming ) ); return x; }();
	return e;
}
inline void stream_fork( stream_t main, stream_t side ) { ENGINE_CHECK( cudaEventRecord( fork_event(), main ) ); ENGINE_CHECK( cudaStreamWaitEvent( side, fork_event(), 0 ) ); }
inline void stream_join( stream_t main, stream_t side ) { ENGINE_CHECK( cudaEventRecord( fork_event(), side ) ); ENGINE_CHECK( cudaStreamWaitEvent( main, fork_event(), 0 ) ); }
inline void device_sync() { ENGINE_CHECK( cudaDeviceSynchronize() ); }

inline void memcpy_async( void* dst, const void* src, size_t bytes, Copy dir, stream_t s ) {
	ENGINE_CHECK( cudaMemcpyAsync( dst, src, bytes, copy_kind( dir ), s ) );
}
inline void memcpy2d_async( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, Copy dir, stream_t s ) {
	ENGINE_CHECK( cudaMemcpy2DAsync( dst, dpitch, src, spitch, width, height, copy_kind( dir ), s ) );
}
inline void memcpy_sync( void* dst, const void* src, size_t bytes, Copy dir ) {
	ENGINE_CHECK( cudaMemcpy( dst, src, bytes, copy_kind( dir ) ) );
}
inline void peer_copy_async( void* dst, int dst_device, const void* src, int src_device, size_t bytes, stream_t s ) {
	ENGINE_CHECK( cudaMemcpyPeerAsync( dst, dst_device, src, src_device, bytes, s ) );
}
inline void memset_async( void* dst, int value, size_t bytes, stream_t s ) { ENGINE_CHECK( cudaMemsetAsync( dst, value, bytes, s ) ); }
inline void memset_sync( void* dst, int value, size_t bytes ) { ENGINE_CHECK( cudaMemset( dst, value, bytes ) ); }

// Pinned host memory (nullptr on failure). Mapped flavors are device-visible (collective flags, tap rings, embedding table)
inline void* host_alloc_pinned( size_t bytes, Pin mode ) {
	void* p = nullptr;
	const unsigned flags = mode == Pin::MappedPortable ? ( cudaHostAllocMapped | cudaHostAllocPortable ) : mode == Pin::Mapped ? cudaHostAllocMapped : cudaHostAllocDefault;
	return cudaHostAlloc( &p, bytes, flags ) == cudaSuccess ? p : nullptr;
}
inline void host_free_pinned( void* p ) { cudaFreeHost( p ); }

// Last launch error (kernel launches are asynchronous; this reports configuration failures)
inline bool last_error_ok() { return cudaGetLastError() == cudaSuccess; }
inline void check_last_error() { ENGINE_CHECK( cudaGetLastError() ); }
inline void clear_last_error() { (void) cudaGetLastError(); }
inline void device_reset() { cudaDeviceReset(); }

// Non-aborting forms for callers that report failure to their own caller (arena creation, loaders)
inline bool set_device_try( int device ) { return cudaSetDevice( device ) == cudaSuccess; }
// Free device memory in bytes on `device`; 0 when unknown
inline size_t device_free_bytes( int device ) { size_t f = 0, t = 0; return ( cudaSetDevice( device ) == cudaSuccess && cudaMemGetInfo( &f, &t ) == cudaSuccess ) ? f : 0; }
inline void* device_malloc_try( size_t bytes ) { void* p = nullptr; return cudaMalloc( &p, bytes ) == cudaSuccess ? p : nullptr; }
inline bool memcpy_async_try( void* dst, const void* src, size_t bytes, Copy dir, stream_t s ) {
	return cudaMemcpyAsync( dst, src, bytes, copy_kind( dir ), s ) == cudaSuccess;
}

inline void* device_malloc( size_t bytes ) { void* p = nullptr; ENGINE_CHECK( cudaMalloc( &p, bytes ) ); return p; }
inline void device_free( void* p ) { (void) cudaFree( p ); }

// Graph capture and replay (the decode body is captured once per row count and replayed)
inline void graph_capture_begin( stream_t s ) { ENGINE_CHECK( cudaStreamBeginCapture( s, cudaStreamCaptureModeThreadLocal ) ); }
inline graph_t graph_capture_end( stream_t s ) { graph_t g = nullptr; ENGINE_CHECK( cudaStreamEndCapture( s, &g ) ); return g; }
inline graph_exec_t graph_instantiate( graph_t g ) {
	graph_exec_t e = nullptr;
	ENGINE_CHECK( cudaGraphInstantiate( &e, g, 0ULL ) );
	return e;
}
inline size_t graph_node_count( graph_t g ) { size_t n = 0; ENGINE_CHECK( cudaGraphGetNodes( g, nullptr, &n ) ); return n; }
inline void graph_launch( graph_exec_t e, stream_t s ) { ENGINE_CHECK( cudaGraphLaunch( e, s ) ); }
inline void graph_destroy( graph_t g ) { ENGINE_CHECK( cudaGraphDestroy( g ) ); }
inline void graph_exec_destroy( graph_exec_t e ) { cudaGraphExecDestroy( e ); }

} // namespace engine::backend
