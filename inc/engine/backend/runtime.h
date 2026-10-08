#pragma once

#include "engine/backend/types.h"

// Thin runtime layer the models, runtime and core use instead of naming the vendor API. In the CUDA backend every
// function is a one-line inline forward, so the generated code is the same as calling cuda* directly.
#if defined(ENGINE_BACKEND_CUDA) && ENGINE_BACKEND_CUDA == 1

	#define ENGINE_CHECK(call) CUDA_CHECK(call)

namespace engine::backend {

inline cudaMemcpyKind copy_kind( Copy c ) {
	switch ( c ) {
		case Copy::H2D: return cudaMemcpyHostToDevice;
		case Copy::D2H: return cudaMemcpyDeviceToHost;
		default: return cudaMemcpyDeviceToDevice;
	}
}

inline void set_device( int device ) { ENGINE_CHECK( cudaSetDevice( device ) ); }
inline int get_device() { int d = 0; ENGINE_CHECK( cudaGetDevice( &d ) ); return d; }
inline int device_count() { int n = 0; ENGINE_CHECK( cudaGetDeviceCount( &n ) ); return n; }

inline stream_t stream_create() { stream_t s = nullptr; ENGINE_CHECK( cudaStreamCreate( &s ) ); return s; }
inline void stream_destroy( stream_t s ) { ENGINE_CHECK( cudaStreamDestroy( s ) ); }
inline void stream_sync( stream_t s ) { ENGINE_CHECK( cudaStreamSynchronize( s ) ); }
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

// Pinned host memory. mapped = device-visible and portable across devices (collective flags, tap rings)
inline void* host_alloc_pinned( size_t bytes, bool mapped ) {
	void* p = nullptr;
	cudaError_t err = cudaHostAlloc( &p, bytes, mapped ? ( cudaHostAllocMapped | cudaHostAllocPortable ) : cudaHostAllocDefault );
	return err == cudaSuccess ? p : nullptr;
}
inline void host_free_pinned( void* p ) { cudaFreeHost( p ); }

// Last launch error (kernel launches are asynchronous; this reports configuration failures)
inline bool last_error_ok() { return cudaGetLastError() == cudaSuccess; }

inline void* device_malloc( size_t bytes ) { void* p = nullptr; ENGINE_CHECK( cudaMalloc( &p, bytes ) ); return p; }
inline void device_free( void* p ) { ENGINE_CHECK( cudaFree( p ) ); }

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
#endif
