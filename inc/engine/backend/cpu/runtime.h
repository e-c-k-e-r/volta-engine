#pragma once

#include "engine/config.h"
#include "engine/backend/types.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// CPU backend runtime layer: one logical device, all "device" memory is 64-byte aligned host memory, copies are memcpy,
// streams are no-ops (work executes eagerly on the calling thread, so there is nothing to wait for).
// Graphs: there is no capture. ENGINE_BACKEND_HAS_GRAPHS is 0 and engine::core::GraphCache runs the body directly; the
// graph_* functions exist only so the vocabulary matches and abort if reached.
#define ENGINE_BACKEND_HAS_GRAPHS 0

// Argument is a bool expression that must hold
#define ENGINE_CHECK(call) \
	do { \
		if ( ENGINE_UNLIKELY( !( call ) ) ) { \
			std::fprintf( stderr, "[CPU ERROR] %s (%s:%d)\n", #call, __FILE__, __LINE__ ); \
			std::_Exit( 1 ); \
		} \
	} while ( 0 )

namespace engine::backend {

inline void set_device( int device ) { ENGINE_CHECK( device == 0 ); }
inline int get_device() { return 0; }
inline int device_count() { return 1; }

inline stream_t stream_create() { return nullptr; }
inline void stream_destroy( stream_t ) {}
inline void stream_sync( stream_t ) {}
inline void device_sync() {}

inline void memcpy_async( void* dst, const void* src, size_t bytes, Copy, stream_t ) { if ( bytes ) std::memmove( dst, src, bytes ); }
inline void memcpy2d_async( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, Copy, stream_t ) {
	for ( size_t r = 0; r < height; ++r ) {
		std::memcpy( static_cast<char*>( dst ) + r * dpitch, static_cast<const char*>( src ) + r * spitch, width );
	}
}
inline void memcpy_sync( void* dst, const void* src, size_t bytes, Copy ) { if ( bytes ) std::memmove( dst, src, bytes ); }
inline void peer_copy_async( void* dst, int, const void* src, int, size_t bytes, stream_t ) { if ( bytes ) std::memmove( dst, src, bytes ); }
inline void memset_async( void* dst, int value, size_t bytes, stream_t ) { if ( bytes ) std::memset( dst, value, bytes ); }
inline void memset_sync( void* dst, int value, size_t bytes ) { if ( bytes ) std::memset( dst, value, bytes ); }

inline void* aligned_alloc64( size_t bytes ) {
	void* p = nullptr;
	return posix_memalign( &p, 64, bytes ? bytes : 64 ) == 0 ? p : nullptr;
}

// Pinned/mapped flavors are plain host memory (everything is host-visible)
inline void* host_alloc_pinned( size_t bytes, Pin ) { return aligned_alloc64( bytes ); }
inline void host_free_pinned( void* p ) { std::free( p ); }

inline bool last_error_ok() { return true; }
inline void check_last_error() {}
inline void clear_last_error() {}
inline void device_reset() {}

inline bool set_device_try( int device ) { return device == 0; }
inline void* device_malloc_try( size_t bytes ) { return aligned_alloc64( bytes ); }
inline bool memcpy_async_try( void* dst, const void* src, size_t bytes, Copy dir, stream_t s ) { memcpy_async( dst, src, bytes, dir, s ); return true; }

inline void* device_malloc( size_t bytes ) {
	void* p = aligned_alloc64( bytes );
	ENGINE_CHECK( p != nullptr );
	return p;
}
inline void device_free( void* p ) { std::free( p ); }

inline void graph_capture_begin( stream_t ) { ENGINE_CHECK( !"CPU backend has no graph capture" ); }
inline graph_t graph_capture_end( stream_t ) { ENGINE_CHECK( !"CPU backend has no graph capture" ); return nullptr; }
inline graph_exec_t graph_instantiate( graph_t ) { ENGINE_CHECK( !"CPU backend has no graph capture" ); return nullptr; }
inline size_t graph_node_count( graph_t ) { return 0; }
inline void graph_launch( graph_exec_t, stream_t ) { ENGINE_CHECK( !"CPU backend has no graph capture" ); }
inline void graph_destroy( graph_t ) {}
inline void graph_exec_destroy( graph_exec_t ) {}

} // namespace engine::backend
