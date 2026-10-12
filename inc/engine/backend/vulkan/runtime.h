#pragma once

#include "engine/backend/types.h"
#include "engine/config.h"
#include <cstddef>

// Vulkan backend runtime layer (selected by engine/backend/runtime.h): the same function set as the CUDA header, implemented
// in src/backend/vulkan. Streams record command buffers that are submitted on sync; a "graph" is a pre-recorded command
// buffer. Device pointers are buffer device addresses (BDA) carried in void*.
//
// Several devices: set_device(d) makes device d current for the calling thread (one context per device). A device pointer is only valid on the
// device that allocated it, and other devices may hand out the same addresses, so a copy between devices must be peer_copy_async (through the host)
// and never a plain memcpy_async of a foreign pointer. Not implemented (asserts): mapped pinned host memory (returns nullptr), memset with
// unaligned address/size, host transfers inside a graph capture.
#define ENGINE_CHECK( call ) ::engine::backend::check_result( int( call ), #call, __FILE__, __LINE__ )

namespace engine::backend {

// Aborts when result != 0 (VkResult style: VK_SUCCESS == 0)
void check_result( int result, const char* expr, const char* file, int line );

void set_device( int device );
int get_device();
int device_count();

stream_t stream_create();
void stream_destroy( stream_t s );
void stream_sync( stream_t s );
// One queue: side streams are the main stream, so ordering is implicit
inline void stream_fork( stream_t, stream_t ) {}
inline void stream_join( stream_t, stream_t ) {}
void device_sync();

void memcpy_async( void* dst, const void* src, size_t bytes, Copy dir, stream_t s );
void memcpy2d_async( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, Copy dir, stream_t s );
void memcpy_sync( void* dst, const void* src, size_t bytes, Copy dir );
void peer_copy_async( void* dst, int dst_device, const void* src, int src_device, size_t bytes, stream_t s );
void memset_async( void* dst, int value, size_t bytes, stream_t s );
void memset_sync( void* dst, int value, size_t bytes );

void* host_alloc_pinned( size_t bytes, Pin mode );
void host_free_pinned( void* p );

bool last_error_ok();
void check_last_error();
void clear_last_error();
void device_reset();

bool set_device_try( int device );
size_t device_free_bytes( int device );   // 0 when unknown
void* device_malloc_try( size_t bytes );
bool memcpy_async_try( void* dst, const void* src, size_t bytes, Copy dir, stream_t s );

void* device_malloc( size_t bytes );
void device_free( void* p );

void graph_capture_begin( stream_t s );
graph_t graph_capture_end( stream_t s );
graph_exec_t graph_instantiate( graph_t g );
size_t graph_node_count( graph_t g );
void graph_launch( graph_exec_t e, stream_t s );
void graph_destroy( graph_t g );
void graph_exec_destroy( graph_exec_t e );

} // namespace engine::backend
