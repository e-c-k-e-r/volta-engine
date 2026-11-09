#pragma once

#include "engine/backend/runtime.h"

// Vulkan-only layer below the portable runtime.h: availability, compute pipelines from SPIR-V and dispatch recording.
// The kernels take their parameters as a push-constant block (Slang `uniform Params p`, std430 layout: natural C layout with
// 8-byte device pointers) and read buffers through physical-storage-buffer pointers, so there are no descriptors.
namespace engine::backend::vk {

struct Pipeline;

// True when an instance and a device with the required features (BDA, Float16, Int64, timeline semaphores) exist
bool available();
const char* device_name();
uint32_t max_push_constants_bytes();
// Packed 4 x int8 dot products (VK_KHR_shader_integer_dot_product) are enabled on the current device
bool int_dot();

// SPIR-V file (entry point "main"); push_bytes = sizeof( Params ), at most max_push_constants_bytes(). nullptr on failure
Pipeline* pipeline_create( const char* spv_path, uint32_t push_bytes );
void pipeline_destroy( Pipeline* p );
// A pipeline whose shader reads `buffers` storage buffers at bindings 0 .. buffers - 1 of set 0 (GLSL shaders, e.g. the vendored ggml ones),
// with spec_count 32-bit specialization constants (constant_id 0 .. spec_count - 1). Needs bound_pipelines(); nullptr on failure
Pipeline* pipeline_create_bound( const char* spv_path, uint32_t push_bytes, uint32_t buffers, const uint32_t* spec, uint32_t spec_count );
// VK_KHR_push_descriptor is enabled: pipelines with bound buffers can be created and recorded (also into graphs)
bool bound_pipelines();

// Cross-device synchronization without the host (the multi-device collective): a host allocation every device imports as addressable memory
// (VK_EXT_external_memory_host) and timeline semaphores one device signals and the others wait on (VK_KHR_external_semaphore_fd)
bool shared_sync();
size_t host_import_alignment();   // the host pointer and size of import_host_memory must be multiples of it
// The host block as a device pointer on the current device (nullptr on failure); memcpy and kernels reach it over the bus
void* import_host_memory( void* host, size_t bytes );
void* shared_timeline_create();          // an exportable timeline semaphore of the current device
int shared_timeline_fd( void* sem );     // a new fd for importing it elsewhere (each import consumes one)
void* shared_timeline_import( int fd );  // that semaphore on the current device
// What is recorded on s so far is submitted, signalling sem = value when it completes
void stream_signal( stream_t s, void* sem, uint64_t value );
// What is recorded on s from now on waits until every sems[i] reaches values[i]
void stream_wait( stream_t s, void* const* sems, const uint64_t* values, int n );

// Record bind + push constants + dispatch( gx, gy, gz workgroups ) + a compute/transfer barrier into the stream
// (nullptr = default stream). While a graph capture is active this records into the graph.
void dispatch( stream_t s, Pipeline* p, const void* push, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1 );
// dispatch for a pipeline_create_bound pipeline: buffers[i] (device pointers, 16-byte aligned) bound at binding i, each up to the end of
// its allocation
void dispatch_bound( stream_t s, Pipeline* p, const void* push, const void* const* buffers, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1 );

} // namespace engine::backend::vk
