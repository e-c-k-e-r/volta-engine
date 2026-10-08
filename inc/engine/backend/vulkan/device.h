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

// SPIR-V file (entry point "main"); push_bytes = sizeof( Params ), at most max_push_constants_bytes(). nullptr on failure
Pipeline* pipeline_create( const char* spv_path, uint32_t push_bytes );
void pipeline_destroy( Pipeline* p );

// Record bind + push constants + dispatch( gx, gy, gz workgroups ) + a compute/transfer barrier into the stream
// (nullptr = default stream). While a graph capture is active this records into the graph.
void dispatch( stream_t s, Pipeline* p, const void* push, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1 );

} // namespace engine::backend::vk
