#pragma once

#include "engine/config.h"
#include "engine/core/tensor.h"
#include <cstddef>

namespace pod {

struct MemoryArena {
	void*  base_ptr;       // Device pointer returned by single cudaMalloc
	size_t total_bytes;    // Total allocated VRAM (e.g. 30,720 MiB)
	size_t offset_bytes;   // Bump offset for static reservations
};

} // namespace pod

namespace engine::core {

bool arena_create( pod::MemoryArena& arena, int device_id, size_t total_bytes );
void* arena_alloc( pod::MemoryArena& arena, size_t bytes, size_t alignment = 256 );
pod::TensorView arena_alloc_tensor( pod::MemoryArena& arena, pod::DType dtype, const int* shape, int ndim, size_t alignment = 256 );
void arena_reset( pod::MemoryArena& arena );
void arena_destroy( pod::MemoryArena& arena );

void* host_alloc_pinned( size_t bytes );
void host_free_pinned( void* ptr );

} // namespace engine::core
