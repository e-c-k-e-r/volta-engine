#pragma once

#include "engine/core/arena.h"
#include <cstddef>
#include <cstdint>

namespace engine::core {

// Device memory a kernel module keeps between calls: split-K and attention partials, a dequantized weight, activations quantized for a
// GEMV. Once the cluster attaches a device's arena every block comes out of it, so nothing a model runs allocates device memory after the
// arena is sized; with no arena attached (kernel tests, tools) a block is allocated from the device directly.
// budget: bytes taken from the arena at once (its first allocation, so the model sizes everything else from what is left) for the blocks; a
// block that does not fit in it comes from the arena's free space and is logged.
void scratch_attach( int device, pod::MemoryArena* arena, size_t budget );
// Forgets the device's blocks: the arena's go with the arena, direct ones are freed
void scratch_detach( int device );

// The block of `site` (a name, compared by address) and `sub` (e.g. a stream) on the current device, of at least `bytes`. A request larger than
// the block takes a new one and leaves the old one allocated, since a graph captured with it still reads it.
void* scratch( const char* site, size_t bytes, uintptr_t sub = 0 );

// Arena bytes the device's blocks hold
size_t scratch_taken( int device );

} // namespace engine::core
