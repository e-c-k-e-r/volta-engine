#include "engine/core/arena.h"
#include "engine/backend/runtime.h"

namespace engine::core {

bool arena_create( pod::MemoryArena& arena, int device_id, size_t total_bytes ) {
	arena_destroy( arena );

	if ( ENGINE_UNLIKELY( !backend::set_device_try( device_id ) ) ) {
		return false;
	}

	void* ptr = backend::device_malloc_try( total_bytes );
	if ( ENGINE_UNLIKELY( ptr == nullptr ) ) {
		return false;
	}

	arena.base_ptr = ptr;
	arena.total_bytes = total_bytes;
	arena.offset_bytes = 0;
	arena.pinned = false;
	return true;
}

bool arena_create_pinned( pod::MemoryArena& arena, size_t total_bytes ) {
	arena_destroy( arena );
	void* ptr = backend::host_alloc_pinned( total_bytes, backend::Pin::Mapped );
	if ( ptr == nullptr ) {
		return false;
	}
	arena.base_ptr = ptr;
	arena.total_bytes = total_bytes;
	arena.offset_bytes = 0;
	arena.pinned = true;
	return true;
}

void* arena_alloc( pod::MemoryArena& arena, size_t bytes, size_t alignment ) {
	ENGINE_ASSERT( arena.base_ptr != nullptr, "Arena not initialized" );
	ENGINE_ASSERT( alignment > 0 && ( alignment & ( alignment - 1 ) ) == 0, "Alignment must be power of 2" );

	size_t aligned_offset = ( arena.offset_bytes + alignment - 1 ) & ~( alignment - 1 );
	size_t new_offset = aligned_offset + bytes;

	if ( ENGINE_UNLIKELY( new_offset > arena.total_bytes ) ) {
		ENGINE_ASSERT( false, "MemoryArena out of memory" );
		return nullptr;
	}

	void* res = static_cast<char*>( arena.base_ptr ) + aligned_offset;
	arena.offset_bytes = new_offset;
	return res;
}

pod::TensorView arena_alloc_tensor( pod::MemoryArena& arena, pod::DType dtype, const int* shape, int ndim, size_t alignment ) {
	ENGINE_ASSERT( ndim >= 1 && ndim <= 4, "Shape must have 1 to 4 dimensions" );

	size_t numel = 1;
	for ( int i = 0; i < ndim; ++i ) {
		ENGINE_ASSERT( shape[i] > 0, "Dimension size must be positive" );
		numel *= static_cast<size_t>( shape[i] );
	}

	size_t bytes = numel * pod::dtype_bytes( dtype );
	void* data_ptr = arena_alloc( arena, bytes, alignment );

	switch ( ndim ) {
		case 1: {
			return pod::make_tensor_1d( data_ptr, dtype, shape[0] );
		} break;
		case 2: {
			return pod::make_tensor_2d( data_ptr, dtype, shape[0], shape[1] );
		} break;
		case 3: {
			return pod::make_tensor_3d( data_ptr, dtype, shape[0], shape[1], shape[2] );
		} break;
		case 4: {
			return pod::make_tensor_4d( data_ptr, dtype, shape[0], shape[1], shape[2], shape[3] );
		} break;
		default: {
			return pod::TensorView{};
		} break;
	}
}

void arena_reset( pod::MemoryArena& arena ) {
	arena.offset_bytes = 0;
}

void arena_destroy( pod::MemoryArena& arena ) {
	if ( arena.base_ptr != nullptr ) {
		if ( arena.pinned ) backend::host_free_pinned( arena.base_ptr );
		else backend::device_free( arena.base_ptr );
		arena.pinned = false;
		arena.base_ptr = nullptr;
		arena.total_bytes = 0;
		arena.offset_bytes = 0;
	}
}

void* host_alloc_pinned( size_t bytes ) {
	return backend::host_alloc_pinned( bytes, backend::Pin::Default );
}

void host_free_pinned( void* ptr ) {
	if ( ptr != nullptr ) {
		backend::host_free_pinned( ptr );
	}
}

} // namespace engine::core
