#pragma once

#include "engine/config.h"
#include "engine/core/arena.h"
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

// Aborts on a cuBLAS error (the caller includes <cublas_v2.h>)
#define CUBLAS_CHECK( call ) \
	do { \
		cublasStatus_t st_ = ( call ); \
		if ( ENGINE_UNLIKELY( st_ != CUBLAS_STATUS_SUCCESS ) ) { \
			std::fprintf( stderr, "[CUBLAS ERROR] %s (%s:%d): %d\n", #call, __FILE__, __LINE__, static_cast<int>( st_ ) ); \
			std::_Exit( 1 ); \
		} \
	} while ( 0 )

namespace engine::video {

// A scratch arena that starts zeroed. A fresh process is handed zeroed device memory, a long-running one (the server) reuses blocks freed by earlier stages, so a
// kernel that reads scratch before writing it (padding rows, register tokens) would give different results from job to job. Weights are fully written and need no fill.
inline bool h3_scratch_arena_create( pod::MemoryArena& arena, int device, size_t bytes ) {
	if ( !engine::core::arena_create( arena, device, bytes ) ) { return false; }
	CUDA_CHECK( cudaMemset( arena.base_ptr, 0, arena.total_bytes ) );
	// the models run on non-blocking streams, which the legacy default stream does not order against: wait for the fill, or it can land after a first write
	CUDA_CHECK( cudaStreamSynchronize( nullptr ) );
	return true;
}

// FNV-1a over the bits of a float array (fingerprints in the logs: equal inputs and settings give equal values)
inline uint64_t h3_fingerprint( const std::vector<float>& v, uint64_t h = 1469598103934665603ULL ) {
	for ( float f : v ) {
		uint32_t b;
		std::memcpy( &b, &f, 4 );
		h = ( h ^ b ) * 1099511628211ULL;
	}
	return h;
}

} // namespace engine::video
