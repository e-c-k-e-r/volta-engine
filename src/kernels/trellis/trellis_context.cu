#include "trellis_context.h"
#include "exl3_gemm.cuh"
#include <mutex>
#include <cstdio>

namespace engine::modules {

static int* g_locks[ENGINE_NUM_GPUS] = { nullptr };
static void* g_workspace[ENGINE_NUM_GPUS] = { nullptr };
static size_t g_workspace_size[ENGINE_NUM_GPUS] = { 0 };
static std::mutex g_trellis_ctx_mtx;

int trellis_get_num_sms( int device ) {
	(void) device;
	return ENGINE_SM_COUNT;
}

int trellis_get_smem_request( int device ) {
	(void) device;
	return ENGINE_MAX_SMEM;
}

int trellis_get_cc( int device ) {
	(void) device;
	return 1;
}

int* trellis_get_locks( int device ) {
	if ( device < 0 || device >= ENGINE_NUM_GPUS ) {
		device = 0;
	}
	if ( g_locks[device] == nullptr ) {
		std::lock_guard<std::mutex> lock( g_trellis_ctx_mtx );
		if ( g_locks[device] == nullptr ) {
			int orig_dev = 0;
			cudaGetDevice( &orig_dev );
			cudaSetDevice( device );
			size_t size = ( MAX_TILES_C + 2 * MAX_BARRIERS + 64 ) * sizeof( int );
			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &g_locks[device] ), size ) );
			CUDA_CHECK( cudaMemset( g_locks[device], 0, size ) );
			cudaSetDevice( orig_dev );
		}
	}
	return g_locks[device];
}

half* trellis_get_workspace( int device, size_t bytes ) {
	if ( device < 0 || device >= ENGINE_NUM_GPUS ) {
		device = 0;
	}
	if ( g_workspace[device] == nullptr || g_workspace_size[device] < bytes ) {
		std::lock_guard<std::mutex> lock( g_trellis_ctx_mtx );
		if ( g_workspace[device] == nullptr || g_workspace_size[device] < bytes ) {
			int orig_dev = 0;
			cudaGetDevice( &orig_dev );
			cudaSetDevice( device );
			if ( g_workspace[device] != nullptr ) {
				CUDA_CHECK( cudaFree( g_workspace[device] ) );
				g_workspace[device] = nullptr;
			}
			size_t alloc_bytes = bytes < ( 16 * 1024 * 1024 ) ? ( 16 * 1024 * 1024 ) : bytes;
			CUDA_CHECK( cudaMalloc( &g_workspace[device], alloc_bytes ) );
			g_workspace_size[device] = alloc_bytes;
			cudaSetDevice( orig_dev );
		}
	}
	return static_cast<half*>( g_workspace[device] );
}

void trellis_context_init_all() {
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		trellis_get_workspace( d, 16 * 1024 * 1024 );
		trellis_get_locks( d );
		exl3_gemm_init_all_attrs( d );
	}
}

void trellis_context_destroy() {
	std::lock_guard<std::mutex> lock( g_trellis_ctx_mtx );
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		if ( g_locks[d] != nullptr ) {
			cudaSetDevice( d );
			cudaFree( g_locks[d] );
			g_locks[d] = nullptr;
		}
		if ( g_workspace[d] != nullptr ) {
			cudaSetDevice( d );
			cudaFree( g_workspace[d] );
			g_workspace[d] = nullptr;
			g_workspace_size[d] = 0;
		}
	}
}

} // namespace engine::modules
