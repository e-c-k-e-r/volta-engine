#include "trellis_context.h"
#include "exl3_gemm.cuh"
#include "exl3_gemv_int8.cuh"
#include <mutex>
#include <cstdio>

namespace engine::modules {

static int* g_locks[ENGINE_NUM_GPUS] = { nullptr };
static int* g_int8_ws[ENGINE_NUM_GPUS] = { nullptr };
static void* g_workspace[ENGINE_NUM_GPUS] = { nullptr };
static size_t g_workspace_size[ENGINE_NUM_GPUS] = { 0 };
static void* g_recon_ws[ENGINE_NUM_GPUS] = { nullptr };
static size_t g_recon_ws_size[ENGINE_NUM_GPUS] = { 0 };
static cublasHandle_t g_cublas[ENGINE_NUM_GPUS] = { nullptr };
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

int* trellis_get_int8_ws( int device ) {
	if ( device < 0 || device >= ENGINE_NUM_GPUS ) {
		device = 0;
	}
	if ( g_int8_ws[device] == nullptr ) {
		std::lock_guard<std::mutex> lock( g_trellis_ctx_mtx );
		if ( g_int8_ws[device] == nullptr ) {
			int orig_dev = 0;
			cudaGetDevice( &orig_dev );
			cudaSetDevice( device );
			// Zeroed so the sq kernel's per-256-column completion counters start at zero; the finishing
			// block resets them, so they stay zero between calls and across graph replays.
			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &g_int8_ws[device] ), INT8_WS_BYTES ) );
			CUDA_CHECK( cudaMemset( g_int8_ws[device], 0, INT8_WS_BYTES ) );
			cudaSetDevice( orig_dev );
		}
	}
	return g_int8_ws[device];
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

cublasHandle_t trellis_get_cublas( int device ) {
	if ( device < 0 || device >= ENGINE_NUM_GPUS ) device = 0;
	if ( g_cublas[device] == nullptr ) {
		std::lock_guard<std::mutex> lock( g_trellis_ctx_mtx );
		if ( g_cublas[device] == nullptr ) {
			int orig_dev = 0;
			cudaGetDevice( &orig_dev );
			cudaSetDevice( device );
			ENGINE_ASSERT( cublasCreate( &g_cublas[device] ) == CUBLAS_STATUS_SUCCESS, "cublasCreate failed" );
			cublasSetMathMode( g_cublas[device], CUBLAS_TENSOR_OP_MATH );
			cudaSetDevice( orig_dev );
		}
	}
	return g_cublas[device];
}

half* trellis_get_recon_ws( int device, size_t bytes ) {
	if ( device < 0 || device >= ENGINE_NUM_GPUS ) device = 0;
	if ( g_recon_ws[device] == nullptr || g_recon_ws_size[device] < bytes ) {
		std::lock_guard<std::mutex> lock( g_trellis_ctx_mtx );
		if ( g_recon_ws[device] == nullptr || g_recon_ws_size[device] < bytes ) {
			int orig_dev = 0;
			cudaGetDevice( &orig_dev );
			cudaSetDevice( device );
			if ( g_recon_ws[device] != nullptr ) {
				CUDA_CHECK( cudaFree( g_recon_ws[device] ) );
				g_recon_ws[device] = nullptr;
			}
			CUDA_CHECK( cudaMalloc( &g_recon_ws[device], bytes ) );
			g_recon_ws_size[device] = bytes;
			cudaSetDevice( orig_dev );
		}
	}
	return static_cast<half*>( g_recon_ws[device] );
}

void trellis_context_init_all() {
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		trellis_get_workspace( d, 16 * 1024 * 1024 );
		trellis_get_locks( d );
		trellis_get_int8_ws( d );
		exl3_gemm_init_all_attrs( d );
		exl3_gemv_int8_init( d );
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
		if ( g_int8_ws[d] != nullptr ) {
			cudaSetDevice( d );
			cudaFree( g_int8_ws[d] );
			g_int8_ws[d] = nullptr;
		}
		if ( g_cublas[d] != nullptr ) {
			cudaSetDevice( d );
			cublasDestroy( g_cublas[d] );
			g_cublas[d] = nullptr;
		}
		if ( g_recon_ws[d] != nullptr ) {
			cudaSetDevice( d );
			cudaFree( g_recon_ws[d] );
			g_recon_ws[d] = nullptr;
			g_recon_ws_size[d] = 0;
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
