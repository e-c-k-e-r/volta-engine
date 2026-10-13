#include "trellis_context.h"
#include "exl3_gemm.cuh"
#include "exl3_gemv_int8.cuh"
#include "exl3_kernel_map.cuh"
#include <mutex>
#include <cstdio>

void exl3_gemm_reset_caches();
void exl3_gemv_reset_caches();
void exl3_gemv_int8_reset_caches();
namespace ext { void exl3_moe_reset_caches(); }

namespace engine::modules {

static int* g_locks[ENGINE_NUM_GPUS] = { nullptr };
static int* g_int8_ws[ENGINE_NUM_GPUS] = { nullptr };
static void* g_workspace[ENGINE_NUM_GPUS] = { nullptr };
static size_t g_workspace_size[ENGINE_NUM_GPUS] = { 0 };
static void* g_recon_ws[ENGINE_NUM_GPUS] = { nullptr };
static size_t g_recon_ws_size[ENGINE_NUM_GPUS] = { 0 };
static cublasHandle_t g_cublas[ENGINE_NUM_GPUS] = { nullptr };
static std::mutex g_trellis_ctx_mtx;

// Persistent and cooperative launches size their grids to the SM count, which must be the card's own: 80 on a V100, 82 on a 3090,
// 60 on a 4070 Ti (a cooperative grid larger than what fits co-resident fails to launch, a smaller one leaves SMs idle). A nonzero
// ENGINE_SM_COUNT replaces the queried count, but never above it
int trellis_get_num_sms( int device ) {
	static int cached[ENGINE_NUM_GPUS] = {};
	if ( device < 0 || device >= ENGINE_NUM_GPUS ) device = 0;
	if ( cached[device] == 0 ) {
		int n = 0;
		if ( cudaDeviceGetAttribute( &n, cudaDevAttrMultiProcessorCount, device ) != cudaSuccess || n <= 0 ) n = ENGINE_SM_COUNT > 0 ? ENGINE_SM_COUNT : 80;
		else if ( ENGINE_SM_COUNT > 0 && ENGINE_SM_COUNT < n ) n = ENGINE_SM_COUNT;
		cached[device] = n;
	}
	return cached[device];
}

int trellis_get_smem_request( int device ) {
	(void) device;
	return ENGINE_MAX_SMEM;
}

// Architecture class the EXL3 launchers key their shape and path choices on (CC_OLD = Volta / Turing, as upstream exllamav3's
// DevCtx::get_cc maps it); cached per device
int trellis_get_cc( int device ) {
	static int cached[ENGINE_NUM_GPUS] = {};
	if ( device < 0 || device >= ENGINE_NUM_GPUS ) device = 0;
	if ( cached[device] == 0 ) {
#if defined(USE_ROCM)
		// HIP reports the gfx generation as the major version (gfx11 -> 11), which the NVIDIA thresholds would read as Blackwell. RDNA
		// takes the oldest class's choices, as upstream does on ROCm (EXL3 GEMM shapes are scored separately there: exl3_kernel_map.cu)
		cached[device] = CC_OLD;
		return cached[device];
#endif
		int major = 0, minor = 0;
		(void) cudaDeviceGetAttribute( &major, cudaDevAttrComputeCapabilityMajor, device );
		(void) cudaDeviceGetAttribute( &minor, cudaDevAttrComputeCapabilityMinor, device );
		if ( major >= 10 ) cached[device] = CC_BLACKWELL;
		else if ( major >= 9 ) cached[device] = CC_HOPPER;
		else if ( major == 8 && minor >= 9 ) cached[device] = CC_ADA;
		else if ( major >= 8 ) cached[device] = CC_AMPERE;
		else cached[device] = CC_OLD;
	}
	return cached[device];
}

int* trellis_get_locks( int device ) {
	if ( device < 0 || device >= ENGINE_NUM_GPUS ) {
		device = 0;
	}
	if ( g_locks[device] == nullptr ) {
		std::lock_guard<std::mutex> lock( g_trellis_ctx_mtx );
		if ( g_locks[device] == nullptr ) {
			int orig_dev = 0;
			(void) cudaGetDevice( &orig_dev );
			(void) cudaSetDevice( device );
			size_t size = ( MAX_TILES_C + 2 * MAX_BARRIERS + 64 ) * sizeof( int );
			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &g_locks[device] ), size ) );
			CUDA_CHECK( cudaMemset( g_locks[device], 0, size ) );
			(void) cudaSetDevice( orig_dev );
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
			(void) cudaGetDevice( &orig_dev );
			(void) cudaSetDevice( device );
			// Zeroed so the sq kernel's per-256-column completion counters start at zero; the finishing
			// block resets them, so they stay zero between calls and across graph replays.
			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &g_int8_ws[device] ), INT8_WS_BYTES ) );
			CUDA_CHECK( cudaMemset( g_int8_ws[device], 0, INT8_WS_BYTES ) );
			(void) cudaSetDevice( orig_dev );
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
			(void) cudaGetDevice( &orig_dev );
			(void) cudaSetDevice( device );
			// A decode graph captured earlier holds the old pointer (and only ever needs its size), so the old buffer is
			// retired, not freed: a large prefill chunk that outgrows 16 MiB must not leave graph replays dangling.
			// Growth happens a handful of times, so the retired buffers cost a few tens of MiB at most.
			g_workspace[device] = nullptr;
			size_t alloc_bytes = bytes < ( 16 * 1024 * 1024 ) ? ( 16 * 1024 * 1024 ) : bytes;
			CUDA_CHECK( cudaMalloc( &g_workspace[device], alloc_bytes ) );
			g_workspace_size[device] = alloc_bytes;
			(void) cudaSetDevice( orig_dev );
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
			(void) cudaGetDevice( &orig_dev );
			(void) cudaSetDevice( device );
			ENGINE_ASSERT( cublasCreate( &g_cublas[device] ) == CUBLAS_STATUS_SUCCESS, "cublasCreate failed" );
			cublasSetMathMode( g_cublas[device], CUBLAS_TENSOR_OP_MATH );
			(void) cudaSetDevice( orig_dev );
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
			(void) cudaGetDevice( &orig_dev );
			(void) cudaSetDevice( device );
			if ( g_recon_ws[device] != nullptr ) {
				CUDA_CHECK( cudaFree( g_recon_ws[device] ) );
				g_recon_ws[device] = nullptr;
			}
			CUDA_CHECK( cudaMalloc( &g_recon_ws[device], bytes ) );
			g_recon_ws_size[device] = bytes;
			(void) cudaSetDevice( orig_dev );
		}
	}
	return static_cast<half*>( g_recon_ws[device] );
}

// Empty kernel: whether it has an image for a device says whether the build targets that device's architecture
static __global__ void arch_probe_kernel() {}

// A card the build has no code for fails on its first launch with "no kernel image"; say which architecture to add instead. Not fatal
// here: a machine may hold a card that the configured tensor-parallel ranks never use
static bool check_device_arch( int device ) {
	cudaDeviceProp prop{};
	CUDA_CHECK( cudaGetDeviceProperties( &prop, device ) );
	cudaFuncAttributes attr{};
	int orig_dev = 0;
	(void) cudaGetDevice( &orig_dev );
	(void) cudaSetDevice( device );
	const cudaError_t err = cudaFuncGetAttributes( &attr, arch_probe_kernel );
	(void) cudaSetDevice( orig_dev );
	if ( err != cudaSuccess ) {
		(void) cudaGetLastError();
#if defined(USE_ROCM)
		std::fprintf( stderr, "[WARN] GPU %d (%s, %s) has no kernels in this build (%s): any work placed on it will fail. Reconfigure with "
			"-DCMAKE_HIP_ARCHITECTURES=%s (or a ;-list including it) and rebuild\n", device, prop.name, prop.gcnArchName, cudaGetErrorString( err ),
			prop.gcnArchName );
		return false;
#endif
		std::fprintf( stderr, "[WARN] GPU %d (%s) is SM_%d%d, which this build has no kernels for (%s): any work placed on it will "
			"fail. Reconfigure with -DCMAKE_CUDA_ARCHITECTURES=%d%d (or a ;-list including it) and rebuild\n", device, prop.name,
			prop.major, prop.minor, cudaGetErrorString( err ), prop.major, prop.minor );
		return false;
	}
#if defined(USE_ROCM)
	std::printf( "[GPU %d] %s, %s, %d CUs, %.1f GiB, %d KB LDS per workgroup, wave%d\n", device, prop.name, prop.gcnArchName,
		prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0, rocm_smem_optin_max() / 1024, prop.warpSize );
	ENGINE_ASSERT( prop.warpSize == 32, "ROCm: the kernels need wave32 (RDNA); this device runs wave64" );
	ENGINE_ASSERT( rocm_smem_optin_max() >= ENGINE_MAX_SMEM, "ROCm: the device grants a workgroup less LDS than ENGINE_MAX_SMEM (64 KB)" );
#else
	std::printf( "[GPU %d] %s, SM_%d%d, %d SMs, %.1f GiB, %zu KB shared memory per block (opt-in)\n", device, prop.name, prop.major,
		prop.minor, prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0, prop.sharedMemPerBlockOptin / 1024 );
	ENGINE_ASSERT( prop.sharedMemPerBlockOptin >= static_cast<size_t>( ENGINE_MAX_SMEM ),
		"the GPU allows less opt-in shared memory per block than ENGINE_MAX_SMEM (default 96 KB, which SM_70, SM_86 and SM_89 parts all allow)" );
#endif
	return true;
}

void trellis_context_init_all() {
	int devices = 0;
	(void) cudaGetDeviceCount( &devices );
	for ( int d = 0; d < devices && d < ENGINE_NUM_GPUS; ++d ) {
		if ( !check_device_arch( d ) ) continue;
		trellis_get_workspace( d, 16 * 1024 * 1024 );
		trellis_get_locks( d );
		trellis_get_int8_ws( d );
		exl3_gemm_init_all_attrs( d );
		exl3_gemv_int8_init( d );
	}
}

void trellis_reset_launch_caches() {
	exl3_gemm_reset_caches();
	exl3_gemv_reset_caches();
	exl3_gemv_int8_reset_caches();
	::ext::exl3_moe_reset_caches();
}

void trellis_context_destroy() {
	trellis_reset_launch_caches();
	std::lock_guard<std::mutex> lock( g_trellis_ctx_mtx );
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		if ( g_locks[d] != nullptr ) {
			(void) cudaSetDevice( d );
			(void) cudaFree( g_locks[d] );
			g_locks[d] = nullptr;
		}
		if ( g_int8_ws[d] != nullptr ) {
			(void) cudaSetDevice( d );
			(void) cudaFree( g_int8_ws[d] );
			g_int8_ws[d] = nullptr;
		}
		if ( g_cublas[d] != nullptr ) {
			(void) cudaSetDevice( d );
			cublasDestroy( g_cublas[d] );
			g_cublas[d] = nullptr;
		}
		if ( g_recon_ws[d] != nullptr ) {
			(void) cudaSetDevice( d );
			(void) cudaFree( g_recon_ws[d] );
			g_recon_ws[d] = nullptr;
			g_recon_ws_size[d] = 0;
		}
		if ( g_workspace[d] != nullptr ) {
			(void) cudaSetDevice( d );
			(void) cudaFree( g_workspace[d] );
			g_workspace[d] = nullptr;
			g_workspace_size[d] = 0;
		}
	}
}

} // namespace engine::modules
