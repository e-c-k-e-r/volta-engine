#pragma once
// ROCm (HIP) compatibility layer, force-included (-include) into every translation unit of the ROCM configuration (host C++ and HIP
// alike; CMakeLists.txt). The CUDA sources are compiled unchanged as HIP: this header maps the CUDA runtime names they use onto HIP,
// and src/backend/rocm/include holds stand-ins for the CUDA headers (cuda_runtime.h, cublas_v2.h, mma.h, ...). Device-side PTX has
// per-site ROCm forms (USE_ROCM) in the kernels themselves. Modelled on upstream exllamav3's exllamav3_ext/rocm/compat.h, which
// solves the same problems for its kernels (whose ROCm port this engine's EXL3 kernels share).
//
// Assumes wave32 (RDNA; the CMake configuration rejects wave64 targets): every kernel is written for 32-lane warps.

#if !defined(USE_ROCM)
	#error "src/backend/rocm/compat.h is for the ROCM configuration only"
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <type_traits>

// ---- Types and constants ----------------------------------------------------------------------------------------------------------

using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;
using cudaGraph_t = hipGraph_t;
using cudaGraphExec_t = hipGraphExec_t;
using cudaGraphNode_t = hipGraphNode_t;
using cudaDeviceProp = hipDeviceProp_t;
using cudaFuncAttributes = hipFuncAttributes;
using cudaMemcpyKind = hipMemcpyKind;
using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;
using nv_bfloat16 = __hip_bfloat16;
using nv_bfloat162 = __hip_bfloat162;

#define cudaSuccess hipSuccess
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaHostAllocPortable hipHostMallocPortable
#define cudaHostAllocMapped hipHostMallocMapped
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributePreferredSharedMemoryCarveout hipFuncAttributePreferredSharedMemoryCarveout
#define cudaSharedmemCarveoutMaxShared 100
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor

// ---- Runtime API ------------------------------------------------------------------------------------------------------------------

#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceReset hipDeviceReset
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaMemGetInfo hipMemGetInfo
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaHostAlloc hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2D hipMemcpy2D
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyPeer hipMemcpyPeer
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphGetNodes hipGraphGetNodes
#define cudaLaunchKernel hipLaunchKernel
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor

// CUDA 12's three-argument form (flags) and the older five-argument one
inline hipError_t cudaGraphInstantiate( hipGraphExec_t* exec, hipGraph_t graph, unsigned long long flags ) {
	return hipGraphInstantiateWithFlags( exec, graph, flags );
}
inline hipError_t cudaGraphInstantiate( hipGraphExec_t* exec, hipGraph_t graph, hipGraphNode_t* error_node, char* log, size_t log_size ) {
	return hipGraphInstantiate( exec, graph, error_node, log, log_size );
}

// HIP takes kernels as const void*; CUDA's templates accept typed kernel pointers. A dynamic shared memory request above what the
// device grants a workgroup (64 KB on RDNA) is clamped: the launches that need more are routed elsewhere on ROCm, and an
// unguarded one fails at launch with a resource error rather than here
inline int rocm_smem_optin_max() {
	int device = 0, v = 0;
	(void) hipGetDevice( &device );
	if ( hipDeviceGetAttribute( &v, hipDeviceAttributeSharedMemPerBlockOptin, device ) != hipSuccess || v <= 0 ) {
		(void) hipGetLastError();
		v = 65536;
	}
	return v;
}
template <typename F>
inline hipError_t cudaFuncSetAttribute( F* func, hipFuncAttribute attr, int value ) {
	// The shared-memory carveout is a hint HIP does not implement (it fails with an invalid value that sticks to the next
	// cudaPeekAtLastError): nothing to do
	if ( attr == hipFuncAttributePreferredSharedMemoryCarveout ) return hipSuccess;
	if ( attr == hipFuncAttributeMaxDynamicSharedMemorySize && value > rocm_smem_optin_max() ) value = rocm_smem_optin_max();
	return hipFuncSetAttribute( reinterpret_cast<const void*>( func ), attr, value );
}
template <typename F>
inline hipError_t cudaFuncGetAttributes( hipFuncAttributes* attr, F* func ) {
	return hipFuncGetAttributes( attr, reinterpret_cast<const void*>( func ) );
}
template <typename F>
inline hipError_t cudaLaunchCooperativeKernel( F* func, dim3 grid, dim3 block, void** args, size_t smem, hipStream_t stream ) {
	return hipLaunchCooperativeKernel( reinterpret_cast<const void*>( func ), grid, block, args, static_cast<unsigned int>( smem ), stream );
}
template <typename T>
inline hipError_t cudaGetSymbolAddress( void** ptr, const T& symbol ) {
	return hipGetSymbolAddress( ptr, HIP_SYMBOL( symbol ) );
}

// ---- Device side ------------------------------------------------------------------------------------------------------------------

// Warp-sync intrinsics onto HIP's plain wave intrinsics (as upstream exllamav3): HIP's own *_sync forms take a 64-bit mask and run a
// convergence check before each operation. Every call site passes the full warp or the lanes active at that point, and RDNA runs
// wave32, so dropping the mask is exact. __syncwarp keeps HIP's mask-free form
#define __shfl_sync( mask, var, ... ) __shfl( var, __VA_ARGS__ )
#define __shfl_up_sync( mask, var, ... ) __shfl_up( var, __VA_ARGS__ )
#define __shfl_down_sync( mask, var, ... ) __shfl_down( var, __VA_ARGS__ )
#define __shfl_xor_sync( mask, var, ... ) __shfl_xor( var, __VA_ARGS__ )
#define __ballot_sync( mask, pred ) ( (unsigned) ( __ballot( pred ) & (unsigned long long) (unsigned) ( mask ) ) )
#define __all_sync( mask, pred ) __all( pred )
#define __any_sync( mask, pred ) __any( pred )
#define __syncwarp( ... ) __syncwarp()

#if defined(__HIPCC__)

// __nanosleep only paces spin-waits; s_sleep takes an immediate (units of 64 clocks)
__device__ __forceinline__ void __nanosleep( unsigned int ) {
#if defined(__HIP_DEVICE_COMPILE__)
	__builtin_amdgcn_s_sleep( 1 );
#endif
}

// __dp4a, unsigned form (byte sums against 0x01010101, the codebook decode): v_dot4_u32_u8 on RDNA3 / RDNA4
__device__ __forceinline__ unsigned int __dp4a( unsigned int a, unsigned int b, unsigned int c ) {
#if defined(__HIP_DEVICE_COMPILE__) && ( defined(__GFX11__) || defined(__GFX12__) )
	return __builtin_amdgcn_udot4( a, b, c, false );
#endif
	#pragma unroll
	for ( int i = 0; i < 4; ++i ) c += ( ( a >> ( 8 * i ) ) & 0xffu ) * ( ( b >> ( 8 * i ) ) & 0xffu );
	return c;
}

// dp4a.u32.s32 (unsigned bytes of a times signed bytes of b): no mixed-sign dot instruction on RDNA3, so byte by byte
__device__ __forceinline__ unsigned int rocm_dp4a_us( unsigned int a, unsigned int b, unsigned int c ) {
	int s = static_cast<int>( c );
	#pragma unroll
	for ( int i = 0; i < 4; ++i ) s += static_cast<int>( ( a >> ( 8 * i ) ) & 0xffu ) * static_cast<int>( static_cast<int8_t>( ( b >> ( 8 * i ) ) & 0xffu ) );
	return static_cast<unsigned int>( s );
}

// __ldcs (streaming) is only a cache hint: a nontemporal load. __ldcg (L2, bypassing L1) is load-bearing where kernels read what other
// blocks wrote, and RDNA's per-CU caches are as incoherent as NVIDIA's L1: an agent-scope relaxed atomic load reads at the coherent level
template <typename T>
__device__ __forceinline__ T __ldcs( const T* p ) {
	// The builtin takes scalars and native vectors; HIP's uint2 / uint4 / half2 are structs: load the same bytes as a native type
	typedef uint32_t u32x2 __attribute__( ( ext_vector_type( 2 ) ) );
	typedef uint32_t u32x4 __attribute__( ( ext_vector_type( 4 ) ) );
	using N = typename std::conditional<sizeof( T ) == 1, uint8_t, typename std::conditional<sizeof( T ) == 2, uint16_t,
		typename std::conditional<sizeof( T ) == 4, uint32_t, typename std::conditional<sizeof( T ) == 8, u32x2, u32x4>::type>::type>::type>::type;
	static_assert( sizeof( T ) <= 16 && sizeof( N ) == sizeof( T ), "__ldcs: 1, 2, 4, 8 or 16 bytes" );
	const N v = __builtin_nontemporal_load( reinterpret_cast<const N*>( p ) );
	T r;
	__builtin_memcpy( &r, &v, sizeof( T ) );
	return r;
}
template <typename T>
__device__ __forceinline__ T __ldcg( const T* p ) {
	if constexpr ( sizeof( T ) == 4 || sizeof( T ) == 8 ) {
		using U = typename std::conditional<sizeof( T ) == 4, uint32_t, uint64_t>::type;
		const U u = __hip_atomic_load( reinterpret_cast<const U*>( p ), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT );
		T v;
		__builtin_memcpy( &v, &u, sizeof( T ) );
		return v;
	}
	else {
		static_assert( sizeof( T ) % 4 == 0, "__ldcg: 4-byte multiple" );
		T v;
		const uint32_t* src = reinterpret_cast<const uint32_t*>( p );
		uint32_t* dst = reinterpret_cast<uint32_t*>( &v );
		#pragma unroll
		for ( int i = 0; i < static_cast<int>( sizeof( T ) / 4 ); ++i ) dst[i] = __hip_atomic_load( src + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT );
		return v;
	}
}

// HIP has __hmax2 / __hmin2 for bfloat162 only
__device__ __forceinline__ __half2 __hmax2( const __half2 a, const __half2 b ) {
	return __halves2half2( __hmax( __low2half( a ), __low2half( b ) ), __hmax( __high2half( a ), __high2half( b ) ) );
}
__device__ __forceinline__ __half2 __hmin2( const __half2 a, const __half2 b ) {
	return __halves2half2( __hmin( __low2half( a ), __low2half( b ) ), __hmin( __high2half( a ), __high2half( b ) ) );
}

// atomicAdd on a half2 (the MoE kernel's output accumulation): a 32-bit compare-and-swap loop (RDNA3 has no packed FP16 global atomic)
__device__ __forceinline__ __half2 atomicAdd( __half2* address, __half2 val ) {
	unsigned int* p = reinterpret_cast<unsigned int*>( address );
	unsigned int old = *p, assumed;
	do {
		assumed = old;
		__half2 cur;
		__builtin_memcpy( &cur, &assumed, 4 );
		const __half2 sum = __hadd2( cur, val );
		unsigned int next;
		__builtin_memcpy( &next, &sum, 4 );
		old = atomicCAS( p, assumed, next );
	} while ( old != assumed );
	__half2 r;
	__builtin_memcpy( &r, &old, 4 );
	return r;
}

#endif // __HIPCC__

// bfloat16 conversion with an explicit rounding mode: HIP's __float2bfloat16 rounds to nearest even
__host__ __device__ inline __hip_bfloat16 __float2bfloat16_rn( const float f ) { return __float2bfloat16( f ); }

// Host C++ translation units (tests, benches): FP16 helpers CUDA also provides on the host
#if !defined(__HIPCC__)
inline unsigned short __half_as_ushort( const __half h ) { unsigned short u; __builtin_memcpy( &u, &h, 2 ); return u; }
inline __half __ushort_as_half( const unsigned short u ) { __half h; __builtin_memcpy( &h, &u, 2 ); return h; }
inline __half __hadd( const __half a, const __half b ) { return __float2half( __half2float( a ) + __half2float( b ) ); }
inline double rsqrt( double x ) { return 1.0 / std::sqrt( x ); }
#endif

// HIP declares rsqrtf for the device only; host code computes softmax scales with it
#if defined(__HIPCC__)
__host__ inline float rsqrtf( float x ) { return 1.0f / std::sqrt( x ); }
#else
inline float rsqrtf( float x ) { return 1.0f / std::sqrt( x ); }
#endif
