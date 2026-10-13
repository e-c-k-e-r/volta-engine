#pragma once
#include <cuda/atomic>
#include "arch.cuh"
#if defined(USE_ROCM)
	// ROCm forms of the PTX below: tensor-core operations emulated in NVIDIA's fragment layouts (mma_emu.h, checked on the CPU by
	// test_mma_frag), memory-model operations as HIP atomics with the matching scope, cp.async as a synchronous copy (as on SM_70)
	#include "../mma_emu.h"
	#define PTX_SYNC_COPY 1
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
	#define PTX_SYNC_COPY 1
#else
	#define PTX_SYNC_COPY 0
#endif

template <typename T, int n>
struct Vec {
	T elems[n];
	__device__ T& operator[]( int i ) { return elems[i]; }
};

using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragC_h = Vec<half2, 2>;

__device__ inline void ptx_mma_m8n8k4(
	const Vec<half2, 2>& frag_a,
	const Vec<half2, 2>& frag_b,
	Vec<float, 8>& frag_c
) {
	const uint32_t* a = reinterpret_cast<const uint32_t*>( &frag_a );
	const uint32_t* b = reinterpret_cast<const uint32_t*>( &frag_b );
	float* c = reinterpret_cast<float*>( &frag_c );
#if defined(USE_ROCM)
	mma_emu_m8n8k4_f32( a, b, c, threadIdx.x & 31, rocm_shfl() );
#else
	const float* d = reinterpret_cast<const float*>( &frag_c );

	asm (
		"mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
		"{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
		: "=f"( c[0] ), "=f"( c[1] ), "=f"( c[2] ), "=f"( c[3] ), "=f"( c[4] ), "=f"( c[5] ), "=f"( c[6] ), "=f"( c[7] )
		: "r"( a[0] ), "r"( a[1] ),
		  "r"( b[0] ), "r"( b[1] ),
		  "f"( d[0] ), "f"( d[1] ), "f"( d[2] ), "f"( d[3] ), "f"( d[4] ), "f"( d[5] ), "f"( d[6] ), "f"( d[7] )
	);
#endif
}

__device__ inline void volta_mma_m8n8k4(
	uint32_t a0,
	uint32_t a1,
	uint32_t b0,
	uint32_t b1,
	float* d
) {
#if defined(USE_ROCM)
	const uint32_t a[2] = { a0, a1 }, b[2] = { b0, b1 };
	mma_emu_m8n8k4_f32( a, b, d, threadIdx.x & 31, rocm_shfl() );
#else
	asm (
		"mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
		"{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
		: "=f"( d[0] ), "=f"( d[1] ), "=f"( d[2] ), "=f"( d[3] ),
		  "=f"( d[4] ), "=f"( d[5] ), "=f"( d[6] ), "=f"( d[7] )
		: "r"( a0 ), "r"( a1 ),
		  "r"( b0 ), "r"( b1 ),
		  "f"( d[0] ), "f"( d[1] ), "f"( d[2] ), "f"( d[3] ),
		  "f"( d[4] ), "f"( d[5] ), "f"( d[6] ), "f"( d[7] )
	);
#endif
}

__device__ inline void volta_d_slot( int lane, int reg, int* m, int* n ) {
	int t = lane & 3;
	*m = ( t & 1 ) + 2 * ( ( reg >> 1 ) & 1 ) + 4 * ( lane >= 16 );
	*n = ( reg & 1 ) + 2 * ( t >> 1 ) + 4 * ( reg >> 2 );
}

__device__ __forceinline__ float hdot2_f( const half2 a, const half2 b ) {
	return __low2float( a ) * __low2float( b ) + __high2float( a ) * __high2float( b );
}

__device__ inline void ptx_mma_m16n8k16(
	const FragA& frag_a,
	const FragB& frag_b,
	FragC& frag_c
) {
#if defined(USE_ROCM)
	mma_emu_m16n8k16_f32( reinterpret_cast<const uint32_t*>( &frag_a ), reinterpret_cast<const uint32_t*>( &frag_b ),
		reinterpret_cast<float*>( &frag_c ), threadIdx.x & 31, rocm_shfl() );
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
	const int lane = threadIdx.x & 31;
	const int g = lane >> 2, tig = lane & 3;
	const half2* fa = reinterpret_cast<const half2*>( &frag_a );
	const half2* fb = reinterpret_cast<const half2*>( &frag_b );
	half2 a00[4], a02[4], a10[4], a12[4];
	#pragma unroll
	for ( int tt = 0; tt < 4; ++tt ) {
		a00[tt] = __shfl_sync( 0xffffffffu, fa[0], g * 4 + tt );
		a02[tt] = __shfl_sync( 0xffffffffu, fa[2], g * 4 + tt );
		a10[tt] = __shfl_sync( 0xffffffffu, fa[1], g * 4 + tt );
		a12[tt] = __shfl_sync( 0xffffffffu, fa[3], g * 4 + tt );
	}
	half2 bx0[4], bx8[4], by0[4], by8[4];
	#pragma unroll
	for ( int tt = 0; tt < 4; ++tt ) {
		bx0[tt] = __shfl_sync( 0xffffffffu, fb[0], 8 * tig + tt );
		bx8[tt] = __shfl_sync( 0xffffffffu, fb[1], 8 * tig + tt );
		by0[tt] = __shfl_sync( 0xffffffffu, fb[0], 8 * tig + 4 + tt );
		by8[tt] = __shfl_sync( 0xffffffffu, fb[1], 8 * tig + 4 + tt );
	}
	half2 s0 = __hmul2( a00[0], bx0[0] );
	half2 s1 = __hmul2( a00[0], by0[0] );
	half2 s2 = __hmul2( a10[0], bx0[0] );
	half2 s3 = __hmul2( a10[0], by0[0] );
	#pragma unroll
	for ( int tt = 1; tt < 4; ++tt ) {
		s0 = __hfma2( a00[tt], bx0[tt], s0 );
		s1 = __hfma2( a00[tt], by0[tt], s1 );
		s2 = __hfma2( a10[tt], bx0[tt], s2 );
		s3 = __hfma2( a10[tt], by0[tt], s3 );
	}
	#pragma unroll
	for ( int tt = 0; tt < 4; ++tt ) {
		s0 = __hfma2( a02[tt], bx8[tt], s0 );
		s1 = __hfma2( a02[tt], by8[tt], s1 );
		s2 = __hfma2( a12[tt], bx8[tt], s2 );
		s3 = __hfma2( a12[tt], by8[tt], s3 );
	}
	frag_c[0] += __low2float( s0 ) + __high2float( s0 );
	frag_c[1] += __low2float( s1 ) + __high2float( s1 );
	frag_c[2] += __low2float( s2 ) + __high2float( s2 );
	frag_c[3] += __low2float( s3 ) + __high2float( s3 );
	return;
#else
	const uint32_t* a = reinterpret_cast<const uint32_t*>( &frag_a );
	const uint32_t* b = reinterpret_cast<const uint32_t*>( &frag_b );
	float* c = reinterpret_cast<float*>( &frag_c );
	const float* d = reinterpret_cast<const float*>( &frag_c );

	asm (
		"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
		"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
		: "=f"( c[0] ), "=f"( c[1] ), "=f"( c[2] ), "=f"( c[3] )
		: "r"( a[0] ), "r"( a[1] ), "r"( a[2] ), "r"( a[3] ),
		  "r"( b[0] ), "r"( b[1] ),
		  "f"( d[0] ), "f"( d[1] ), "f"( d[2] ), "f"( d[3] )
	);
#endif
}

__device__ inline void ptx_mma_m16n8k16(
	const FragA& frag_a,
	const FragB& frag_b,
	FragC_h& frag_c
) {
#if defined(USE_ROCM)
	mma_emu_m16n8k16_f16( reinterpret_cast<const uint32_t*>( &frag_a ), reinterpret_cast<const uint32_t*>( &frag_b ),
		reinterpret_cast<uint32_t*>( &frag_c ), threadIdx.x & 31, rocm_shfl() );
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
	const int lane = threadIdx.x & 31;
	const int g = lane >> 2, tig = lane & 3;
	const half2* fa = reinterpret_cast<const half2*>( &frag_a );
	const half2* fb = reinterpret_cast<const half2*>( &frag_b );
	half2 a00[4], a02[4], a10[4], a12[4];
	#pragma unroll
	for ( int tt = 0; tt < 4; ++tt ) {
		a00[tt] = __shfl_sync( 0xffffffffu, fa[0], g * 4 + tt );
		a02[tt] = __shfl_sync( 0xffffffffu, fa[2], g * 4 + tt );
		a10[tt] = __shfl_sync( 0xffffffffu, fa[1], g * 4 + tt );
		a12[tt] = __shfl_sync( 0xffffffffu, fa[3], g * 4 + tt );
	}
	half2 bx0[4], bx8[4], by0[4], by8[4];
	#pragma unroll
	for ( int tt = 0; tt < 4; ++tt ) {
		bx0[tt] = __shfl_sync( 0xffffffffu, fb[0], 8 * tig + tt );
		bx8[tt] = __shfl_sync( 0xffffffffu, fb[1], 8 * tig + tt );
		by0[tt] = __shfl_sync( 0xffffffffu, fb[0], 8 * tig + 4 + tt );
		by8[tt] = __shfl_sync( 0xffffffffu, fb[1], 8 * tig + 4 + tt );
	}
	float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
	#pragma unroll
	for ( int tt = 0; tt < 4; ++tt ) {
		s0 += hdot2_f( a00[tt], bx0[tt] ) + hdot2_f( a02[tt], bx8[tt] );
		s1 += hdot2_f( a00[tt], by0[tt] ) + hdot2_f( a02[tt], by8[tt] );
		s2 += hdot2_f( a10[tt], bx0[tt] ) + hdot2_f( a12[tt], bx8[tt] );
		s3 += hdot2_f( a10[tt], by0[tt] ) + hdot2_f( a12[tt], by8[tt] );
	}
	frag_c[0].x = __hadd( frag_c[0].x, __float2half_rn( s0 ) );
	frag_c[0].y = __hadd( frag_c[0].y, __float2half_rn( s1 ) );
	frag_c[1].x = __hadd( frag_c[1].x, __float2half_rn( s2 ) );
	frag_c[1].y = __hadd( frag_c[1].y, __float2half_rn( s3 ) );
	return;
#else
	const uint32_t* a = reinterpret_cast<const uint32_t*>( &frag_a );
	const uint32_t* b = reinterpret_cast<const uint32_t*>( &frag_b );
	uint32_t* c = reinterpret_cast<uint32_t*>( &frag_c );
	const uint32_t* d = reinterpret_cast<const uint32_t*>( &frag_c );

	asm (
		"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
		"{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
		: "=r"( c[0] ), "=r"( c[1] )
		: "r"( a[0] ), "r"( a[1] ), "r"( a[2] ), "r"( a[3] ),
		  "r"( b[0] ), "r"( b[1] ),
		  "r"( d[0] ), "r"( d[1] )
	);
#endif
}

__device__ inline void barrier_acquire( int* lock, int stage ) {
	if ( threadIdx.x == 0 ) {
		volatile int state = -1;
		do {
#if defined(USE_ROCM)
			state = __hip_atomic_load( lock, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT );
#else
			asm volatile ( "ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"( state ) : "l"( lock ) );
#endif
		} while ( state != stage );
	}
	__syncthreads();
}

__device__ inline void barrier_release( int* lock, int val, bool reset ) {
	__syncthreads();
	if ( threadIdx.x == 0 ) {
		if ( reset ) {
			*lock = 0;
			return;
		}
#if defined(USE_ROCM)
		__hip_atomic_fetch_add( lock, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT );
#else
		asm volatile ( "fence.acq_rel.gpu;\n" );
		asm volatile ( "red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"( lock ), "r"( val ) );
#endif
	}
}

__device__ inline void cp_async_pred( void* smem_ptr, const void* glob_ptr, bool pred = true ) {
#if PTX_SYNC_COPY
	if ( pred ) {
		uint4 v = *reinterpret_cast<const uint4*>( glob_ptr );
		*reinterpret_cast<uint4*>( smem_ptr ) = v;
	}
#else
	const int bytes = 16;
	uint32_t smem = static_cast<uint32_t>( __cvta_generic_to_shared( smem_ptr ) );
	asm volatile (
		"{\n"
		"   .reg .pred p;\n"
		"   setp.ne.b32 p, %0, 0;\n"
		"   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
		"}\n" :: "r"( (int) pred ), "r"( smem ), "l"( glob_ptr ), "n"( bytes )
	);
#endif
}

__device__ inline void cp_async( void* smem_ptr, const void* glob_ptr ) {
#if PTX_SYNC_COPY
	uint4 v = *reinterpret_cast<const uint4*>( glob_ptr );
	*reinterpret_cast<uint4*>( smem_ptr ) = v;
#else
	const int bytes = 16;
	uint32_t smem = static_cast<uint32_t>( __cvta_generic_to_shared( smem_ptr ) );
	asm volatile (
		"{\n"
		"   cp.async.cg.shared.global [%0], [%1], %2;\n"
		"}\n" :: "r"( smem ), "l"( glob_ptr ), "n"( bytes )
	);
#endif
}

__device__ inline void cp_async_stream( void* smem_ptr, const void* glob_ptr ) {
#if PTX_SYNC_COPY
	uint4 v = *reinterpret_cast<const uint4*>( glob_ptr );
	*reinterpret_cast<uint4*>( smem_ptr ) = v;
#else
	uint32_t smem = static_cast<uint32_t>( __cvta_generic_to_shared( smem_ptr ) );
	const int bytes = 16;
	asm volatile (
		"{\n"
		"   .reg .b64 p;\n"
		"   createpolicy.fractional.L2::evict_first.b64 p, 1.0;\n"
		"   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
		"}\n" :: "r"( smem ), "l"( glob_ptr ), "n"( bytes )
	);
#endif
}

__device__ inline void cp_async_fence() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
	asm volatile ( "cp.async.commit_group;\n" :: );
#endif
}

template <int n>
__device__ inline void cp_async_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
	asm volatile ( "cp.async.wait_group %0;\n" :: "n"( n ) );
#endif
}

__device__ inline void ldsm4( FragA& frag_a, const void* smem_ptr ) {
#if defined(USE_ROCM)
	// ldmatrix.x4: lanes 8i..8i+7 supply the row addresses of matrix i; lane l takes, of each, row l / 4, halves 2 (l % 4) and + 1
	const int lane = threadIdx.x & 31;
	const unsigned long long base = reinterpret_cast<unsigned long long>( smem_ptr );
	uint32_t* r = reinterpret_cast<uint32_t*>( &frag_a );
	#pragma unroll
	for ( int j = 0; j < 4; ++j ) {
		const unsigned long long row = __shfl( base, j * 8 + ( lane >> 2 ) );
		r[j] = *reinterpret_cast<const uint32_t*>( row + ( lane & 3 ) * 4 );
	}
	return;
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750)
	uint32_t addr = static_cast<uint32_t>( __cvta_generic_to_shared( smem_ptr ) );
	const int lane = threadIdx.x & 31;
	uint32_t* r = reinterpret_cast<uint32_t*>( &frag_a );
	#pragma unroll
	for ( int j = 0; j < 4; ++j ) {
		uint32_t aj = __shfl_sync( 0xffffffffu, addr, j * 8 + ( lane >> 2 ) );
		asm volatile ( "ld.shared.b32 %0, [%1];" : "=r"( r[j] ) : "r"( aj + ( lane & 3 ) * 4 ) );
	}
	return;
#else
	uint32_t* a = reinterpret_cast<uint32_t*>( &frag_a );
	uint32_t smem = static_cast<uint32_t>( __cvta_generic_to_shared( smem_ptr ) );
	asm volatile (
		"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
		: "=r"( a[0] ), "=r"( a[1] ), "=r"( a[2] ), "=r"( a[3] ) : "r"( smem )
	);
#endif
}

#if defined(USE_ROCM)
__device__ inline uint32_t mul_lo_u32( uint32_t x, uint32_t y ) { return x * y; }
__device__ inline uint32_t mul_hi_u32( uint32_t x, uint32_t y ) { return __umulhi( x, y ); }
// Write-through / volatile (st.wt, ld.cv) and acquire / release at system scope: the cross-device flags of the collectives
__device__ __forceinline__ void stg_wt_u32( uint32_t* p, uint32_t v ) { __hip_atomic_store( p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM ); }
__device__ __forceinline__ void stg_wt_u128( uint4* p, const uint4 v ) {
	uint32_t* q = reinterpret_cast<uint32_t*>( p );
	__hip_atomic_store( q + 0, v.x, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM );
	__hip_atomic_store( q + 1, v.y, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM );
	__hip_atomic_store( q + 2, v.z, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM );
	__hip_atomic_store( q + 3, v.w, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM );
}
__device__ __forceinline__ uint32_t ldg_cv_u32( const uint32_t* p ) { return __hip_atomic_load( p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM ); }
__device__ __forceinline__ uint4 ldg_cv_u128( const uint4* p ) {
	const uint32_t* q = reinterpret_cast<const uint32_t*>( p );
	uint4 v;
	v.x = __hip_atomic_load( q + 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM );
	v.y = __hip_atomic_load( q + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM );
	v.z = __hip_atomic_load( q + 2, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM );
	v.w = __hip_atomic_load( q + 3, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM );
	return v;
}
__device__ __forceinline__ uint32_t ldg_acquire_sys_u32( const uint32_t* p ) { return __hip_atomic_load( p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM ); }
__device__ __forceinline__ uint64_t ldg_acquire_sys_u64( const uint64_t* p ) { return __hip_atomic_load( p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM ); }
__device__ __forceinline__ void stg_release_sys_u32( uint32_t* p, uint32_t v ) { __hip_atomic_store( p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM ); }
__device__ __forceinline__ void stg_release_sys_u64( uint64_t* p, uint64_t v ) { __hip_atomic_store( p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM ); }
// Nanoseconds from the constant-rate counter (100 MHz on RDNA: hipDeviceAttributeWallClockRate). Diagnostics only
__device__ __forceinline__ uint64_t globaltimer_ns() { return wall_clock64() * 10ull; }
static __forceinline__ __device__ uint32_t bfe64( uint32_t lo, uint32_t hi, int offset, int length ) {
	const uint64_t value = ( static_cast<uint64_t>( hi ) << 32 ) | static_cast<uint64_t>( lo );
	return static_cast<uint32_t>( ( value >> offset ) & ( length >= 64 ? ~0ull : ( ( 1ull << length ) - 1 ) ) );
}
#define FSHF_IMM(dst, lo, hi, imm) ( (dst) = __funnelshift_r( (lo), (hi), (imm) ) )
#define BFE16_IMM(dst, src, imm) ( (dst) = ( (uint32_t) (src) >> (imm) ) & 0xffffu )
#else
__device__ inline uint32_t mul_lo_u32( uint32_t x, uint32_t y ) {
	uint32_t w;
	asm volatile (
		"mul.lo.u32 %0, %1, %2;"
		: "=r"( w )
		: "r"( x ), "r"( y )
	);
	return w;
}

__device__ inline uint32_t mul_hi_u32( uint32_t x, uint32_t y ) {
	uint32_t w;
	asm volatile (
		"mul.hi.u32 %0, %1, %2;"
		: "=r"( w )
		: "r"( x ), "r"( y )
	);
	return w;
}

__device__ __forceinline__ void stg_wt_u32( uint32_t* p, uint32_t v ) {
	asm volatile ( "st.global.wt.u32 [%0], %1;" :: "l"( p ), "r"( v ) );
}

__device__ __forceinline__ void stg_wt_u128( uint4* p, const uint4 v ) {
	asm volatile (
		"st.global.wt.v4.u32 [%0], {%1,%2,%3,%4};"
		:: "l"( p ), "r"( v.x ), "r"( v.y ), "r"( v.z ), "r"( v.w )
	);
}

__device__ __forceinline__ uint32_t ldg_cv_u32( const uint32_t* p ) {
	uint32_t v;
	asm volatile ( "ld.global.cv.u32 %0, [%1];" : "=r"( v ) : "l"( p ) );
	return v;
}

__device__ __forceinline__ uint4 ldg_cv_u128( const uint4* p ) {
	uint4 v;
	asm volatile (
		"ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
		: "=r"( v.x ), "=r"( v.y ), "=r"( v.z ), "=r"( v.w )
		: "l"( p )
	);
	return v;
}

__device__ __forceinline__ uint32_t ldg_acquire_sys_u32( const uint32_t* p ) {
	uint32_t v;
	asm volatile ( "ld.global.acquire.sys.u32 %0, [%1];" : "=r"( v ) : "l"( p ) : "memory" );
	return v;
}

__device__ __forceinline__ uint64_t ldg_acquire_sys_u64( const uint64_t* p ) {
	uint64_t v;
	asm volatile ( "ld.global.acquire.sys.u64 %0, [%1];" : "=l"( v ) : "l"( p ) : "memory" );
	return v;
}

__device__ __forceinline__ void stg_release_sys_u32( uint32_t* p, uint32_t v ) {
	asm volatile ( "st.global.release.sys.u32 [%0], %1;" :: "l"( p ), "r"( v ) : "memory" );
}

__device__ __forceinline__ void stg_release_sys_u64( uint64_t* p, uint64_t v ) {
	asm volatile ( "st.global.release.sys.u64 [%0], %1;" :: "l"( p ), "l"( v ) : "memory" );
}

__device__ __forceinline__ uint64_t globaltimer_ns() {
	uint64_t t;
	asm volatile ( "mov.u64 %0, %%globaltimer;" : "=l"( t ) );
	return t;
}

static __forceinline__ __device__ uint32_t bfe64( uint32_t lo, uint32_t hi, int offset, int length ) {
	uint64_t value = ( static_cast<uint64_t>( hi ) << 32 ) | static_cast<uint64_t>( lo );
	uint64_t result64;
	asm ( "bfe.u64 %0, %1, %2, %3;"
		: "=l"( result64 )
		: "l"( value ), "r"( offset ), "r"( length ) );
	return static_cast<uint32_t>( result64 );
}

#define FSHF_IMM(dst, lo, hi, imm) asm("shf.r.wrap.b32 %0, %1, %2, " #imm ";" : "=r"(dst) : "r"(lo), "r"(hi))
#define BFE16_IMM(dst, src, imm) asm("bfe.u32 %0, %1, " #imm ", 16;" : "=r"(dst) : "r"(src))
#endif // USE_ROCM

__device__ inline void group_barrier(
	int group_id,
	int group_size,
	int* barrier_counters_sense
) {
	__syncthreads();

	if ( threadIdx.x == 0 ) {
		cuda::atomic_ref<int, cuda::thread_scope_device> counter( barrier_counters_sense[group_id * 2] );
		cuda::atomic_ref<int, cuda::thread_scope_device> sense( barrier_counters_sense[group_id * 2 + 1] );

		int old_sense = sense.load( cuda::memory_order_relaxed );
		int old = counter.fetch_add( 1, cuda::memory_order_acq_rel );

		if ( old == group_size - 1 ) {
			counter.store( 0, cuda::memory_order_relaxed );
			sense.store( 1 - old_sense, cuda::memory_order_release );
		}
		else {
			while ( sense.load( cuda::memory_order_acquire ) == old_sense ) {
				__nanosleep( 32 );
			}
		}
	}

	__syncthreads();
}
