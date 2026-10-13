#include <cstdio>
#include <cuda_fp16.h>
#include "exl3_gemm.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "util.cuh"
#include "exl3_gemm_kernel.cuh"
#include "exl3_kernel_map.cuh"
#include "exl3_gemv.cuh"
#include "trellis_context.h"
#include <set>

static std::set<void*> g_gemm_attr_set[ENGINE_NUM_GPUS];

void exl3_gemm_reset_caches() {
	for ( auto& st : g_gemm_attr_set ) st.clear();
}

void exl3_gemm_init_all_attrs( int device ) {
	if ( device < 0 || device >= ENGINE_NUM_GPUS ) device = 0;
	int orig_dev = 0;
	(void) cudaGetDevice( &orig_dev );
	(void) cudaSetDevice( device );
	for ( int K = 1; K <= 8; ++K ) {
		for ( int cb = 0; cb <= 2; ++cb ) {
			for ( int shape_idx = 1; shape_idx <= 4; ++shape_idx ) {
				fp_exl3_gemm_kernel k_fp16 = get_gemm_kernel_ptr( K, shape_idx, false, cb );
				if ( k_fp16 && g_gemm_attr_set[device].find( (void*) k_fp16 ) == g_gemm_attr_set[device].end() ) {
					(void) cudaFuncSetAttribute( (void*) k_fp16, cudaFuncAttributeMaxDynamicSharedMemorySize, ENGINE_MAX_SMEM );
					g_gemm_attr_set[device].insert( (void*) k_fp16 );
				}
				fp_exl3_gemm_kernel k_fp32 = get_gemm_kernel_ptr( K, shape_idx, true, cb );
				if ( k_fp32 && g_gemm_attr_set[device].find( (void*) k_fp32 ) == g_gemm_attr_set[device].end() ) {
					(void) cudaFuncSetAttribute( (void*) k_fp32, cudaFuncAttributeMaxDynamicSharedMemorySize, ENGINE_MAX_SMEM );
					g_gemm_attr_set[device].insert( (void*) k_fp32 );
				}
			}
		}
	}
	(void) cudaSetDevice( orig_dev );
}

void exl3_gemm(
	const half* A,
	const uint16_t* B,
	void* C,
	const half* suh,
	half* A_had,
	const half* svh,
	int size_m,
	int size_k,
	int size_n,
	int K,
	bool mcg,
	bool mul1,
	bool c_fp32,
	cudaStream_t stream,
	int force_shape
) {
	int device = 0;
	(void) cudaGetDevice( &device );
	int num_sms = engine::modules::trellis_get_num_sms( device );
	int cc = engine::modules::trellis_get_cc( device );
	int* locks = engine::modules::trellis_get_locks( device );

	int cb = 0;
	if ( mcg ) cb = 1;
	if ( mul1 ) cb = 2;

	void* kernel_args[] = {
		(void*) &A,
		(void*) &B,
		(void*) &C,
		(void*) &size_m,
		(void*) &size_k,
		(void*) &size_n,
		(void*) &locks,
		(void*) &suh,
		(void*) &A_had,
		(void*) &svh
	};

	if ( force_shape <= 0 && size_m <= 8 && size_n < 4096 && size_k % 128 == 0 && size_n % 128 == 0 ) {
		if ( exl3_gemv_try_launch(
			kernel_args,
			size_m,
			size_k,
			size_n,
			K,
			cb,
			c_fp32,
			( suh != nullptr && A_had != nullptr && svh != nullptr ),
			device,
			stream,
			nullptr,
			false
		) ) {
			return;
		}
	}

	int block_dim = 0;
	int shape_idx = force_shape > 0 ? force_shape : select_gemm_shape( cc, size_m, size_k, size_n, K, false, 1, 1 );
	fp_exl3_gemm_kernel kernel = select_exl3_gemm_kernel(
		cc,
		size_m,
		size_k,
		size_n,
		K,
		c_fp32,
		shape_idx,
		&block_dim,
		&shape_idx,
		&num_sms,
		cb
	);

	if ( kernel == nullptr ) {
		std::fprintf( stderr, "exl3_gemm: no kernel for m=%d k=%d n=%d bits=%d cb=%d fp32=%d\n", size_m, size_k, size_n, K, cb, c_fp32 ? 1 : 0 );
	}
	ENGINE_ASSERT( kernel != nullptr, "exl3_gemm: failed to select kernel" );

	size_t actual_smem = exl3_gemm_shape_smem( shape_idx, K );
	if ( g_gemm_attr_set[device].find( (void*) kernel ) == g_gemm_attr_set[device].end() ) {
		CUDA_CHECK( cudaFuncSetAttribute( (void*) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ENGINE_MAX_SMEM ) );
		g_gemm_attr_set[device].insert( (void*) kernel );
	}

	CUDA_CHECK( cudaLaunchCooperativeKernel(
		(void*) kernel,
		num_sms,
		block_dim,
		kernel_args,
		actual_smem,
		stream
	) );
}
