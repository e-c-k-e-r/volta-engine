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
	cudaStream_t stream
) {
	int device = 0;
	cudaGetDevice( &device );
	int num_sms = ENGINE_SM_COUNT;
	int cc = CC_OLD;
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

	if ( size_m <= 8 && size_n < 4096 && size_k % 128 == 0 && size_n % 128 == 0 ) {
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
	int shape_idx = select_gemm_shape( cc, size_m, size_k, size_n, K, false, 1, 1 );
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

	ENGINE_ASSERT( kernel != nullptr, "exl3_gemm: failed to select kernel" );

	size_t actual_smem = exl3_gemm_shape_smem( shape_idx, K );
	if ( g_gemm_attr_set[device].find( (void*) kernel ) == g_gemm_attr_set[device].end() ) {
		CUDA_CHECK( cudaFuncSetAttribute( (void*) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ENGINE_MAX_SMEM ) );
		g_gemm_attr_set[device].insert( (void*) kernel );
	}

	int max_active_blocks = 0;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor( &max_active_blocks, (void*) kernel, block_dim, actual_smem );
	int max_coop_grid = max_active_blocks * 80;
	if ( max_coop_grid > 0 && num_sms > max_coop_grid ) {
		num_sms = max_coop_grid;
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
