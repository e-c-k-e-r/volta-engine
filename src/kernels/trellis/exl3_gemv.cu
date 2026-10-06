#include <cuda_fp16.h>
#include "exl3_gemv.cuh"
#include "exl3_gemm.cuh"
#include "exl3_gemv_int8.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "util.cuh"
#include "exl3_gemv_kernel.cuh"
#include "trellis_context.h"
#include <map>
#include <cstdlib>

static int exl3_gemv_env_mode() {
	const char* env = std::getenv( "EXL3_GEMV" );
	if ( !env ) return 1;
	return atoi( env );
}

static int exl3_gemv_env_smem() {
	const char* env = std::getenv( "EXL3_GEMV_SMEM" );
	if ( !env ) return -1;
	return atoi( env );
}

static int exl3_gemv_cfg( int cc, int size_m, int size_k, int size_n, int K, int cb, int mode, int narrow_coresident ) {
	(void) narrow_coresident;
	if ( mode == 0 ) return -1;
	if ( K < 2 || K > 6 ) return -1;
	if ( K != 4 && K != 6 && cb == 0 ) return -1;
	if ( size_m > EXL3_GEMV_MAX_M ) return -1;
	if ( size_k % 128 || size_n % 128 ) return -1;
	if ( cc == CC_OLD ) {
		if ( mode == 1 && size_n >= 4096 ) return -1;
		if ( K == 5 ) return 1;
		if ( K == 2 || K == 3 || K == 4 || K == 6 ) return ( size_n <= 8192 ) ? 0 : 1;
		return -1;
	}
	if ( mode == 2 ) return size_n <= 8192 ? 0 : 1;
	if ( mode == 3 ) return 0;
	if ( mode == 4 ) return 1;
	if ( K == 2 ) return size_n <= 8192 ? 0 : 1;
	if ( K == 5 ) return 1;
	if ( size_k <= 2048 && size_n <= 8192 ) return 0;
	if ( K == 3 ) return -1;
	if ( size_n >= 8192 && size_k <= 4096 ) return 1;
	return -1;
}

static void* exl3_gemv_select_kernel( int bits, int cb, bool c_fp32, int mmode, int cfg, bool smem ) {
	#define SEL(bits_, cb_, fp32_, mm_, cfg_, sm_) \
		if ( bits == bits_ && cb == cb_ && c_fp32 == fp32_ && mmode == mm_ && cfg == cfg_ && smem == sm_ ) \
			return (void*) exl3_gemv_kernel<bits_, fp32_, cb_, mm_, cfg_, sm_>;
	#define SEL_GRID(bits_, cb_, sm_) \
		SEL(bits_, cb_, false, 0, 0, sm_) SEL(bits_, cb_, false, 0, 1, sm_) \
		SEL(bits_, cb_, false, 1, 0, sm_) SEL(bits_, cb_, false, 1, 1, sm_) \
		SEL(bits_, cb_, true,  0, 0, sm_) SEL(bits_, cb_, true,  0, 1, sm_) \
		SEL(bits_, cb_, true,  1, 0, sm_) SEL(bits_, cb_, true,  1, 1, sm_)
	#define SEL_CFG1(bits_, cb_, sm_) \
		SEL(bits_, cb_, false, 0, 1, sm_) \
		SEL(bits_, cb_, false, 1, 1, sm_) \
		SEL(bits_, cb_, true,  0, 1, sm_) \
		SEL(bits_, cb_, true,  1, 1, sm_)

	SEL_GRID( 4, 0, false ) SEL_GRID( 4, 1, false ) SEL_GRID( 4, 2, false )
	SEL_GRID( 2, 1, false ) SEL_GRID( 2, 2, false ) SEL_GRID( 2, 1, true ) SEL_GRID( 2, 2, true )
	SEL_GRID( 3, 1, false ) SEL_GRID( 3, 2, false ) SEL_GRID( 3, 1, true ) SEL_GRID( 3, 2, true )
	SEL_CFG1( 5, 1, true )  SEL_CFG1( 5, 2, true )
	SEL_GRID( 6, 1, true )  SEL_GRID( 6, 2, true )

	#undef SEL_CFG1
	#undef SEL_GRID
	#undef SEL
	return nullptr;
}

bool exl3_gemv_try_launch(
	void** kernel_args,
	int size_m,
	int size_k,
	int size_n,
	int K,
	int cb,
	bool c_fp32,
	bool has_su_sv,
	int device,
	cudaStream_t stream,
	void** launched_kernel,
	bool force
) {
	if ( !has_su_sv ) return false;
	if ( K < 2 || K > 6 ) return false;
	if ( K != 4 && K != 6 && cb == 0 ) return false;
	if ( size_m > EXL3_GEMV_MAX_M ) return false;
	if ( size_k % 128 || size_n % 128 ) return false;

	int mode = force ? 2 : exl3_gemv_env_mode();
	if ( mode == 0 ) return false;
	int cc = CC_OLD;
	int mmode = size_m == 1 ? 0 : 1;
	int num_sms = ENGINE_SM_COUNT;

	static std::map<void*, int> occ_cache[ENGINE_NUM_GPUS];
	auto& cache = occ_cache[device < ENGINE_NUM_GPUS ? device : 0];
	auto occupancy = [&]( void* kern, int bdim ) -> int {
		auto it = cache.find( kern );
		if ( it != cache.end() ) return it->second;
		int blocks_per_sm = 1;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor( &blocks_per_sm, kern, bdim, 0 );
		cache[kern] = blocks_per_sm;
		return blocks_per_sm;
	};

	bool smem = ( K == 5 || K == 6 ) ? true : ( exl3_gemv_env_smem() == 1 );
	void* kernel = nullptr;
	int cfg = -1;

	if ( K == 5 ) {
		cfg = exl3_gemv_cfg( cc, size_m, size_k, size_n, K, cb, mode, 0 );
		if ( cfg != 1 ) return false;
		kernel = exl3_gemv_select_kernel( K, cb, c_fp32, mmode, 1, smem );
		if ( !kernel ) return false;
	}
	else {
		void* narrow_kernel = exl3_gemv_select_kernel( K, cb, c_fp32, mmode, 0, smem );
		if ( !narrow_kernel ) return false;
		int narrow_coresident = occupancy( narrow_kernel, 512 ) * num_sms;
		cfg = exl3_gemv_cfg( cc, size_m, size_k, size_n, K, cb, mode, narrow_coresident );
		if ( cfg < 0 ) return false;
		kernel = ( cfg == 0 ) ? narrow_kernel : exl3_gemv_select_kernel( K, cb, c_fp32, mmode, cfg, smem );
		if ( !kernel ) return false;
	}

	int block_dim = ( cfg == 0 ) ? 512 : 256;
	int cols = ( cfg == 0 ) ? 32 : 64;

	int max_blocks = occupancy( kernel, block_dim ) * num_sms;
	int gx = MIN( size_n / cols, max_blocks );
	int gy = 1;

	if ( cc == CC_OLD && gx < num_sms && size_k >= 1024 ) {
		gy = MIN( 4, max_blocks / gx );
		gy = MIN( gy, MAX( 1, size_k / 512 ) );
	}
	int total_grid = gx * gy;
	if ( total_grid < 1 || total_grid > max_blocks ) return false;

	uintptr_t flags = 0;
	int* flags_ptr = (int*) flags;
	void* saved_arg6 = kernel_args[6];
	kernel_args[6] = (void*) &flags_ptr;

	cudaError_t err = cudaLaunchCooperativeKernel(
		kernel,
		dim3( gx, gy, 1 ),
		dim3( block_dim ),
		kernel_args,
		0,
		stream
	);

	kernel_args[6] = saved_arg6;
	if ( err != cudaSuccess ) return false;

	if ( launched_kernel ) *launched_kernel = kernel;
	return true;
}

void exl3_gemv(
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
	int cb = 0;
	if ( mcg ) cb = 1;
	if ( mul1 ) cb = 2;

	int device = 0;
	cudaGetDevice( &device );

	// dp4a int8 path first: at M=1 it beats both the fp16 GEMV and the Shape 4 GEMM on every
	// per-rank Qwen shape, and it is a plain launch with a fixed workspace, so it captures cleanly.
	if ( exl3_gemv_int8( A, B, C, suh, A_had, svh, size_m, size_k, size_n, K, mcg, mul1, c_fp32, device, stream ) )
		return;

	int* locks = engine::modules::trellis_get_locks( device );

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

	bool ok = exl3_gemv_try_launch(
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
		true
	);
	if ( !ok ) {
		exl3_gemm( A, B, C, suh, A_had, svh, size_m, size_k, size_n, K, mcg, mul1, c_fp32, stream );
	}
}
