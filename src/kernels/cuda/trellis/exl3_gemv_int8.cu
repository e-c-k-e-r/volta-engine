#include <cuda_fp16.h>
#include "exl3_gemv_int8.cuh"
#include "exl3_gemv_int8_kernel.cuh"
#include "trellis_context.h"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>

namespace {

// Fixed-size per-device workspace (INT8_WS_BYTES, see trellis_context.h), allocated once at init and
// never reallocated: the pointer is baked as a kernel argument into captured CUDA graphs, so growing
// the buffer would leave every previously captured graph with a dangling pointer. Zeroed at
// allocation so the completion counters (which the finishing block resets) start at zero. Work that
// does not fit is rejected by the launcher.
std::set<void*> int8_attr_set[ENGINE_NUM_GPUS];
std::map<std::pair<void*, size_t>, int> int8_occ_cache[ENGINE_NUM_GPUS];

}

void exl3_gemv_int8_reset_caches() {
	for ( auto& st : int8_attr_set ) st.clear();
	for ( auto& m : int8_occ_cache ) m.clear();
}

namespace {

// 0: disabled; 1: int8 + error-feedback residual pass (~15-16 bit activation precision);
// 2: plain int8 (cheaper, lossy).
int int8_mode() {
	return 1;
}

// Smem-staged unit routing. Upstream measured K = 3, 5, 7 on Ampere, where cp.async hides the
// staging cost. ptx.cuh degrades cp_async to a synchronous global->register->shared copy below
// sm_80, so the staged unit is a net loss on Volta for K = 5.
bool int8_smem_for( int bits ) {
	return gemv_int8_stage_smem( bits );
}

// Highest K the int8 path accepts. Measured on V100: 4 bpw MLP 57.7 -> 51.5 us and 5 bpw 85.6 ->
// 75.9 us in residual mode, both wins; above this the fp16 tensor-core path wins.
int int8_max_k() {
	return 8;
}

template <int bits, int M, bool c_fp32, bool residual, bool stage_smem>
void* sel() {
	return (void*) exl3_gemv_int8_sq_kernel<bits, M, c_fp32, residual, stage_smem>;
}

template <int bits, int M, bool c_fp32, bool residual>
void* sel_stage( bool stage_smem ) {
	if constexpr ( bits == 3 || bits == 5 || bits == 7 ) {
		return stage_smem ? sel<bits, M, c_fp32, residual, true>()
		                  : sel<bits, M, c_fp32, residual, false>();
	}
	else {
		(void) stage_smem;
		return sel<bits, M, c_fp32, residual, false>();
	}
}

template <int bits, int M>
void* sel_c( bool c_fp32, bool residual ) {
	if ( c_fp32 ) return residual ? sel_stage<bits, M, true, true>( false )
	                              : sel_stage<bits, M, true, false>( int8_smem_for( bits ) );
	return residual ? sel_stage<bits, M, false, true>( false )
	                : sel_stage<bits, M, false, false>( int8_smem_for( bits ) );
}

void* select_kernel( int K, int M, bool c_fp32, bool residual ) {
	// M == 2 only reaches this path in plain int8 mode, and only the bit widths the models actually
	// use are instantiated: the two-row variants double compile time for a shape we do not decode.
	if ( M == 2 ) {
		switch ( K ) {
			case 4: return sel_c<4, 2>( c_fp32, residual );
			case 5: return sel_c<5, 2>( c_fp32, residual );
			default: return nullptr;
		}
	}
	if ( M != 1 ) return nullptr;
	switch ( K ) {
		case 2: return sel_c<2, 1>( c_fp32, residual );
		case 3: return sel_c<3, 1>( c_fp32, residual );
		case 4: return sel_c<4, 1>( c_fp32, residual );
		case 5: return sel_c<5, 1>( c_fp32, residual );
		case 6: return sel_c<6, 1>( c_fp32, residual );
		case 7: return sel_c<7, 1>( c_fp32, residual );
		case 8: return sel_c<8, 1>( c_fp32, residual );
		default: return nullptr;
	}
}

// Mirror of the kernel's work decomposition, needed before launch to size shared memory and to bound
// the workspace. The kernel recomputes rows_per from the same constants, so the host cannot ask for
// less without desyncing the two.
struct Decomp {
	int ksplit;
	int rows_per;
};

Decomp decomp( int grid, int size_k, int size_n, int M, bool residual ) {
	int rows_total = size_k / 16;
	int nb256 = size_n / 256;
	int r = CEIL_DIVIDE( rows_total * nb256, grid );
	int rows_per = ( MAX( r, MIN( 2 * r, 32 ) ) + 7 ) & ~7;
	rows_per = MAX( rows_per, SQ_MINROWS );
	rows_per = MIN( rows_per, gemv_int8_sq_rows_max( M, residual ) );
	rows_per = MIN( rows_per, ( rows_total + 7 ) & ~7 );
	return { CEIL_DIVIDE( rows_total, rows_per ), rows_per };
}

size_t smem_for( int rows_per, int size_n, int M, bool residual, bool smem, int bits ) {
	size_t stage = smem ? (size_t) 8 * GEMV_STAGE_D * 16 * bits * 4 : 0;
	return (size_t) rows_per * 16 * 2 + (size_t) rows_per * 16 * 4 * M * ( residual ? 2 : 1 )
	     + stage + (size_t) 2 * M * 128 * 4;
}

} // namespace

int exl3_gemv_int8_max_k( int device ) {
	(void) device;
	return int8_max_k();
}

bool exl3_gemv_int8_enabled() {
	return int8_mode() != 0;
}

void exl3_gemv_int8_init( int device ) {
	if ( device < 0 || device >= ENGINE_NUM_GPUS ) return;
	engine::modules::trellis_get_int8_ws( device );
	if ( int8_mode() == 0 ) return;
	// Shared-memory opt-in is a per-device property, and wide shapes (the 248320-column LM head needs
	// ~52 KB) launch above the 48 KB default. Without the attribute on the right device the launch
	// fails, the error is swallowed and the fallback kernel inherits a poisoned capture stream.
	int cur = 0;
	CUDA_CHECK( cudaGetDevice( &cur ) );
	CUDA_CHECK( cudaSetDevice( device ) );
	for ( int bits : { 2, 3, 4, 5, 6, 7, 8 } ) {
		for ( int M : { 1, 2 } ) {
			for ( bool fp32 : { false, true } ) {
				for ( bool residual : { false, true } ) {
					{
						void* fn = select_kernel( bits, M, fp32, residual );
						if ( !fn ) continue;
						if ( int8_attr_set[device].count( fn ) ) continue;
						bool smem = gemv_int8_stage_smem( bits );
						int rows_max = gemv_int8_sq_rows_max( M, residual );
						size_t smem_max = smem_for( rows_max, 0, M, residual, smem, bits );
						cudaFuncSetAttribute( fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_max );
						// Match the tensor-core kernels' shared-memory carveout: these interleave with them
						// (hundreds of launches per decoded token), and a smaller carveout makes the GPU
						// drain and reconfigure the SMs on every transition.
						cudaFuncSetAttribute( fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared );
						int8_attr_set[device].insert( fn );
						CUDA_CHECK( cudaPeekAtLastError() );
					}
				}
			}
		}
	}
	CUDA_CHECK( cudaSetDevice( cur ) );
}

bool exl3_gemv_int8(
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
	int device,
	cudaStream_t stream
) {
	(void) A_had;
	if ( int8_mode() == 0 ) return false;
	if ( mcg || !mul1 ) return false;          // dp4a path is mul1 (cb 2) only
	if ( suh == nullptr || svh == nullptr ) return false;
	if ( size_n % 256 ) return false;          // 256-column work unit
	if ( size_k % 128 ) return false;
	if ( K < 1 || K > int8_max_k() ) return false;
	if ( size_m < 1 || size_m > 2 ) return false;

	bool residual = ( int8_mode() == 1 );
	// Residual mode now serves two rows too. More rows lose to the tensor-core GEMM: the dp4a chains (two
	// per row in residual mode) are issue-bound, so time grows ~linearly in M (M=4 315 us, M=8 457 us on
	// the 5120x17408 5-bit shape against 217 us for the GEMM).
	const int M = size_m;
	bool smem = gemv_int8_stage_smem( K );

	// Both the sq kernel and the launcher size shared memory against the ~80 KB budget baked into
	// gemv_int8_sq_rows_max(). On a part with less opt-in shared memory the cudaFuncSetAttribute would
	// fail, so decline instead: this is a performance loss over the fp16 kernel, not a correctness one.
	if ( ENGINE_MAX_SMEM < 80 * 1024 ) return false;

	int* ws = engine::modules::trellis_get_int8_ws( device );
	if ( ws == nullptr ) return false;

	void* fn = select_kernel( K, M, c_fp32, residual );
	if ( !fn ) return false;

	int num_sms = engine::modules::trellis_get_num_sms( device );

	// The grid is sized from the single-row variant's occupancy for every row count, so the k-slice
	// decomposition (and with it every per-slice activation scale and the fixed-order reduction) is
	// identical across M: row r of an M-row call is bit-identical to the same row run alone. Wider
	// variants may then exceed their own resident-block count, which is fine (blocks never wait on
	// each other; the last finisher of a column group runs its epilogue).
	void* fn1 = select_kernel( K, 1, c_fp32, residual );
	if ( !fn1 ) fn1 = fn;
	const int M1 = ( fn1 == fn ) ? M : 1;
	Decomp d = decomp( 6 * num_sms, size_k, size_n, M1, residual );
	size_t smem_guess = smem_for( d.rows_per, size_n, M1, residual, smem, K );

	int maxb = 1;
	auto occ_key = std::make_pair( fn1, smem_guess );
	auto occ_it = int8_occ_cache[device].find( occ_key );
	if ( occ_it != int8_occ_cache[device].end() ) {
		maxb = occ_it->second;
	}
	else {
		maxb = 1;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxb, fn1, NUM_THREADS, smem_guess );
		int8_occ_cache[device][occ_key] = maxb;
	}
	int grid = MIN( MAX( maxb, 1 ) * num_sms, 1024 );
	d = decomp( grid, size_k, size_n, M, residual );
	size_t smem_bytes = smem_for( d.rows_per, size_n, M, residual, smem, K );
	if ( d.ksplit > SQ_KSPLIT_CAP ) return false;
	if ( size_n / 256 > SQ_COUNTERS_CAP ) return false;

	int pstride = size_n * ( residual ? 2 : 1 );
	size_t ws_ints = SQ_WS_RESERVED + (size_t) d.ksplit * M * pstride;
	if ( ws_ints > (size_t) INT8_WS_INTS ) return false;
	int* ws_ptr = ws;

	void* kernel_args[] = {
		(void*) &A,
		(void*) &B,
		(void*) &C,
		(void*) &size_m,
		(void*) &size_k,
		(void*) &size_n,
		(void*) &ws_ptr,
		(void*) &suh,
		(void*) &A_had,
		(void*) &svh
	};

	cudaError_t err = cudaLaunchKernel( fn, dim3( grid ), dim3( NUM_THREADS ), kernel_args, smem_bytes, stream );
	if ( err != cudaSuccess ) {
		// Declining the fast path is only safe if nothing was enqueued. Swallowing the error here
		// would leave the caller's fallback launch reporting a poisoned stream instead of the real
		// cause, so surface it and let the caller fall through with the state cleaned up.
		std::printf( "[int8] launch failed (K=%d M=%d n=%d k=%d smem=%zu grid=%d): %s\n",
			K, M, size_n, size_k, smem_bytes, grid, cudaGetErrorString( err ) );
		std::fflush( stdout );
		cudaGetLastError();
		return false;
	}
	return true;
}
