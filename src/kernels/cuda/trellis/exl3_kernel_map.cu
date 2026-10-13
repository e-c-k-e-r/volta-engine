#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "util.cuh"
#include "ptx.cuh"
#include "exl3_kernel_map.cuh"
#include "trellis_context.h"

#if defined(USE_ROCM)
extern int exl3_gemm_tilesize_n[];
extern int exl3_gemm_blockdim[];

// Blocks of the (bits, shape) instance resident per compute unit, from the runtime's occupancy query (cached: it is not cheap and
// shape selection is on the launch path). 0 when the shape's shared memory exceeds what the device grants a workgroup
static int occ_blocks_per_cu( int bits, int shape_idx ) {
	static int cache[9][EXL3_GEMM_NUM_SHAPES + 1];
	static bool init = false;
	if ( !init ) {
		for ( auto& row : cache )
			for ( int& v : row ) v = -1;
		init = true;
	}
	int& cached = cache[bits][shape_idx];
	if ( cached >= 0 ) return cached;
	fp_exl3_gemm_kernel k = get_gemm_kernel_ptr( bits, shape_idx, false, 0 );
	const int smem = exl3_gemm_shape_smem( shape_idx, bits );
	int blocks = 0;
	if ( k && smem <= ENGINE_MAX_SMEM ) {
		if ( cudaOccupancyMaxActiveBlocksPerMultiprocessor( &blocks, (const void*) k, exl3_gemm_blockdim[shape_idx], smem ) != cudaSuccess ) {
			(void) cudaGetLastError();
			blocks = 0;
		}
	}
	cached = blocks;
	return cached;
}

// RDNA (upstream exllamav3's select_gemm_shape for ROCm): the NVIDIA tables refer to that shared memory budget, so the shapes that fit
// are scored instead: wider N tiles amortise the weight read, occupancy dominates where it differs, the grid should fill the device.
// Divisibility is tested per matrix, not on the bszm-scaled widths
int select_gemm_shape( int cc, int size_m, int size_k, int size_n, int K, bool multi, int bszm_in, int bszm_out ) {
	(void) cc;
	const long long eff_k = static_cast<long long>( size_k ) * bszm_in;
	const long long eff_n = static_cast<long long>( size_n ) * bszm_out;
	int device = 0;
	(void) cudaGetDevice( &device );
	const int cu_count = MAX( engine::modules::trellis_get_num_sms( device ), 1 );

	int best_shape = 1;
	long long best_score = -( 1ll << 62 );
	for ( int shape = EXL3_GEMM_NUM_SHAPES; shape >= 1; --shape ) {
		if ( !exl3_gemm_shape_compat( shape, size_m, size_k, size_n, K ) ) continue;
		const int blocks_per_cu = occ_blocks_per_cu( K, shape );
		if ( blocks_per_cu <= 0 ) continue;
		const int tn = exl3_gemm_tilesize_n[shape];
		const long long slices = MAX( ( eff_k / 16 ) * ( eff_n / tn ), 1ll );
		const long long max_resident = static_cast<long long>( cu_count ) * blocks_per_cu;
		const long long underfill = MAX( max_resident - slices, 0ll );
		long long score = 0;
		score += static_cast<long long>( tn ) * 200;
		score += static_cast<long long>( blocks_per_cu ) * 10000;
		score += MIN( slices, max_resident ) * 10;
		score -= underfill * ( multi ? 30 : 10 );
		// Low bit rates read less weight per tile and can afford the wider tiles; high ones cannot
		if ( K <= 4 ) { if ( shape == 4 ) score += 50000; }
		else if ( K <= 6 ) { if ( shape == 3 ) score += 30000; }
		else { if ( shape <= 2 ) score += 20000; }
		if ( score > best_score ) { best_score = score; best_shape = shape; }
	}
	return best_shape;
}

// The NVIDIA tables stay compiled (under another name) so both configurations parse the same code
int select_gemm_shape_cuda_tables( int cc, int size_m, int size_k, int size_n, int K, bool multi, int bszm_in, int bszm_out );
int select_gemm_shape_cuda_tables( int cc, int size_m, int size_k, int size_n, int K, bool multi, int bszm_in, int bszm_out ) {
#else
int select_gemm_shape( int cc, int size_m, int size_k, int size_n, int K, bool multi, int bszm_in, int bszm_out ) {
#endif
	bool mod_256 = ( size_n % 256 == 0 );
	bool mod_512 = ( size_n % 512 == 0 );

	size_k *= bszm_in;
	size_n *= bszm_out;

	switch ( cc ) {
		case CC_OLD: {
			if ( size_m <= 16 ) {
				if ( mod_512 && size_n >= 4096 && K <= 6 ) return 4;
				if ( mod_256 && size_n >= 4096 && K <= 6 ) return 3;
				return 2;
			}
			if ( mod_256 && K <= 4 ) {
				if ( size_n <= 2048 || size_k <= 2048 ) return 2;
				return 3;
			}
			if ( mod_256 && size_n < 4096 ) {
				return size_k > 8192 ? 3 : 2;
			}
			if ( mod_512 && ( size_n * size_k ) > ( 4096 * 4096 ) && K <= 6 ) return 4;
			if ( mod_256 ) return 3;
			return 2;
		} break;
		// The rest are upstream exllamav3's tables (exl3_kernel_map.cu, select_gemm_shape), tuned on its reference parts
		case CC_AMPERE: {
			if ( mod_256 && K <= 4 ) {
				if ( size_n <= 2048 || size_k <= 2048 ) return 2;
				return 3;
			}
			if ( mod_256 && size_n < 4096 ) return size_k > 8192 ? 3 : 2;
			if ( mod_512 && ( size_n * size_k ) > ( 4096 * 4096 ) && K <= 6 ) return 4;
			if ( mod_256 ) return 3;
			return 2;
		} break;
		case CC_ADA: {
			if ( mod_256 && K <= 3 ) {
				if ( size_k <= 2048 && !multi ) return 2;
				if ( size_n < 4096 && size_k <= 12288 ) return 2;
				return 3;
			}
			if ( size_n <= 16384 ) return 2;
			if ( mod_512 && size_n >= 32768 ) return 4;
			if ( mod_256 ) return 3;
			return 2;
		} break;
		default: {   // CC_HOPPER, CC_BLACKWELL
			if ( ( K == 4 || K == 2 ) && !multi ) {
				if ( size_k <= 2048 ) return 1;
			}
			if ( K >= 7 ) {
				if ( mod_256 && size_n <= 8192 ) return size_k > 32768 ? 3 : 2;
				if ( mod_512 && size_n > 32768 ) return 4;
				return 2;
			}
			if ( mod_256 && size_n <= 4096 ) return size_k > 8192 && K >= 3 ? 3 : 2;
			if ( mod_512 && size_n > 16384 ) return 4;
			if ( mod_256 ) return 3;
			return 2;
		} break;
	}
}

int exl3_gemm_num_kernel_shapes() {
	return EXL3_GEMM_NUM_SHAPES;
}

int exl3_gemm_tilesize_k[] = { EXL3_GEMM_TILESIZE_K };
int exl3_gemm_tilesize_n[] = { EXL3_GEMM_TILESIZE_N };
int exl3_gemm_blockdim[] = { EXL3_GEMM_BLOCKDIM };

int exl3_gemm_shape_smem( int shape_idx, int K ) {
	return exl3_gemm_smem_bytes_for_shape( shape_idx, K, true );
}

bool exl3_gemm_shape_compat( int shape_idx, int size_m, int size_k, int size_n, int K ) {
	(void) size_m;
	int tilesize_k = exl3_gemm_tilesize_k[shape_idx];
	int tilesize_n = exl3_gemm_tilesize_n[shape_idx];
	if ( size_k % tilesize_k || size_n % tilesize_n ) return false;
	return exl3_gemm_shape_smem( shape_idx, K ) <= ENGINE_MAX_SMEM;
}

#define EXL3_KERNEL_TABLE_ROW(fp, K) \
	{ tfp_exl3_gemm_kernel_##fp##_b##K##_cb0, tfp_exl3_gemm_kernel_##fp##_b##K##_cb1, tfp_exl3_gemm_kernel_##fp##_b##K##_cb2 }
#define EXL3_MKERNEL_TABLE_ROW(fp, K) \
	{ tfp_exl3_mgemm_kernel_##fp##_b##K##_cb0, tfp_exl3_mgemm_kernel_##fp##_b##K##_cb1, tfp_exl3_mgemm_kernel_##fp##_b##K##_cb2 }

static fp_exl3_gemm_kernel* const tab_gemm_fp32[9][3] = {
	{ nullptr, nullptr, nullptr },
	EXL3_KERNEL_TABLE_ROW( fp32, 1 ), EXL3_KERNEL_TABLE_ROW( fp32, 2 ), EXL3_KERNEL_TABLE_ROW( fp32, 3 ),
	EXL3_KERNEL_TABLE_ROW( fp32, 4 ), EXL3_KERNEL_TABLE_ROW( fp32, 5 ), EXL3_KERNEL_TABLE_ROW( fp32, 6 ),
	EXL3_KERNEL_TABLE_ROW( fp32, 7 ), EXL3_KERNEL_TABLE_ROW( fp32, 8 )
};

static fp_exl3_gemm_kernel* const tab_gemm_fp16[9][3] = {
	{ nullptr, nullptr, nullptr },
	EXL3_KERNEL_TABLE_ROW( fp16, 1 ), EXL3_KERNEL_TABLE_ROW( fp16, 2 ), EXL3_KERNEL_TABLE_ROW( fp16, 3 ),
	EXL3_KERNEL_TABLE_ROW( fp16, 4 ), EXL3_KERNEL_TABLE_ROW( fp16, 5 ), EXL3_KERNEL_TABLE_ROW( fp16, 6 ),
	EXL3_KERNEL_TABLE_ROW( fp16, 7 ), EXL3_KERNEL_TABLE_ROW( fp16, 8 )
};

static fp_exl3_mgemm_kernel* const tab_mgemm_fp32[9][3] = {
	{ nullptr, nullptr, nullptr },
	EXL3_MKERNEL_TABLE_ROW( fp32, 1 ), EXL3_MKERNEL_TABLE_ROW( fp32, 2 ), EXL3_MKERNEL_TABLE_ROW( fp32, 3 ),
	EXL3_MKERNEL_TABLE_ROW( fp32, 4 ), EXL3_MKERNEL_TABLE_ROW( fp32, 5 ), EXL3_MKERNEL_TABLE_ROW( fp32, 6 ),
	EXL3_MKERNEL_TABLE_ROW( fp32, 7 ), EXL3_MKERNEL_TABLE_ROW( fp32, 8 )
};

static fp_exl3_mgemm_kernel* const tab_mgemm_fp16[9][3] = {
	{ nullptr, nullptr, nullptr },
	EXL3_MKERNEL_TABLE_ROW( fp16, 1 ), EXL3_MKERNEL_TABLE_ROW( fp16, 2 ), EXL3_MKERNEL_TABLE_ROW( fp16, 3 ),
	EXL3_MKERNEL_TABLE_ROW( fp16, 4 ), EXL3_MKERNEL_TABLE_ROW( fp16, 5 ), EXL3_MKERNEL_TABLE_ROW( fp16, 6 ),
	EXL3_MKERNEL_TABLE_ROW( fp16, 7 ), EXL3_MKERNEL_TABLE_ROW( fp16, 8 )
};

fp_exl3_gemm_kernel select_exl3_gemm_kernel(
	int cc,
	int size_m,
	int size_k,
	int size_n,
	int K,
	bool c_fp32,
	int force_shape_idx,
	int* out_block_dim,
	int* out_shape_idx,
	int* num_sms,
	int cb
) {
	int shape_idx = force_shape_idx <= 0 ? select_gemm_shape( cc, size_m, size_k, size_n, K, false, 1, 1 ) : force_shape_idx;

	ENGINE_ASSERT( shape_idx > 0 && shape_idx <= EXL3_GEMM_NUM_SHAPES, "exl3_gemm: invalid shape index" );
	if ( out_shape_idx ) *out_shape_idx = shape_idx;
	if ( out_block_dim ) *out_block_dim = exl3_gemm_blockdim[shape_idx];

	if ( num_sms ) {
		int tilesize_k = exl3_gemm_tilesize_k[shape_idx];
		int tilesize_n = exl3_gemm_tilesize_n[shape_idx];
		int max_slices = size_k / tilesize_k * size_n / tilesize_n;
		*num_sms = MAX( MIN( max_slices, *num_sms ), 1 );
	}

	ENGINE_ASSERT( K >= 1 && K <= 8 && cb >= 0 && cb <= 2, "No kernel for GEMM shape" );
	return ( c_fp32 ? tab_gemm_fp32 : tab_gemm_fp16 )[K][cb][shape_idx];
}

fp_exl3_gemm_kernel get_gemm_kernel_ptr( int K, int shape_idx, bool c_fp32, int cb ) {
	ENGINE_ASSERT( K >= 1 && K <= 8 && cb >= 0 && cb <= 2, "No kernel for GEMM shape" );
	return ( c_fp32 ? tab_gemm_fp32 : tab_gemm_fp16 )[K][cb][shape_idx];
}

fp_exl3_mgemm_kernel get_mgemm_kernel_ptr( int K, int shape_idx, bool c_fp32, int cb ) {
	ENGINE_ASSERT( K >= 1 && K <= 8 && cb >= 0 && cb <= 2, "No kernel for GEMM shape" );
	return ( c_fp32 ? tab_mgemm_fp32 : tab_mgemm_fp16 )[K][cb][shape_idx];
}
