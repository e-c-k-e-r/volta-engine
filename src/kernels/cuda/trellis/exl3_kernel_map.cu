#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "util.cuh"
#include "ptx.cuh"
#include "exl3_kernel_map.cuh"
#include "trellis_context.h"

int select_gemm_shape( int cc, int size_m, int size_k, int size_n, int K, bool multi, int bszm_in, int bszm_out ) {
	(void) multi;
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
		default: {
			if ( size_m <= 16 ) {
				if ( mod_512 && size_n >= 8192 && K <= 6 ) return 4;
				if ( mod_256 && size_n >= 4096 && K <= 6 ) return 3;
				return 2;
			}
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
