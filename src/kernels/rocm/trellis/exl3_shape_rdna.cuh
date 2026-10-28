// RDNA EXL3 GEMM shape selection, included by src/kernels/cuda/trellis/exl3_kernel_map.cu on ROCm
#pragma once

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

