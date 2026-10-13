#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/modules/trellis.h"
#include "reconstruct.cuh"
#include "trellis_context.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <atomic>

static bool g_tracking_allocations = false;
// Zero host heap allocations per forward is a property of the CUDA runtime. HIP's launch path allocates a command object on the host
// for every kernel launch, so on ROCm the count is reported, not asserted
#if defined(USE_ROCM)
	#define ASSERT_NO_ALLOCS( n, msg ) ( (void) ( n ) )
	#define ALLOCS_TARGET "HIP runtime launch objects, not checked on ROCm"
#else
	#define ASSERT_NO_ALLOCS( n, msg ) ENGINE_ASSERT( ( n ) == 0, msg )
	#define ALLOCS_TARGET "target == 0"
#endif
static std::atomic<size_t> g_allocation_count{ 0 };

void* operator new( size_t size ) {
	if ( g_tracking_allocations ) {
		g_allocation_count.fetch_add( 1, std::memory_order_relaxed );
	}
	void* ptr = std::malloc( size );
	if ( !ptr ) std::abort();
	return ptr;
}

void operator delete( void* ptr ) noexcept {
	std::free( ptr );
}

void* operator new[]( size_t size ) {
	if ( g_tracking_allocations ) {
		g_allocation_count.fetch_add( 1, std::memory_order_relaxed );
	}
	void* ptr = std::malloc( size );
	if ( !ptr ) std::abort();
	return ptr;
}

void operator delete[]( void* ptr ) noexcept {
	std::free( ptr );
}

static float host_half_to_float( half h ) {
	return __half2float( h );
}

static half host_float_to_half( float f ) {
	return __float2half_rn( f );
}

static void run_parity_test_k(
	int K,
	bool mcg,
	int m,
	int k_dim,
	int n_dim,
	pod::MemoryArena& arena,
	cudaStream_t stream,
	bool mul1 = false
) {
	std::printf( "[*] Testing K=%d (%d bpw), mcg=%s%s, M=%d, K=%d, N=%d...\n",
		K, K, mcg ? "true" : "false", mul1 ? ", mul1" : "", m, k_dim, n_dim );

	size_t x_numel = static_cast<size_t>( m ) * k_dim;
	size_t w_numel = static_cast<size_t>( k_dim ) * n_dim;
	size_t out_numel = static_cast<size_t>( m ) * n_dim;
	size_t trellis_numel = static_cast<size_t>( k_dim / 16 ) * ( n_dim / 16 ) * ( 16 * K );

	std::vector<half> h_x( x_numel );
	std::vector<half> h_suh( k_dim );
	std::vector<half> h_svh( n_dim );
	std::vector<uint16_t> h_trellis( trellis_numel );
	std::vector<half> h_out_gpu( out_numel );
	std::vector<half> h_w_recon( w_numel );

	unsigned int seed = 12345 + K * 101;
	for ( size_t i = 0; i < x_numel; ++i ) {
		seed = seed * 1664525u + 1013904223u;
		float val = ( static_cast<float>( seed & 0xffff ) / 65536.0f - 0.5f ) * 0.4f;
		h_x[i] = host_float_to_half( val );
	}
	for ( int i = 0; i < k_dim; ++i ) {
		seed = seed * 1664525u + 1013904223u;
		float val = ( static_cast<float>( seed & 0xffff ) / 65536.0f - 0.5f ) * 0.2f;
		h_suh[i] = host_float_to_half( val );
	}
	for ( int i = 0; i < n_dim; ++i ) {
		seed = seed * 1664525u + 1013904223u;
		float val = ( static_cast<float>( seed & 0xffff ) / 65536.0f - 0.5f ) * 0.2f;
		h_svh[i] = host_float_to_half( val );
	}
	for ( size_t i = 0; i < trellis_numel; ++i ) {
		seed = seed * 1664525u + 1013904223u;
		h_trellis[i] = static_cast<uint16_t>( seed & 0xffff );
	}

	engine::core::arena_reset( arena );

	int x_shape[2] = { m, k_dim };
	pod::TensorView x_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, x_shape, 2 );

	int trellis_shape[3] = { k_dim / 16, n_dim / 16, 16 * K };
	pod::TensorView trellis_tv = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, trellis_shape, 3 );

	int suh_shape[1] = { k_dim };
	pod::TensorView suh_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, suh_shape, 1 );

	int svh_shape[1] = { n_dim };
	pod::TensorView svh_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, svh_shape, 1 );

	int out_shape[2] = { m, n_dim };
	pod::TensorView out_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, out_shape, 2 );

	int w_shape[2] = { k_dim, n_dim };
	pod::TensorView w_recon_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, w_shape, 2 );

	CUDA_CHECK( cudaMemcpyAsync( x_tv.data, h_x.data(), x_numel * sizeof( half ), cudaMemcpyHostToDevice, stream ) );
	CUDA_CHECK( cudaMemcpyAsync( trellis_tv.data, h_trellis.data(), trellis_numel * sizeof( uint16_t ), cudaMemcpyHostToDevice, stream ) );
	CUDA_CHECK( cudaMemcpyAsync( suh_tv.data, h_suh.data(), k_dim * sizeof( half ), cudaMemcpyHostToDevice, stream ) );
	CUDA_CHECK( cudaMemcpyAsync( svh_tv.data, h_svh.data(), n_dim * sizeof( half ), cudaMemcpyHostToDevice, stream ) );

	// Reconstruct the exact weight matrix on GPU via reconstruct_had_slice
	reconstruct_had_slice(
		w_recon_tv.ptr<half>(),
		trellis_tv.ptr<const uint16_t>(),
		suh_tv.ptr<const half>(),
		svh_tv.ptr<const half>(),
		K,
		mcg,
		mul1,
		k_dim,
		n_dim,
		0,
		stream
	);
	CUDA_CHECK( cudaStreamSynchronize( stream ) );

	CUDA_CHECK( cudaMemcpy( h_w_recon.data(), w_recon_tv.data, w_numel * sizeof( half ), cudaMemcpyDeviceToHost ) );

	// Compute FP32 reference on CPU
	std::vector<float> h_out_ref( out_numel, 0.0f );
	for ( int i = 0; i < m; ++i ) {
		for ( int j = 0; j < n_dim; ++j ) {
			double acc = 0.0;
			for ( int p = 0; p < k_dim; ++p ) {
				float a = host_half_to_float( h_x[i * k_dim + p] );
				float b = host_half_to_float( h_w_recon[p * n_dim + j] );
				acc += static_cast<double>( a * b );
			}
			h_out_ref[i * n_dim + j] = static_cast<float>( acc );
		}
	}

	// Warm-up execution pass
	engine::modules::LinearTrellis::forward(
		x_tv,
		trellis_tv,
		suh_tv,
		svh_tv,
		out_tv,
		K,
		mcg,
		mul1,
		stream
	);
	CUDA_CHECK( cudaStreamSynchronize( stream ) );

	// Execution pass with 0-allocation assertion
	g_tracking_allocations = true;
	g_allocation_count.store( 0 );

	engine::modules::LinearTrellis::forward(
		x_tv,
		trellis_tv,
		suh_tv,
		svh_tv,
		out_tv,
		K,
		mcg,
		mul1,
		stream
	);
	CUDA_CHECK( cudaStreamSynchronize( stream ) );

	g_tracking_allocations = false;
	size_t dynamic_allocs = g_allocation_count.load();
	ASSERT_NO_ALLOCS( dynamic_allocs, "Execution pass performed dynamic heap allocations!" );

	CUDA_CHECK( cudaMemcpy( h_out_gpu.data(), out_tv.data, out_numel * sizeof( half ), cudaMemcpyDeviceToHost ) );

	// Metrics: Cosine Similarity and Max Absolute Relative Error
	double dot = 0.0;
	double norm_gpu_sq = 0.0;
	double norm_ref_sq = 0.0;
	double max_rel_err = 0.0;
	double diff_sq_sum = 0.0;
	double ref_sq_sum = 0.0;

	for ( size_t i = 0; i < out_numel; ++i ) {
		float gpu_val = host_half_to_float( h_out_gpu[i] );
		float ref_val = h_out_ref[i];

		dot += static_cast<double>( gpu_val * ref_val );
		norm_gpu_sq += static_cast<double>( gpu_val * gpu_val );
		norm_ref_sq += static_cast<double>( ref_val * ref_val );

		double diff = std::fabs( static_cast<double>( gpu_val - ref_val ) );
		double denom = std::fabs( static_cast<double>( ref_val ) ) + 1e-4;
		double rel_err = diff / denom;
		if ( rel_err > max_rel_err ) {
			max_rel_err = rel_err;
		}

		diff_sq_sum += diff * diff;
		ref_sq_sum += static_cast<double>( ref_val * ref_val );
	}

	double cos_sim = dot / ( std::sqrt( norm_gpu_sq ) * std::sqrt( norm_ref_sq ) + 1e-12 );
	double rms_rel_err = std::sqrt( diff_sq_sum ) / ( std::sqrt( ref_sq_sum ) + 1e-12 );

	std::printf( "    -> Cosine Similarity: %.6f (target >= 0.99990)\n", cos_sim );
	std::printf( "    -> RMS Relative Error: %.6f (target <= 0.0050)\n", rms_rel_err );
	std::printf( "    -> Max Relative Error: %.6f\n", max_rel_err );
	std::printf( "    -> Dynamic Allocations: %zu (" ALLOCS_TARGET ")\n", dynamic_allocs );
	for ( int dbg = 0; dbg < 5; ++dbg ) {
		std::printf( "       [%d] GPU: %f, REF: %f\n", dbg, host_half_to_float( h_out_gpu[dbg] ), h_out_ref[dbg] );
	}
	std::fflush( stdout );

	ENGINE_ASSERT( cos_sim >= 0.99990, "Cosine similarity below 0.9999" );
	ENGINE_ASSERT( rms_rel_err <= 0.0050, "RMS relative error exceeds 0.005" );
	ASSERT_NO_ALLOCS( dynamic_allocs, "Non-zero dynamic allocations during execution" );

	std::printf( "[+] K=%d PASS.\n", K );
}

int main( int argc, char** argv ) {
	(void) argc;
	(void) argv;

	std::printf( "========================================================\n" );
	std::printf( "   volta-engine Phase 0: Trellis Parity Test Suite       \n" );
	std::printf( "========================================================\n" );

	int dev = 0;
	CUDA_CHECK( cudaSetDevice( dev ) );

	cudaDeviceProp prop;
	CUDA_CHECK( cudaGetDeviceProperties( &prop, dev ) );
	std::printf( "[*] Device %d: %s (SM %d.%d, %d SMs, %zu MB VRAM)\n",
		dev, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
		prop.totalGlobalMem / ( 1024 * 1024 ) );

	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );

	pod::MemoryArena arena;
	size_t arena_bytes = 256 * 1024 * 1024; // 256 MB static arena
	bool ok = engine::core::arena_create( arena, dev, arena_bytes );
	ENGINE_ASSERT( ok, "Failed to create static arena" );

	// Sweep across 2 .. 8 bpw (the M <= 8 GEMV kernels exist for 2 .. 6; 1, 7 and 8 take the GEMM kernel)
	int bit_widths[] = { 2, 3, 4, 5, 6, 7, 8 };
	for ( int K : bit_widths ) {
		bool mcg = ( K != 4 );
		// Test GEMV shape (m = 1 or 4, small N)
		run_parity_test_k( K, mcg, 4, 512, 512, arena, stream );
		// Test GEMM shape (m = 16 or wide N)
		run_parity_test_k( K, mcg, 16, 512, 1024, arena, stream );
	}
	// Full-width GEMV (a model projection's N): the grid reaches the device's co-resident limit, where the cooperative kernel's
	// grid.sync needs every block resident (the N = 512 cases above launch 16 blocks)
	run_parity_test_k( 4, false, 1, 4096, 8192, arena, stream );
	run_parity_test_k( 3, true, 1, 4096, 8192, arena, stream );
	// Odd row counts (ROCm's GEMV takes every M <= 8; CUDA the GEMM)
	run_parity_test_k( 4, false, 3, 4096, 1024, arena, stream );
	run_parity_test_k( 6, true, 7, 2048, 1024, arena, stream );
	// Multiplier codebook (cb 2): what the Flash-Next 7 / 8 bpw projections use
	for ( int K : { 5, 7, 8 } ) {
		run_parity_test_k( K, false, 4, 512, 512, arena, stream, true );
		run_parity_test_k( K, false, 16, 512, 1024, arena, stream, true );
	}

	CUDA_CHECK( cudaStreamDestroy( stream ) );

	// Teardown everything cleanly to ensure 0 MiB VRAM across all 4 GPUs
	engine::core::arena_destroy( arena );
	engine::modules::trellis_context_destroy();

	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		(void) cudaSetDevice( d );
		(void) cudaDeviceReset();
	}

	std::printf( "========================================================\n" );
	std::printf( "[+] ALL TRELLIS PARITY TESTS PASSED SUCCESSFULLY!\n" );
	std::printf( "========================================================\n" );

	return 0;
}
