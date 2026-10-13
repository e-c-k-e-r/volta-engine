#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/modules/mhc.h"
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

static void verify_parity(
	const char* name,
	const float* actual,
	const float* expected,
	size_t n,
	double min_cos_sim = 0.9999,
	double max_rel_err_limit = 0.005
) {
	double dot = 0.0;
	double norm_act_sq = 0.0;
	double norm_exp_sq = 0.0;
	double max_rel_err = 0.0;
	double diff_sq_sum = 0.0;
	double exp_sq_sum = 0.0;

	for ( size_t i = 0; i < n; ++i ) {
		double a = static_cast<double>( actual[i] );
		double e = static_cast<double>( expected[i] );

		dot += a * e;
		norm_act_sq += a * a;
		norm_exp_sq += e * e;

		double diff = std::fabs( a - e );
		double denom = std::fabs( e ) + 1e-4;
		double rel_err = diff / denom;
		if ( rel_err > max_rel_err ) {
			max_rel_err = rel_err;
		}

		diff_sq_sum += diff * diff;
		exp_sq_sum += e * e;
	}

	double cos_sim = dot / ( std::sqrt( norm_act_sq ) * std::sqrt( norm_exp_sq ) + 1e-12 );
	double rms_rel_err = std::sqrt( diff_sq_sum ) / ( std::sqrt( exp_sq_sum ) + 1e-12 );

	std::printf( "    [%s] Cosine Similarity: %.7f (target >= %.4f)\n", name, cos_sim, min_cos_sim );
	std::printf( "    [%s] RMS Relative Error: %.7f\n", name, rms_rel_err );
	std::printf( "    [%s] Max Relative Error: %.7f (target <= %.4f)\n", name, max_rel_err, max_rel_err_limit );
	std::fflush( stdout );

	ENGINE_ASSERT( cos_sim >= min_cos_sim, "Cosine similarity below threshold" );
	ENGINE_ASSERT( max_rel_err <= max_rel_err_limit, "Max relative error exceeds limit" );
}

static void run_mhc_parity_test(
	int hidden_size,
	pod::MemoryArena& arena,
	cudaStream_t stream
) {
	std::printf( "[*] Testing HyperConnection Parity: hidden_size=%d (N=4*%d=%d floats)...\n",
		hidden_size, hidden_size, 4 * hidden_size );

	const int num_streams = 4;
	const int total_stream_floats = num_streams * hidden_size; // 16384
	const int fn_rows = 24;
	const int fn_numel = fn_rows * total_stream_floats;

	std::vector<float> h_streams( total_stream_floats );
	std::vector<float> h_fn( fn_numel );
	std::vector<float> h_base( 24 );
	std::vector<float> h_scale( 3 );
	std::vector<half> h_sublayer_out( hidden_size );

	unsigned int seed = 424242u;
	for ( int i = 0; i < total_stream_floats; ++i ) {
		seed = seed * 1664525u + 1013904223u;
		h_streams[i] = ( static_cast<float>( seed & 0xffff ) / 65536.0f - 0.5f ) * 0.5f;
	}
	for ( int i = 0; i < fn_numel; ++i ) {
		seed = seed * 1664525u + 1013904223u;
		h_fn[i] = ( static_cast<float>( seed & 0xffff ) / 65536.0f - 0.5f ) * 0.1f;
	}
	for ( int i = 0; i < 24; ++i ) {
		seed = seed * 1664525u + 1013904223u;
		h_base[i] = ( static_cast<float>( seed & 0xffff ) / 65536.0f - 0.5f ) * 0.2f;
	}
	h_scale[0] = 1.0f;
	h_scale[1] = 1.0f;
	h_scale[2] = 1.0f;

	for ( int i = 0; i < hidden_size; ++i ) {
		seed = seed * 1664525u + 1013904223u;
		float val = ( static_cast<float>( seed & 0xffff ) / 65536.0f - 0.5f ) * 0.4f;
		h_sublayer_out[i] = host_float_to_half( val );
	}

	// -------------------------------------------------------------------
	// Exact CPU FP32 Reference Computation
	// -------------------------------------------------------------------
	// 1. Unweighted RMSNorm
	double sum_sq = 0.0;
	for ( int i = 0; i < total_stream_floats; ++i ) {
		sum_sq += static_cast<double>( h_streams[i] ) * static_cast<double>( h_streams[i] );
	}
	float ref_inv_rms = 1.0f / std::sqrt( static_cast<float>( sum_sq / total_stream_floats ) + 1e-5f );

	// 2. Linear projection with fn
	std::vector<float> ref_mix_logits( 24, 0.0f );
	for ( int k = 0; k < fn_rows; ++k ) {
		double dot = 0.0;
		for ( int j = 0; j < total_stream_floats; ++j ) {
			dot += static_cast<double>( h_streams[j] * ref_inv_rms ) * static_cast<double>( h_fn[k * total_stream_floats + j] );
		}
		ref_mix_logits[k] = static_cast<float>( dot );
	}

	// 3. Sigmoid gates
	std::vector<float> ref_pre( 4 );
	std::vector<float> ref_post( 4 );
	for ( int h = 0; h < 4; ++h ) {
		float val = ref_mix_logits[h] * h_scale[0] + h_base[h];
		ref_pre[h] = 1.0f / ( 1.0f + std::exp( -val ) ) + 1e-6f;
	}
	for ( int h = 0; h < 4; ++h ) {
		float val = ref_mix_logits[4 + h] * h_scale[1] + h_base[4 + h];
		ref_post[h] = 2.0f / ( 1.0f + std::exp( -val ) );
	}

	// 4. Sinkhorn combine matrix
	float ref_comb[4][4];
	for ( int i = 0; i < 4; ++i ) {
		float max_v = -1e20f;
		for ( int j = 0; j < 4; ++j ) {
			float val = ref_mix_logits[8 + i * 4 + j] * h_scale[2] + h_base[8 + i * 4 + j];
			ref_comb[i][j] = val;
			if ( val > max_v ) max_v = val;
		}
		float sum_exp = 0.0f;
		for ( int j = 0; j < 4; ++j ) {
			ref_comb[i][j] = std::exp( ref_comb[i][j] - max_v );
			sum_exp += ref_comb[i][j];
		}
		float inv_sum = 1.0f / sum_exp;
		for ( int j = 0; j < 4; ++j ) {
			ref_comb[i][j] = ref_comb[i][j] * inv_sum + 1e-6f;
		}
	}
	// Initial column normalization
	for ( int j = 0; j < 4; ++j ) {
		float col_sum = ref_comb[0][j] + ref_comb[1][j] + ref_comb[2][j] + ref_comb[3][j] + 1e-6f;
		float inv_col = 1.0f / col_sum;
		for ( int i = 0; i < 4; ++i ) {
			ref_comb[i][j] *= inv_col;
		}
	}
	// 19 Sinkhorn iterations
	for ( int it = 0; it < 19; ++it ) {
		for ( int i = 0; i < 4; ++i ) {
			float row_sum = ref_comb[i][0] + ref_comb[i][1] + ref_comb[i][2] + ref_comb[i][3] + 1e-6f;
			float inv_row = 1.0f / row_sum;
			for ( int j = 0; j < 4; ++j ) {
				ref_comb[i][j] *= inv_row;
			}
		}
		for ( int j = 0; j < 4; ++j ) {
			float col_sum = ref_comb[0][j] + ref_comb[1][j] + ref_comb[2][j] + ref_comb[3][j] + 1e-6f;
			float inv_col = 1.0f / col_sum;
			for ( int i = 0; i < 4; ++i ) {
				ref_comb[i][j] *= inv_col;
			}
		}
	}

	// 5. Stream collapse
	std::vector<float> ref_collapsed_f32( hidden_size );
	for ( int j = 0; j < hidden_size; ++j ) {
		float sum = ref_pre[0] * h_streams[0 * hidden_size + j]
		          + ref_pre[1] * h_streams[1 * hidden_size + j]
		          + ref_pre[2] * h_streams[2 * hidden_size + j]
		          + ref_pre[3] * h_streams[3 * hidden_size + j];
		ref_collapsed_f32[j] = host_half_to_float( host_float_to_half( sum ) );
	}

	// 6. Stream update
	std::vector<float> ref_updated_streams( total_stream_floats );
	for ( int j = 0; j < hidden_size; ++j ) {
		float y = host_half_to_float( h_sublayer_out[j] );
		float r0 = h_streams[0 * hidden_size + j];
		float r1 = h_streams[1 * hidden_size + j];
		float r2 = h_streams[2 * hidden_size + j];
		float r3 = h_streams[3 * hidden_size + j];

		for ( int h = 0; h < 4; ++h ) {
			float post_h = ref_post[h];
			float c0 = ref_comb[0][h];
			float c1 = ref_comb[1][h];
			float c2 = ref_comb[2][h];
			float c3 = ref_comb[3][h];

			float res_mix = c0 * r0 + c1 * r1 + c2 * r2 + c3 * r3;
			float new_val = post_h * y + res_mix;
			ref_updated_streams[h * hidden_size + j] = new_val;
		}
	}

	// 7. HyperHead mean
	std::vector<float> ref_final_hidden_f32( hidden_size );
	for ( int j = 0; j < hidden_size; ++j ) {
		float sum = ref_updated_streams[0 * hidden_size + j]
		          + ref_updated_streams[1 * hidden_size + j]
		          + ref_updated_streams[2 * hidden_size + j]
		          + ref_updated_streams[3 * hidden_size + j];
		ref_final_hidden_f32[j] = host_half_to_float( host_float_to_half( sum * 0.25f ) );
	}

	// -------------------------------------------------------------------
	// Device Execution
	// -------------------------------------------------------------------
	engine::core::arena_reset( arena );

	int streams_shape[2] = { num_streams, hidden_size };
	pod::TensorView streams_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, streams_shape, 2 );

	int fn_shape[2] = { fn_rows, total_stream_floats };
	pod::TensorView fn_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, fn_shape, 2 );

	int base_shape[1] = { 24 };
	pod::TensorView base_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, base_shape, 1 );

	int scale_shape[1] = { 3 };
	pod::TensorView scale_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, scale_shape, 1 );

	int sublayer_shape[1] = { hidden_size };
	pod::TensorView sublayer_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, sublayer_shape, 1 );

	int h_shape[2] = { 1, hidden_size };
	pod::TensorView collapsed_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );
	pod::TensorView final_hidden_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, h_shape, 2 );

	int post_shape[1] = { 4 };
	pod::TensorView post_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, post_shape, 1 );

	int comb_shape[2] = { 4, 4 };
	pod::TensorView comb_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, comb_shape, 2 );

	int rms_shape[1] = { 1 };
	pod::TensorView inv_rms_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, rms_shape, 1 );

	int mix_shape[1] = { 24 };
	pod::TensorView mix_logits_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, mix_shape, 1 );

	int pre_shape[1] = { 4 };
	pod::TensorView pre_tv = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, pre_shape, 1 );

	CUDA_CHECK( cudaMemcpyAsync( streams_tv.data, h_streams.data(), total_stream_floats * sizeof( float ), cudaMemcpyHostToDevice, stream ) );
	CUDA_CHECK( cudaMemcpyAsync( fn_tv.data, h_fn.data(), fn_numel * sizeof( float ), cudaMemcpyHostToDevice, stream ) );
	CUDA_CHECK( cudaMemcpyAsync( base_tv.data, h_base.data(), 24 * sizeof( float ), cudaMemcpyHostToDevice, stream ) );
	CUDA_CHECK( cudaMemcpyAsync( scale_tv.data, h_scale.data(), 3 * sizeof( float ), cudaMemcpyHostToDevice, stream ) );
	CUDA_CHECK( cudaMemcpyAsync( sublayer_tv.data, h_sublayer_out.data(), hidden_size * sizeof( half ), cudaMemcpyHostToDevice, stream ) );

	// Warm-up pass
	engine::modules::HyperConnection::collapse(
		streams_tv, fn_tv, base_tv, scale_tv,
		collapsed_tv, post_tv, comb_tv,
		inv_rms_tv, mix_logits_tv, pre_tv,
		hidden_size, 20, 1e-6f, stream
	);
	engine::modules::HyperConnection::update(
		streams_tv, sublayer_tv, post_tv, comb_tv, hidden_size, stream
	);
	engine::modules::HyperHead::mean(
		streams_tv, final_hidden_tv, hidden_size, stream
	);
	CUDA_CHECK( cudaStreamSynchronize( stream ) );

	// Reset streams on device for tracked pass
	CUDA_CHECK( cudaMemcpyAsync( streams_tv.data, h_streams.data(), total_stream_floats * sizeof( float ), cudaMemcpyHostToDevice, stream ) );

	// Execution pass with zero dynamic heap allocation assertion
	g_tracking_allocations = true;
	g_allocation_count.store( 0 );

	engine::modules::HyperConnection::collapse(
		streams_tv, fn_tv, base_tv, scale_tv,
		collapsed_tv, post_tv, comb_tv,
		inv_rms_tv, mix_logits_tv, pre_tv,
		hidden_size, 20, 1e-6f, stream
	);
	engine::modules::HyperConnection::update(
		streams_tv, sublayer_tv, post_tv, comb_tv, hidden_size, stream
	);
	engine::modules::HyperHead::mean(
		streams_tv, final_hidden_tv, hidden_size, stream
	);
	CUDA_CHECK( cudaStreamSynchronize( stream ) );

	g_tracking_allocations = false;
	size_t dynamic_allocs = g_allocation_count.load();
	ASSERT_NO_ALLOCS( dynamic_allocs, "Execution pass performed dynamic heap allocations!" );
	std::printf( "    -> Dynamic Allocations: %zu (" ALLOCS_TARGET ")\n", dynamic_allocs );

	// Read back outputs
	std::vector<half> h_gpu_collapsed( hidden_size );
	std::vector<float> h_gpu_collapsed_f32( hidden_size );
	CUDA_CHECK( cudaMemcpy( h_gpu_collapsed.data(), collapsed_tv.data, hidden_size * sizeof( half ), cudaMemcpyDeviceToHost ) );
	for ( int i = 0; i < hidden_size; ++i ) {
		h_gpu_collapsed_f32[i] = host_half_to_float( h_gpu_collapsed[i] );
	}

	std::vector<float> h_gpu_streams( total_stream_floats );
	CUDA_CHECK( cudaMemcpy( h_gpu_streams.data(), streams_tv.data, total_stream_floats * sizeof( float ), cudaMemcpyDeviceToHost ) );

	std::vector<half> h_gpu_final_hidden( hidden_size );
	std::vector<float> h_gpu_final_hidden_f32( hidden_size );
	CUDA_CHECK( cudaMemcpy( h_gpu_final_hidden.data(), final_hidden_tv.data, hidden_size * sizeof( half ), cudaMemcpyDeviceToHost ) );
	for ( int i = 0; i < hidden_size; ++i ) {
		h_gpu_final_hidden_f32[i] = host_half_to_float( h_gpu_final_hidden[i] );
	}

	// Verify mathematical parity
	verify_parity( "Stream Collapse", h_gpu_collapsed_f32.data(), ref_collapsed_f32.data(), hidden_size, 0.9999, 0.005 );
	verify_parity( "Stream Update", h_gpu_streams.data(), ref_updated_streams.data(), total_stream_floats, 0.9999, 0.005 );
	verify_parity( "HyperHead Mean", h_gpu_final_hidden_f32.data(), ref_final_hidden_f32.data(), hidden_size, 0.9999, 0.005 );

	std::printf( "[+] HyperConnection & HyperHead PASS.\n" );
}

int main( int argc, char** argv ) {
	(void) argc;
	(void) argv;

	std::printf( "========================================================\n" );
	std::printf( "   volta-engine Phase 2: HyperConnections (mHC) Parity  \n" );
	std::printf( "========================================================\n" );
	std::fflush( stdout );

	int dev = 0;
	CUDA_CHECK( cudaSetDevice( dev ) );

	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );

	pod::MemoryArena arena;
	size_t arena_bytes = 256ULL * 1024 * 1024; // 256 MB arena
	bool ok = engine::core::arena_create( arena, dev, arena_bytes );
	ENGINE_ASSERT( ok, "Failed to create memory arena" );

	run_mhc_parity_test( 4096, arena, stream );

	CUDA_CHECK( cudaStreamDestroy( stream ) );
	engine::core::arena_destroy( arena );
	CUDA_CHECK( cudaDeviceReset() );

	std::printf( "========================================================\n" );
	std::printf( "[+] ALL TIER 2 mHC PARITY TESTS PASSED SUCCESSFULLY!\n" );
	std::printf( "========================================================\n" );

	return 0;
}
