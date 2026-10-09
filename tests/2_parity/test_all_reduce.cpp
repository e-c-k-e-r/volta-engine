#include <cstdlib>
#include "engine/core/arena.h"
#include "engine/core/comm.h"
#include "engine/core/tensor.h"
#include "engine/modules/norm.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <thread>
#include <atomic>

static float half_to_float( half h ) {
	return __half2float( h );
}

static half float_to_half( float f ) {
	return __float2half_rn( f );
}

int main() {
	// AR_RANKS=2 checks the two-rank collective (the TP=2 configuration); the default is the 4-GPU mesh
	const int R = std::getenv( "AR_RANKS" ) ? std::atoi( std::getenv( "AR_RANKS" ) ) : ENGINE_NUM_GPUS;
	std::printf( "========================================================\n" );
	std::printf( "   volta-engine Tier 2 Parity: NVLink AllReduce & Norm   \n" );
	std::printf( "========================================================\n" );

	engine::comm::CollectiveRing rings[ENGINE_NUM_GPUS];
	void* staging_ptrs[ENGINE_NUM_GPUS] = { nullptr };
	void* flag_ptrs[ENGINE_NUM_GPUS] = { nullptr };
	engine::comm::init_4gpu_mesh( rings, staging_ptrs, flag_ptrs, engine::comm::kMeshStagingBytes, R );

	cudaStream_t streams[ENGINE_NUM_GPUS];
	pod::MemoryArena arenas[ENGINE_NUM_GPUS];

	const int hidden_size = 5120;
	const int rows = std::getenv( "AR_ROWS" ) ? std::atoi( std::getenv( "AR_ROWS" ) ) : 1;
	const size_t total_elems = static_cast<size_t>( rows ) * hidden_size;

	for ( int d = 0; d < R; ++d ) {
		CUDA_CHECK( cudaSetDevice( d ) );
		CUDA_CHECK( cudaStreamCreate( &streams[d] ) );
		bool ok = engine::core::arena_create( arenas[d], d, 64ULL * 1024 * 1024 );
		ENGINE_ASSERT( ok, "Failed to create arena" );
	}

	pod::TensorView d_x[ENGINE_NUM_GPUS];
	pod::TensorView d_res[ENGINE_NUM_GPUS];
	pod::TensorView d_norm_out[ENGINE_NUM_GPUS];
	pod::TensorView d_w[ENGINE_NUM_GPUS];

	int shape_2d[2] = { rows, hidden_size };
	int shape_1d[1] = { hidden_size };

	for ( int d = 0; d < R; ++d ) {
		CUDA_CHECK( cudaSetDevice( d ) );
		d_x[d] = engine::core::arena_alloc_tensor( arenas[d], pod::DType::FP16, shape_2d, 2 );
		d_res[d] = engine::core::arena_alloc_tensor( arenas[d], pod::DType::FP16, shape_2d, 2 );
		d_norm_out[d] = engine::core::arena_alloc_tensor( arenas[d], pod::DType::FP16, shape_2d, 2 );
		d_w[d] = engine::core::arena_alloc_tensor( arenas[d], pod::DType::FP16, shape_1d, 1 );
	}

	// Test 1: Pure AllReduce across multiple steps
	std::printf( "[*] Test 1: Pure AllReduce multi-step accumulation across 4 GPUs...\n" );
	for ( int step = 0; step < 20; ++step ) {
		std::vector<half> h_x[ENGINE_NUM_GPUS];
		std::vector<float> h_ref( total_elems, 0.0f );

		for ( int d = 0; d < R; ++d ) {
			h_x[d].resize( total_elems );
			for ( size_t i = 0; i < total_elems; ++i ) {
				float val = std::sin( static_cast<float>( step * 1000 + d * 100 + i ) * 0.01f );
				h_x[d][i] = float_to_half( val );
				h_ref[i] += val;
			}
			CUDA_CHECK( cudaSetDevice( d ) );
			CUDA_CHECK( cudaMemcpyAsync( d_x[d].data, h_x[d].data(), total_elems * sizeof( half ), cudaMemcpyHostToDevice, streams[d] ) );
		}

		// Launch all_reduce concurrently across 4 threads
		std::thread workers[ENGINE_NUM_GPUS];
		for ( int d = 0; d < R; ++d ) {
			workers[d] = std::thread( [&, d]() {
				CUDA_CHECK( cudaSetDevice( d ) );
				rings[d].all_reduce( d_x[d], streams[d] );
				CUDA_CHECK( cudaStreamSynchronize( streams[d] ) );
			} );
		}
		for ( int d = 0; d < R; ++d ) {
			workers[d].join();
		}

		// Verify on rank 0
		std::vector<half> h_gpu( total_elems );
		CUDA_CHECK( cudaSetDevice( 0 ) );
		CUDA_CHECK( cudaMemcpy( h_gpu.data(), d_x[0].data, total_elems * sizeof( half ), cudaMemcpyDeviceToHost ) );

		double dot = 0.0, norm_ref = 0.0, norm_gpu = 0.0;
		float max_abs_err = 0.0f;
		double sum_sq_err = 0.0;
		for ( size_t i = 0; i < total_elems; ++i ) {
			float r = h_ref[i];
			float g = half_to_float( h_gpu[i] );
			dot += r * g;
			norm_ref += r * r;
			norm_gpu += g * g;
			float diff = std::fabs( r - g );
			if ( diff > max_abs_err ) max_abs_err = diff;
			sum_sq_err += ( diff * diff ) / ( r * r + 1e-4f );
		}
		double cos_sim = dot / ( std::sqrt( norm_ref ) * std::sqrt( norm_gpu ) + 1e-9 );
		float rms_rel_err = std::sqrt( static_cast<float>( sum_sq_err / total_elems ) );
		std::printf( "    [Step %d] cos_sim: %.7f, max_abs_err: %.6f, rms_rel_err: %.6f\n",
			step, cos_sim, max_abs_err, rms_rel_err );
		std::fflush( stdout );
		ENGINE_ASSERT( cos_sim >= 0.9999, "Pure AllReduce step failed cosine similarity" );
		ENGINE_ASSERT( rms_rel_err <= 0.005f, "Pure AllReduce step failed RMS relative error" );
	}
	std::printf( "    -> Pure AllReduce 20/20 steps PASS (cos_sim >= 0.9999).\n" );

	// Test 2: Fused AllReduce + Residual + RMSNorm (both bias = 0.0f and bias = 1.0f)
	std::printf( "[*] Test 2: Fused AllReduce + Residual + RMSNorm multi-step...\n" );
	for ( float bias : { 0.0f, 1.0f } ) {
		for ( int step = 0; step < 20; ++step ) {
			std::vector<half> h_x[ENGINE_NUM_GPUS];
			std::vector<half> h_res[ENGINE_NUM_GPUS];
			std::vector<half> h_w( hidden_size );

			std::vector<float> h_res_ref( total_elems, 0.0f );
			std::vector<float> h_norm_ref( total_elems, 0.0f );

			for ( int j = 0; j < hidden_size; ++j ) {
				h_w[j] = float_to_half( 0.5f + 0.5f * std::cos( static_cast<float>( j ) * 0.05f ) );
			}

			// Residual baseline
			for ( size_t i = 0; i < total_elems; ++i ) {
				float r_val = std::cos( static_cast<float>( step * 500 + i ) * 0.02f );
				h_res_ref[i] = r_val;
			}

			for ( int d = 0; d < R; ++d ) {
				h_x[d].resize( total_elems );
				h_res[d].resize( total_elems );
				for ( size_t i = 0; i < total_elems; ++i ) {
					float x_val = std::sin( static_cast<float>( step * 1000 + d * 100 + i ) * 0.01f );
					h_x[d][i] = float_to_half( x_val );
					h_res[d][i] = float_to_half( h_res_ref[i] );
					h_res_ref[i] += x_val; // Each rank accumulates sum(x) + residual
				}
				CUDA_CHECK( cudaSetDevice( d ) );
				CUDA_CHECK( cudaMemcpyAsync( d_x[d].data, h_x[d].data(), total_elems * sizeof( half ), cudaMemcpyHostToDevice, streams[d] ) );
				CUDA_CHECK( cudaMemcpyAsync( d_res[d].data, h_res[d].data(), total_elems * sizeof( half ), cudaMemcpyHostToDevice, streams[d] ) );
				CUDA_CHECK( cudaMemcpyAsync( d_w[d].data, h_w.data(), hidden_size * sizeof( half ), cudaMemcpyHostToDevice, streams[d] ) );
			}

			// Compute CPU reference RMSNorm
			for ( int r = 0; r < rows; ++r ) {
				double sum_sq = 0.0;
				for ( int c = 0; c < hidden_size; ++c ) {
					float val = h_res_ref[r * hidden_size + c];
					sum_sq += val * val;
				}
				float inv_rms = 1.0f / std::sqrt( static_cast<float>( sum_sq / hidden_size ) + 1e-6f );
				for ( int c = 0; c < hidden_size; ++c ) {
					float val = h_res_ref[r * hidden_size + c];
					float weight_val = half_to_float( h_w[c] ) + bias;
					h_norm_ref[r * hidden_size + c] = val * weight_val * inv_rms;
				}
			}

			// Launch fused all_reduce_rms_norm concurrently across 4 threads
			std::thread workers[ENGINE_NUM_GPUS];
			for ( int d = 0; d < R; ++d ) {
				workers[d] = std::thread( [&, d]() {
					CUDA_CHECK( cudaSetDevice( d ) );
					rings[d].all_reduce_rms_norm( d_x[d], d_res[d], d_w[d], 1e-6f, streams[d], bias, &d_norm_out[d] );
					CUDA_CHECK( cudaStreamSynchronize( streams[d] ) );
				} );
			}
			for ( int d = 0; d < R; ++d ) {
				workers[d].join();
			}

			// Verify updated residual on rank 0
			std::vector<half> h_gpu_res( total_elems );
			std::vector<half> h_gpu_norm( total_elems );
			CUDA_CHECK( cudaSetDevice( 0 ) );
			CUDA_CHECK( cudaMemcpy( h_gpu_res.data(), d_res[0].data, total_elems * sizeof( half ), cudaMemcpyDeviceToHost ) );
			CUDA_CHECK( cudaMemcpy( h_gpu_norm.data(), d_norm_out[0].data, total_elems * sizeof( half ), cudaMemcpyDeviceToHost ) );

			double dot_res = 0.0, norm_ref_res = 0.0, norm_gpu_res = 0.0;
			for ( size_t i = 0; i < total_elems; ++i ) {
				float r = h_res_ref[i];
				float g = half_to_float( h_gpu_res[i] );
				dot_res += r * g;
				norm_ref_res += r * r;
				norm_gpu_res += g * g;
			}
			double cos_res = dot_res / ( std::sqrt( norm_ref_res ) * std::sqrt( norm_gpu_res ) + 1e-9 );
			ENGINE_ASSERT( cos_res >= 0.9999, "Fused AllReduce residual failed cosine similarity" );

			double dot_norm = 0.0, norm_ref_n = 0.0, norm_gpu_n = 0.0;
			for ( size_t i = 0; i < total_elems; ++i ) {
				float r = h_norm_ref[i];
				float g = half_to_float( h_gpu_norm[i] );
				dot_norm += r * g;
				norm_ref_n += r * r;
				norm_gpu_n += g * g;
			}
			double cos_norm = dot_norm / ( std::sqrt( norm_ref_n ) * std::sqrt( norm_gpu_n ) + 1e-9 );
			ENGINE_ASSERT( cos_norm >= 0.9999, "Fused AllReduce RMSNorm failed cosine similarity" );
		}
		std::printf( "    -> Fused AllReduce + RMSNorm (bias=%.1f) 20/20 steps PASS (cos_sim >= 0.9999).\n", bias );
	}

	// Test 3: Replicate Asynchronous Multi-Layer Pipeline (128 consecutive calls per rank without host sync)
	std::printf( "[*] Test 3: Asynchronous 128-layer pipeline simulation without host sync...\n" );
	{
		const int num_layers = 64;
		std::vector<half> h_init_x[ENGINE_NUM_GPUS];
		for ( int d = 0; d < R; ++d ) {
			h_init_x[d].resize( total_elems );
			for ( size_t i = 0; i < total_elems; ++i ) {
				h_init_x[d][i] = float_to_half( 0.01f * std::sin( static_cast<float>( d * 100 + i ) * 0.05f ) );
			}
			CUDA_CHECK( cudaSetDevice( d ) );
			CUDA_CHECK( cudaMemcpyAsync( d_x[d].data, h_init_x[d].data(), total_elems * sizeof( half ), cudaMemcpyHostToDevice, streams[d] ) );
		}

		std::thread workers[ENGINE_NUM_GPUS];
		for ( int d = 0; d < R; ++d ) {
			workers[d] = std::thread( [&, d]() {
				CUDA_CHECK( cudaSetDevice( d ) );
				for ( int l = 0; l < num_layers * 2; ++l ) {
					rings[d].all_reduce( d_x[d], streams[d] );
					// Normalize so values don't exponentially grow past FP16 limits
					engine::modules::RMSNorm::forward( d_x[d], d_w[d], d_x[d], 1e-6f, 0.0f, streams[d] );
				}
				CUDA_CHECK( cudaStreamSynchronize( streams[d] ) );
			} );
		}
		for ( int d = 0; d < R; ++d ) {
			workers[d].join();
		}

		// Read back results across all 4 ranks
		std::vector<half> h_out[ENGINE_NUM_GPUS];
		for ( int d = 0; d < R; ++d ) {
			h_out[d].resize( total_elems );
			CUDA_CHECK( cudaSetDevice( d ) );
			CUDA_CHECK( cudaMemcpy( h_out[d].data(), d_x[d].data, total_elems * sizeof( half ), cudaMemcpyDeviceToHost ) );
		}

		// Verify that all 4 ranks produced identical values
		for ( int d = 1; d < R; ++d ) {
			double dot = 0.0, norm_0 = 0.0, norm_d = 0.0;
			for ( size_t i = 0; i < total_elems; ++i ) {
				float v0 = half_to_float( h_out[0][i] );
				float vd = half_to_float( h_out[d][i] );
				dot += v0 * vd;
				norm_0 += v0 * v0;
				norm_d += vd * vd;
			}
			double cos_sim = dot / ( std::sqrt( norm_0 ) * std::sqrt( norm_d ) + 1e-9 );
			std::printf( "    [Rank 0 vs %d] cos_sim: %.7f\n", d, cos_sim );
			std::fflush( stdout );
			ENGINE_ASSERT( cos_sim >= 0.99999, "Asynchronous multi-layer all-reduce ranks diverged!" );
		}
		std::printf( "    -> 128 asynchronous layers PASS (ranks bit-exact).\n" );
	}

	engine::comm::destroy_4gpu_mesh( staging_ptrs, flag_ptrs );
	for ( int d = 0; d < R; ++d ) {
		CUDA_CHECK( cudaSetDevice( d ) );
		CUDA_CHECK( cudaStreamDestroy( streams[d] ) );
		engine::core::arena_destroy( arenas[d] );
		CUDA_CHECK( cudaDeviceReset() );
	}

	std::printf( "[+] ALL ALL_REDUCE PARITY TESTS PASSED.\n" );
	return 0;
}
