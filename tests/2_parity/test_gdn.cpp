#include "engine/core/arena.h"
#include "engine/modules/gdn.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <cmath>

static float half_to_float( half h ) {
	return __half2float( h );
}

static half float_to_half( float f ) {
	return __float2half_rn( f );
}

int main() {
	std::printf( "========================================================\n" );
	std::printf( "   volta-engine Tier 2 Parity: GDN Linear Attention     \n" );
	std::printf( "========================================================\n" );

	CUDA_CHECK( cudaSetDevice( 0 ) );
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );

	pod::MemoryArena arena;
	bool ok = engine::core::arena_create( arena, 0, 128ULL * 1024 * 1024 );
	ENGINE_ASSERT( ok, "Failed to create arena" );

	// Test Qwen TP=4 shape: 4 K heads, 12 V heads, head_dim 128
	const int h_k = 4;
	const int h_v = 12;
	const int d = 128;
	const int qkv_channels = ( 2 * h_k + h_v ) * d; // (8 + 12) * 128 = 2560
	const int out_channels = h_v * d; // 1536

	int qkv_shape[2] = { 1, qkv_channels };
	int gate_shape[1] = { h_v };
	int beta_shape[1] = { h_v };
	int out_shape[2] = { 1, out_channels };
	int z_shape[2] = { 1, out_channels };
	int norm_shape[1] = { d };

	pod::TensorView d_qkv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, qkv_shape, 2 );
	pod::TensorView d_gate = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, gate_shape, 1 );
	pod::TensorView d_beta = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, beta_shape, 1 );
	pod::TensorView d_out = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, out_shape, 2 );
	pod::TensorView d_z = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, z_shape, 2 );
	pod::TensorView d_norm_w = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, norm_shape, 1 );

	size_t gdn_state_elems = static_cast<size_t>( h_v ) * d * d;
	float* d_gdn_state = static_cast<float*>( engine::core::arena_alloc( arena, gdn_state_elems * sizeof( float ) ) );
	CUDA_CHECK( cudaMemset( d_gdn_state, 0, gdn_state_elems * sizeof( float ) ) );

	std::vector<float> cpu_state( gdn_state_elems, 0.0f );
	std::vector<half> h_norm_w( d );
	for ( int i = 0; i < d; ++i ) {
		h_norm_w[i] = float_to_half( 0.8f + 0.4f * std::sin( static_cast<float>( i ) * 0.1f ) );
	}
	CUDA_CHECK( cudaMemcpy( d_norm_w.data, h_norm_w.data(), d * sizeof( half ), cudaMemcpyHostToDevice ) );

	// Simulate 20 sequential recurrent steps
	std::printf( "[*] Simulating 20 sequential GDN recurrent steps...\n" );
	for ( int step = 0; step < 20; ++step ) {
		std::vector<half> h_qkv( qkv_channels );
		std::vector<float> h_gate( h_v );
		std::vector<float> h_beta( h_v );
		std::vector<half> h_z( out_channels );

		for ( int i = 0; i < qkv_channels; ++i ) {
			h_qkv[i] = float_to_half( std::sin( static_cast<float>( step * 500 + i ) * 0.03f ) );
		}
		for ( int h = 0; h < h_v; ++h ) {
			h_gate[h] = -0.5f - 0.2f * std::cos( static_cast<float>( h ) * 0.5f );
			h_beta[h] = 0.5f + 0.3f * std::sin( static_cast<float>( h ) * 0.7f );
		}
		for ( int i = 0; i < out_channels; ++i ) {
			h_z[i] = float_to_half( 0.5f * std::cos( static_cast<float>( i ) * 0.04f ) );
		}

		CUDA_CHECK( cudaMemcpyAsync( d_qkv.data, h_qkv.data(), qkv_channels * sizeof( half ), cudaMemcpyHostToDevice, stream ) );
		CUDA_CHECK( cudaMemcpyAsync( d_gate.data, h_gate.data(), h_v * sizeof( float ), cudaMemcpyHostToDevice, stream ) );
		CUDA_CHECK( cudaMemcpyAsync( d_beta.data, h_beta.data(), h_v * sizeof( float ), cudaMemcpyHostToDevice, stream ) );
		CUDA_CHECK( cudaMemcpyAsync( d_z.data, h_z.data(), out_channels * sizeof( half ), cudaMemcpyHostToDevice, stream ) );

		// 1. L2 norm on Q and K heads
		engine::modules::GDN::l2_norm_heads( d_qkv, h_k, d, stream, false );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );

		std::vector<half> h_qkv_after_l2( qkv_channels );
		CUDA_CHECK( cudaMemcpy( h_qkv_after_l2.data(), d_qkv.data, qkv_channels * sizeof( half ), cudaMemcpyDeviceToHost ) );

		// 2. Recurrent step
		engine::modules::GDN::recurrent_step( d_gdn_state, d_qkv, d_gate, d_beta, d_out, h_k, h_v, d, stream );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );

		std::vector<half> h_out_after_rec( out_channels );
		CUDA_CHECK( cudaMemcpy( h_out_after_rec.data(), d_out.data, out_channels * sizeof( half ), cudaMemcpyDeviceToHost ) );

		// 3. Gated RMSNorm
		engine::modules::GDN::gated_rmsnorm( d_out, d_z, d_norm_w, h_v, d, stream );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );

		// Compute CPU reference
		std::vector<float> cpu_qkv( qkv_channels );
		for ( int i = 0; i < qkv_channels; ++i ) {
			cpu_qkv[i] = half_to_float( h_qkv[i] );
		}

		// CPU L2 norm on Q heads (0..h_k-1) and K heads (h_k..2*h_k-1)
		for ( int head = 0; head < 2 * h_k; ++head ) {
			double sum_sq = 0.0;
			for ( int j = 0; j < d; ++j ) {
				float val = cpu_qkv[head * d + j];
				sum_sq += val * val;
			}
			float inv_l2 = 1.0f / std::sqrt( static_cast<float>( sum_sq ) + 1e-6f );
			for ( int j = 0; j < d; ++j ) {
				cpu_qkv[head * d + j] *= inv_l2;
			}
		}

		// CPU Recurrent step
		std::vector<float> cpu_rec_out( out_channels, 0.0f );
		std::vector<float> cpu_out( out_channels, 0.0f );
		float scale = 1.0f / std::sqrt( static_cast<float>( d ) );
		int h_k_ratio = h_v / h_k;

		for ( int vh = 0; vh < h_v; ++vh ) {
			int qh = vh / h_k_ratio;
			const float* q_ptr = cpu_qkv.data() + qh * d;
			const float* k_ptr = cpu_qkv.data() + h_k * d + qh * d;
			const float* v_ptr = cpu_qkv.data() + 2 * h_k * d + vh * d;

			float g_val = std::exp( h_gate[vh] );
			float beta_val = h_beta[vh];

			for ( int col = 0; col < d; ++col ) {
				double kv_col = 0.0;
				for ( int row = 0; row < d; ++row ) {
					kv_col += cpu_state[( static_cast<size_t>( vh ) * d + row ) * d + col] * k_ptr[row];
				}
				float delta_col = ( v_ptr[col] - g_val * static_cast<float>( kv_col ) ) * beta_val;

				double attn_col = 0.0;
				for ( int row = 0; row < d; ++row ) {
					float new_s = g_val * cpu_state[( static_cast<size_t>( vh ) * d + row ) * d + col] + k_ptr[row] * delta_col;
					cpu_state[( static_cast<size_t>( vh ) * d + row ) * d + col] = new_s;
					attn_col += new_s * q_ptr[row];
				}
				cpu_rec_out[vh * d + col] = static_cast<float>( attn_col ) * scale;
				cpu_out[vh * d + col] = static_cast<float>( attn_col ) * scale;
			}

			// Gated RMSNorm on head
			double head_sum_sq = 0.0;
			for ( int j = 0; j < d; ++j ) {
				float val = cpu_out[vh * d + j];
				head_sum_sq += val * val;
			}
			float inv_rms = 1.0f / std::sqrt( static_cast<float>( head_sum_sq / d ) + 1e-6f );
			for ( int j = 0; j < d; ++j ) {
				float val = cpu_out[vh * d + j];
				float gamma = half_to_float( h_norm_w[j] );
				float z_val = half_to_float( h_z[vh * d + j] );
				float silu_z = z_val / ( 1.0f + std::exp( -z_val ) );
				cpu_out[vh * d + j] = val * inv_rms * gamma * silu_z;
			}
		}

		// Read back GPU output
		std::vector<half> h_gpu_out( out_channels );
		CUDA_CHECK( cudaMemcpy( h_gpu_out.data(), d_out.data, out_channels * sizeof( half ), cudaMemcpyDeviceToHost ) );

		double dot = 0.0, norm_ref = 0.0, norm_gpu = 0.0;
		for ( size_t i = 0; i < cpu_out.size(); ++i ) {
			float r = cpu_out[i];
			float g = half_to_float( h_gpu_out[i] );
			dot += r * g;
			norm_ref += r * r;
			norm_gpu += g * g;
		}
		double cos_sim = dot / ( std::sqrt( norm_ref ) * std::sqrt( norm_gpu ) + 1e-9 );
		std::printf( "    [Step %d] cos_sim: %.7f\n", step, cos_sim );
		std::fflush( stdout );
		ENGINE_ASSERT( cos_sim >= 0.9999, "GDN recurrent step failed cosine similarity" );
	}

	std::printf( "    -> GDN 20/20 steps PASS (cos_sim >= 0.9999).\n" );

	CUDA_CHECK( cudaStreamDestroy( stream ) );
	engine::core::arena_destroy( arena );
	CUDA_CHECK( cudaDeviceReset() );

	std::printf( "[+] ALL GDN PARITY TESTS PASSED.\n" );
	return 0;
}
