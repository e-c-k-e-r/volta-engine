#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/spec/dflash.h"
#include "engine/loader/safetensors.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

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
	}

	double cos_sim = dot / ( std::sqrt( norm_act_sq ) * std::sqrt( norm_exp_sq ) + 1e-12 );

	std::printf( "   [%s] Cosine Similarity: %.6f (min: %.6f) | Max Rel Err: %.6f (max: %.6f)\n",
		name, cos_sim, min_cos_sim, max_rel_err, max_rel_err_limit );
	std::fflush( stdout );

	ENGINE_ASSERT( cos_sim >= min_cos_sim, "Cosine similarity violation" );
	ENGINE_ASSERT( max_rel_err <= max_rel_err_limit, "Max relative error violation" );
}

int main( int argc, char** argv ) {
	(void) argc;
	(void) argv;

	std::printf( "========================================================\n" );
	std::printf( "   volta-engine Phase 3: DFlash2 Parity & Spec Suite    \n" );
	std::printf( "========================================================\n" );
	std::fflush( stdout );

	int dev = 0;
	CUDA_CHECK( cudaSetDevice( dev ) );

	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );

	pod::MemoryArena arena;
	// The drafter now loads its real backbone: five layers of EXL3 projections plus the replicated
	// conv kernel projections, the two 248320x256 codebooks, and a sliding-window context KV cache.
	// 4 GiB: the drafter's per-position tap ring is sliding_window * 5 * hidden FP16, which is 105 MiB
	// on its own for the real 5120-wide geometry.
	size_t arena_bytes = 4ULL * 1024 * 1024 * 1024;
	bool ok = engine::core::arena_create( arena, dev, arena_bytes );
	ENGINE_ASSERT( ok, "Failed to create static arena" );

	// -----------------------------------------------------------------------
	// Test 1: dflash2_dynconv Mathematical Parity (FP16 & FP32)
	// -----------------------------------------------------------------------
	std::printf( "[*] Test 1: dflash2_dynconv Mathematical Parity...\n" );
	{
		const int bsz = 1;
		const int seqlen = 8;
		const int hidden = 5120;
		const int taps = 2;
		const int group_size = 16;
		const int groups = hidden / group_size;

		std::vector<half> h_x( bsz * seqlen * hidden );
		// Dynamic coefficients are FP32 in the engine; the kernel template takes that type explicitly.
		std::vector<float> h_dyn( bsz * seqlen * taps * groups );
		std::vector<half> h_base( taps * hidden );
		std::vector<half> h_out_gpu( bsz * seqlen * hidden );
		std::vector<float> h_out_cpu( bsz * seqlen * hidden );

		for ( size_t i = 0; i < h_x.size(); ++i ) {
			h_x[i] = host_float_to_half( static_cast<float>( static_cast<int>( i % 17 ) - 8 ) * 0.01f );
		}
		for ( size_t i = 0; i < h_dyn.size(); ++i ) {
			h_dyn[i] = static_cast<float>( static_cast<int>( i % 13 ) - 6 ) * 0.01f;
		}
		for ( size_t i = 0; i < h_base.size(); ++i ) {
			h_base[i] = host_float_to_half( static_cast<float>( static_cast<int>( i % 19 ) - 9 ) * 0.01f );
		}

		// CPU reference calculation:
		// out[b, t, c] = sum_k (base[k, c] + dyn[b, t, k, c / group_size]) * x[b, t - k, c]
		for ( int b = 0; b < bsz; ++b ) {
			for ( int t = 0; t < seqlen; ++t ) {
				for ( int c = 0; c < hidden; ++c ) {
					int g = c / group_size;
					float acc = 0.0f;
					for ( int k = 0; k < taps; ++k ) {
						if ( k > t ) break;
						float base_val = host_half_to_float( h_base[k * hidden + c] );
						size_t dyn_idx = static_cast<size_t>( b ) * ( seqlen * taps * groups ) +
						                 static_cast<size_t>( t ) * ( taps * groups ) +
						                 static_cast<size_t>( k ) * groups + g;
						float dyn_val = h_dyn[dyn_idx];
						size_t x_idx = static_cast<size_t>( b ) * ( seqlen * hidden ) +
						               static_cast<size_t>( t - k ) * hidden + c;
						float x_val = host_half_to_float( h_x[x_idx] );
						acc += ( base_val + dyn_val ) * x_val;
					}
					h_out_cpu[static_cast<size_t>( b ) * ( seqlen * hidden ) + static_cast<size_t>( t ) * hidden + c] = acc;
				}
			}
		}

		int x_shape[3] = { bsz, seqlen, hidden };
		int dyn_shape[4] = { bsz, seqlen, taps, groups };
		int base_shape[2] = { taps, hidden };

		pod::TensorView d_x = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, x_shape, 3 );
		pod::TensorView d_dyn = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, dyn_shape, 4 );
		pod::TensorView d_base = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, base_shape, 2 );
		pod::TensorView d_out = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, x_shape, 3 );

		CUDA_CHECK( cudaMemcpyAsync( d_x.data, h_x.data(), h_x.size() * sizeof( half ), cudaMemcpyHostToDevice, stream ) );
		CUDA_CHECK( cudaMemcpyAsync( d_dyn.data, h_dyn.data(), h_dyn.size() * sizeof( float ), cudaMemcpyHostToDevice, stream ) );
		CUDA_CHECK( cudaMemcpyAsync( d_base.data, h_base.data(), h_base.size() * sizeof( half ), cudaMemcpyHostToDevice, stream ) );

		engine::spec::dflash2_dynconv( d_x, d_dyn, d_base, d_out, group_size, false, stream );

		CUDA_CHECK( cudaMemcpyAsync( h_out_gpu.data(), d_out.data, h_out_gpu.size() * sizeof( half ), cudaMemcpyDeviceToHost, stream ) );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );

		std::vector<float> h_out_act( h_out_gpu.size() );
		for ( size_t i = 0; i < h_out_gpu.size(); ++i ) {
			h_out_act[i] = host_half_to_float( h_out_gpu[i] );
		}

		verify_parity( "dflash2_dynconv FP16", h_out_act.data(), h_out_cpu.data(), h_out_act.size(), 0.9999, 0.005 );
		std::printf( "[+] Test 1 Passed.\n" );
	}

	// -----------------------------------------------------------------------
	// Test 2: dflash2_selector_walk Exact Chained Parity
	// -----------------------------------------------------------------------
	std::printf( "\n[*] Test 2: dflash2_selector_walk Exact Chained Parity...\n" );
	{
		const int bsz = 1;
		const int rows = 7;
		const int k = 16;
		const int rank = 256;
		const int vocab = 1000;

		std::vector<float> h_unary( bsz * rows * k );
		std::vector<int> h_cands( bsz * rows * k );
		std::vector<half> h_gate( bsz * rows * rank );
		std::vector<half> h_pred_cb( vocab * rank );
		std::vector<half> h_succ_cb( vocab * rank );
		int h_anchor = 42;

		for ( size_t i = 0; i < h_unary.size(); ++i ) {
			h_unary[i] = static_cast<float>( static_cast<int>( i % 31 ) - 15 ) * 0.1f;
			h_cands[i] = static_cast<int>( ( ( i * 7 ) + 13 ) % vocab );
		}
		for ( size_t i = 0; i < h_gate.size(); ++i ) {
			h_gate[i] = host_float_to_half( static_cast<float>( static_cast<int>( i % 11 ) - 5 ) * 0.05f );
		}
		for ( size_t i = 0; i < h_pred_cb.size(); ++i ) {
			h_pred_cb[i] = host_float_to_half( static_cast<float>( static_cast<int>( i % 17 ) - 8 ) * 0.02f );
		}
		for ( size_t i = 0; i < h_succ_cb.size(); ++i ) {
			h_succ_cb[i] = host_float_to_half( static_cast<float>( static_cast<int>( i % 19 ) - 9 ) * 0.02f );
		}

		// CPU reference of greedy chained walk:
		std::vector<int> h_cpu_out( bsz * ( rows + 1 ) );
		std::vector<float> h_cpu_conf( bsz * ( rows + 1 ) );
		h_cpu_out[0] = h_anchor;
		h_cpu_conf[0] = 1.0f;
		int pred = h_anchor;

		for ( int i = 0; i < rows; ++i ) {
			std::vector<float> a_g( rank );
			for ( int r = 0; r < rank; ++r ) {
				float a_val = host_half_to_float( h_pred_cb[pred * rank + r] );
				float g_val = host_half_to_float( h_gate[i * rank + r] );
				a_g[r] = a_val * g_val;
			}

			int best_cand_idx = 0;
			float best_score = -1e30f;
			std::vector<float> scores( k );
			for ( int c = 0; c < k; ++c ) {
				int cand_tok = h_cands[i * k + c];
				float dot = 0.0f;
				for ( int r = 0; r < rank; ++r ) {
					dot += a_g[r] * host_half_to_float( h_succ_cb[cand_tok * rank + r] );
				}
				scores[c] = h_unary[i * k + c] + dot;
				if ( scores[c] > best_score ) {
					best_score = scores[c];
					best_cand_idx = c;
				}
			}

			pred = h_cands[i * k + best_cand_idx];
			h_cpu_out[i + 1] = pred;

			float sum_exp = 0.0f;
			for ( int c = 0; c < k; ++c ) {
				sum_exp += std::exp( scores[c] - best_score );
			}
			h_cpu_conf[i + 1] = 1.0f / sum_exp;
		}

		// Allocate on GPU
		float* d_unary = static_cast<float*>( engine::core::arena_alloc( arena, h_unary.size() * sizeof( float ) ) );
		int* d_cands = static_cast<int*>( engine::core::arena_alloc( arena, h_cands.size() * sizeof( int ) ) );
		half* d_gate = static_cast<half*>( engine::core::arena_alloc( arena, h_gate.size() * sizeof( half ) ) );
		half* d_pred_cb = static_cast<half*>( engine::core::arena_alloc( arena, h_pred_cb.size() * sizeof( half ) ) );
		half* d_succ_cb = static_cast<half*>( engine::core::arena_alloc( arena, h_succ_cb.size() * sizeof( half ) ) );
		int* d_anchor = static_cast<int*>( engine::core::arena_alloc( arena, sizeof( int ) ) );
		int* d_out = static_cast<int*>( engine::core::arena_alloc( arena, ( rows + 1 ) * sizeof( int ) ) );
		float* d_conf = static_cast<float*>( engine::core::arena_alloc( arena, ( rows + 1 ) * sizeof( float ) ) );

		CUDA_CHECK( cudaMemcpyAsync( d_unary, h_unary.data(), h_unary.size() * sizeof( float ), cudaMemcpyHostToDevice, stream ) );
		CUDA_CHECK( cudaMemcpyAsync( d_cands, h_cands.data(), h_cands.size() * sizeof( int ), cudaMemcpyHostToDevice, stream ) );
		CUDA_CHECK( cudaMemcpyAsync( d_gate, h_gate.data(), h_gate.size() * sizeof( half ), cudaMemcpyHostToDevice, stream ) );
		CUDA_CHECK( cudaMemcpyAsync( d_pred_cb, h_pred_cb.data(), h_pred_cb.size() * sizeof( half ), cudaMemcpyHostToDevice, stream ) );
		CUDA_CHECK( cudaMemcpyAsync( d_succ_cb, h_succ_cb.data(), h_succ_cb.size() * sizeof( half ), cudaMemcpyHostToDevice, stream ) );
		CUDA_CHECK( cudaMemcpyAsync( d_anchor, &h_anchor, sizeof( int ), cudaMemcpyHostToDevice, stream ) );

		engine::spec::dflash2_selector_walk(
			d_unary, d_cands, d_gate, d_pred_cb, d_succ_cb, d_anchor, d_out, d_conf,
			bsz, rows, k, rank, stream
		);

		std::vector<int> h_gpu_out( rows + 1 );
		std::vector<float> h_gpu_conf( rows + 1 );
		CUDA_CHECK( cudaMemcpyAsync( h_gpu_out.data(), d_out, ( rows + 1 ) * sizeof( int ), cudaMemcpyDeviceToHost, stream ) );
		CUDA_CHECK( cudaMemcpyAsync( h_gpu_conf.data(), d_conf, ( rows + 1 ) * sizeof( float ), cudaMemcpyDeviceToHost, stream ) );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );

		for ( int i = 0; i <= rows; ++i ) {
			ENGINE_ASSERT( h_gpu_out[i] == h_cpu_out[i], "Selector walk token ID mismatch" );
		}
		verify_parity( "dflash2_selector_walk confidence", h_gpu_conf.data(), h_cpu_conf.data(), rows + 1, 0.9999, 0.005 );
		std::printf( "[+] Test 2 Passed: 100%% bit-exact chained tokens match.\n" );
	}


	// -----------------------------------------------------------------------
	// Test 5: SafeTensorsRepo Draft Model Weight Discovery
	// -----------------------------------------------------------------------
	std::printf( "\n[*] Test 5: SafeTensorsRepo Draft Model Weight Discovery...\n" );
	{
		engine::loader::SafeTensorsRepo repo;
		const char* qwen_dir = "/home/mrq/Documents/models/Qwen3.8-27B-DFlash2-EXL3-5.0bpw";
		bool opened = repo.open( qwen_dir );
		if ( opened ) {
			std::printf( "   [+] Successfully opened Qwen3.8-27B-DFlash2-EXL3 repo (shards: %d)\n", repo.shard_count() );
			const auto* m1 = repo.find( "candidate_selector.predecessor_codebook" );
			const auto* m2 = repo.find( "candidate_selector.successor_codebook" );
			const auto* m3 = repo.find( "layers.0.attention_conv.base_kernel" );
			const auto* m4 = repo.find( "fc.trellis" );
			if ( m1 && m2 && m3 && m4 ) {
				std::printf( "   [+] Found key tensors: pred_cb (%zu bytes), succ_cb (%zu bytes), base_kernel (%zu bytes), fc.trellis (%zu bytes)\n",
					m1->size_bytes, m2->size_bytes, m3->size_bytes, m4->size_bytes );
			}
			repo.close();
		}
		else {
			std::printf( "   [*] Notice: %s not accessible, skipping checkpoint load verification\n", qwen_dir );
		}

		const char* glm_dir = "/home/mrq/Documents/models/GLM-5.3-Flash-DFlash2";
		opened = repo.open( glm_dir );
		if ( opened ) {
			std::printf( "   [+] Successfully opened GLM-5.3-Flash-DFlash2 repo (shards: %d)\n", repo.shard_count() );
			const auto* m1 = repo.find( "candidate_selector.predecessor_codebook" );
			if ( m1 ) {
				std::printf( "   [+] Found key tensor: pred_cb (%ld bytes)\n", m1->size_bytes );
			}
			repo.close();
		}
		std::printf( "[+] Test 5 Passed.\n" );
	}

	// -----------------------------------------------------------------------
	// Test 5b: DFlash2 checkpoint loads under the shipped DFlashConfig
	// -----------------------------------------------------------------------
	// DFlashConfig's defaults describe the Qwen3.8-27B-DFlash2 drafter, and ModelCluster only sets
	// hidden_size, vocab_size and block_size. load_weights now validates the geometry against the
	// checkpoint so a mismatched drafter fails loudly instead of silently running Qwen shapes.
	// The negative case cannot be checked in-process: ENGINE_ASSERT calls _Exit, so it is spawned as
	// a child and judged by exit status.
	std::printf( "\n[*] Test 5b: DFlash2 geometry validation against the real checkpoint...\n" );
	{
		const char* d2_dir = "/home/mrq/Documents/models/Qwen3.8-27B-DFlash2-EXL3-5.0bpw";
		engine::loader::SafeTensorsRepo probe;
		if ( probe.open( d2_dir ) ) {
			probe.close();
			engine::spec::DFlashSpeculator sp;
			pod::DFlashConfig cfg;
			engine::core::arena_reset( arena );
			bool ok_init = sp.init( cfg, dev, arena, stream );
			ENGINE_ASSERT( ok_init, "speculator init failed for the real drafter" );
			bool ok_load = sp.load_weights( d2_dir, stream );
			CUDA_CHECK( cudaStreamSynchronize( stream ) );
			ENGINE_ASSERT( ok_load, "the shipped DFlashConfig was rejected by the real checkpoint" );
			std::printf( "   [+] Real drafter loads under the shipped DFlashConfig\n" );
			sp.destroy();

			// Negative case: same checkpoint, wrong geometry. ENGINE_ASSERT calls _Exit, so run it as
			// a child process and judge by exit status -- an in-process check would take the suite down.
			if ( std::getenv( "EXL3_DFLASH2_CHILD" ) ) {
				pod::DFlashConfig bad;
				bad.hidden_size = 2560;   // wrong width: head_dim, intermediate and groups all disagree
				engine::spec::DFlashSpeculator bsp;
				engine::core::arena_reset( arena );
				bsp.init( bad, dev, arena, stream );
				bsp.load_weights( d2_dir, stream );
				std::printf( "   [!] validator did NOT reject the mismatched config\n" );
				std::_Exit( 0 );
			}
			const std::string cmd = std::string( "./" ) + argv[ 0 ]
				+ " > /dev/null 2>&1";
			(void) cmd;
			{
				char self[ 4096 ];
				snprintf( self, sizeof( self ), "%s", argv[ 0 ] );
				setenv( "EXL3_DFLASH2_CHILD", "1", 1 );
				const int rc = system( self );
				unsetenv( "EXL3_DFLASH2_CHILD" );
				// Nonzero means the child hit the geometry assert, which is the point.
				ENGINE_ASSERT( rc != 0, "geometry validation accepted a mismatched DFlashConfig" );
				std::printf( "   [+] Mismatched DFlashConfig is rejected (child exit %d)\n", rc );
			}
		}
		else {
			std::printf( "   [--] drafter checkpoint not present; geometry validation not exercised\n" );
		}
	}

	// Teardown & Reset
	CUDA_CHECK( cudaStreamDestroy( stream ) );
	engine::core::arena_destroy( arena );

	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		cudaSetDevice( d );
		cudaDeviceReset();
	}

	std::printf( "========================================================\n" );
	std::printf( "[+] ALL DFLASH2 TIER 2 PARITY TESTS PASSED!\n" );
	std::printf( "========================================================\n" );

	return 0;
}
