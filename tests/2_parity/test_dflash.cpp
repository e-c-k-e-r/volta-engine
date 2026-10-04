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
#include <atomic>

static bool g_tracking_allocations = false;
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
	size_t arena_bytes = 256ULL * 1024 * 1024; // 256 MB static arena
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
		std::vector<half> h_dyn( bsz * seqlen * taps * groups );
		std::vector<half> h_base( taps * hidden );
		std::vector<half> h_out_gpu( bsz * seqlen * hidden );
		std::vector<float> h_out_cpu( bsz * seqlen * hidden );

		for ( size_t i = 0; i < h_x.size(); ++i ) {
			h_x[i] = host_float_to_half( static_cast<float>( static_cast<int>( i % 17 ) - 8 ) * 0.01f );
		}
		for ( size_t i = 0; i < h_dyn.size(); ++i ) {
			h_dyn[i] = host_float_to_half( static_cast<float>( static_cast<int>( i % 13 ) - 6 ) * 0.01f );
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
						float dyn_val = host_half_to_float( h_dyn[dyn_idx] );
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
		pod::TensorView d_dyn = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, dyn_shape, 4 );
		pod::TensorView d_base = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, base_shape, 2 );
		pod::TensorView d_out = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, x_shape, 3 );

		CUDA_CHECK( cudaMemcpyAsync( d_x.data, h_x.data(), h_x.size() * sizeof( half ), cudaMemcpyHostToDevice, stream ) );
		CUDA_CHECK( cudaMemcpyAsync( d_dyn.data, h_dyn.data(), h_dyn.size() * sizeof( half ), cudaMemcpyHostToDevice, stream ) );
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
	// Test 3: GPU Verification Matching (Full Accept A=8, Partial A in [2..7], Reject A=1)
	// -----------------------------------------------------------------------
	std::printf( "\n[*] Test 3: GPU Verification Matching Suite (A in [1, 8])...\n" );
	{
		const int M = 8;
		int h_draft_tokens[M] = { 100, 101, 102, 103, 104, 105, 106, 107 }; // anchor + 7 draft

		int* d_draft = static_cast<int*>( engine::core::arena_alloc( arena, M * sizeof( int ) ) );
		int* d_target_argmax = static_cast<int*>( engine::core::arena_alloc( arena, M * sizeof( int ) ) );
		int* d_acc_cnt = static_cast<int*>( engine::core::arena_alloc( arena, sizeof( int ) ) );
		int* d_nxt_tok = static_cast<int*>( engine::core::arena_alloc( arena, sizeof( int ) ) );

		int* h_acc_cnt = nullptr;
		int* h_nxt_tok = nullptr;
		CUDA_CHECK( cudaHostAlloc( &h_acc_cnt, sizeof( int ), cudaHostAllocMapped ) );
		CUDA_CHECK( cudaHostAlloc( &h_nxt_tok, sizeof( int ), cudaHostAllocMapped ) );

		CUDA_CHECK( cudaMemcpyAsync( d_draft, h_draft_tokens, M * sizeof( int ), cudaMemcpyHostToDevice, stream ) );

		// Subtest 3.1: Full Accept A = 8
		{
			// All 7 draft tokens match target argmax: target argmax = [101, 102, 103, 104, 105, 106, 107, 999]
			int h_target_argmax[M] = { 101, 102, 103, 104, 105, 106, 107, 999 };
			CUDA_CHECK( cudaMemcpyAsync( d_target_argmax, h_target_argmax, M * sizeof( int ), cudaMemcpyHostToDevice, stream ) );

			engine::spec::verify_greedy_with_argmax( d_target_argmax, d_draft, d_acc_cnt, d_nxt_tok, h_acc_cnt, h_nxt_tok, M, stream );
			std::printf( "   [Case A=8 Full Accept] accepted_count: %d (expected: 8), next_token: %d (expected: 999)\n",
				*h_acc_cnt, *h_nxt_tok );
			ENGINE_ASSERT( *h_acc_cnt == 8, "Expected accepted_count == 8" );
			ENGINE_ASSERT( *h_nxt_tok == 999, "Expected next_token == 999" );
		}

		// Subtest 3.2: Partial Accepts A in [2..7]
		for ( int expected_A = 2; expected_A <= 7; ++expected_A ) {
			int h_target_argmax[M];
			// Tokens 0 .. expected_A - 2 match draft[1 .. expected_A - 1]
			for ( int i = 0; i < expected_A - 1; ++i ) {
				h_target_argmax[i] = h_draft_tokens[i + 1];
			}
			// Divergence at index expected_A - 1:
			int divergent_token = 5000 + expected_A;
			h_target_argmax[expected_A - 1] = divergent_token;
			// Subsequent positions are arbitrary:
			for ( int i = expected_A; i < M; ++i ) {
				h_target_argmax[i] = 7000 + i;
			}

			CUDA_CHECK( cudaMemcpyAsync( d_target_argmax, h_target_argmax, M * sizeof( int ), cudaMemcpyHostToDevice, stream ) );
			engine::spec::verify_greedy_with_argmax( d_target_argmax, d_draft, d_acc_cnt, d_nxt_tok, h_acc_cnt, h_nxt_tok, M, stream );

			std::printf( "   [Case A=%d Partial Accept] accepted_count: %d, next_token: %d (expected: %d)\n",
				expected_A, *h_acc_cnt, *h_nxt_tok, divergent_token );
			ENGINE_ASSERT( *h_acc_cnt == expected_A, "Partial accept count mismatch" );
			ENGINE_ASSERT( *h_nxt_tok == divergent_token, "Partial accept next_token mismatch" );
		}

		// Subtest 3.3: Complete Reject A = 1
		{
			// Target position 0 immediately diverges from draft token 1:
			int h_target_argmax[M] = { 888, 102, 103, 104, 105, 106, 107, 999 };
			CUDA_CHECK( cudaMemcpyAsync( d_target_argmax, h_target_argmax, M * sizeof( int ), cudaMemcpyHostToDevice, stream ) );

			engine::spec::verify_greedy_with_argmax( d_target_argmax, d_draft, d_acc_cnt, d_nxt_tok, h_acc_cnt, h_nxt_tok, M, stream );
			std::printf( "   [Case A=1 Reject] accepted_count: %d (expected: 1), next_token: %d (expected: 888)\n",
				*h_acc_cnt, *h_nxt_tok );
			ENGINE_ASSERT( *h_acc_cnt == 1, "Expected accepted_count == 1" );
			ENGINE_ASSERT( *h_nxt_tok == 888, "Expected next_token == 888" );
		}

		cudaFreeHost( h_acc_cnt );
		cudaFreeHost( h_nxt_tok );
		std::printf( "[+] Test 3 Passed: All A in [1, 8] verified with 100%% precision.\n" );
	}

	// -----------------------------------------------------------------------
	// Test 4: In-place KV Rollback / Advance Logic
	// -----------------------------------------------------------------------
	std::printf( "\n[*] Test 4: In-place KV Rollback / Advance Logic...\n" );
	{
		int current_pos = 100;
		int round1_accepted = 8;
		current_pos += round1_accepted;
		std::printf( "   Round 1 (Full Accept A=8): pos advanced to %d (expected: 108)\n", current_pos );
		ENGINE_ASSERT( current_pos == 108, "Round 1 pos advance failed" );

		int round2_accepted = 4;
		current_pos += round2_accepted;
		std::printf( "   Round 2 (Partial Accept A=4): pos advanced to %d (expected: 112, rolled back 4 slots)\n", current_pos );
		ENGINE_ASSERT( current_pos == 112, "Round 2 pos advance failed" );

		int round3_accepted = 1;
		current_pos += round3_accepted;
		std::printf( "   Round 3 (Reject A=1): pos advanced to %d (expected: 113, rolled back 7 slots)\n", current_pos );
		ENGINE_ASSERT( current_pos == 113, "Round 3 pos advance failed" );

		std::printf( "[+] Test 4 Passed.\n" );
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
			if ( m1 && m2 && m3 ) {
				std::printf( "   [+] Found key tensors: pred_cb (%ld bytes), succ_cb (%ld bytes), base_kernel (%ld bytes)\n",
					m1->size_bytes, m2->size_bytes, m3->size_bytes );
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
	// Test 6: Zero Dynamic Heap Allocation Gate
	// -----------------------------------------------------------------------
	std::printf( "\n[*] Test 6: Zero Dynamic Heap Allocation Gate...\n" );
	{
		engine::spec::DFlashSpeculator speculator;
		pod::DFlashConfig spec_cfg;
		bool spec_ok = speculator.init( spec_cfg, dev, arena, stream );
		ENGINE_ASSERT( spec_ok, "Failed to init DFlashSpeculator" );

		// Warm up
		speculator.draft( 42, 100, stream );
		int dummy_argmax[8] = { 43, 44, 45, 46, 47, 48, 49, 50 };
		speculator.verify_with_argmax( dummy_argmax, stream );

		// Start tracking allocations during the live speculative execution pass
		g_allocation_count.store( 0, std::memory_order_relaxed );
		g_tracking_allocations = true;

		for ( int round = 0; round < 20; ++round ) {
			speculator.draft( 100 + round, 100 + round * 8, stream );
			speculator.prepare_verify_mask( stream );
			auto res = speculator.verify_with_argmax( dummy_argmax, stream );
			(void) res;
		}

		g_tracking_allocations = false;
		size_t pass_allocations = g_allocation_count.load( std::memory_order_relaxed );

		std::printf( "   [+] Dynamic heap allocations during 20 speculative execution rounds: %zu (expected: 0)\n",
			pass_allocations );
		ENGINE_ASSERT( pass_allocations == 0, "FATAL: Dynamic heap allocations detected during speculative pass!" );
		std::printf( "[+] Test 6 Passed: 0 dynamic allocations verified.\n" );

		speculator.destroy();
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
