// Losslessness precondition for speculative decoding: the target's verify forward runs several rows
// at once, greedy runs one row at a time. Accept-while-match is only lossless if those two produce
// the same argmax, i.e. if the M>1 forward is numerically consistent with the M=1 forward. This
// measures that directly, per row, and reports the logit margin at which they disagree.
// Both paths prefill the same prompt; the batched path then runs the candidates as one causal
// block, the sequential path as M=1 steps carrying state. Logits are read full-vocabulary through
// ModelCluster::read_logits (shard-aware), and each path's on-device sampler is checked against
// the host argmax of the logits it sampled.
//
// Usage: test_spec_target_equiv [--prompt-tokens N] [--rows N]
#include "engine/runtime/test_driver.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

using namespace engine::runtime;

int main( int argc, char** argv ) {
	int n_prompt = 24;
	int rows = 8;
	for ( int i = 1; i < argc - 1; ++i ) {
		if ( !std::strcmp( argv[i], "--prompt-tokens" ) ) n_prompt = atoi( argv[++i] );
		else if ( !std::strcmp( argv[i], "--rows" ) ) rows = atoi( argv[++i] );
	}
	rows = MIN( rows, pod::MAX_BATCH );

	ModelCluster mc;
	ClusterConfig cfg;
	cfg.model_type = "qwen3_5";
	cfg.model_dir = test::QWEN_DIR;
	ENGINE_ASSERT( mc.init( cfg ), "ModelCluster init failed" );
	mc.warm_graphs( rows );

	std::vector<int> prompt( n_prompt );
	for ( int i = 0; i < n_prompt; ++i ) prompt[i] = 1000 + i * 7;

	// Both paths see identical inputs: the same prompt prefilled into the slot, then the same
	// candidate tokens at positions [base_pos, base_pos + rows). The batched path evaluates them as
	// one causal verify block; the sequential path as M=1 steps carrying state forward (greedy decode).
	const int V = mc.metadata().vocab_size;
	const int base_pos = n_prompt;
	int ids[pod::MAX_BATCH];
	for ( int r = 0; r < rows; ++r ) ids[r] = 5000 + r * 11;

	mc.reset_slot( 0 );
	mc.prefill( prompt.data(), n_prompt, 0 );
	mc.forward( ids, rows, base_pos, 0 );
	std::vector<half> batched( static_cast<size_t>( rows ) * V );
	mc.read_logits( rows, batched.data() );
	int batched_sampled[pod::MAX_BATCH];
	mc.sample( batched_sampled, rows );

	// Sequential reference: prefill again, then one row at a time with state carried forward.
	mc.reset_slot( 0 );
	mc.prefill( prompt.data(), n_prompt, 0 );
	std::vector<half> seq_all( static_cast<size_t>( rows ) * V );
	int seq_sampled[pod::MAX_BATCH];
	for ( int r = 0; r < rows; ++r ) {
		mc.forward( &ids[r], 1, base_pos + r, 0 );
		mc.read_logits( 1, seq_all.data() + static_cast<size_t>( r ) * V );
		seq_sampled[r] = mc.sample_one();
	}

	int batched_arg[pod::MAX_BATCH];
	float batched_top[pod::MAX_BATCH], batched_second[pod::MAX_BATCH];
	for ( int r = 0; r < rows; ++r ) {
		const half* row = batched.data() + static_cast<size_t>( r ) * mc.metadata().vocab_size;
		int bi = 0;
		float b1 = -1e30f, b2 = -1e30f;
		for ( int j = 0; j < mc.metadata().vocab_size; ++j ) {
			float v = __half2float( row[j] );
			if ( v > b1 ) { b2 = b1; b1 = v; bi = j; }
			else if ( v > b2 ) { b2 = v; }
		}
		batched_arg[r] = bi;
		batched_top[r] = b1;
		batched_second[r] = b2;
	}

	int mismatch = 0;
	float worst_margin = 1e30f;
	int worst_row = -1;
	std::printf( "row | batched_argmax | seq_argmax | batched_margin | max|dlogit|\n" );
	for ( int r = 0; r < rows; ++r ) {
		const half* seq = seq_all.data() + static_cast<size_t>( r ) * V;
		int si = 0;
		float s1 = -1e30f;
		float maxd = 0.0f;
		for ( int j = 0; j < mc.metadata().vocab_size; ++j ) {
			float v = __half2float( seq[j] );
			if ( v > s1 ) { s1 = v; si = j; }
			maxd = fmaxf( maxd, fabsf( v - __half2float(
				batched[static_cast<size_t>( r ) * mc.metadata().vocab_size + j] ) ) );
		}
		// Margin = top1 - top2 of the batched row (>= 0); a mismatch below it is a near-tie flip
		const float margin = batched_top[r] - batched_second[r];
		// The engine's own samplers must agree with the host argmax of the logits they sampled
		const bool sampler_ok = ( batched_sampled[r] == batched_arg[r] ) && ( seq_sampled[r] == si );
		const bool bad = ( si != batched_arg[r] ) || !sampler_ok;
		if ( bad ) {
			mismatch++;
			if ( margin < worst_margin ) { worst_margin = margin; worst_row = r; }
		}
		std::printf( "%3d | %14d | %10d | %14.6f | %.6f | sampled %d/%d%s\n", r, batched_arg[r], si, margin, maxd,
			batched_sampled[r], seq_sampled[r], bad ? "   <-- MISMATCH" : "" );
	}
	std::printf( "\nrows=%d  mismatches=%d\n", rows, mismatch );
	if ( worst_row >= 0 ) {
		std::printf( "worst margin at row %d: %.6f\n", worst_row, worst_margin );
	}
	mc.shutdown();
	return mismatch == 0 ? 0 : 1;
}
