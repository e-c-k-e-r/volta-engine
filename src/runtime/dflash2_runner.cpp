#include "engine/runtime/dflash2_runner.h"
#include "engine/runtime/cluster.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace engine::runtime {

namespace impl {

double now_ms() {
	return std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now().time_since_epoch() ).count();
}

} // namespace impl

bool DFlash2Runner::init( ModelCluster& mc, const std::string& draft_dir ) {
	mc_ = &mc;
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		ENGINE_ASSERT( mc.model( d ).draft_hooks().valid(), "DFlash2Runner: model exposes no draft hooks" );
	}
	const auto& meta = mc.metadata();
	pod::DFlashConfig cfg;
	cfg.hidden_size = meta.hidden_size;
	cfg.vocab_size = meta.vocab_size;
	cfg.block_size = models::kDraftM;
	cfg.mask_token_id = 248070;
	// Full rotary over head_dim: the drafter checkpoint declares rope_type "default" with no
	// partial_rotary_factor, and a partial-rotation default would quietly poison every prediction.
	cfg.rope_dim_sections[0] = cfg.head_dim;
	cfg.rope_dim_sections[1] = 0;
	cfg.rope_dim_sections[2] = 0;
	cfg.rope_dim_sections[3] = 0;
	ENGINE_ASSERT( cfg.selector_top_k == models::kDraftTopK, "DFlash2Runner: selector_top_k != kDraftTopK" );

	const char* tp_env = std::getenv( "EXL3_DRAFT_TP" );
	tp_ = ( tp_env && tp_env[0] == '0' ) ? 1 : ENGINE_NUM_GPUS;
	bool loaded = true;
	for ( int d = 0; d < tp_; ++d ) {
		CUDA_CHECK( cudaSetDevice( d ) );
		spec_[d].set_candidate_ranks( ENGINE_NUM_GPUS );
		if ( tp_ > 1 ) spec_[d].set_ring( mc.ring( d ) );
		if ( !spec_[d].init( cfg, d, mc.arena( d ), mc.stream( d ), d, tp_ ) ) return false;
		loaded = spec_[d].load_weights( draft_dir.c_str(), mc.stream( d ) ) && loaded;
	}
	std::printf( "[+] DFlash2 speculator: %s (%s), drafter tp=%d\n", loaded ? "loaded" : "STANDALONE", draft_dir.c_str(), tp_ );
	return true;
}

void DFlash2Runner::shutdown() {
	for ( int d = 0; d < tp_; ++d ) {
		CUDA_CHECK( cudaSetDevice( d ) );
		spec_[d].destroy();
	}
	mc_ = nullptr;
	owner_ = -1;
}

void DFlash2Runner::claim( int slot ) {
	owner_ = slot;
	for ( int d = 0; d < tp_; ++d ) spec_[d].reset_context();
	for ( int i = 0; i < 8; ++i ) att_[i] = hit_[i] = 0;
	// cond_ survives: per-step survival is a property of the drafter against this target, not of the
	// sequence being decoded, so resetting it per request would re-learn it at full depth every time.
	mc_->run_ranks( [&]( int d ) {
		const int v = slot;
		const auto& hk = mc_->model( d ).draft_hooks();
		// Pageable source: staged before return, and the sync below orders it before any forward.
		CUDA_CHECK( cudaMemcpyAsync( hk.taps_owner, &v, sizeof( int ), cudaMemcpyHostToDevice, mc_->stream( d ) ) );
		CUDA_CHECK( cudaStreamSynchronize( mc_->stream( d ) ) );
	} );
}

void DFlash2Runner::step_stats( int* attempts, int* hits ) const {
	for ( int i = 0; i < 8; ++i ) {
		attempts[i] = att_[i];
		hits[i] = hit_[i];
	}
}

SpecResult DFlash2Runner::speculate( int slot, int anchor, int pos, float depth_threshold ) {
	ENGINE_ASSERT( owner_ == slot, "DFlash2Runner::speculate: slot does not own the drafter context" );
	constexpr int M = models::kDraftM;
	constexpr int K = M - 1;
	ModelCluster& mc = *mc_;
	const auto& meta = mc.metadata();
	const cudaStream_t s0 = mc.stream( 0 );
	SpecResult res;
	prof_last_ = SpecProfile::DRAFT;
	prof_t0_ = impl::now_ms();

	// 1. Drafter backbone over the block [anchor, mask x K] and the target's tapped context, then each rank's
	// head shard + local top-k over the drafter hidden. Every rank's candidate set is the top-k of its
	// vocabulary shard (exact: a rank's local top-k is a superset of the global top-k restricted to its shard).
	const auto& hk0 = mc.model( 0 ).draft_hooks();
	const size_t hid_bytes = static_cast<size_t>( M ) * mc.metadata().hidden_size * sizeof( half );
	if ( tp_ > 1 ) {
		// Sharded drafter: all ranks run the backbone in lockstep and end with the same full hidden.
		mc.run_ranks( [&]( int d ) {
			const auto& hk = mc.model( d ).draft_hooks();
			spec_[d].draft_backbone( *hk.embedding, anchor, hk.taps, pos, hk.taps_cap, mc.stream( d ) );
			CUDA_CHECK( cudaMemcpyAsync( hk.lm_in, spec_[d].draft_hidden().data, hid_bytes, cudaMemcpyDeviceToDevice, mc.stream( d ) ) );
			mc.model( d ).draft_head_topk( M, mc.stream( d ) );
			CUDA_CHECK( cudaStreamSynchronize( mc.stream( d ) ) );
		} );
	}
	else {
		CUDA_CHECK( cudaSetDevice( 0 ) );
		spec_[0].draft_backbone( *hk0.embedding, anchor, hk0.taps, pos, hk0.taps_cap, s0 );
		// Peer the hidden rows to every rank, then run each rank's head shard on its own worker/stream.
		const pod::TensorView& dh = spec_[0].draft_hidden();
		for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
			CUDA_CHECK( cudaMemcpyPeerAsync( mc.model( d ).draft_hooks().lm_in, d, dh.ptr<const half>(), 0, hid_bytes, s0 ) );
		}
		CUDA_CHECK( cudaStreamSynchronize( s0 ) );
		mc.run_ranks( [&]( int d ) {
			mc.model( d ).draft_head_topk( M, mc.stream( d ) );
			CUDA_CHECK( cudaStreamSynchronize( mc.stream( d ) ) );
		} );
	}

	// Bring the small [M, k] results to rank 0 and merge.
	const size_t val_bytes = static_cast<size_t>( M ) * models::kDraftTopK * sizeof( float );
	const size_t idx_bytes = static_cast<size_t>( M ) * models::kDraftTopK * sizeof( int );
	CUDA_CHECK( cudaSetDevice( 0 ) );
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		const auto& hk = mc.model( d ).draft_hooks();
		CUDA_CHECK( cudaMemcpyPeerAsync( spec_[0].merged_cand_values_ptr( d ), 0, hk.cand_vals, d, val_bytes, s0 ) );
		CUDA_CHECK( cudaMemcpyPeerAsync( spec_[0].merged_cand_indices_ptr( d ), 0, hk.cand_idx, d, idx_bytes, s0 ) );
	}
	engine::spec::dflash2_topk_merge(
		spec_[0].merged_cand_values_ptr( 0 ), spec_[0].merged_cand_indices_ptr( 0 ),
		ENGINE_NUM_GPUS, M, models::kDraftTopK,
		spec_[0].merged_final_values_ptr(), spec_[0].merged_final_indices_ptr(), M, s0 );

	// 3. Selector gate + bilinear lattice walk over the merged candidates.
	mark( SpecProfile::WALK );
	spec_[0].project_gate( K, s0 );
	spec_[0].set_walk_anchor( anchor, s0 );
	spec_[0].draft_select( spec_[0].merged_final_values_ptr(), spec_[0].merged_final_indices_ptr(), K, s0 );
	int h_draft[M];
	CUDA_CHECK( cudaMemcpyAsync( h_draft, spec_[0].d_draft_tokens(), sizeof( h_draft ), cudaMemcpyDeviceToHost, s0 ) );
	CUDA_CHECK( cudaStreamSynchronize( s0 ) );
	CUDA_CHECK( cudaGetLastError() );

	// Depth: the whole block by default (what DFlash2 does natively). depth_threshold > 0 enables the
	// survival-EMA truncation; it is a throughput option, off by default so acceptance is measurable.
	int active = K;
	if ( depth_threshold > 0.0f ) {
		active = 1;
		float run = cond_[0];
		for ( int k = 1; k < K && run >= depth_threshold; ++k ) {
			run *= cond_[k];
			active = k + 1;
		}
	}
	active = MIN( MAX( active, 1 ), meta.max_seq_len - pos - 2 );
	if ( active < 1 ) {
		// No room to verify a block: fall back to the target's own prediction for the anchor.
		mc.forward( &anchor, 1, pos, slot );
		res.emitted[res.emitted_count++] = mc.sample_one();
		res.next_pos = pos + 1;
		return res;
	}
	res.proposed = active;
	const int* cands = h_draft + 1;

	// 4. Batched verify of [anchor, c1..c_active] at positions [pos, pos + active]. Row 0 re-feeds the
	// anchor (exllamav3 generator/job.py:384), so one forward yields both the draft verification and
	// the anchor's own successor: row j's logits predict position pos+j+1, so cands[k] is checked
	// against preds[k] with no shift.
	int block[pod::MAX_BATCH];
	block[0] = anchor;
	for ( int i = 0; i < active; ++i ) block[1 + i] = cands[i];

	mark( SpecProfile::CHECKPOINT );
	mc.set_spec_snapshot( true );
	mark( SpecProfile::VERIFY );
	mc.forward( block, active + 1, pos, slot );
	int preds[pod::MAX_BATCH];
	mc.sample( preds, active + 1 );
	mark( SpecProfile::REPAIR );
	++prof_.rounds;

	// 5. Greedy acceptance.
	for ( int k = 0; k < active; ++k ) {
		++att_[k];
		if ( cands[k] != preds[k] ) {
			res.emitted[res.emitted_count++] = preds[k]; // correction token
			break;
		}
		++hit_[k];
		res.emitted[res.emitted_count++] = cands[k];
		res.accepted++;
	}
	// Per-step survival EMA, observed only where the step was proposed.
	constexpr float kEma = 0.25f;
	for ( int k = 0; k < res.proposed; ++k ) {
		cond_[k] = ( 1.0f - kEma ) * cond_[k] + kEma * ( k < res.accepted ? 1.0f : 0.0f );
	}
	// Bonus token: the target's prediction just past the accepted prefix.
	if ( res.accepted == active ) {
		res.emitted[res.emitted_count++] = preds[active];
	}

	// 6. State repair. The next round expects exactly accepted + 1 rows consumed ([anchor, c1..c_accepted]),
	// which is the snapshot published after row `accepted` of the verify forward. A full accept already
	// leaves the live state there, so only a partial accept copies it in. No re-forward.
	if ( res.accepted < active ) {
		++prof_.commits;
		mc.commit_spec_row( slot, res.accepted );
	}
	mc.set_spec_snapshot( false );
	res.next_pos = pos + 1 + res.accepted;
	// Close the round: account the trailing segment and rewind the cursor.
	if ( profile_on_ ) {
		const double now = impl::now_ms();
		prof_.ms[prof_last_] += now - prof_t0_;
		prof_last_ = SpecProfile::DRAFT;
		prof_t0_ = now;
	}
	return res;
}

// Closes the segment that began at the previous mark, attributing it to the stage just finished, and
// arms the clock for `stage`. Only accumulates when profiling is on.
void DFlash2Runner::mark( SpecProfile::Stage stage ) {
	if ( !profile_on_ ) return;
	const double now = impl::now_ms();
	if ( stage != prof_last_ ) prof_.ms[prof_last_] += now - prof_t0_;
	prof_last_ = stage;
	prof_t0_ = now;
}

void DFlash2Runner::profile_dump() {
	static const char* names[SpecProfile::COUNT] = {
		"draft(backbone+head+merge)", "walk(gate+lattice)", "snapshot_on", "verify_fwd", "commit"
	};
	double total = 0.0;
	for ( int i = 0; i < SpecProfile::COUNT; ++i ) total += prof_.ms[i];
	const int n = prof_.rounds;
	std::printf( "\n--- speculative round profile: %d rounds, %d commits (%.0f%%) ---\n",
		n, prof_.commits, n ? 100.0 * prof_.commits / n : 0.0 );
	for ( int i = 0; i < SpecProfile::COUNT; ++i ) {
		std::printf( "  %-26s %8.2f ms total  %7.3f ms/round  %5.1f%%\n", names[i], prof_.ms[i],
			n ? prof_.ms[i] / n : 0.0, total > 0.0 ? 100.0 * prof_.ms[i] / total : 0.0 );
	}
	std::printf( "  %-26s %8.2f ms total  %7.3f ms/round\n", "TOTAL", total, n ? total / n : 0.0 );
	std::fflush( stdout );
	prof_ = SpecProfile{};
}

} // namespace engine::runtime
